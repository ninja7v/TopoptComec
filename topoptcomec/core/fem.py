# topoptcomec/core/fem.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Finite Element Method (FEM) class for topology optimization.

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import spsolve, cg, LinearOperator

from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Region, Support

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

#: Accepted linear solver identifiers.
SOLVERS = ("Direct", "Iterative", "Auto")
#: Accepted filter identifiers.
FILTERS = ("Sensitivity", "Density", "None")
#: Relative minimum stiffness (E_min = MIN_STIFFNESS_RATIO * E_max).
MIN_STIFFNESS_RATIO = 1e-9


class FEM:
    """
    Finite Element Method solver for topology optimization.

    Supports 2D (Q4) and 3D (H8) continuum elements with multiple load cases,
    SIMP-based material interpolation, and sensitivity/density filtering.

    The discrete equilibrium equation solved for each load case is:

    .. math::

        \\mathbf{K}(\\boldsymbol{\\rho})\\mathbf{u} = \\mathbf{f}

    with the assembled stiffness matrix:

    .. math::

        \\mathbf{K}(\\boldsymbol{\\rho}) =
        \\sum_{e=1}^{n_e} E_e(\\rho_e) \\mathbf{K}_e^0

    Parameters
    ----------
    grid : StructuredGrid
        Structured mesh definition.
    E : Sequence[float]
        Young's modulus of each material.
    nu : Sequence[float]
        Poisson's ratio of each material.
    penal : float
        SIMP penalization exponent.
    solver : str
        One of ``"Direct"``, ``"Iterative"``, ``"Auto"``.
    filter_type : str
        One of ``"Sensitivity"``, ``"Density"``, ``"None"``.
    filter_radius : float
        Filter radius in element units; ``<= 0`` disables filtering.
    """

    def __init__(
        self,
        grid: StructuredGrid,
        E: Sequence[float] = (1.0,),
        nu: Sequence[float] = (0.3,),
        penal: float = 3.0,
        solver: str = "Auto",
        filter_type: str = "Sensitivity",
        filter_radius: float = 0.0,
    ) -> None:
        if solver not in SOLVERS:
            raise ValueError(f"Unknown solver {solver!r}; expected one of {SOLVERS}")
        if filter_type not in FILTERS:
            raise ValueError(
                f"Unknown filter_type {filter_type!r}; expected one of {FILTERS}"
            )

        # Geometry and Grid Setup
        self.grid: StructuredGrid = grid
        self.nelx: int = grid.nelx
        self.nely: int = grid.nely
        self.nelz: int = grid.nelz
        self.is_3d: bool = grid.is_3d
        self.nel: int = grid.nel
        self.elemndof: int = grid.dofs_per_node
        self.dim_mul: int = grid.dofs_per_node
        self.eldof: int = 8 * (self.elemndof if self.is_3d else 1)
        self.ndof: int = grid.ndof

        # Materials and Optimization Params
        self.E_max: FloatArray = np.atleast_1d(np.asarray(E, dtype=np.float64))
        self.nb_mat: int = self.E_max.size
        # Relative minimum stiffness keeps the system conditioned for any
        # physical unit of E (C4: E_min proportional to E_max).
        self.E_min: FloatArray = MIN_STIFFNESS_RATIO * self.E_max
        self.nu: FloatArray = np.atleast_1d(np.asarray(nu, dtype=np.float64))
        self.penal: float = float(penal)
        self.solver_type: str = solver
        self.filter_type: str = filter_type
        self.filter_radius: float = float(filter_radius)

        # Pre-compute Constant Matrices (KE, DOF Maps, Filter)
        self.KE: FloatArray = self._get_lk_stiffness()
        self._KE_flat: FloatArray = self.KE.ravel(order="F")  # reused at every assembly
        self.edofMat, self.iK, self.jK = self._build_dof_map()
        self.H, self.Hs = self._build_filter()

        # State placeholders for BCs
        self.fixed_dofs: IntArray = np.array([], dtype=np.int64)
        self.free_dofs: IntArray = np.array([], dtype=np.int64)
        self.loads_in: list[Load] = []
        self.loads_out: list[Load] = []
        self.in_dofs: list[int] = []
        self.out_dofs: list[int] = []
        self.forces_i: FloatArray | None = None
        self.forces_o: FloatArray | None = None
        self._spring_dofs: IntArray = np.array([], dtype=np.int64)
        self._spring_vals: FloatArray = np.array([], dtype=np.float64)

    def setup_boundary_conditions(
        self,
        loads_in: Sequence[Load],
        loads_out: Sequence[Load] = (),
        supports: Sequence[Support] = (),
    ) -> None:
        """
        Build force vectors and fixed DOFs from typed boundary conditions.

        Parameters
        ----------
        loads_in : Sequence[Load]
            Actuation loads. One FEM load case per entry.
        loads_out : Sequence[Load]
            Output ports (compliant mechanisms). Each acts as a dummy/adjoint
            load and defines the desired output direction. Empty for rigid
            (compliance-minimization) problems.
        supports : Sequence[Support]
            Fixed nodes.
        """
        self.loads_in = list(loads_in)
        self.loads_out = list(loads_out)
        self.forces_i, self.in_dofs = self._assemble_loads(self.loads_in)
        self.forces_o, self.out_dofs = self._assemble_loads(self.loads_out)

        # Artificial springs at load locations (classic Sigmund formulation):
        # extra diagonal triplets appended at every assembly.
        spring_dofs: list[int] = []
        spring_vals: list[float] = []
        for dofs, loads in (
            (self.in_dofs, self.loads_in),
            (self.out_dofs, self.loads_out),
        ):
            for dof, load in zip(dofs, loads):
                if load.spring_stiffness > 0:
                    spring_dofs.append(dof)
                    spring_vals.append(load.spring_stiffness)
        self._spring_dofs: IntArray = np.asarray(spring_dofs, dtype=np.int64)
        self._spring_vals: FloatArray = np.asarray(spring_vals, dtype=np.float64)

        fixed: list[int] = []
        for sup in supports:
            nodes_to_fix = [self.grid.node_index(sup.x, sup.y, sup.z)]
            if sup.radius > 0:
                nodes_to_fix.extend(self._nodes_within_radius(sup))
            for node_idx in nodes_to_fix:
                node_dof = self.dim_mul * node_idx
                if sup.fix_x:
                    fixed.append(node_dof)
                if sup.fix_y:
                    fixed.append(node_dof + 1)
                if self.is_3d and sup.fix_z:
                    fixed.append(node_dof + 2)

        self.fixed_dofs = np.unique(np.asarray(fixed, dtype=np.int64))
        self.free_dofs = np.setdiff1d(
            np.arange(self.ndof, dtype=np.int64), self.fixed_dofs
        )

    def _nodes_within_radius(self, sup: Support) -> list[int]:
        """All node indices within ``sup.radius`` of the support center."""
        radius = float(sup.radius)
        x_range = range(
            max(0, int(sup.x - radius)), min(self.nelx + 1, int(sup.x + radius + 1))
        )
        y_range = range(
            max(0, int(sup.y - radius)), min(self.nely + 1, int(sup.y + radius + 1))
        )
        z_range = (
            range(
                max(0, int(sup.z - radius)),
                min(self.nelz + 1, int(sup.z + radius + 1)),
            )
            if self.is_3d
            else range(1)
        )
        nodes: list[int] = []
        for z in z_range:
            for x in x_range:
                for y in y_range:
                    dist_sq = (
                        (x - sup.x) ** 2
                        + (y - sup.y) ** 2
                        + ((z - sup.z) ** 2 if self.is_3d else 0)
                    )
                    if dist_sq <= radius**2:
                        nodes.append(self.grid.node_index(x, y, z))
        return nodes

    def _assemble_loads(self, loads: Sequence[Load]) -> tuple[FloatArray, list[int]]:
        """
        Assemble one force vector column per load.

        Returns
        -------
        tuple[FloatArray, list[int]]
            (force_matrix of shape (ndof, n_loads), loaded DOF indices).
        """
        f_vec = np.zeros((self.ndof, len(loads)), dtype=np.float64)
        dofs: list[int] = []
        for col, load in enumerate(loads):
            node = self.grid.node_index(load.x, load.y, load.z if self.is_3d else 0)
            dof = self.dim_mul * node + load.axis
            f_vec[dof, col] = load.signed_magnitude()
            dofs.append(dof)
        return f_vec, dofs

    def apply_regions(self, x: FloatArray, regions: Sequence[Region]) -> FloatArray:
        """
        Apply geometric constraints (Regions) to the density field.

        .. math::

            \\rho_e =
            \\begin{cases}
                \\rho_{\\min}, & e \\in \\Omega_{void} \\\\
                1, & e \\in \\Omega_{solid} \\\\
                \\rho_e, & \\text{otherwise}
            \\end{cases}

        Notes
        -----
        Coverage convention (kept identical to the GUI preview and historic
        presets): a region spans element indices ``[int(c - r), int(c + r))``
        per axis; spheres additionally require
        ``dist(element, center) <= r``.

        Parameters
        ----------
        x : FloatArray
            Current design variable vector (flat element ordering).
        regions : Sequence[Region]
            Geometric constraints.

        Returns
        -------
        FloatArray
            Modified density field with regions applied.
        """
        xPhys = x.copy()
        for region in regions:
            val = 1.0 if region.solid else 1e-6
            r = float(region.radius)
            z_range = (
                range(
                    max(0, int(region.z - r)),
                    min(self.nelz, int(region.z + r)),
                )
                if self.is_3d
                else range(1)
            )
            x_range = range(
                max(0, int(region.x - r)),
                min(self.nelx, int(region.x + r)),
            )
            y_range = range(
                max(0, int(region.y - r)),
                min(self.nely, int(region.y + r)),
            )
            for ez in z_range:
                for ex in x_range:
                    for ey in y_range:
                        if region.shape == "sphere":
                            dist_sq = (
                                (ex - region.x) ** 2
                                + (ey - region.y) ** 2
                                + ((ez - region.z) ** 2 if self.is_3d else 0)
                            )
                            if dist_sq > r**2:
                                continue
                        xPhys[self.grid.element_index(ex, ey, ez)] = val
        return xPhys

    def solve(self, xPhys: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Assemble stiffness matrix and solve for displacements.

        The effective stiffness per element follows SIMP interpolation. For
        several materials, contributions are summed:

        .. math::

            E_e = \\sum_{m=1}^{n_m}
            \\left(E_{\\min,m} + \\rho_{m,e}^{p}
            (E_{0,m} - E_{\\min,m})\\right)

        The finite element system is then solved on the free DOFs:

        .. math::

            \\mathbf{K}_{ff}\\mathbf{u}_f = \\mathbf{f}_f

        Parameters
        ----------
        xPhys : FloatArray
            Physical (filtered) element densities. Shape (nel,) or (nb_mat, nel).

        Returns
        -------
        Tuple[FloatArray, FloatArray]
            (ui, uo) - Displacement arrays for input and output loads.
        """
        E_eff = self._effective_stiffness(xPhys)
        sK = self._KE_flat[:, None] * E_eff

        # Element triplets + artificial spring triplets in a single assembly.
        K = coo_matrix(
            (
                np.concatenate([sK.ravel(order="F"), self._spring_vals]),
                (
                    np.concatenate([self.iK, self._spring_dofs]),
                    np.concatenate([self.jK, self._spring_dofs]),
                ),
            ),
            shape=(self.ndof, self.ndof),
        ).tocsc()

        # Solving
        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        ui = np.zeros((self.ndof, len(self.loads_in)), dtype=np.float64)
        uo = np.zeros((self.ndof, len(self.loads_out)), dtype=np.float64)
        self._solve_linear_system(K_free, self.forces_i, ui)
        self._solve_linear_system(K_free, self.forces_o, uo)

        return ui, uo

    def _effective_stiffness(self, xPhys: FloatArray) -> FloatArray:
        """SIMP effective stiffness per element, shape (nel,)."""
        xP = np.atleast_2d(np.asarray(xPhys, dtype=np.float64))
        return (
            self.E_min[:, None] + xP**self.penal * (self.E_max - self.E_min)[:, None]
        ).sum(axis=0)

    def compute_ce(self, ui: FloatArray, uo: FloatArray) -> FloatArray:
        """
        Compute element-wise energy terms used by the objective sensitivities.

        For rigid structures:

        .. math::

            c_e =
            \\sum_{i=1}^{n_i}
            (\\mathbf{u}_{e,i}^{in})^T
            \\mathbf{K}_e^0
            \\mathbf{u}_{e,i}^{in}

        For compliant mechanisms:

        .. math::

            c_e =
            \\sum_{i=1}^{n_i}\\sum_{o=1}^{n_o}
            (\\mathbf{u}_{e,i}^{in})^T
            \\mathbf{K}_e^0
            \\mathbf{u}_{e,o}^{out}

        In both expressions :math:`\\mathbf{K}_e^0` is the normalized element
        stiffness matrix. The SIMP stiffness factor is applied later when the
        objective or sensitivity is assembled.

        Parameters
        ----------
        ui : FloatArray
            Displacements due to input loads, shape (ndof, n_inputs).
        uo : FloatArray
            Displacements due to output (adjoint) loads, shape (ndof, n_outputs).

        Returns
        -------
        FloatArray
            Element energy values, shape (nel,).
        """
        ce_total = np.zeros(self.nel, dtype=np.float64)

        if not self.loads_out:  # Rigid structure (minimize compliance)
            for i_in in range(len(self.loads_in)):
                Ue = ui[self.edofMat, i_in]
                ce_total += np.sum((Ue @ self.KE) * Ue, axis=1)
        else:  # Compliant mechanism
            for i_in in range(len(self.loads_in)):
                Ue_in = ui[self.edofMat, i_in]
                for i_out in range(len(self.loads_out)):
                    Ue_out = uo[self.edofMat, i_out]
                    ce_total += np.sum((Ue_in @ self.KE) * Ue_out, axis=1)

        return ce_total

    def compute_sensitivities(
        self,
        xPhys: FloatArray,
        ui: FloatArray,
        uo: FloatArray,
        mat_idx: int = 0,
    ) -> tuple[FloatArray, FloatArray]:
        """
        Calculate sensitivity derivatives for optimization.

        For compliance minimization with SIMP interpolation, the density
        derivative has the standard form:

        .. math::

            \\frac{\\partial C}{\\partial \\rho_e}
            = -p\\rho_e^{p-1}(E_0 - E_{\\min})c_e

        For the compliant-mechanism objective (output displacement
        :math:`u_{out}` computed with the adjoint solve ``uo``), the exact
        derivative is:

        .. math::

            \\frac{\\partial u_{out}}{\\partial \\rho_e}
            = -p\\rho_e^{p-1}(E_0 - E_{\\min})c_e

        which is returned with a positive sign so that the OC update (a
        minimizer) maximizes the output displacement.

        Parameters
        ----------
        xPhys : FloatArray
            Physical element densities of material ``mat_idx``, shape (nel,).
        ui, uo : FloatArray
            Displacement solutions.
        mat_idx : int
            Material index used for the stiffness scale factor.

        Returns
        -------
        Tuple[FloatArray, FloatArray]
            (dc, dv) - Objective sensitivity and volume sensitivity.
        """
        ce_total = self.compute_ce(ui, uo)
        dE = self.E_max[mat_idx] - self.E_min[mat_idx]
        scale = self.penal * (xPhys ** (self.penal - 1)) * dE

        if not self.loads_out:  # Rigid structure (minimize compliance)
            dc = -scale * ce_total
        else:  # Compliant mechanism (maximize output displacement)
            dc = scale * ce_total

        # Volume Sensitivity (dv)
        dv: FloatArray = np.ones(self.nel, dtype=np.float64)

        # Filtering
        return self._apply_filter(xPhys, dc, dv)

    def compute_objective(
        self, xPhys: FloatArray, ui: FloatArray, uo: FloatArray
    ) -> float:
        """
        Compute objective function value.

        Rigid designs minimize compliance:

        .. math::

            C(\\boldsymbol{\\rho}) =
            \\sum_{e=1}^{n_e} E_e(\\rho_e)c_e

        Compliant mechanisms report the signed output displacement under the
        input loads, averaged over load-case/output pairs (positive when the
        output moves in the intended direction):

        .. math::

            J(\\boldsymbol{\\rho}) =
            \\frac{1}{n_i n_o}\\sum_{i=1}^{n_i}\\sum_{o=1}^{n_o}
            s_o\\, u_{i}[d_o]

        Parameters
        ----------
        xPhys, ui, uo : FloatArray
            As per solve() and compute_ce().

        Returns
        -------
        float
            Objective function value.
        """
        if not self.loads_out:
            # Rigid structure (minimize compliance)
            E_eff = self._effective_stiffness(xPhys)
            ce_total = self.compute_ce(ui, uo)
            return float((E_eff * ce_total).sum())

        # Compliant mechanism: signed output displacement under input loads.
        if ui.shape[1] == 0:
            return 0.0
        vals = [
            load.sign * ui[dof, i_in]
            for dof, load in zip(self.out_dofs, self.loads_out)
            for i_in in range(ui.shape[1])
        ]
        return float(np.mean(vals))

    # --- Internal Helper Methods ---

    def _solve_linear_system(
        self,
        K_free: csc_matrix,
        F: FloatArray | None,
        U_full: FloatArray,
    ) -> None:
        """
        Solve the linear system Ku = F using direct or iterative methods.

        Boundary conditions reduce the full system to free DOFs:

        .. math::

            \\mathbf{K}_{ff}\\mathbf{U}_f = \\mathbf{F}_f,
            \\qquad \\mathbf{U}_c = \\mathbf{0}

        Parameters
        ----------
        K_free : csc_matrix
            Reduced stiffness matrix (free DOFs only).
        F : FloatArray | None
            Force matrix, shape (ndof, n_cases).
        U_full : FloatArray
            Displacement matrix to fill (modified in place).
        """
        if F is None or F.shape[1] == 0:
            return

        use_direct = self.solver_type == "Direct" or (
            self.solver_type == "Auto" and K_free.shape[0] < 10000
        )

        if use_direct:
            sol = spsolve(K_free, F[self.free_dofs, :])
            if sol.ndim == 1:
                U_full[self.free_dofs, 0] = sol
            else:
                U_full[self.free_dofs, :] = sol
        else:
            # Iterative Solver (CG with Jacobi Preconditioner)
            D_inv = 1.0 / K_free.diagonal()
            M = LinearOperator(K_free.shape, lambda x: D_inv * x)
            for i in range(F.shape[1]):
                if np.any(F[self.free_dofs, i]):
                    u_sol, info = cg(
                        K_free,
                        F[self.free_dofs, i],
                        M=M,
                        rtol=1e-8,
                        maxiter=K_free.shape[0],
                    )
                    if info != 0 and self.solver_type == "Auto":
                        # Fallback
                        try:
                            U_full[self.free_dofs, i] = spsolve(
                                K_free, F[self.free_dofs, i]
                            )
                        except Exception as e:
                            print(
                                f"Direct solver failed: {e}. Using partial CG solution."
                            )
                            U_full[self.free_dofs, i] = u_sol
                    else:
                        U_full[self.free_dofs, i] = u_sol

    def _apply_filter(
        self, x: FloatArray, dc: FloatArray, dv: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """
        Apply sensitivity or density filtering to sensitivities.

        Sensitivity filtering uses:

        .. math::

            \\widehat{\\frac{\\partial C}{\\partial x_i}} =
            \\frac{1}{\\max(\\gamma, x_i)\\sum_j H_{ij}}
            \\sum_j H_{ij}x_j\\frac{\\partial C}{\\partial x_j}

        Density filtering applies the normalized filter directly:

        .. math::

            \\widehat{x}_i = \\frac{\\sum_j H_{ij}x_j}{\\sum_j H_{ij}}

        Parameters
        ----------
        x : FloatArray
            Design variables.
        dc : FloatArray
            Objective sensitivities.
        dv : FloatArray
            Volume sensitivities.

        Returns
        -------
        tuple[FloatArray, FloatArray]
            (filtered_dc, filtered_dv) - Filtered sensitivity arrays.
        """
        if self.filter_type == "Sensitivity":
            Hx_dc = self.H @ (x * dc)
            dc = np.asarray(Hx_dc, dtype=np.float64) / self.Hs / np.maximum(0.001, x)

        elif self.filter_type == "Density":
            dc = np.asarray(self.H @ (dc / self.Hs), dtype=np.float64)
            dv = np.asarray(self.H @ (dv / self.Hs), dtype=np.float64)

        return dc, dv

    def update_xPhys(self, x: FloatArray) -> FloatArray:
        """
        Calculate physical density from design variables using filter.

        .. math::

            \\rho_i =
            \\frac{\\sum_j H_{ij}x_j}{\\sum_j H_{ij}}

        Parameters
        ----------
        x : FloatArray
            Design variable vector.

        Returns
        -------
        FloatArray
            Filtered physical density.
        """
        if self.filter_type == "Density":
            return (self.H @ x).ravel() / np.asarray(self.Hs).ravel()
        return x

    def _get_lk_stiffness(self) -> FloatArray:
        """
        Compute the element stiffness matrix for Q4 (2D) or H8 (3D) elements.

        .. math::

            \\mathbf{K}_e^0 =
            \\int_{\\Omega_e} \\mathbf{B}^T\\mathbf{D}\\mathbf{B}\\,d\\Omega

        Returns
        -------
        FloatArray
            Element stiffness matrix, shape (8, 8) for 2D or (24, 24) for 3D.
        """
        E, nu = 1.0, float(self.nu[0])  # Normalized E for KE
        if self.is_3d:
            A = np.array(
                [
                    [32, 6, -8, 6, -6, 4, 3, -6, -10, 3, -3, -3, -4, -8],
                    [-48, 0, 0, -24, 24, 0, 0, 0, 12, -12, 0, 12, 12, 12],
                ],
                dtype=np.float64,
            )
            k = 1 / 72 * (A.T @ np.array([1.0, nu], dtype=np.float64))
            K_blocks = self._build_3d_blocks(k)
            # The classic top3d coefficient table corresponds to a 2x2x2
            # element; the 0.5 factor rescales it to the unit cube so 2D and
            # 3D elements share the same unit-element convention (verified
            # against Gauss-quadrature integration in tests/test_numerics.py).
            return (0.5 * E / ((nu + 1) * (1 - 2 * nu)) * K_blocks).astype(np.float64)
        else:
            k = np.array(
                [
                    1 / 2 - nu / 6,
                    1 / 8 + nu / 8,
                    -1 / 4 - nu / 12,
                    -1 / 8 + 3 * nu / 8,
                    -1 / 4 + nu / 12,
                    -1 / 8 - nu / 8,
                    nu / 6,
                    1 / 8 - 3 * nu / 8,
                ],
                dtype=np.float64,
            )
            return (
                E
                / (1 - nu**2)
                * np.array(
                    [
                        [k[0], k[1], k[2], k[3], k[4], k[5], k[6], k[7]],
                        [k[1], k[0], k[7], k[6], k[5], k[4], k[3], k[2]],
                        [k[2], k[7], k[0], k[5], k[6], k[3], k[4], k[1]],
                        [k[3], k[6], k[5], k[0], k[7], k[2], k[1], k[4]],
                        [k[4], k[5], k[6], k[7], k[0], k[1], k[2], k[3]],
                        [k[5], k[4], k[3], k[2], k[1], k[0], k[7], k[6]],
                        [k[6], k[3], k[4], k[1], k[2], k[7], k[0], k[5]],
                        [k[7], k[2], k[1], k[4], k[3], k[6], k[5], k[0]],
                    ],
                    dtype=np.float64,
                )
            )

    def _build_3d_blocks(self, k: FloatArray) -> FloatArray:
        """
        Build the 3D stiffness matrix blocks from coefficients.

        Parameters
        ----------
        k : FloatArray
            Stiffness coefficients vector.

        Returns
        -------
        FloatArray
            Assembled 3D element stiffness matrix (24x24).
        """
        K1 = np.array(
            [
                [k[0], k[1], k[1], k[2], k[4], k[4]],
                [k[1], k[0], k[1], k[3], k[5], k[6]],
                [k[1], k[1], k[0], k[3], k[6], k[5]],
                [k[2], k[3], k[3], k[0], k[7], k[7]],
                [k[4], k[5], k[6], k[7], k[0], k[1]],
                [k[4], k[6], k[5], k[7], k[1], k[0]],
            ],
            dtype=np.float64,
        )
        K2 = np.array(
            [
                [k[8], k[7], k[11], k[5], k[3], k[6]],
                [k[7], k[8], k[11], k[4], k[2], k[4]],
                [k[9], k[9], k[12], k[6], k[3], k[5]],
                [k[5], k[4], k[10], k[8], k[1], k[9]],
                [k[3], k[2], k[4], k[1], k[8], k[11]],
                [k[10], k[3], k[5], k[11], k[9], k[12]],
            ],
            dtype=np.float64,
        )
        K3 = np.array(
            [
                [k[5], k[6], k[3], k[8], k[11], k[7]],
                [k[6], k[5], k[3], k[9], k[12], k[9]],
                [k[4], k[4], k[2], k[7], k[11], k[8]],
                [k[8], k[9], k[1], k[5], k[10], k[4]],
                [k[11], k[12], k[9], k[10], k[5], k[3]],
                [k[1], k[11], k[8], k[3], k[4], k[2]],
            ],
            dtype=np.float64,
        )
        K4 = np.array(
            [
                [k[13], k[10], k[10], k[12], k[9], k[9]],
                [k[10], k[13], k[10], k[11], k[8], k[7]],
                [k[10], k[10], k[13], k[11], k[7], k[8]],
                [k[12], k[11], k[11], k[13], k[6], k[6]],
                [k[9], k[8], k[7], k[6], k[13], k[10]],
                [k[9], k[7], k[8], k[6], k[10], k[13]],
            ],
            dtype=np.float64,
        )
        K5 = np.array(
            [
                [k[0], k[1], k[7], k[2], k[4], k[3]],
                [k[1], k[0], k[7], k[3], k[5], k[10]],
                [k[7], k[7], k[0], k[4], k[10], k[5]],
                [k[2], k[3], k[4], k[0], k[7], k[1]],
                [k[4], k[5], k[10], k[7], k[0], k[7]],
                [k[3], k[10], k[5], k[1], k[7], k[0]],
            ],
            dtype=np.float64,
        )
        K6 = np.array(
            [
                [k[13], k[10], k[6], k[12], k[9], k[11]],
                [k[10], k[13], k[6], k[11], k[8], k[1]],
                [k[6], k[6], k[13], k[9], k[1], k[8]],
                [k[12], k[11], k[9], k[13], k[6], k[10]],
                [k[9], k[8], k[1], k[6], k[13], k[6]],
                [k[11], k[1], k[8], k[10], k[6], k[13]],
            ],
            dtype=np.float64,
        )
        return np.block(
            [
                [K1, K2, K3, K4],
                [K2.T, K5, K6, K3.T],
                [K3.T, K6, K5.T, K2.T],
                [K4, K3, K2, K1.T],
            ]
        )

    def _build_dof_map(self) -> tuple[IntArray, IntArray, IntArray]:
        """
        Build the DOF mapping arrays for element connectivity.

        Returns
        -------
        tuple[IntArray, IntArray, IntArray]
            (edofMat, iK, jK) - Element DOF matrix and sparse matrix indices.
        """
        size = 8 * (self.elemndof if self.is_3d else 1)
        ex, ey, ez = self.grid.element_coordinates()

        if self.is_3d:
            n1 = ez * (self.nelx + 1) * (self.nely + 1) + ex * (self.nely + 1) + ey

            # Bottom + Top face nodes
            off = (self.nelx + 1) * (self.nely + 1)
            base_nodes = np.array(
                [
                    1,
                    self.nely + 2,
                    self.nely + 1,
                    0,
                    1 + off,
                    self.nely + 2 + off,
                    self.nely + 1 + off,
                    off,
                ],
                dtype=np.int64,
            )
        else:
            n1 = ex * (self.nely + 1) + ey
            base_nodes = np.array([1, self.nely + 2, self.nely + 1, 0], dtype=np.int64)

        # 2. Map node combinations using broadcasting
        node_mat = n1[:, None] + base_nodes[None, :]
        dof_offsets = np.arange(self.dim_mul, dtype=np.int64)

        # Multiply by DOF multiplier and apply offsets, returning a flattened structure
        edofMat = (
            (node_mat[:, :, None] * self.dim_mul + dof_offsets[None, None, :])
            .reshape(self.nel, -1)
            .astype(np.int64)
        )

        # Fast equivalent of np.kron for DOF mappings
        iK = edofMat.repeat(size, axis=0).flatten()
        jK = edofMat.repeat(size, axis=1).flatten()

        return edofMat, iK, jK

    def _build_filter(self) -> tuple[csc_matrix, FloatArray]:
        """
        Build the sensitivity/density filter matrix.

        .. math::

            H_{ij} = \\max(0, r_{\\min} - \\lVert \\mathbf{x}_i -
            \\mathbf{x}_j \\rVert)

        Returns
        -------
        tuple
            (H, Hs) - Sparse filter matrix and row sums for normalization.
        """
        if self.filter_radius <= 0 or self.filter_type == "None":
            # No error raised since it is equivalent to no filtering.
            H = eye(self.nel, format="csc", dtype=np.float64)
            Hs = np.ones(self.nel, dtype=np.float64)
            return H, Hs

        ex, ey, ez = self.grid.element_coordinates()
        el = np.arange(self.nel)

        # Pre-calculate the localized relative neighbor meshgrid
        r = int(np.ceil(self.filter_radius))
        if self.is_3d:
            dx, dy, dz = np.meshgrid(
                np.arange(-r, r + 1),
                np.arange(-r, r + 1),
                np.arange(-r, r + 1),
                indexing="ij",
            )
            dist = np.sqrt(dx**2 + dy**2 + dz**2)
            valid = dist < self.filter_radius
            dx, dy, dz = dx[valid], dy[valid], dz[valid]
        else:
            dx, dy = np.meshgrid(
                np.arange(-r, r + 1), np.arange(-r, r + 1), indexing="ij"
            )
            dist = np.sqrt(dx**2 + dy**2)
            valid = dist < self.filter_radius
            dx, dy = dx[valid], dy[valid]

        vals = self.filter_radius - dist[valid]
        iH, jH, sH = [], [], []
        for n in range(len(dx)):
            nx = ex + dx[n]
            ny = ey + dy[n]

            if self.is_3d:
                nz = ez + dz[n]
                mask = (
                    (nx >= 0)
                    & (nx < self.nelx)
                    & (ny >= 0)
                    & (ny < self.nely)
                    & (nz >= 0)
                    & (nz < self.nelz)
                )
                el2 = (
                    nz[mask] * (self.nelx * self.nely) + nx[mask] * self.nely + ny[mask]
                )
            else:
                mask = (nx >= 0) & (nx < self.nelx) & (ny >= 0) & (ny < self.nely)
                el2 = nx[mask] * self.nely + ny[mask]

            iH.append(el[mask])
            jH.append(el2)
            sH.append(np.full(np.count_nonzero(mask), vals[n]))

        # Concatenate chunks to build the sparse matrix
        H = coo_matrix(
            (np.concatenate(sH), (np.concatenate(iH), np.concatenate(jH))),
            shape=(self.nel, self.nel),
        ).tocsc()

        return H, np.asarray(H.sum(1)).ravel()
