# topoptcomec/core/fem.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Finite Element Method (FEM) class for topology optimization.

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csc_matrix, eye
from scipy.sparse.linalg import LinearOperator, cg, splu

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
#: Auto solver: estimated fill-in budget of the direct factorisation
#: (``n_free * bandwidth``). Above it the iterative solver is used, which keeps
#: memory bounded for large meshes.
AUTO_DIRECT_WORK_2D = 2e7
AUTO_DIRECT_WORK_3D = 2e6
#: Diagonal regularisation applied to every free stiffness matrix.
#: It pins any unconstrained rigid-body modes without changing the strain of
#: a self-equilibrated load case (the load vector is orthogonal to them), and
#: keeps the solution deterministic when supports fix only some directions.
RIGID_BODY_REGULARISATION = 1e-12


class FEM:
    """
    Finite Element Method solver for topology optimization.

    Supports 2D (Q4) and 3D (H8) continuum elements with simultaneous point
    loads, SIMP-based material interpolation, and sensitivity/density filtering.

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
        self._validate_materials()
        self.penal: float = float(penal)
        self.solver_type: str = solver
        self.filter_type: str = filter_type
        self.filter_radius: float = float(filter_radius)

        # Pre-compute Constant Matrices (KE, DOF Maps, Filter).
        # One element matrix per material: the Poisson ratio may differ
        # between materials, so a single shared matrix is not physical.
        self.KE_mats: FloatArray = np.stack(
            [self._get_lk_stiffness(float(nu_m)) for nu_m in self.nu]
        )
        #: Element stiffness of the first material (historic attribute).
        self.KE: FloatArray = self.KE_mats[0]
        self._KE_flat_mats: FloatArray = self.KE_mats.reshape(
            self.nb_mat, -1
        )  # reused at every assembly
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
        self._no_support_warned: bool = False

    def _validate_materials(self) -> None:
        """Reject material constants that would produce an invalid system."""
        if self.nu.size != self.nb_mat:
            raise ValueError(
                f"E and nu must have the same length (got {self.nb_mat} and "
                f"{self.nu.size})."
            )
        if not np.all(np.isfinite(self.E_max)) or np.any(self.E_max <= 0.0):
            raise ValueError("Young's modulus must be finite and strictly positive.")
        if (
            not np.all(np.isfinite(self.nu))
            or np.any(self.nu < 0.0)
            or np.any(self.nu >= 0.5)
        ):
            raise ValueError(
                "Poisson's ratio must satisfy 0 <= nu < 0.5 for every material."
            )

    def _validate_node(self, x: int, y: int, z: int, what: str) -> None:
        """Reject coordinates that do not address a node of the grid."""
        if not (0 <= x <= self.nelx and 0 <= y <= self.nely):
            raise ValueError(
                f"{what} at ({x}, {y}, {z}) is outside the grid "
                f"[0, {self.nelx}] x [0, {self.nely}]."
            )
        if self.is_3d and not 0 <= z <= self.nelz:
            raise ValueError(
                f"{what} at ({x}, {y}, {z}) is outside the grid "
                f"[0, {self.nelx}] x [0, {self.nely}] x [0, {self.nelz}]."
            )

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
            Actuation loads applied simultaneously in one load case.
        loads_out : Sequence[Load]
            Output ports (compliant mechanisms). Each acts as a dummy/adjoint
            direction with equal weight in the mean output objective. Empty
            for rigid (compliance-minimization) problems.
        supports : Sequence[Support]
            Fixed nodes.
        """
        self.loads_in = list(loads_in)
        self.loads_out = list(loads_out)
        for load in self.loads_in + self.loads_out:
            self._validate_node(load.x, load.y, load.z, "Force")
            if not 0 <= load.axis < self.dim_mul:
                raise ValueError(
                    f"Force axis {load.axis} is invalid for a "
                    f"{'3D' if self.is_3d else '2D'} problem."
                )
        self.forces_i, self.in_dofs = self._assemble_loads(self.loads_in)
        self.forces_o, self.out_dofs = self._assemble_loads(self.loads_out)
        if self.loads_out:
            # The output objective is the mean signed port displacement.
            # Output magnitudes still define port spring stiffnesses below.
            self.forces_o.fill(0.0)
            weight = 1.0 / len(self.loads_out)
            for col, (dof, load) in enumerate(zip(self.out_dofs, self.loads_out)):
                self.forces_o[dof, col] = load.sign * weight

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
            self._validate_node(sup.x, sup.y, sup.z, "Support")
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
        if (
            self.fixed_dofs.size == 0
            and (self.loads_in or self.loads_out)
            and not self._no_support_warned
        ):
            self._no_support_warned = True
            warnings.warn(
                "No support is defined: the model has rigid-body modes. The "
                "drift is pinned by a tiny diagonal regularisation, but the "
                "load case is only meaningful when the applied forces are "
                "self-equilibrated.",
                RuntimeWarning,
                stacklevel=2,
            )

    def region_element_mask(self, region: Region) -> npt.NDArray[np.bool_]:
        """
        Boolean element mask covered by a region.

        Uses the same coverage convention as :meth:`apply_regions`: element
        indices ``[int(c - r), int(c + r))`` per axis, and spheres require the
        element coordinate to lie within ``r`` of the centre.

        Parameters
        ----------
        region : Region
            Geometric region.

        Returns
        -------
        npt.NDArray[np.bool_]
            Mask of shape ``(nel,)`` in flat element ordering.
        """
        r = float(region.radius)
        z_lo = max(0, int(region.z - r)) if self.is_3d else 0
        z_hi = min(self.nelz, int(region.z + r)) if self.is_3d else 1
        x_lo, x_hi = max(0, int(region.x - r)), min(self.nelx, int(region.x + r))
        y_lo, y_hi = max(0, int(region.y - r)), min(self.nely, int(region.y + r))
        mask = np.zeros(self.nel, dtype=bool)
        if x_hi <= x_lo or y_hi <= y_lo or z_hi <= z_lo:
            return mask
        ez, ex, ey = np.meshgrid(
            np.arange(z_lo, z_hi),
            np.arange(x_lo, x_hi),
            np.arange(y_lo, y_hi),
            indexing="ij",
        )
        if region.shape == "sphere":
            dist_sq = (
                (ex - region.x) ** 2
                + (ey - region.y) ** 2
                + ((ez - region.z) ** 2 if self.is_3d else 0)
            )
            inside = dist_sq <= r**2
            ez, ex, ey = ez[inside], ex[inside], ey[inside]
        idx = ez.ravel() * (self.nelx * self.nely) + ex.ravel() * self.nely + ey.ravel()
        mask[idx] = True
        return mask

    def region_masks(
        self, regions: Sequence[Region], mat_idx: int = 0
    ) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
        """
        Passive-solid and passive-void masks for one material slice.

        Parameters
        ----------
        regions : Sequence[Region]
            Geometric constraints.
        mat_idx : int
            Material slice being processed (0-based).

        Returns
        -------
        tuple of masks
            ``(solid, void)`` boolean arrays of shape ``(nel,)``. Both are
            ``False`` where the material is designable. A void region always
            wins over a solid one when the two overlap.
        """
        solid = np.zeros(self.nel, dtype=bool)
        void = np.zeros(self.nel, dtype=bool)
        for region in regions:
            mask = self.region_element_mask(region)
            if region.material == mat_idx:
                solid |= mask
            else:
                void |= mask
        solid &= ~void
        return solid, void

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

    def apply_regions(
        self, x: FloatArray, regions: Sequence[Region], mat_idx: int = 0
    ) -> FloatArray:
        """
        Force geometric regions to a target density in the physical field.

        For multi-material runs each call operates on a single material
        slice (``mat_idx``): a region whose ``material`` matches ``mat_idx``
        is forced to solid (1.0), any region targeting a *different* material
        or void (``-1``) is forced to void (1e-6) within this slice.

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
        mat_idx : int
            Index of the material slice being processed (0-based).

        Returns
        -------
        FloatArray
            Modified density field with regions applied.
        """
        xPhys = x.copy()
        solid, void = self.region_masks(regions, mat_idx)
        xPhys[solid] = 1.0
        xPhys[void] = 1e-6
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
        E_eff = self._material_stiffness(xPhys)  # (nb_mat, nel)
        sK = np.zeros((self.KE.size, self.nel), dtype=np.float64)
        for m in range(self.nb_mat):
            sK += self._KE_flat_mats[m][:, None] * E_eff[m]

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

        # Solving: one factorisation / Krylov solve for all load cases.
        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        if K_free.shape[0] > 0:
            # Pin any unconstrained rigid-body modes so the factorisation is
            # deterministic across platforms. Supports can fix only some
            # directions (e.g. a roller), so a non-empty fixed_dofs array does
            # not guarantee a non-singular system. The load vector is
            # orthogonal to the rigid modes for a self-equilibrated case, so
            # the strain is unchanged; for well-constrained systems the added
            # diagonal is negligible.
            eps = RIGID_BODY_REGULARISATION * float(
                np.max(np.abs(K_free.diagonal())) or 1.0
            )
            K_free = K_free + eps * eye(K_free.shape[0], format="csc")

        ui = np.zeros(self.ndof, dtype=np.float64)
        uo = np.zeros(self.ndof, dtype=np.float64)
        rhs = self._load_case_matrix()
        if rhs.shape[1] > 0:
            solutions = self._solve_all(K_free, rhs)
            ui[self.free_dofs] = solutions[:, 0]
            if solutions.shape[1] > 1:
                uo[self.free_dofs] = solutions[:, 1]

        return ui, uo

    def _load_case_matrix(self) -> FloatArray:
        """Simultaneous input load and (optional) adjoint output load.

        Returns
        -------
        FloatArray
            Matrix of shape ``(n_free, k)`` with the free-DOF forces. The first
            column is the physical input load; the second column, when present,
            is the normalized adjoint (output) load.
        """
        columns: list[FloatArray] = []
        for forces in (self.forces_i, self.forces_o):
            if forces is None or forces.shape[1] == 0:
                continue
            columns.append(forces[self.free_dofs].sum(axis=1))
        if not columns:
            return np.zeros((self.free_dofs.size, 0), dtype=np.float64)
        return np.column_stack(columns)

    def _material_stiffness(self, xPhys: FloatArray) -> FloatArray:
        """
        SIMP effective stiffness per element and material, shape (nb_mat, nel).
        """
        xP = np.atleast_2d(np.asarray(xPhys, dtype=np.float64))
        if xP.shape[0] != self.nb_mat:
            raise ValueError(
                f"Expected {self.nb_mat} density slice(s), got {xP.shape[0]}."
            )
        return self.E_min[:, None] + xP**self.penal * (self.E_max - self.E_min)[:, None]

    def _effective_stiffness(self, xPhys: FloatArray) -> FloatArray:
        """Summed SIMP effective stiffness per element, shape (nel,)."""
        return self._material_stiffness(xPhys).sum(axis=0)

    def compute_ce(
        self, ui: FloatArray, uo: FloatArray, mat_idx: int = 0
    ) -> FloatArray:
        """
        Compute element-wise energy terms used by the objective sensitivities.

        For rigid structures:

        .. math::

            c_e =
            (\\mathbf{u}_{e}^{in})^T
            \\mathbf{K}_{e,m}^0
            \\mathbf{u}_{e}^{in}

        For compliant mechanisms:

        .. math::

            c_e =
            (\\mathbf{u}_{e}^{in})^T
            \\mathbf{K}_{e,m}^0
            \\mathbf{u}_{e}^{out}

        In both expressions :math:`\\mathbf{K}_{e,m}^0` is the normalized
        element stiffness matrix of material ``mat_idx`` (it depends on that
        material's Poisson ratio). The SIMP stiffness factor is applied later
        when the objective or sensitivity is assembled.

        Parameters
        ----------
        ui : FloatArray
            Displacements due to the simultaneous input loads, shape (ndof,).
        uo : FloatArray
            Displacements due to the normalized output adjoint, shape (ndof,).
        mat_idx : int
            Material whose element stiffness is used.

        Returns
        -------
        FloatArray
            Element energy values, shape (nel,).
        """
        KE = self.KE_mats[mat_idx]
        if not self.loads_out:  # Rigid structure (minimize compliance)
            Ue = ui[self.edofMat]
            return np.sum((Ue @ KE) * Ue, axis=1)
        # Compliant mechanism
        Ue_in = ui[self.edofMat]
        Ue_out = uo[self.edofMat]
        return np.sum((Ue_in @ KE) * Ue_out, axis=1)

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
        ce_total = self.compute_ce(ui, uo, mat_idx)
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

            C(\\boldsymbol{\\rho}) = \\mathbf{f}^T\\mathbf{u}

        Compliant mechanisms report the signed output displacement under the
        combined input load, averaged over output ports (positive when the
        output moves in the intended direction):

        .. math::

            J(\\boldsymbol{\\rho}) =
            \\frac{1}{n_o}\\sum_{o=1}^{n_o}
            s_o\\, u[d_o]

        where ``u`` is the single combined-load displacement field.

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
            if self.forces_i is None:
                return 0.0
            return float(self.forces_i.sum(axis=1) @ ui)

        # Compliant mechanism: signed output displacement under input loads.
        if not self.in_dofs:
            return 0.0
        vals = [load.sign * ui[dof] for dof, load in zip(self.out_dofs, self.loads_out)]
        return float(np.mean(vals)) if vals else 0.0

    # --- Internal Helper Methods ---

    def _use_direct_solver(self, size: int) -> bool:
        """Whether the direct factorisation should be used for this system.

        Sparse direct factorisation wins as soon as two load cases share one
        factorisation, and Q4 (2D) meshes have a small bandwidth. The decision
        is based on the estimated fill-in ``size * bandwidth`` so that large
        meshes do not exhaust memory; 3D factors fill in much faster than 2D
        ones, hence the smaller budget.

        Parameters
        ----------
        size : int
            Number of free degrees of freedom.

        Returns
        -------
        bool
            ``True`` to use the direct solver.
        """
        if self.solver_type == "Direct":
            return True
        if self.solver_type == "Iterative":
            return False
        bandwidth = 2 * (self.nely + 1)
        limit = AUTO_DIRECT_WORK_2D
        if self.is_3d:
            bandwidth *= self.nelx + 1
            limit = AUTO_DIRECT_WORK_3D
        return float(size) * float(bandwidth) <= limit

    def _solve_all(self, K_free: csc_matrix, F: FloatArray) -> FloatArray:
        """Solve ``K_free x = F`` for every column of ``F``.

        Parameters
        ----------
        K_free : csc_matrix
            Reduced stiffness matrix (free DOFs only).
        F : FloatArray
            Right-hand sides, shape ``(n_free, n_cases)``.

        Returns
        -------
        FloatArray
            Solutions, shape ``(n_free, n_cases)``.

        Raises
        ------
        RuntimeError
            If the iterative solver does not converge in ``Iterative`` mode.
        """
        size, n_cases = F.shape
        if n_cases == 0:
            return np.zeros((size, 0), dtype=np.float64)

        if self._use_direct_solver(size):
            try:
                factors = splu(K_free.tocsc())
                return np.asarray(factors.solve(F), dtype=np.float64).reshape(
                    size, n_cases
                )
            except RuntimeError as error:  # singular factorisation
                if self.solver_type == "Direct":
                    raise RuntimeError(
                        f"Direct solver failed (singular system): {error}"
                    ) from error
                warnings.warn(
                    f"Direct solver failed ({error}); falling back to the "
                    "iterative solver.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        # Iterative solver (CG with a Jacobi preconditioner).
        diagonal = K_free.diagonal()
        if np.any(diagonal == 0):
            diagonal = np.where(diagonal == 0, 1.0, diagonal)
        D_inv = 1.0 / diagonal
        M = LinearOperator(K_free.shape, lambda x: D_inv * x)
        solutions = np.zeros((size, n_cases), dtype=np.float64)
        for column in range(n_cases):
            u_sol, info = cg(
                K_free,
                F[:, column],
                M=M,
                rtol=1e-10,
                maxiter=max(1000, size),
            )
            if info != 0:
                message = (
                    f"Iterative solver did not converge (info={info}) after "
                    f"{size} iterations."
                )
                if self.solver_type == "Iterative":
                    raise RuntimeError(message)
                warnings.warn(
                    f"{message} Falling back to the direct solver.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                factors = splu(K_free.tocsc())
                return np.asarray(factors.solve(F), dtype=np.float64).reshape(
                    size, n_cases
                )
            solutions[:, column] = u_sol
        return solutions

    def _solve_linear_system(
        self,
        K_free: csc_matrix,
        F: FloatArray | None,
        U_full: FloatArray,
    ) -> None:
        """
        Solve the reduced system and write the result into ``U_full``.

        Parameters
        ----------
        K_free : csc_matrix
            Reduced stiffness matrix (free DOFs only).
        F : FloatArray | None
            Force-component matrix, shape (ndof, n_components). Columns are
            summed into one simultaneous load vector.
        U_full : FloatArray
            Displacement vector to fill, shape (ndof,), modified in place.
        """
        if F is None or F.shape[1] == 0:
            return
        F_sum = F[self.free_dofs].sum(axis=1)[:, np.newaxis]
        U_full[self.free_dofs] = self._solve_all(K_free, F_sum)[:, 0]

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

    def _get_lk_stiffness(self, nu: float | None = None) -> FloatArray:
        """
        Compute the element stiffness matrix for Q4 (2D) or H8 (3D) elements.

        .. math::

            \\mathbf{K}_e^0 =
            \\int_{\\Omega_e} \\mathbf{B}^T\\mathbf{D}\\mathbf{B}\\,d\\Omega

        Parameters
        ----------
        nu : float, optional
            Poisson's ratio of the material. Defaults to the first material.

        Returns
        -------
        FloatArray
            Element stiffness matrix, shape (8, 8) for 2D or (24, 24) for 3D.
        """
        E, nu = 1.0, float(self.nu[0] if nu is None else nu)  # Normalized E
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
