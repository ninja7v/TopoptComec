# app/core/fem.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Finite Element Method (FEM) class for topology optimization.

from __future__ import annotations
from collections.abc import Sequence
import numpy as np
import numpy.typing as npt
from scipy.sparse import coo_matrix, csc_matrix
from scipy.sparse.linalg import spsolve, cg, LinearOperator

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


class FEM:
    def __init__(self, Dimensions: dict, Materials: dict, Optimizer: dict) -> None:
        # Geometry and Grid Setup
        self.nelxyz: Sequence[int] = Dimensions.get("nelxyz", [1, 1, 1])
        self.nelx, self.nely, self.nelz = (int(v) for v in self.nelxyz)
        self.is_3d: bool = self.nelz > 0
        self.nel: int = self.nelx * self.nely * (self.nelz if self.is_3d else 1)
        self.elemndof: int = 3 if self.is_3d else 2
        self.dim_mul: int = self.elemndof
        self.eldof: int = 8 * (self.elemndof if self.is_3d else 1)
        self.ndof: int = (
            self.dim_mul
            * (self.nelx + 1)
            * (self.nely + 1)
            * ((self.nelz + 1) if self.is_3d else 1)
        )

        # Materials and Optimization Params
        self.E_max: FloatArray = np.atleast_1d(
            np.asarray(Materials.get("E", [1.0]), dtype=np.float64)
        )
        self.nb_mat: int = self.E_max.size
        self.E_min: FloatArray = np.full(self.nb_mat, 1e-9, dtype=np.float64)
        self.nu: FloatArray = np.atleast_1d(
            np.asarray(Materials.get("nu", [0.3]), dtype=np.float64)
        )
        self.penal: float = float(Optimizer.get("penal", 3.0))
        self.solver_type: str = Optimizer.get("solver", "default")
        self.filter_type: str = Optimizer.get("filter_type", 0)
        self.filter_radius: float = float(Optimizer.get("filter_radius_min", 0.0))

        # Pre-compute Constant Matrices (KE, DOF Maps, Filter)
        self.KE: FloatArray = self._get_lk_stiffness()
        self.edofMat, self.iK, self.jK = self._build_dof_map()
        self.H, self.Hs = self._build_filter()

        # State placeholders for BCs
        self.fixed_dofs: IntArray = np.array([], dtype=np.int64)
        self.free_dofs: IntArray = np.array([], dtype=np.int64)
        self.fi_indices: list[int] = []
        self.fo_indices: list[int] = []
        self.di_indices: list[int] = []
        self.do_indices: list[int] = []
        self.finorm: Sequence[float] = []
        self.fonorm: Sequence[float] = []
        self.forces_i: FloatArray | None = None
        self.forces_o: FloatArray | None = None

    def setup_boundary_conditions(
        self, forces: dict, supports: dict | None = None
    ) -> None:
        """Parses Forces and Supports dicts to create force vectors and fixed DOFs."""
        # Forces
        di: list[int]
        do: list[int]
        self.forces_i, self.fi_indices, di = self._parse_forces(
            forces, "fix", "fiy", "fiz", "fidir", "finorm"
        )
        self.forces_o, self.fo_indices, do = self._parse_forces(
            forces, "fox", "foy", "foz", "fodir", "fonorm"
        )

        self.di_indices = di  # For artificial stiffness addition
        self.do_indices = do  # For artificial stiffness addition
        self.finorm = forces.get("finorm", [])
        self.fonorm = forces.get("fonorm", [])

        # Supports
        fixed: list[int] = []
        if supports:
            sx: list[float] = supports.get("sx", [])
            sy: list[float] = supports.get("sy", [])
            sz: list[float] = supports.get("sz", [])
            sr: list[float] = supports.get("sr", [0.0] * len(sx))
            sdim: list[str] = supports.get("sdim", [])
            active_sup: list[int] = [i for i, val in enumerate(sdim) if val != "-"]

            for i in active_sup:
                center_node_idx = self._get_node_idx(
                    int(sx[i]), int(sy[i]), int(sz[i]) if self.is_3d else 0
                )
                nodes_to_fix = [center_node_idx]

                # If radius > 0, find all nodes within radius
                if i < len(sr) and sr[i] > 0:
                    radius = float(sr[i])
                    # Determine range to search
                    x_range = range(
                        max(0, int(sx[i] - radius)),
                        min(self.nelx + 1, int(sx[i] + radius + 1)),
                    )
                    y_range = range(
                        max(0, int(sy[i] - radius)),
                        min(self.nely + 1, int(sy[i] + radius + 1)),
                    )
                    z_range = (
                        range(
                            max(0, int(sz[i] - radius)),
                            min(self.nelz + 1, int(sz[i] + radius + 1)),
                        )
                        if self.is_3d
                        else range(1)
                    )

                    for z in z_range:
                        for x in x_range:
                            for y in y_range:
                                dist_sq = (
                                    (x - sx[i]) ** 2
                                    + (y - sy[i]) ** 2
                                    + ((z - sz[i]) ** 2 if self.is_3d else 0)
                                )
                                if dist_sq <= radius**2:
                                    n_idx = self._get_node_idx(
                                        x, y, z if self.is_3d else 0
                                    )
                                    nodes_to_fix.append(n_idx)

                for node_idx in nodes_to_fix:
                    node_dof = self.dim_mul * node_idx
                    if "X" in sdim[i]:
                        fixed.append(node_dof)
                    if "Y" in sdim[i]:
                        fixed.append(node_dof + 1)
                    if self.is_3d and "Z" in sdim[i]:
                        fixed.append(node_dof + 2)

        self.fixed_dofs = np.unique(fixed).astype(np.int64)
        self.free_dofs = np.setdiff1d(
            np.arange(self.ndof, dtype=np.int64), self.fixed_dofs
        )

    def apply_regions(self, x: FloatArray, regions: dict) -> FloatArray:
        """Applies geometric constraints (Regions) to the density field."""
        xPhys = x.copy()
        rshape: list[str] = regions.get("rshape", [])
        if not rshape:
            return xPhys

        rx: list[float] = regions.get("rx", [])
        ry: list[float] = regions.get("ry", [])
        rz: list[float] = regions.get("rz", [])
        rradius: list[float] = regions.get("rradius", [])
        rstate: list[str] = regions.get("rstate", [])

        active_regions: list[int] = [i for i, s in enumerate(rshape) if s != "-"]

        for i in active_regions:
            val = 1e-6 if rstate[i] == "Void" else 1.0
            r = float(rradius[i])
            z_range = (
                range(max(0, int(rz[i] - r)), min(self.nelz, int(rz[i] + r)))
                if self.is_3d
                else range(1)
            )
            for ez in z_range:
                for ex in range(max(0, int(rx[i] - r)), min(self.nelx, int(rx[i] + r))):
                    for ey in range(
                        max(0, int(ry[i] - r)), min(self.nely, int(ry[i] + r))
                    ):
                        # Geometric check
                        if rshape[i] == "◯":
                            dist_sq = (
                                (ex - rx[i]) ** 2
                                + (ey - ry[i]) ** 2
                                + ((ez - rz[i]) ** 2 if self.is_3d else 0)
                            )
                            if dist_sq > r**2:
                                continue

                        idx = (
                            (ez * self.nelx * self.nely if self.is_3d else 0)
                            + ex * self.nely
                            + ey
                        )
                        xPhys[idx] = val
        return xPhys

    def solve(self, xPhys: FloatArray) -> tuple[FloatArray, FloatArray]:
        """Assembles K and solves for Input and Output forces."""
        # Assembly
        E_eff = (
            self.E_min[:, None] + xPhys**self.penal * (self.E_max - self.E_min)[:, None]
        )
        E_eff = E_eff.sum(axis=0)
        sK = (self.KE.flatten()[np.newaxis]).T * E_eff
        K = coo_matrix(
            (sK.flatten(order="F"), (self.iK, self.jK)), shape=(self.ndof, self.ndof)
        ).tocsc()
        # Add artificial stiffness at force locations
        self._add_artificial_springs(K, self.di_indices, self.fi_indices, self.finorm)
        self._add_artificial_springs(K, self.do_indices, self.fo_indices, self.fonorm)

        # Solving
        K_free = K[np.ix_(self.free_dofs, self.free_dofs)]
        ui = np.zeros((self.ndof, len(self.fi_indices)), dtype=np.float64)
        uo = np.zeros((self.ndof, len(self.fo_indices)), dtype=np.float64)
        self._solve_linear_system(K_free, self.forces_i, self.fi_indices, ui)
        self._solve_linear_system(K_free, self.forces_o, self.fo_indices, uo)

        return ui, uo

    def compute_ce(self, ui: FloatArray, uo: FloatArray) -> FloatArray:
        """Compute element compliance ce = u^T KE u for all elements.

        For rigid mechanisms (no output forces): ce = sum_i u_i^T KE u_i
        For compliant mechanisms: ce = sum_i sum_o u_in^T KE u_out
        """
        nb_out = len(self.fo_indices)
        ce_total = np.zeros(self.nel, dtype=np.float64)

        if nb_out == 0:  # Rigid Mechanism (Minimize Compliance)
            for i_in in range(len(self.fi_indices)):
                Ue = ui[self.edofMat, i_in]
                ce_total += np.sum((Ue @ self.KE) * Ue, axis=1)
        else:  # Compliant Mechanism
            for i_in in range(len(self.fi_indices)):
                Ue_in = ui[self.edofMat, i_in]
                for i_out in range(len(self.fo_indices)):
                    Ue_out = uo[self.edofMat, i_out]
                    ce_total += np.sum((Ue_in @ self.KE) * Ue_out, axis=1)

        return ce_total

    def compute_sensitivities(
        self, xPhys: FloatArray, ui: FloatArray, uo: FloatArray
    ) -> tuple[FloatArray, FloatArray]:
        """Calculates Sensitivity (dc) and Volume Sensitivity (dv)."""
        nb_out = len(self.fo_indices)
        ce_total = self.compute_ce(ui, uo)

        # Compliance / Objective Calculation
        if nb_out == 0:  # Rigid Mechanism (Minimize Compliance)
            dc = -self.penal * (xPhys ** (self.penal - 1)) * ce_total
        else:  # Compliant Mechanism
            dc = self.penal * (xPhys ** (self.penal - 1)) * ce_total

        # Volume Sensitivity (dv)
        dv: FloatArray = np.ones(self.nel, dtype=np.float64)

        # Filtering
        return self._apply_filter(xPhys, dc, dv)

    def compute_objective(
        self, xPhys: FloatArray, ui: FloatArray, uo: FloatArray
    ) -> float:
        """Compute objective value based on current displacements."""
        nb_out = len(self.fo_indices)
        obj_val = 0.0

        if nb_out == 0:  # Rigid Mechanism (Minimize Compliance)
            E_eff = np.full(self.nel, self.E_min[0], dtype=np.float64)
            for i, E_i in enumerate(self.E_max):
                E_eff += xPhys[i] ** self.penal * (E_i - self.E_min[i])
            ce_total = self.compute_ce(ui, uo)
            obj_val = (E_eff * ce_total).sum()
        else:  # Compliant Mechanism
            # Sum of absolute output displacements
            for idx, dof_indices in enumerate(self.do_indices):
                obj_val += sum(abs(uo[dof, idx]) for dof in [dof_indices]) / nb_out

        return obj_val

    # --- Internal Helper Methods ---

    def _get_node_idx(self, x: int, y: int, z: int) -> int:
        return (
            (z * (self.nelx + 1) * (self.nely + 1) if self.is_3d else 0)
            + x * (self.nely + 1)
            + y
        )

    def _parse_forces(
        self,
        Forces: dict,
        kx: str,
        ky: str,
        kz: str,
        kdir: str,
        knorm: str,
    ) -> tuple[FloatArray, list[int], list[int]]:
        fx = Forces.get(kx, [])
        fy = Forces.get(ky, [])
        fz = Forces.get(kz, []) if self.is_3d else []
        fdir = Forces.get(kdir, [])
        fnorm = Forces.get(knorm, [])

        active_indices = [i for i, val in enumerate(fdir) if val != "-"]
        f_vec = np.zeros((self.ndof, len(active_indices)), dtype=np.float64)
        dof_indices: list[int] = []

        for mat_idx, i in enumerate(active_indices):
            node = self._get_node_idx(fx[i], fy[i], fz[i] if self.is_3d else 0)
            val = fnorm[i]
            dof = self.dim_mul * node
            if "X" in fdir[i]:
                if "←" in fdir[i]:
                    val = -val
            elif "Y" in fdir[i]:
                dof += 1
                if "↑" in fdir[i]:
                    val = -val
            elif self.is_3d and "Z" in fdir[i]:
                dof += 2
                if ">" in fdir[i]:
                    val = -val

            f_vec[dof, mat_idx] = val
            dof_indices.append(dof)

        return f_vec, active_indices, dof_indices

    def _add_artificial_springs(
        self,
        K: csc_matrix,
        dofs: list[int],
        active_indices: list[int],
        norms: list[float],
    ) -> None:
        for i, dof in enumerate(dofs):
            original_idx = active_indices[i]
            if original_idx < len(norms) and norms[original_idx] > 0:
                K[dof, dof] += norms[original_idx]

    def _solve_linear_system(
        self,
        K_free: csc_matrix,
        F: FloatArray | None,
        active_indices: list[int],
        U_full: FloatArray,
    ) -> None:
        if not active_indices or F is None:
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
            for i in range(len(active_indices)):
                if np.any(F[self.free_dofs, i]):
                    u_sol, info = cg(
                        K_free,
                        F[self.free_dofs, i],
                        M=M,
                        rtol=1e-6,
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
        if self.filter_type == "Sensitivity":
            # H * (x * dc) / Hs / max(x, 0.001)
            dc = np.asarray((self.H @ (x * dc)) / self.Hs.flatten()) / np.maximum(
                0.001, x
            )
        elif self.filter_type == "Density":
            dc = np.asarray(self.H * (dc[np.newaxis].T / self.Hs))[:, 0]
            dv = np.asarray(self.H * (dv[np.newaxis].T / self.Hs))[:, 0]
        return dc, dv

    def update_xPhys(self, x: FloatArray) -> FloatArray:
        """Calculates physical density based on design variable and filter."""
        if self.filter_type == "Density":
            return (self.H @ x).ravel() / np.asarray(self.Hs).ravel()
        return x

    def _get_lk_stiffness(self) -> FloatArray:
        """Get element stiffness matrix."""
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
            return (E / ((nu + 1) * (1 - 2 * nu)) * K_blocks).astype(np.float64)
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
        size = 8 * (self.elemndof if self.is_3d else 1)
        el = np.arange(self.nel)

        # Vectorize elemental coordinates
        if self.is_3d:
            ez = el // (self.nelx * self.nely)
            rem = el % (self.nelx * self.nely)
            ex = rem // self.nely
            ey = rem % self.nely
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
            ex = el // self.nely
            ey = el % self.nely
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

    def _build_filter(self):
        el = np.arange(self.nel)

        # Map 1D element arrays to 2D/3D indices
        if self.is_3d:
            ez = el // (self.nelx * self.nely)
            rem = el % (self.nelx * self.nely)
            ex = rem // self.nely
            ey = rem % self.nely
        else:
            ex = el // self.nely
            ey = el % self.nely

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

        return H, H.sum(1)
