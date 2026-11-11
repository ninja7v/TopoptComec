# topoptcomec/core/nonlinear.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Geometrically non-linear finite element solver for large displacements.

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Support

# Type aliases
FloatArray = npt.NDArray[np.float64]

#: Energy interpolation (Wang, Lazarov, Sigmund & Jensen, 2014): steepness
#: and threshold of the smooth switch between the linear (void) and the
#: non-linear (solid) strain energy.
INTERPOLATION_BETA = 500.0
INTERPOLATION_THRESHOLD = 0.01
#: Newton convergence: residual norm relative to the applied load norm.
NEWTON_TOLERANCE = 1e-7
#: Newton convergence: step length relative to the displacement norm (the
#: fallback criterion when the residual stalls at round-off level).
NEWTON_STEP_TOLERANCE = 1e-10
#: Newton iterations allowed per load level.
NEWTON_MAX_ITERATIONS = 20
#: Smallest step length of the Newton line search.
LINE_SEARCH_MIN_STEP = 1.0 / 16.0
#: Smallest load increment tried before giving up on the full load.
MIN_LOAD_INCREMENT = 1.0 / 64.0
#: Share of the optimization iterations over which the load is ramped up.
LOAD_RAMP_SHARE = 0.5
#: Failed load increments tolerated by a solve started from rest.
COLD_START_FAILURES = 5
#: Elements processed at once when integrating the element matrices.
ELEMENT_CHUNK = 8192


class NonlinearFEM(FEM):
    """
    Geometrically non-linear (total Lagrangian) finite element solver.

    The kinematics are exact for arbitrarily large displacements and
    rotations: the Green-Lagrange strain

    .. math::

        \\mathbf{E} = \\tfrac{1}{2}(\\mathbf{F}^T\\mathbf{F} - \\mathbf{I}),
        \\qquad \\mathbf{F} = \\mathbf{I} + \\nabla_X\\mathbf{u}

    is combined with the St. Venant-Kirchhoff material
    :math:`\\mathbf{S} = \\mathbf{D}\\,\\mathbf{E}`, which uses the same
    constitutive matrix as the linear solver (plane stress in 2D), so both
    solvers agree in the small-displacement limit.

    Equilibrium :math:`\\mathbf{r}(\\mathbf{u}) = \\mathbf{f}_{int}(\\mathbf{u})
    + \\mathbf{K}_s\\mathbf{u} - \\mathbf{f}_{ext} = \\mathbf{0}` is solved with
    Newton-Raphson iterations and a backtracking line search. Each solve
    starts from the equilibrium of the previous design; when that fails the
    load is applied incrementally with adaptive steps.

    Low-density elements are stabilised with the energy interpolation of
    Wang et al. (2014): the element strain energy is

    .. math::

        \\phi_e = \\phi_{NL}(\\gamma_e\\mathbf{u}_e)
        - \\phi_{L}(\\gamma_e\\mathbf{u}_e) + \\phi_{L}(\\mathbf{u}_e)

    where :math:`\\gamma_e \\in [0, 1]` is a smooth Heaviside function of the
    penalised density. Void elements (:math:`\\gamma_e = 0`) are therefore
    linear and cannot invert, solid elements (:math:`\\gamma_e = 1`) are fully
    non-linear.

    Sensitivities are exact adjoint derivatives of the discrete non-linear
    problem (the tangent stiffness at equilibrium is symmetric, so one extra
    linear solve is needed).

    Parameters
    ----------
    grid, E, nu, penal, solver, filter_type, filter_radius
        See :class:`~topoptcomec.core.fem.FEM`.
    load_factor : float
        Multiplier of the input forces. The artificial port springs are not
        scaled, so the factor directly controls how large the displacements
        are.
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
        load_factor: float = 1.0,
    ) -> None:
        super().__init__(grid, E, nu, penal, solver, filter_type, filter_radius)
        if not np.isfinite(load_factor) or load_factor <= 0.0:
            raise ValueError("The load factor must be finite and strictly positive.")
        self.load_factor: float = float(load_factor)
        self._gradients: FloatArray = self._shape_gradients()
        self._D0: FloatArray = np.stack(
            [self._constitutive_matrix(float(nu_m)) for nu_m in self.nu]
        )
        dim = self.dim_mul
        self._voigt: list[tuple[int, int]] = (
            [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]
            if dim == 3
            else [(0, 0), (1, 1), (0, 1)]
        )
        #: Converged displacement of the previous design (Newton warm start).
        self._state: FloatArray | None = None
        self._state_fraction: float = 0.0
        self._increment: float = 1.0
        #: Fraction of the load reached by the last solve (1.0 = full load).
        self.last_load_fraction: float = 1.0
        #: Largest fraction of the load a solve may apply (load continuation).
        self.load_limit: float = 1.0

    # --- Problem setup ---

    def begin_iteration(self, iteration: int, n_iterations: int) -> bool:
        """
        Load continuation over the optimization iterations.

        The load is ramped up quadratically during the first
        ``LOAD_RAMP_SHARE`` of the iterations. The first designs are soft and
        gray: under the full load they deform far more than the final design
        ever will, which misleads the optimizer. Under a small load the
        analysis is nearly linear, so the topology forms as in a linear
        optimization and is then refined for large displacements.

        Returns
        -------
        bool
            ``False`` while the load is still being ramped up.
        """
        ramp = LOAD_RAMP_SHARE * n_iterations
        if iteration >= ramp:
            self.load_limit = 1.0
        else:
            self.load_limit = max(MIN_LOAD_INCREMENT, (iteration / ramp) ** 2)
        return self.load_limit >= 1.0

    def setup_boundary_conditions(
        self,
        loads_in: Sequence[Load],
        loads_out: Sequence[Load] = (),
        supports: Sequence[Support] = (),
    ) -> None:
        """Same as the linear solver, with the input forces scaled by ``load_factor``."""
        super().setup_boundary_conditions(loads_in, loads_out, supports)
        if self.forces_i is not None:
            self.forces_i *= self.load_factor
        self._state = None

    def _shape_gradients(self) -> FloatArray:
        """
        Shape function gradients at the Gauss points of the unit element.

        Returns
        -------
        FloatArray
            Shape ``(n_gauss, dim, n_nodes)``. The local node order is the
            one of ``edofMat``: (0,0), (1,0), (1,1), (0,1), then the same with
            ``z = 1`` in 3D.
        """
        dim = self.dim_mul
        corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.float64)
        if dim == 3:
            corners = np.vstack(
                [np.column_stack([corners, np.full(4, z)]) for z in (0.0, 1.0)]
            )
        signs = 2.0 * corners - 1.0
        g = 0.5 / np.sqrt(3.0)
        offsets = np.array(np.meshgrid(*([(-g, g)] * dim), indexing="ij"))
        points = 0.5 + offsets.reshape(dim, -1).T  # (n_gauss, dim)
        gradients = np.empty((points.shape[0], dim, corners.shape[0]))
        for q, point in enumerate(points):
            # N_n(x) = prod_a (1 - c_a + s_a x_a) on the unit element.
            factors = (1.0 - corners) + signs * point
            for a in range(dim):
                others = np.prod(np.delete(factors, a, axis=1), axis=1)
                gradients[q, a] = signs[:, a] * others
        return gradients

    def _constitutive_matrix(self, nu: float) -> FloatArray:
        """Unit-modulus constitutive matrix (plane stress in 2D), Voigt form."""
        if self.is_3d:
            lam = nu / ((1 + nu) * (1 - 2 * nu))
            mu = 1 / (2 * (1 + nu))
            D = np.zeros((6, 6))
            D[:3, :3] = lam
            D[np.diag_indices(3)] = lam + 2 * mu
            D[3, 3] = D[4, 4] = D[5, 5] = mu
            return D
        return (
            1 / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
        )

    # --- Energy interpolation ---

    def _interpolation(self, xP: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Energy interpolation factor and its density derivative.

        .. math::

            \\gamma_e = \\frac{\\tanh(\\beta\\rho_0)
            + \\tanh(\\beta(\\bar\\rho_e^{\\,p} - \\rho_0))}
            {\\tanh(\\beta\\rho_0) + \\tanh(\\beta(1 - \\rho_0))}

        with :math:`\\bar\\rho_e` the total physical density of the element.

        Parameters
        ----------
        xP : FloatArray
            Physical densities, shape (nb_mat, nel).

        Returns
        -------
        tuple[FloatArray, FloatArray]
            ``(gamma, dgamma)``, both of shape (nel,); ``dgamma`` is the
            derivative with respect to the density of any material.
        """
        beta, rho0 = INTERPOLATION_BETA, INTERPOLATION_THRESHOLD
        total = np.clip(xP.sum(axis=0), 0.0, 1.0)
        penalised = total**self.penal
        denominator = np.tanh(beta * rho0) + np.tanh(beta * (1.0 - rho0))
        switch = np.tanh(beta * (penalised - rho0))
        gamma = (np.tanh(beta * rho0) + switch) / denominator
        dgamma = (
            beta
            * (1.0 - switch**2)
            / denominator
            * self.penal
            * total ** (self.penal - 1.0)
        )
        return gamma, dgamma

    # --- Element kinematics ---

    def _strain(self, H: FloatArray) -> FloatArray:
        """
        Green-Lagrange strain in Voigt form (engineering shear).

        Computed as :math:`\\tfrac{1}{2}(\\mathbf{H} + \\mathbf{H}^T +
        \\mathbf{H}^T\\mathbf{H})` from the displacement gradient rather than
        :math:`\\tfrac{1}{2}(\\mathbf{F}^T\\mathbf{F} - \\mathbf{I})`: the
        latter subtracts two numbers close to one and loses all accuracy
        for small displacement gradients (Newton then stalls at small loads).

        Parameters
        ----------
        H : FloatArray
            Displacement gradients ``du_i/dX_a``, shape (n, dim, dim).

        Returns
        -------
        FloatArray
            Strains, shape (n, n_voigt).
        """
        strain = np.empty((H.shape[0], len(self._voigt)))
        for v, (a, b) in enumerate(self._voigt):
            # (H^T H)_ab (explicit sums beat tiny batched matmuls)
            quadratic = H[:, 0, a] * H[:, 0, b]
            for i in range(1, self.dim_mul):
                quadratic += H[:, i, a] * H[:, i, b]
            if a == b:
                strain[:, v] = H[:, a, a] + 0.5 * quadratic
            else:
                strain[:, v] = H[:, a, b] + H[:, b, a] + quadratic
        return strain

    def _strain_operator(self, F: FloatArray, gradient: FloatArray) -> FloatArray:
        """
        Variation operator of the Green-Lagrange strain at one Gauss point.

        Parameters
        ----------
        F : FloatArray
            Deformation gradients, shape (n, dim, dim).
        gradient : FloatArray
            Shape function gradients, shape (dim, n_nodes).

        Returns
        -------
        FloatArray
            Operator ``B`` with :math:`\\delta\\mathbf{E} =
            \\mathbf{B}\\,\\delta\\mathbf{u}_e`, shape ``(n, n_voigt, eldof)``.
        """
        n = F.shape[0]
        B = np.empty((n, len(self._voigt), self.eldof))
        for v, (a, b) in enumerate(self._voigt):
            # Entry (node, i) is F[i, a] * dN_node/dX_b (symmetrised for shear).
            rows = gradient[b][None, :, None] * F[:, None, :, a]
            if a != b:
                rows = rows + gradient[a][None, :, None] * F[:, None, :, b]
            B[:, v] = rows.reshape(n, -1)
        return B

    def _stress_tensor(self, S: FloatArray) -> FloatArray:
        """Second Piola-Kirchhoff stress tensors (n, dim, dim) from Voigt form."""
        dim = self.dim_mul
        tensor = np.empty((S.shape[0], dim, dim))
        for v, (a, b) in enumerate(self._voigt):
            tensor[:, a, b] = S[:, v]
            tensor[:, b, a] = S[:, v]
        return tensor

    @staticmethod
    def _displacement_gradient(nodal: FloatArray, gradient: FloatArray) -> FloatArray:
        """Displacement gradients ``du_i/dX_a``, shape (n, dim, dim)."""
        return nodal.transpose(0, 2, 1) @ gradient.T

    def _element_response(
        self,
        u: FloatArray,
        E_eff: FloatArray,
        gamma: FloatArray,
        tangent: bool,
    ) -> tuple[FloatArray, FloatArray | None, float]:
        """
        Element internal forces, (optionally) tangent matrices and energy.

        With :math:`\\hat{\\mathbf{u}}_e = \\gamma_e\\mathbf{u}_e` the
        interpolated element force and tangent are

        .. math::

            \\mathbf{f}_e = \\gamma_e\\mathbf{f}_{NL}(\\hat{\\mathbf{u}}_e)
            + (1 - \\gamma_e^2)\\mathbf{K}_L\\mathbf{u}_e, \\qquad
            \\mathbf{K}_e = \\gamma_e^2\\mathbf{K}_T(\\hat{\\mathbf{u}}_e)
            + (1 - \\gamma_e^2)\\mathbf{K}_L

        Parameters
        ----------
        u : FloatArray
            Global displacement vector, shape (ndof,).
        E_eff : FloatArray
            SIMP stiffness per material and element, shape (nb_mat, nel).
        gamma : FloatArray
            Energy interpolation factors, shape (nel,).
        tangent : bool
            Whether the tangent matrices are needed.

        Returns
        -------
        tuple[FloatArray, FloatArray | None, float]
            Element forces ``(nel, eldof)``, flattened tangent matrices
            ``(nel, eldof**2)`` (``None`` when ``tangent`` is False) and the
            total strain energy.
        """
        dim, eldof = self.dim_mul, self.eldof
        n_nodes = eldof // dim
        weight = 1.0 / self._gradients.shape[0]
        forces = np.empty((self.nel, eldof))
        tangents = np.empty((self.nel, eldof * eldof)) if tangent else None
        identity = np.eye(dim)
        energy = 0.0

        for start in range(0, self.nel, ELEMENT_CHUNK):
            chunk = slice(start, min(start + ELEMENT_CHUNK, self.nel))
            ue = u[self.edofMat[chunk]]
            g = gamma[chunk]
            E_chunk = E_eff[:, chunk]
            n = ue.shape[0]
            nodal = (g[:, None] * ue).reshape(n, n_nodes, dim)
            D = np.einsum("me,mvw->evw", E_chunk, self._D0)
            f_nl = np.zeros((n, n_nodes, dim))
            K_nl = np.zeros((n, eldof, eldof)) if tangent else None
            for gradient in self._gradients:
                H = self._displacement_gradient(nodal, gradient)
                F = identity + H
                strain = self._strain(H)
                S = (D @ strain[:, :, None])[:, :, 0]
                energy += 0.5 * weight * float(np.vdot(strain, S))
                S_tensor = self._stress_tensor(S)
                # Nodal forces from the first Piola-Kirchhoff stress F.S
                f_nl += weight * (F @ S_tensor @ gradient).transpose(0, 2, 1)
                if K_nl is None:
                    continue
                B = self._strain_operator(F, gradient)
                K_nl += weight * (B.transpose(0, 2, 1) @ (D @ B))
                # Geometric stiffness: delta_ij * (grad N_n . S . grad N_m)
                geometric = weight * (gradient.T @ S_tensor @ gradient)
                K_view = K_nl.reshape(n, n_nodes, dim, n_nodes, dim)
                for i in range(dim):
                    K_view[:, :, i, :, i] += geometric

            # Linear part, with the per-material element matrices.
            linear_weight = 1.0 - g**2
            if tangents is not None and K_nl is not None:
                K_lin = (E_chunk.T @ self._KE_flat_mats).reshape(n, eldof, eldof)
                f_lin = (K_lin @ ue[:, :, None])[:, :, 0]
                tangents[chunk] = (
                    (g**2)[:, None, None] * K_nl + linear_weight[:, None, None] * K_lin
                ).reshape(n, -1)
            else:
                f_lin = np.zeros((n, eldof))
                for m in range(self.nb_mat):
                    f_lin += E_chunk[m][:, None] * (ue @ self.KE_mats[m])
            forces[chunk] = (
                g[:, None] * f_nl.reshape(n, eldof) + linear_weight[:, None] * f_lin
            )
            energy += 0.5 * float(linear_weight @ np.einsum("ek,ek->e", f_lin, ue))
        return forces, tangents, energy

    def _residual(
        self,
        u: FloatArray,
        E_eff: FloatArray,
        gamma: FloatArray,
        f_ext: FloatArray,
        tangent: bool,
    ) -> tuple[FloatArray, FloatArray | None, float]:
        """
        Free-DOF residual, element tangents and total potential energy.

        The residual ``f_int + K_s u - f_ext`` is the gradient of the
        potential energy (strain energy, spring energy and load potential),
        which the Newton line search uses as merit function.
        """
        forces, tangents, energy = self._element_response(u, E_eff, gamma, tangent)
        residual = np.bincount(
            self.edofMat.ravel(), weights=forces.ravel(), minlength=self.ndof
        )
        if self._spring_dofs.size:
            spring_force = self._spring_vals * u[self._spring_dofs]
            np.add.at(residual, self._spring_dofs, spring_force)
            energy += 0.5 * float(spring_force @ u[self._spring_dofs])
        energy -= float(f_ext @ u[self.free_dofs])
        return residual[self.free_dofs] - f_ext, tangents, energy

    # --- Linear algebra ---

    def _tangent_solver(
        self, tangents: FloatArray
    ) -> Callable[[FloatArray], FloatArray]:
        """
        Solver of the assembled tangent system.

        Returns
        -------
        Callable[[FloatArray], FloatArray]
            Maps a right-hand side ``(n_free,)`` to the solution.
        """
        K_free = self._assemble_free(tangents)

        def solve(rhs: FloatArray) -> FloatArray:
            return self._solve_all(K_free, rhs[:, None])[:, 0]

        if self._use_direct_solver(K_free.shape[0]):
            try:
                factor = self._direct.factorize(K_free)
            except RuntimeError:
                return solve
            return factor
        return solve

    # --- Equilibrium ---

    def _newton(
        self,
        u: FloatArray,
        E_eff: FloatArray,
        gamma: FloatArray,
        f_ext: FloatArray,
    ) -> tuple[FloatArray, Callable[[FloatArray], FloatArray] | None]:
        """
        Newton-Raphson iterations with a backtracking line search.

        Parameters
        ----------
        u : FloatArray
            Starting displacement, shape (ndof,).
        E_eff, gamma : FloatArray
            Element stiffness and interpolation factors.
        f_ext : FloatArray
            External load on the free DOFs.

        Returns
        -------
        tuple
            ``(u, solver)``: the equilibrium displacement and the solver of
            the tangent system *at equilibrium* (reused by the adjoint), or
            ``(u, None)`` when the iterations did not converge.
        """
        free = self.free_dofs
        u = u.copy()
        tolerance = NEWTON_TOLERANCE * max(float(np.linalg.norm(f_ext)), 1e-300)
        residual, tangents, energy = self._residual(u, E_eff, gamma, f_ext, True)
        for _ in range(NEWTON_MAX_ITERATIONS):
            norm = float(np.linalg.norm(residual))
            if not np.isfinite(norm):
                return u, None
            try:
                solver = self._tangent_solver(tangents)
                if norm <= tolerance:
                    return u, solver
                step = solver(-residual)
            except RuntimeError:
                return u, None
            if not np.all(np.isfinite(step)):
                return u, None
            if np.linalg.norm(step) <= NEWTON_STEP_TOLERANCE * np.linalg.norm(u[free]):
                # The residual is at round-off level: with very small loads it
                # cannot drop below the relative tolerance above.
                u = u.copy()
                u[free] += step
                return u, solver

            # Backtracking (Armijo) line search on the potential energy. When
            # the tangent is indefinite the Newton step is not a descent
            # direction: the full step is taken (plain Newton).
            slope = float(residual @ step)
            alpha = 1.0
            while True:
                trial = u.copy()
                trial[free] += alpha * step
                trial_residual, trial_tangents, trial_energy = self._residual(
                    trial, E_eff, gamma, f_ext, True
                )
                if slope >= 0.0 or alpha <= LINE_SEARCH_MIN_STEP:
                    break
                if np.isfinite(trial_energy) and (
                    trial_energy <= energy + 1e-4 * alpha * slope
                    or float(np.linalg.norm(trial_residual)) <= tolerance
                ):
                    break
                alpha *= 0.5
            u, residual, tangents, energy = (
                trial,
                trial_residual,
                trial_tangents,
                trial_energy,
            )
        return u, None

    def _equilibrium(
        self, E_eff: FloatArray, gamma: FloatArray, f_ext: FloatArray
    ) -> tuple[FloatArray, Callable[[FloatArray], FloatArray], float]:
        """
        Equilibrium displacement under ``f_ext`` with adaptive load control.

        The solve restarts from the equilibrium of the previous design at the
        load level that design reached (the design changes little between two
        optimization iterations), then the load is increased towards the full
        load. An increment that fails - typically because the structure
        buckles or snaps through at that load - is halved, and the increment
        size is remembered for the next design.

        Once some load level is reached, a failed increment ends the ramp for
        this design: the optimization proceeds at the reached level and tries
        to go further on the next design. Soft early designs are therefore
        loaded progressively while the structure forms, at a bounded cost per
        iteration.

        Returns
        -------
        tuple
            ``(u, solver, fraction)``: displacement, tangent solver at that
            state and the fraction of the load that was reached.
        """
        u = np.zeros(self.ndof)
        solver: Callable[[FloatArray], FloatArray] | None = None
        fraction = 0.0
        if self._state is not None and self._state_fraction > 0.0:
            trial, solver = self._newton(
                self._state, E_eff, gamma, self._state_fraction * f_ext
            )
            if solver is not None:
                u, fraction = trial, self._state_fraction

        # A warm solve stops at its first failed increment; a solve started
        # from rest may fail a few times before settling for a partial load.
        cold = fraction == 0.0
        failures_left = COLD_START_FAILURES if cold else 1
        increment = 1.0 if cold else self._increment
        limit = self.load_limit
        if fraction > limit:
            # The load limit was lowered: reload from rest.
            u, solver, fraction = np.zeros(self.ndof), None, 0.0
        while fraction < limit:
            target = min(limit, fraction + increment)
            trial, trial_solver = self._newton(u, E_eff, gamma, target * f_ext)
            if trial_solver is None:
                # The failed attempt reused the factorisation storage: the
                # solver of the last converged state is no longer valid.
                solver = None
                increment = 0.5 * min(increment, limit - fraction)
                failures_left -= 1
                if failures_left <= 0 or increment < MIN_LOAD_INCREMENT:
                    break
                continue
            u, solver, fraction = trial, trial_solver, target
            increment = min(2.0 * increment, 1.0)
        self._increment = max(increment, MIN_LOAD_INCREMENT)
        if solver is None:
            # Tangent at the last converged state (at rest when no increment
            # converged at all).
            _, tangents, _ = self._residual(u, E_eff, gamma, fraction * f_ext, True)
            solver = self._tangent_solver(tangents)
        return u, solver, fraction

    def solve(self, xPhys: FloatArray) -> tuple[FloatArray, FloatArray]:
        """
        Solve the non-linear equilibrium and the adjoint problem.

        When the full load cannot be reached, the state and the adjoint are
        those of the reached load level (see :attr:`last_load_fraction`).

        Parameters
        ----------
        xPhys : FloatArray
            Physical element densities. Shape (nel,) or (nb_mat, nel).

        Returns
        -------
        Tuple[FloatArray, FloatArray]
            ``(u, lam)``: the equilibrium displacement and the adjoint field
            solving :math:`\\mathbf{K}_T(\\mathbf{u})\\boldsymbol{\\lambda} =
            \\mathbf{l}`, where :math:`\\mathbf{l}` is the normalized output
            load (compliant mechanisms) or the input load (rigid structures).
        """
        E_eff = self._material_stiffness(xPhys)
        xP = np.atleast_2d(np.asarray(xPhys, dtype=np.float64))
        gamma, _ = self._interpolation(xP)
        lam = np.zeros(self.ndof, dtype=np.float64)
        rhs = self._load_case_matrix()
        if rhs.shape[1] == 0 or self.forces_i is None or not self.loads_in:
            return np.zeros(self.ndof, dtype=np.float64), lam

        f_ext = rhs[:, 0]
        u, solver, fraction = self._equilibrium(E_eff, gamma, f_ext)
        self.last_load_fraction = fraction
        self._state, self._state_fraction = u.copy(), fraction
        adjoint_load = rhs[:, 1] if rhs.shape[1] > 1 else fraction * f_ext
        lam[self.free_dofs] = solver(adjoint_load)
        return u, lam

    # --- Sensitivities ---

    def compute_sensitivities(
        self,
        xPhys: FloatArray,
        ui: FloatArray,
        uo: FloatArray,
        mat_idx: int = 0,
    ) -> tuple[FloatArray, FloatArray]:
        """Single-material sensitivities (see :meth:`compute_all_sensitivities`)."""
        if self.nb_mat != 1:
            raise ValueError(
                "Use compute_all_sensitivities for multi-material non-linear problems."
            )
        return self.compute_all_sensitivities(xPhys, ui, uo)[mat_idx]

    def compute_all_sensitivities(
        self, xPhys: FloatArray, ui: FloatArray, uo: FloatArray
    ) -> list[tuple[FloatArray, FloatArray]]:
        """
        Adjoint sensitivities of the non-linear objective for every material.

        With the adjoint field :math:`\\boldsymbol{\\lambda}` the derivative
        of the objective :math:`J = \\mathbf{l}^T\\mathbf{u}` is

        .. math::

            \\frac{\\partial J}{\\partial \\rho_{m,e}} =
            -\\boldsymbol{\\lambda}_e^T
            \\frac{\\partial \\mathbf{f}_e}{\\partial \\rho_{m,e}}

        where the element force depends on the density through the SIMP
        stiffness and through the energy interpolation factor
        :math:`\\gamma_e`. As in the linear solver, the result is returned so
        that the OC update (a minimizer) minimizes the compliance of a rigid
        structure and maximizes the output displacement of a mechanism.

        Parameters
        ----------
        xPhys : FloatArray
            Physical densities, shape (nel,) or (nb_mat, nel).
        ui : FloatArray
            Equilibrium displacement returned by :meth:`solve`.
        uo : FloatArray
            Adjoint field returned by :meth:`solve`.

        Returns
        -------
        list[tuple[FloatArray, FloatArray]]
            Filtered ``(dc, dv)`` per material.
        """
        xP = np.atleast_2d(np.asarray(xPhys, dtype=np.float64))
        E_eff = self._material_stiffness(xP)
        gamma, dgamma = self._interpolation(xP)
        dE = (self.E_max - self.E_min)[:, None] * self.penal * xP ** (self.penal - 1)

        dim, eldof = self.dim_mul, self.eldof
        n_nodes = eldof // dim
        weight = 1.0 / self._gradients.shape[0]
        identity = np.eye(dim)
        ue_all = ui[self.edofMat]
        le_all = uo[self.edofMat]

        # lam . f_NL^0 per material (unit modulus) and the gamma derivative
        # lam . (f_NL + gamma K_T u - 2 gamma K_L u).
        force_term = np.zeros((self.nb_mat, self.nel))
        gamma_term = np.zeros(self.nel)
        linear_term = np.empty((self.nb_mat, self.nel))
        for m in range(self.nb_mat):
            linear_term[m] = np.sum((le_all @ self.KE_mats[m]) * ue_all, axis=1)

        for start in range(0, self.nel, ELEMENT_CHUNK):
            chunk = slice(start, min(start + ELEMENT_CHUNK, self.nel))
            ue, le, g = ue_all[chunk], le_all[chunk], gamma[chunk]
            n = ue.shape[0]
            nodal = (g[:, None] * ue).reshape(n, n_nodes, dim)
            D = np.einsum("me,mvw->evw", E_eff[:, chunk], self._D0)
            for gradient in self._gradients:
                H = self._displacement_gradient(nodal, gradient)
                F = identity + H
                strain = self._strain(H)
                B = self._strain_operator(F, gradient)
                d_strain = (B @ le[:, :, None])[:, :, 0]  # B lam
                u_strain = (B @ ue[:, :, None])[:, :, 0]  # B u
                for m in range(self.nb_mat):
                    force_term[m, chunk] += weight * np.einsum(
                        "ev,vw,ew->e", d_strain, self._D0[m], strain
                    )
                S = np.einsum("evw,ew->ev", D, strain)
                grad_l = self._displacement_gradient(
                    le.reshape(n, n_nodes, dim), gradient
                )
                grad_u = self._displacement_gradient(
                    ue.reshape(n, n_nodes, dim), gradient
                )
                gamma_term[chunk] += weight * (
                    np.einsum("ev,ev->e", d_strain, S)
                    + g
                    * (
                        np.einsum("ev,evw,ew->e", d_strain, D, u_strain)
                        + np.einsum(
                            "eia,eab,eib->e", grad_l, self._stress_tensor(S), grad_u
                        )
                    )
                )
        gamma_term -= 2.0 * gamma * np.einsum("me,me->e", E_eff, linear_term)

        sign = 1.0 if self.loads_out else -1.0
        dv_ones = np.ones(self.nel, dtype=np.float64)
        result: list[tuple[FloatArray, FloatArray]] = []
        for m in range(self.nb_mat):
            derivative = (
                dE[m] * (gamma * force_term[m] + (1.0 - gamma**2) * linear_term[m])
                + dgamma * gamma_term
            )
            result.append(self._apply_filter(xP[m], sign * derivative, dv_ones.copy()))
        return result
