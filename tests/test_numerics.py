# tests/test_numerics.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Numerical validation of the FEM core: element stiffness matrices against
# independent Gauss-quadrature integration, sensitivities against finite
# differences, and solver equivalence.

import numpy as np
import pytest

from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Support

# --- Independent element stiffness integration -------------------------------


def _ke_q4_plane_stress(E: float, nu: float) -> np.ndarray:
    """Q4 plane-stress stiffness for a unit square via 2x2 Gauss quadrature.

    Node ordering matches the FEM edofMat convention: (0,0), (1,0), (1,1),
    (0,1) in mathematical (y-up) coordinates.
    """
    nodes = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    g = 1 / np.sqrt(3)
    KE = np.zeros((8, 8))
    for xi in (-g, g):
        for eta in (-g, g):
            dN = 0.25 * np.array(
                [
                    [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                    [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)],
                ]
            )
            J = dN @ nodes
            dNxy = np.linalg.solve(J, dN)
            B = np.zeros((3, 8))
            for k in range(4):
                B[0, 2 * k] = dNxy[0, k]
                B[1, 2 * k + 1] = dNxy[1, k]
                B[2, 2 * k] = dNxy[1, k]
                B[2, 2 * k + 1] = dNxy[0, k]
            KE += B.T @ D @ B * np.linalg.det(J)
    return KE


def _ke_h8(E: float, nu: float) -> np.ndarray:
    """H8 stiffness for a unit cube via 2x2x2 Gauss quadrature."""
    nodes = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ],
        dtype=float,
    )
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    D = np.zeros((6, 6))
    D[:3, :3] = lam
    D[np.diag_indices(3)] = lam + 2 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    signs = 2 * nodes - 1
    g = 1 / np.sqrt(3)
    KE = np.zeros((24, 24))
    for xi in (-g, g):
        for eta in (-g, g):
            for ze in (-g, g):
                dN = np.zeros((3, 8))
                for k in range(8):
                    sx, sy, sz = signs[k]
                    dN[0, k] = 0.125 * sx * (1 + sy * eta) * (1 + sz * ze)
                    dN[1, k] = 0.125 * sy * (1 + sx * xi) * (1 + sz * ze)
                    dN[2, k] = 0.125 * sz * (1 + sx * xi) * (1 + sy * eta)
                J = dN @ nodes
                dNxyz = np.linalg.solve(J, dN)
                B = np.zeros((6, 24))
                for k in range(8):
                    B[0, 3 * k] = dNxyz[0, k]
                    B[1, 3 * k + 1] = dNxyz[1, k]
                    B[2, 3 * k + 2] = dNxyz[2, k]
                    B[3, 3 * k] = dNxyz[1, k]
                    B[3, 3 * k + 1] = dNxyz[0, k]
                    B[4, 3 * k + 1] = dNxyz[2, k]
                    B[4, 3 * k + 2] = dNxyz[1, k]
                    B[5, 3 * k] = dNxyz[2, k]
                    B[5, 3 * k + 2] = dNxyz[0, k]
                KE += B.T @ D @ B * np.linalg.det(J)
    return KE


@pytest.mark.parametrize("nu", [0.25, 0.3, 0.4])
def test_ke_2d_matches_gauss_quadrature(nu: float):
    """The hard-coded Q4 matrix must equal exact numerical integration."""
    fem = FEM(StructuredGrid(1, 1, 0), E=[1.0], nu=[nu])
    np.testing.assert_allclose(fem.KE, _ke_q4_plane_stress(1.0, nu), atol=1e-12)


@pytest.mark.parametrize("nu", [0.25, 0.3, 0.4])
def test_ke_3d_matches_gauss_quadrature(nu: float):
    """The hard-coded H8 matrix must equal exact numerical integration."""
    fem = FEM(StructuredGrid(1, 1, 1), E=[1.0], nu=[nu])
    np.testing.assert_allclose(fem.KE, _ke_h8(1.0, nu), atol=1e-12)


# --- Finite-difference sensitivity checks ------------------------------------


def _cantilever_fem(filter_type: str = "None") -> FEM:
    """Small 2D cantilever clamped at x=0, direct solver, no/with filter."""
    grid = StructuredGrid(4, 3, 0)
    fem = FEM(
        grid,
        E=[1.0],
        nu=[0.3],
        penal=3.0,
        solver="Direct",
        filter_type=filter_type,
        filter_radius=1.5 if filter_type != "None" else 0.0,
    )
    return fem


def _clamp_left_edge(grid: StructuredGrid) -> list[Support]:
    return [Support(x=0, y=j, fix_x=True, fix_y=True) for j in range(grid.nely + 1)]


def test_fd_sensitivity_rigid():
    """dC/drho from the adjoint expression must match finite differences.

    Uses spring-free loads so that compliance == f^T u exactly.
    """
    fem = _cantilever_fem("None")
    supports = _clamp_left_edge(fem.grid)
    loads_in = [Load(x=4, y=1, axis=1, sign=1, magnitude=1.0, spring=0.0)]
    fem.setup_boundary_conditions(loads_in, [], supports)

    rng = np.random.default_rng(3)
    x = rng.uniform(0.3, 0.9, fem.nel)

    ui, uo = fem.solve(x)
    dc, _ = fem.compute_sensitivities(x, ui, uo)

    h = 1e-6
    for e in range(0, fem.nel, 3):  # spot-check a third of the elements
        xp = x.copy()
        xp[e] += h
        uip, uop = fem.solve(xp)
        cp = fem.compute_objective(xp, uip, uop)
        xm = x.copy()
        xm[e] -= h
        uim, uom = fem.solve(xm)
        cm = fem.compute_objective(xm, uim, uom)
        fd = (cp - cm) / (2 * h)
        np.testing.assert_allclose(dc[e], fd, rtol=1e-4, atol=1e-10)


def test_fd_sensitivity_compliant():
    """Compliant-mechanism sensitivity must match -d(u_out)/drho.

    The adjoint identity holds with artificial springs because both state and
    adjoint solves use the same (spring-augmented) stiffness matrix.
    """
    fem = _cantilever_fem("None")
    supports = _clamp_left_edge(fem.grid)
    loads_in = [Load(x=4, y=0, axis=0, sign=1, magnitude=1.0)]
    loads_out = [Load(x=4, y=3, axis=1, sign=-1, magnitude=1.0)]
    fem.setup_boundary_conditions(loads_in, loads_out, supports)

    rng = np.random.default_rng(7)
    x = rng.uniform(0.3, 0.9, fem.nel)

    ui, uo = fem.solve(x)
    dc, _ = fem.compute_sensitivities(x, ui, uo)

    h = 1e-6
    for e in range(0, fem.nel, 3):
        xp = x.copy()
        xp[e] += h
        uip, uop = fem.solve(xp)
        jp = fem.compute_objective(xp, uip, uop)
        xm = x.copy()
        xm[e] -= h
        uim, uom = fem.solve(xm)
        jm = fem.compute_objective(xm, uim, uom)
        fd = (jp - jm) / (2 * h)
        # dc is the *descent* sensitivity: dc = -dJ/drho (J maximized).
        np.testing.assert_allclose(dc[e], -fd, rtol=1e-4, atol=1e-10)


def test_fd_sensitivity_rigid_multiple_simultaneous_loads_with_springs():
    """Combined-load compliance and sensitivity must include port springs."""
    fem = _cantilever_fem("None")
    supports = _clamp_left_edge(fem.grid)
    loads_in = [
        Load(x=4, y=0, axis=1, sign=1, magnitude=0.7),
        Load(x=4, y=3, axis=1, sign=1, magnitude=1.3),
    ]
    fem.setup_boundary_conditions(loads_in, [], supports)

    x = np.linspace(0.35, 0.85, fem.nel)
    ui, uo = fem.solve(x)
    dc, _ = fem.compute_sensitivities(x, ui, uo)

    h = 1e-6
    for e in range(0, fem.nel, 3):
        xp = x.copy()
        xp[e] += h
        uip, uop = fem.solve(xp)
        cp = fem.compute_objective(xp, uip, uop)
        xm = x.copy()
        xm[e] -= h
        uim, uom = fem.solve(xm)
        cm = fem.compute_objective(xm, uim, uom)
        fd = (cp - cm) / (2 * h)
        np.testing.assert_allclose(dc[e], fd, rtol=1e-4, atol=1e-10)


def test_fd_sensitivity_compliant_multiple_ports_with_unequal_springs():
    """Adjoint must match mean output displacement despite unequal springs."""
    fem = _cantilever_fem("None")
    supports = _clamp_left_edge(fem.grid)
    loads_in = [
        Load(x=4, y=0, axis=0, sign=1, magnitude=0.8),
        Load(x=4, y=3, axis=0, sign=1, magnitude=1.2),
    ]
    loads_out = [
        Load(x=4, y=1, axis=1, sign=1, magnitude=0.2),
        Load(x=4, y=2, axis=1, sign=-1, magnitude=0.6),
    ]
    fem.setup_boundary_conditions(loads_in, loads_out, supports)

    x = np.linspace(0.35, 0.85, fem.nel)
    ui, uo = fem.solve(x)
    dc, _ = fem.compute_sensitivities(x, ui, uo)

    h = 1e-6
    for e in range(0, fem.nel, 3):
        xp = x.copy()
        xp[e] += h
        uip, uop = fem.solve(xp)
        jp = fem.compute_objective(xp, uip, uop)
        xm = x.copy()
        xm[e] -= h
        uim, uom = fem.solve(xm)
        jm = fem.compute_objective(xm, uim, uom)
        fd = (jp - jm) / (2 * h)
        np.testing.assert_allclose(dc[e], -fd, rtol=1e-4, atol=1e-10)


# --- Solver equivalence -------------------------------------------------------


def test_cg_matches_direct_solver():
    """Iterative (CG) and direct solutions must agree."""
    grid = StructuredGrid(8, 6, 0)
    supports = [Support(x=0, y=j, fix_x=True, fix_y=True) for j in range(7)]
    loads_in = [Load(x=8, y=3, axis=1, sign=1, magnitude=1.0)]

    rng = np.random.default_rng(11)
    x = rng.uniform(0.2, 1.0, grid.nel)

    results = {}
    for solver in ("Direct", "Iterative"):
        fem = FEM(grid, E=[1.0], nu=[0.3], solver=solver)
        fem.setup_boundary_conditions(loads_in, [], supports)
        ui, _ = fem.solve(x)
        results[solver] = ui

    np.testing.assert_allclose(
        results["Iterative"], results["Direct"], rtol=1e-5, atol=1e-9
    )


# --- Filter behavior ----------------------------------------------------------


def test_density_filter_preserves_uniform_field():
    """A row-normalized density filter must leave a uniform field unchanged."""
    fem = FEM(StructuredGrid(6, 4, 0), filter_type="Density", filter_radius=2.0)
    x = np.full(fem.nel, 0.42)
    np.testing.assert_allclose(fem.update_xPhys(x), x, rtol=1e-12)


def test_density_filter_smooths_checkerboard():
    """The density filter must damp a checkerboard pattern."""
    fem = FEM(StructuredGrid(6, 4, 0), filter_type="Density", filter_radius=2.0)
    ex, ey, _ = fem.grid.element_coordinates()
    x = ((ex + ey) % 2).astype(float)
    xf = fem.update_xPhys(x)
    # Variance must shrink, mean must be preserved approximately.
    assert np.var(xf) < 0.25 * np.var(x)
    np.testing.assert_allclose(np.mean(xf), np.mean(x), atol=0.05)


def test_no_filter_is_identity():
    """filter_type='None' must leave design variables untouched."""
    fem = FEM(StructuredGrid(4, 4, 0), filter_type="None", filter_radius=2.0)
    rng = np.random.default_rng(5)
    x = rng.uniform(0, 1, fem.nel)
    np.testing.assert_array_equal(fem.update_xPhys(x), x)


# --- Assembly and direct solver ----------------------------------------------


@pytest.mark.parametrize("dims", [(5, 4, 0), (3, 2, 2)])
def test_pattern_assembly_matches_triplet_assembly(dims):
    """The cached sparsity pattern must give the same reduced matrix as a
    plain COO assembly followed by the extraction of the free DOFs."""
    from scipy.sparse import coo_matrix

    grid = StructuredGrid(*dims)
    fem = FEM(grid, E=[1.0, 0.4], nu=[0.3, 0.2])
    supports = [Support(x=0, y=1, z=0, radius=1.5, fix_x=True, fix_y=True, fix_z=True)]
    loads = [Load(x=dims[0], y=1, z=0, axis=1, sign=1, magnitude=0.5, spring=0.25)]
    fem.setup_boundary_conditions(loads, loads, supports)

    x = np.random.default_rng(0).uniform(0.05, 0.5, (2, fem.nel))
    values = fem._linear_element_values(x)
    full = coo_matrix(
        (values.ravel(), (fem.iK, fem.jK)), shape=(fem.ndof, fem.ndof)
    ).toarray()
    spring_dof = fem.in_dofs[0]
    full[spring_dof, spring_dof] += 0.25 + 0.25  # input and output springs
    expected = full[np.ix_(fem.free_dofs, fem.free_dofs)]

    assembled = fem._assemble_free(values)
    assert assembled.has_canonical_format
    np.testing.assert_allclose(assembled.toarray(), expected, atol=1e-14)


def test_pattern_is_rebuilt_only_when_the_supports_change():
    grid = StructuredGrid(4, 3, 0)
    fem = FEM(grid)
    supports = _clamp_left_edge(grid)
    fem.setup_boundary_conditions([Load(x=4, y=1, axis=1)], [], supports)
    pattern = fem._assembly_pattern()

    fem.setup_boundary_conditions([Load(x=4, y=2, axis=0)], [], supports)
    assert fem._assembly_pattern() is pattern  # loads moved: same free DOFs

    fem.setup_boundary_conditions([Load(x=4, y=2, axis=0)], [], supports[:2])
    assert fem._assembly_pattern() is not pattern
    assert fem._assembly_pattern().size == fem.free_dofs.size


def test_direct_solver_solves_several_right_hand_sides():
    from topoptcomec.core.direct_solver import DirectSolver

    grid = StructuredGrid(6, 5, 0)
    fem = FEM(grid)
    fem.setup_boundary_conditions([Load(x=6, y=2, axis=1)], [], _clamp_left_edge(grid))
    x = np.random.default_rng(1).uniform(0.2, 1.0, grid.nel)
    K = fem._assemble_free(fem._linear_element_values(x))
    rhs = np.random.default_rng(2).normal(size=(K.shape[0], 3))

    solve = DirectSolver().factorize(K)
    np.testing.assert_allclose(K @ solve(rhs), rhs, atol=1e-9)
    np.testing.assert_allclose(K @ solve(rhs[:, 0]), rhs[:, 0], atol=1e-9)


def test_direct_solver_reports_a_singular_matrix():
    from scipy.sparse import csc_matrix

    from topoptcomec.core.direct_solver import DirectSolver

    with pytest.raises(RuntimeError):
        DirectSolver().factorize(csc_matrix(np.zeros((3, 3))))(np.ones(3))


def test_auto_solver_is_direct_for_large_2d_meshes():
    """2D factorisations stay sparse: Jacobi-CG is never the better choice
    at the sizes the application handles."""
    fem = FEM(StructuredGrid(400, 200, 0), solver="Auto")
    assert fem._use_direct_solver(fem.ndof)
    assert FEM(StructuredGrid(2, 2, 0), solver="Direct")._use_direct_solver(10**9)
    assert not FEM(StructuredGrid(2, 2, 0), solver="Iterative")._use_direct_solver(1)


def test_iterative_solver_warm_starts_from_the_previous_solution():
    grid = StructuredGrid(8, 6, 0)
    fem = FEM(grid, solver="Iterative")
    fem.setup_boundary_conditions([Load(x=8, y=3, axis=1)], [], _clamp_left_edge(grid))
    x = np.random.default_rng(11).uniform(0.2, 1.0, grid.nel)
    first, _ = fem.solve(x)
    assert fem._warm_start is not None
    second, _ = fem.solve(x)  # starts from the converged solution
    np.testing.assert_allclose(second, first, rtol=1e-8, atol=1e-12)


def test_direct_solver_is_refined_to_full_accuracy():
    """SuperLU in symmetric mode pivots weakly; the refined solution must
    still reach a residual close to machine precision on a stiffness matrix
    with void elements (condition number ~1e9)."""
    from topoptcomec.core.direct_solver import DirectSolver

    grid = StructuredGrid(30, 20, 0)
    fem = FEM(grid)
    fem.setup_boundary_conditions(
        [Load(x=30, y=10, axis=1)], [], _clamp_left_edge(grid)
    )
    rng = np.random.default_rng(3)
    x = np.where(rng.random(grid.nel) < 0.4, 1.0, 1e-3)
    K = fem._assemble_free(fem._linear_element_values(x))
    rhs = fem._load_case_matrix()
    solution = DirectSolver().factorize(K)(rhs)
    assert np.linalg.norm(K @ solution - rhs) <= 1e-11 * np.linalg.norm(rhs)
