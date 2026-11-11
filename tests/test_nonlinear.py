# tests/test_nonlinear.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Numerical validation of the geometrically non-linear solver: tangent
# consistency, linear limit, adjoint sensitivities against finite differences,
# load control and the non-linear optimization loop.

import numpy as np
import pytest
from scipy.sparse import coo_matrix

from topoptcomec.core import nonlinear as nonlinear_module
from topoptcomec.core import optimizers
from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Support
from topoptcomec.core.nonlinear import LOAD_RAMP_SHARE, NonlinearFEM
from topoptcomec.parameter_check import ParameterCheck
from topoptcomec.result_cache import params_hash

GRIDS = [(6, 4, 0), (3, 3, 2)]


def _clamped(grid: StructuredGrid) -> list[Support]:
    """Clamp the whole x = 0 face."""
    return [
        Support(x=0, y=j, z=k, fix_x=True, fix_y=True, fix_z=True)
        for j in range(grid.nely + 1)
        for k in range(grid.nelz + 1)
    ]


def _problem(
    cls: type[FEM],
    dims: tuple[int, int, int],
    mechanism: bool,
    n_mat: int = 1,
    filter_type: str = "None",
    **kwargs: float,
) -> FEM:
    """Cantilever loaded at its tip, optionally with an output port."""
    grid = StructuredGrid(*dims)
    fem = cls(
        grid,
        E=[1.0, 0.6][:n_mat],
        nu=[0.3, 0.2][:n_mat],
        solver="Direct",
        filter_type=filter_type,
        filter_radius=1.5 if filter_type != "None" else 0.0,
        **kwargs,
    )
    nx, ny, nz = dims
    loads_in = [
        Load(x=nx, y=ny // 2, z=nz // 2, axis=1, sign=1, magnitude=0.02, spring=0.01)
    ]
    loads_out = (
        [Load(x=nx, y=0, z=0, axis=0, sign=-1, magnitude=0.01, spring=0.005)]
        if mechanism
        else []
    )
    fem.setup_boundary_conditions(loads_in, loads_out, _clamped(grid))
    return fem


def _densities(nel: int, n_mat: int, seed: int = 1) -> np.ndarray:
    """Random densities with solid, gray and void elements, sum <= 1."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.05, 0.45, (n_mat, nel))
    x[0, ::3] = 0.95 - x[1:, ::3].sum(axis=0)
    x[:, 1::7] = 0.01
    return x


# --- Element formulation ------------------------------------------------------


@pytest.mark.parametrize("dims", GRIDS)
@pytest.mark.parametrize("n_mat", [1, 2])
def test_tangent_is_the_derivative_of_the_internal_force(dims, n_mat):
    """The assembled tangent must equal d(f_int)/du (and be symmetric)."""
    fem = _problem(NonlinearFEM, dims, mechanism=True, n_mat=n_mat)
    x = _densities(fem.nel, n_mat)
    E_eff = fem._material_stiffness(x)
    gamma, _ = fem._interpolation(x)
    u = np.random.default_rng(2).normal(0.0, 0.15, fem.ndof)

    def internal_force(disp: np.ndarray) -> np.ndarray:
        forces, _, _ = fem._element_response(disp, E_eff, gamma, tangent=False)
        return np.bincount(fem.edofMat.ravel(), forces.ravel(), minlength=fem.ndof)

    _, tangents, _ = fem._element_response(u, E_eff, gamma, tangent=True)
    K = coo_matrix(
        (tangents.ravel(), (fem.iK, fem.jK)), shape=(fem.ndof, fem.ndof)
    ).toarray()

    h = 1e-6
    K_fd = np.empty_like(K)
    for i in range(fem.ndof):
        step = np.zeros(fem.ndof)
        step[i] = h
        K_fd[:, i] = (internal_force(u + step) - internal_force(u - step)) / (2 * h)

    scale = np.abs(K).max()
    np.testing.assert_allclose(K, K_fd, atol=1e-7 * scale)
    np.testing.assert_allclose(K, K.T, atol=1e-12 * scale)


@pytest.mark.parametrize("dims", GRIDS)
def test_internal_force_is_the_gradient_of_the_energy(dims):
    """The line-search merit (strain energy) must be consistent with f_int."""
    fem = _problem(NonlinearFEM, dims, mechanism=False)
    x = _densities(fem.nel, 1)
    E_eff = fem._material_stiffness(x)
    gamma, _ = fem._interpolation(x)
    rng = np.random.default_rng(4)
    u = rng.normal(0.0, 0.1, fem.ndof)
    direction = rng.normal(0.0, 1.0, fem.ndof)

    forces, _, _ = fem._element_response(u, E_eff, gamma, tangent=False)
    f_int = np.bincount(fem.edofMat.ravel(), forces.ravel(), minlength=fem.ndof)

    h = 1e-6
    e_plus = fem._element_response(u + h * direction, E_eff, gamma, False)[2]
    e_minus = fem._element_response(u - h * direction, E_eff, gamma, False)[2]
    np.testing.assert_allclose(
        (e_plus - e_minus) / (2 * h), f_int @ direction, rtol=1e-6
    )


def test_void_elements_are_linear_and_solid_elements_non_linear():
    """Energy interpolation: gamma is 0 in the void and 1 in solid material."""
    fem = NonlinearFEM(StructuredGrid(2, 1, 0), penal=3.0)
    gamma, _ = fem._interpolation(np.array([[1e-6, 1.0]]))
    np.testing.assert_allclose(gamma, [0.0, 1.0], atol=1e-3)


# --- Equilibrium --------------------------------------------------------------


@pytest.mark.parametrize("dims", GRIDS)
@pytest.mark.parametrize("n_mat", [1, 2])
def test_small_load_matches_the_linear_solver(dims, n_mat):
    """For vanishing loads both solvers (state and adjoint) must agree."""
    factor = 1e-5
    x = _densities(StructuredGrid(*dims).nel, n_mat)
    linear = _problem(FEM, dims, mechanism=True, n_mat=n_mat)
    nonlinear = _problem(
        NonlinearFEM, dims, mechanism=True, n_mat=n_mat, load_factor=factor
    )

    ui, uo = linear.solve(x)
    u, lam = nonlinear.solve(x)

    assert nonlinear.last_load_fraction == 1.0
    np.testing.assert_allclose(u / factor, ui, atol=1e-4 * np.abs(ui).max())
    np.testing.assert_allclose(lam, uo, atol=1e-4 * np.abs(uo).max())


def test_equilibrium_residual_vanishes():
    """The returned state must satisfy f_int + K_s u = f_ext on the free DOFs."""
    fem = _problem(NonlinearFEM, (8, 4, 0), mechanism=False, load_factor=6.0)
    x = np.full(fem.nel, 0.6)
    u, _ = fem.solve(x)

    f_ext = fem._load_case_matrix()[:, 0]
    gamma, _ = fem._interpolation(np.atleast_2d(x))
    residual, _, _ = fem._residual(u, fem._material_stiffness(x), gamma, f_ext, False)

    assert fem.last_load_fraction == 1.0
    assert np.linalg.norm(residual) <= 1e-6 * np.linalg.norm(f_ext)
    assert np.all(u[fem.fixed_dofs] == 0.0)


def test_large_displacement_differs_from_linear_prediction():
    """A transversely loaded cantilever stiffens: the tip moves less than
    the linear solution, and its end shortens (a purely non-linear effect)."""
    dims = (16, 2, 0)
    grid = StructuredGrid(*dims)
    load = [Load(x=16, y=1, axis=1, sign=1, magnitude=0.004, spring=0.0)]
    x = np.ones(grid.nel)

    linear = FEM(grid, E=[1.0], nu=[0.3], solver="Direct")
    linear.setup_boundary_conditions(load, [], _clamped(grid))
    nonlinear = NonlinearFEM(grid, E=[1.0], nu=[0.3], solver="Direct")
    nonlinear.setup_boundary_conditions(load, [], _clamped(grid))

    u_lin, _ = linear.solve(x)
    u_nl, _ = nonlinear.solve(x)
    tip = grid.dofs_per_node * grid.node_index(16, 1)

    assert nonlinear.last_load_fraction == 1.0
    assert abs(u_lin[tip + 1]) > 4.0  # genuinely large displacement
    assert abs(u_nl[tip + 1]) < 0.9 * abs(u_lin[tip + 1])
    assert abs(u_lin[tip]) < 1e-9  # linear theory: no axial motion
    assert u_nl[tip] < -0.5  # the tip moves back towards the clamp


def test_load_factor_scales_forces_but_not_springs():
    fem = _problem(NonlinearFEM, (6, 4, 0), mechanism=True, load_factor=4.0)
    reference = _problem(FEM, (6, 4, 0), mechanism=True)
    np.testing.assert_allclose(fem.forces_i, 4.0 * reference.forces_i)
    np.testing.assert_allclose(fem.forces_o, reference.forces_o)
    np.testing.assert_allclose(fem._spring_vals, reference._spring_vals)


@pytest.mark.parametrize("load_factor", [0.0, -1.0, np.inf, np.nan])
def test_invalid_load_factor_is_rejected(load_factor):
    with pytest.raises(ValueError, match="load factor"):
        NonlinearFEM(StructuredGrid(2, 2, 0), load_factor=load_factor)


def test_unreachable_load_returns_the_reached_level():
    """A column compressed far beyond collapse has no equilibrium: the solver
    must not raise, but report the load level it reached and return the
    equilibrium of that level."""
    grid = StructuredGrid(12, 2, 0)
    fem = NonlinearFEM(grid, E=[1.0], nu=[0.3], solver="Direct")
    crushing = [Load(x=12, y=1, axis=0, sign=-1, magnitude=5.0, spring=0.0)]
    fem.setup_boundary_conditions(crushing, [], _clamped(grid))
    x = np.ones(fem.nel)
    u, lam = fem.solve(x)

    assert 0.0 <= fem.last_load_fraction < 1.0
    assert np.all(np.isfinite(u)) and np.all(np.isfinite(lam))
    f_ext = fem.last_load_fraction * fem._load_case_matrix()[:, 0]
    gamma, _ = fem._interpolation(np.atleast_2d(x))
    residual, _, _ = fem._residual(u, fem._material_stiffness(x), gamma, f_ext, False)
    assert np.linalg.norm(residual) <= 1e-5 * max(np.linalg.norm(f_ext), 1e-12)


def test_warm_start_gives_the_same_equilibrium():
    """Restarting from the previous design must not change the solution."""
    x = np.full(48, 0.6)
    perturbed = x + np.random.default_rng(0).uniform(-0.05, 0.05, x.size)

    warm = _problem(NonlinearFEM, (8, 6, 0), mechanism=True, load_factor=5.0)
    warm.solve(x)
    u_warm, lam_warm = warm.solve(perturbed)
    cold = _problem(NonlinearFEM, (8, 6, 0), mechanism=True, load_factor=5.0)
    u_cold, lam_cold = cold.solve(perturbed)

    np.testing.assert_allclose(u_warm, u_cold, atol=1e-6 * np.abs(u_cold).max())
    np.testing.assert_allclose(lam_warm, lam_cold, atol=1e-5 * np.abs(lam_cold).max())


# --- Sensitivities ------------------------------------------------------------


@pytest.mark.parametrize("dims,load_factor", [((6, 4, 0), 6.0), ((3, 3, 2), 25.0)])
@pytest.mark.parametrize("mechanism", [True, False])
@pytest.mark.parametrize("filter_type", ["None", "Density", "Sensitivity"])
@pytest.mark.parametrize("n_mat", [1, 2])
def test_sensitivities_match_finite_differences(
    dims, load_factor, mechanism, filter_type, n_mat, monkeypatch
):
    """Adjoint sensitivities of the non-linear objective against central
    finite differences, in a regime where the non-linearity matters."""
    # Finite differences need a state converged far below the perturbation.
    monkeypatch.setattr(nonlinear_module, "NEWTON_TOLERANCE", 1e-12)
    fem = _problem(
        NonlinearFEM,
        dims,
        mechanism,
        n_mat=n_mat,
        filter_type=filter_type,
        load_factor=load_factor,
    )
    x = _densities(fem.nel, n_mat)

    def objective(design: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        physical = np.vstack([fem.update_xPhys(row) for row in design])
        u, lam = fem.solve(physical)
        assert fem.last_load_fraction == 1.0
        return fem.compute_objective(physical, u, lam), u, lam

    _, u, lam = objective(x)
    linear_u, _ = _problem(FEM, dims, mechanism, n_mat=n_mat).solve(x)
    assert np.abs(u).max() > 0.5  # large displacements ...
    assert not np.allclose(u, load_factor * linear_u, rtol=0.05)  # ... non-linear

    physical = np.vstack([fem.update_xPhys(row) for row in x])
    sensitivities = fem.compute_all_sensitivities(physical, u, lam)
    if filter_type == "Sensitivity":
        # The sensitivity filter is a heuristic: it is not the derivative of
        # anything, so only its shape and finiteness can be checked.
        for dc, dv in sensitivities:
            assert dc.shape == dv.shape == (fem.nel,)
            assert np.all(np.isfinite(dc))
        return

    # The optimizer minimizes: dc is -dJ/dx for mechanisms, +dJ/dx otherwise.
    sign = -1.0 if mechanism else 1.0
    rng = np.random.default_rng(5)
    h = 1e-5
    for m in range(n_mat):
        for e in rng.choice(fem.nel, 5, replace=False):
            plus, minus = x.copy(), x.copy()
            plus[m, e] += h
            minus[m, e] -= h
            fd = (objective(plus)[0] - objective(minus)[0]) / (2 * h)
            np.testing.assert_allclose(
                sign * sensitivities[m][0][e], fd, rtol=2e-3, atol=1e-9
            )


def test_single_material_sensitivities_use_the_common_interface():
    fem = _problem(NonlinearFEM, (6, 4, 0), mechanism=True, load_factor=3.0)
    x = _densities(fem.nel, 1)[0]
    u, lam = fem.solve(x)
    dc, dv = fem.compute_sensitivities(x, u, lam)
    dc_all, dv_all = fem.compute_all_sensitivities(x, u, lam)[0]
    np.testing.assert_array_equal(dc, dc_all)
    np.testing.assert_array_equal(dv, dv_all)


# --- Load continuation --------------------------------------------------------


def test_load_is_ramped_over_the_first_iterations():
    fem = NonlinearFEM(StructuredGrid(2, 2, 0))
    n_it = 40
    limits = []
    final = []
    for iteration in range(1, n_it + 1):
        final.append(fem.begin_iteration(iteration, n_it))
        limits.append(fem.load_limit)

    ramp = int(LOAD_RAMP_SHARE * n_it)
    assert limits == sorted(limits)  # never decreases
    assert 0.0 < limits[0] < 0.1
    assert all(limit == 1.0 for limit in limits[ramp - 1 :])
    assert final == [limit == 1.0 for limit in limits]
    assert FEM(StructuredGrid(2, 2, 0)).begin_iteration(1, n_it) is True


def test_load_limit_caps_the_applied_load():
    fem = _problem(NonlinearFEM, (6, 4, 0), mechanism=False, load_factor=3.0)
    x = np.full(fem.nel, 0.6)
    fem.load_limit = 0.25
    u_quarter, _ = fem.solve(x)
    assert fem.last_load_fraction == 0.25
    fem.load_limit = 1.0
    u_full, _ = fem.solve(x)
    assert fem.last_load_fraction == 1.0
    assert np.abs(u_full).max() > 2.0 * np.abs(u_quarter).max()


# --- Optimization -------------------------------------------------------------


def _inverter(analysis: str, load_factor: float = 1.0, n_it: int = 24) -> dict:
    return {
        "Dimensions": {"nelxyz": [30, 20, 0], "volfrac": 0.3},
        "Forces": {
            "fix": [0],
            "fiy": [10],
            "fiz": [0],
            "fidir": ["X:\u2192"],
            "finorm": [0.01],
            "fox": [30],
            "foy": [10],
            "foz": [0],
            "fodir": ["X:\u2190"],
            "fonorm": [0.01],
        },
        "Supports": {
            "sx": [0, 0],
            "sy": [0, 20],
            "sz": [0, 0],
            "sdim": ["XY", "XY"],
            "sr": [1, 1],
        },
        "Materials": {"E": [1.0], "nu": [0.3], "percent": [100], "init_type": 0},
        "Optimizer": {
            "filter_type": "Sensitivity",
            "filter_radius_min": 1.5,
            "penal": 3.0,
            "eta": 0.3,
            "max_change": 0.1,
            "n_it": n_it,
            "solver": "Auto",
            "analysis": analysis,
            "load_factor": load_factor,
        },
    }


def _nonlinear_output(params: dict, x: np.ndarray, load_factor: float) -> float:
    """Output displacement of a design under the full non-linear analysis."""
    optimizer = {**params["Optimizer"], "analysis": "Nonlinear"}
    optimizer["load_factor"] = load_factor
    grid, loads_in, loads_out, supports, _ = optimizers._translate_problem(
        params["Dimensions"], params["Forces"], params["Supports"], None
    )
    fem = optimizers.make_fem(grid, params["Materials"], optimizer)
    fem.setup_boundary_conditions(loads_in, loads_out, supports)
    u, lam = fem.solve(x)
    assert fem.last_load_fraction == 1.0
    return fem.compute_objective(x, u, lam)


def test_nonlinear_optimization_builds_a_working_mechanism(capsys):
    load_factor = 4.0
    params = _inverter("Nonlinear", load_factor)
    frames: list[tuple[int, float]] = []

    xPhys, u = optimizers.optimize(
        **params,
        progress_callback=lambda it, obj, _c, _x: bool(frames.append((it, obj))),
    )

    assert xPhys.shape == (600,)
    assert np.all(np.isfinite(xPhys)) and np.all(np.isfinite(u))
    assert xPhys.min() >= 0.0 and xPhys.max() <= 1.0
    np.testing.assert_allclose(xPhys.mean(), 0.3, atol=1e-3)
    # The inverter works under the full large-displacement analysis ...
    output = _nonlinear_output(params, xPhys, load_factor)
    assert output > 0.2
    np.testing.assert_allclose(frames[-1][1], output, rtol=1e-6)
    # ... and the run reports the load level while ramping.
    log = capsys.readouterr().out
    assert "Load: 100%" in log
    assert "only reached" not in log


@pytest.mark.parametrize("factor", [1e-6, 1e-9])
def test_nonlinear_optimization_matches_linear_for_small_loads(factor):
    """With a vanishing load factor both analyses see the same physics, and
    the optimization is scale invariant, so the designs must be identical
    (up to solver round-off) - including during the load ramp.

    Regression: the strain was computed as (F^T F - I) / 2, which loses all
    digits for tiny displacement gradients; Newton then stalled, the load
    level fell to zero and the design received zero sensitivities.
    """
    linear, _ = optimizers.optimize(**_inverter("Linear"), verbose=False)
    nonlinear, _ = optimizers.optimize(**_inverter("Nonlinear", factor), verbose=False)
    np.testing.assert_allclose(nonlinear, linear, atol=1e-3)


@pytest.mark.parametrize("dims", GRIDS)
def test_newton_converges_for_tiny_loads(dims):
    """The ramp starts at 1/64 of the load: with a tiny load factor that is a
    displacement gradient of ~1e-12, which Newton must still resolve."""
    fem = _problem(NonlinearFEM, dims, mechanism=True, load_factor=1e-8)
    fem.load_limit = 1.0 / 64.0
    x = _densities(fem.nel, 1)
    u, _ = fem.solve(x)
    assert fem.last_load_fraction == fem.load_limit
    linear_u, _ = _problem(FEM, dims, mechanism=True).solve(x)
    np.testing.assert_allclose(
        u, 1e-8 / 64.0 * linear_u, atol=1e-6 * 1e-8 / 64.0 * np.abs(linear_u).max()
    )


def test_strain_is_accurate_for_small_gradients():
    fem = NonlinearFEM(StructuredGrid(1, 1, 1))
    H = np.random.default_rng(0).normal(size=(4, 3, 3)) * 1e-12
    strain = fem._strain(H)
    linear = np.column_stack(
        [H[:, 0, 0], H[:, 1, 1], H[:, 2, 2], H[:, 1, 2] + H[:, 2, 1]]
        + [H[:, 0, 2] + H[:, 2, 0], H[:, 0, 1] + H[:, 1, 0]]
    )
    np.testing.assert_allclose(strain, linear, rtol=1e-9)


def test_nonlinear_multimaterial_optimization_runs():
    params = _inverter("Nonlinear", 3.0, n_it=6)
    params["Materials"] = {
        "E": [1.0, 0.5],
        "nu": [0.3, 0.4],
        "percent": [70, 30],
        "init_type": 0,
    }
    xPhys, u = optimizers.optimize(**params, verbose=False, multimaterial=True)
    assert xPhys.shape == (2, 600)
    assert np.all(np.isfinite(xPhys)) and np.all(np.isfinite(u))
    assert xPhys.sum(axis=0).max() <= 1.0 + 1e-9
    np.testing.assert_allclose(xPhys.mean(axis=1), [0.21, 0.09], atol=1e-3)


def test_nonlinear_3d_optimization_runs():
    params = _inverter("Nonlinear", 2.0, n_it=4)
    params["Dimensions"]["nelxyz"] = [8, 6, 4]
    params["Forces"].update(fiy=[3], fiz=[2], fox=[8], foy=[3], foz=[2])
    params["Supports"].update(sx=[0, 0], sy=[0, 6], sz=[0, 0], sdim=["XYZ", "XYZ"])
    xPhys, u = optimizers.optimize(**params, verbose=False)
    assert xPhys.shape == (192,)
    assert np.all(np.isfinite(xPhys)) and np.all(np.isfinite(u))


# --- Parameter plumbing -------------------------------------------------------


def test_make_fem_selects_the_analysis():
    grid = StructuredGrid(2, 2, 0)
    materials = {"E": [1.0], "nu": [0.3]}
    assert type(optimizers.make_fem(grid, materials, {})) is FEM
    assert type(optimizers.make_fem(grid, materials, {"analysis": "Linear"})) is FEM
    fem = optimizers.make_fem(
        grid, materials, {"analysis": "Nonlinear", "load_factor": 7.0}
    )
    assert isinstance(fem, NonlinearFEM) and fem.load_factor == 7.0
    with pytest.raises(ValueError, match="Unknown analysis"):
        optimizers.make_fem(grid, materials, {"analysis": "Quadratic"})


def test_parameter_check_validates_the_analysis():
    check = ParameterCheck()
    assert check.validate(_inverter("Nonlinear", 2.0)) is None
    assert check.validate(_inverter("Linear")) is None
    assert "Unknown analysis" in check.validate(_inverter("Cubic"))
    assert "Load factor" in check.validate(_inverter("Nonlinear", 0.0))
    # The load factor is ignored by the linear analysis.
    assert check.validate(_inverter("Linear", 0.0)) is None
    bad = _inverter("Nonlinear")
    bad["Optimizer"]["load_factor"] = "big"
    assert "load_factor" in check.validate(bad)
    bad = _inverter("Nonlinear")
    bad["Optimizer"]["analysis"] = 3
    assert "analysis" in check.validate(bad)


def test_cache_hash_ignores_the_defaults_of_the_linear_analysis():
    """Presets written before the non-linear analysis existed keep their hash."""
    legacy = _inverter("Linear")
    del legacy["Optimizer"]["analysis"], legacy["Optimizer"]["load_factor"]

    assert params_hash(_inverter("Linear", 1.0)) == params_hash(legacy)
    assert params_hash(_inverter("Linear", 9.0)) == params_hash(legacy)
    assert params_hash(_inverter("Nonlinear", 1.0)) != params_hash(legacy)
    assert params_hash(_inverter("Nonlinear", 1.0)) != params_hash(
        _inverter("Nonlinear", 2.0)
    )
    implicit = _inverter("Nonlinear", 1.0)
    del implicit["Optimizer"]["load_factor"]
    assert params_hash(implicit) == params_hash(_inverter("Nonlinear", 1.0))
