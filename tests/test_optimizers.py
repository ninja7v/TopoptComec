# tests/test_optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the optimizers.

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from topoptcomec.core import initializers, optimizers

REFERENCES_DIR = Path(__file__).parent / "references"
# Loose enough to absorb BLAS/library differences across platforms, tight
# enough to catch real regressions (ordering bugs, sign errors, formula
# changes produce O(1) relative deviations).
REFERENCE_RTOL = 1e-3
REFERENCE_ATOL = 1e-6


def test_oc_update_rule():
    """Unit Test: Checks the Optimality Criteria function."""
    nel = 4
    x = np.array([0.5, 0.5, 0.5, 0.5])
    eta = 0.3
    max_change = 0.05
    dc = np.array([-1.0, -0.5, 0.0, 0.5])
    dv = np.array([1.0, 1.0, 1.0, 1.0])
    target = float(x.sum())

    xnew, gt = optimizers._oc(nel, x, eta, max_change, dc, dv, target)

    # Check shape
    assert isinstance(xnew, np.ndarray), "oc should return a NumPy array"
    assert xnew.shape == (nel,), "Returned array should have shape (nel,)"

    # Check bounds
    assert np.all(xnew >= 1e-6), "All values should be >= 1e-6"
    assert np.all(xnew <= 1.0), "All values should be <= 1.0"

    # Check different
    assert not np.allclose(xnew, x), (
        "Updated values should differ from original if sensitivities are nonzero"
    )

    # The volume constraint residual must be small (target sum preserved).
    assert abs(gt) < 1e-3, "The design variables must sum to the target"
    assert abs(xnew.sum() - target) < 1e-3

    # Passive elements must keep their value.
    free = np.array([True, True, False, False])
    target_free = float(x[free].sum())
    xnew_free, gt_free = optimizers._oc(
        nel, x, eta, max_change, dc, dv, target_free, free
    )
    np.testing.assert_array_equal(xnew_free[~free], x[~free])
    assert abs(gt_free) < 1e-6


def test_oc_respects_upper_bound_and_custom_volume():
    """The OC bound enforces the partition of unity; volume can be physical."""
    nel = 6
    x = np.full(nel, 0.4)
    dc = -np.linspace(1.0, 2.0, nel)
    dv = np.ones(nel)
    upper = np.array([0.42, 0.42, 1.0, 1.0, 1.0, 1.0])
    weights = np.linspace(0.5, 1.5, nel)

    def volume(values):
        return float(np.sum(weights * values))

    target = volume(x)
    xnew, gt = optimizers._oc(
        nel, x, 0.5, 0.2, dc, dv, target, upper=upper, volume=volume
    )
    assert np.all(xnew <= upper + 1e-15)
    assert abs(gt) < 1e-6 * target
    assert volume(xnew) == pytest.approx(target, rel=1e-6)


def _volume_case(filter_type: str, regions: dict | None, materials: dict) -> dict:
    return {
        "Dimensions": {"nelxyz": [30, 12, 0], "volfrac": 0.3},
        "Forces": {
            "fix": [30],
            "fiy": [12],
            "fiz": [0],
            "fidir": ["Y:\u2193"],
            "finorm": [1.0],
            "fox": [],
            "foy": [],
            "foz": [],
            "fodir": [],
            "fonorm": [],
        },
        "Supports": {
            "sx": [0, 0],
            "sy": [0, 12],
            "sz": [0, 0],
            "sdim": ["XYZ", "XYZ"],
            "sr": [0, 0],
        },
        "Regions": regions,
        "Materials": materials,
        "Optimizer": {
            "filter_type": filter_type,
            "filter_radius_min": 1.5,
            "penal": 3.0,
            "eta": 0.5,
            "max_change": 0.2,
            "n_it": 8,
            "solver": "Direct",
        },
    }


@pytest.mark.parametrize("filter_type", ["Sensitivity", "Density", "None"])
@pytest.mark.parametrize("region_state", [None, "Material 1", "Void"])
@pytest.mark.parametrize("n_mat", [1, 2])
def test_volume_constraint_is_exact(filter_type, region_state, n_mat):
    """Every material meets its target on the physical field, for every filter.

    The case is a compliance problem, so the volume constraint is active and
    must hold with equality. Regression for F01: the accumulated linearized
    constraint drifted with regions, non-uniform starts and multi-material
    normalization.
    """
    regions = None
    if region_state is not None:
        regions = {
            "rshape": ["\u25a1"],
            "rstate": [region_state],
            "rradius": [3],
            "rx": [15],
            "ry": [6],
            "rz": [0],
        }
    if n_mat == 1:
        materials = {"E": [1.0], "nu": [0.3], "init_type": 1}
    else:
        materials = {
            "E": [1.0, 0.5],
            "nu": [0.3, 0.45],
            "percent": [70, 30],
            "init_type": 1,
        }
    params = _volume_case(filter_type, regions, materials)
    result, _ = optimizers.optimize(**params, verbose=False, multimaterial=n_mat > 1)
    field = result if n_mat > 1 else result[np.newaxis, :]
    _, free = _expected_volumes(params, n_mat, 0.3, field.shape[1])
    shares = np.array(materials.get("percent", [100]), dtype=float) / 100
    for m in range(n_mat):
        assert field[m][free].mean() == pytest.approx(0.3 * shares[m], abs=1e-6)
    # Partition of unity on the physical field.
    assert np.all(field.sum(axis=0) <= 1.0 + 1e-6)


# Helper function to load the presets file for the test
def _load_presets():
    """Finds and loads the presets.json file."""
    # Go up two directories from this test file to find the project root
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r") as f:
        presets_data = json.load(f)

    # Return the presets as a list of tuples for pytest
    return presets_data.items()


def _expected_volumes(
    params: dict, n_mat: int, volfrac: float, nel: int
) -> tuple[np.ndarray, np.ndarray]:
    """Expected per-material mean density, accounting for passive regions."""
    from topoptcomec.core import preset_format
    from topoptcomec.core.fem import FEM
    from topoptcomec.core.grid import StructuredGrid

    nelx, nely, nelz = params["Dimensions"]["nelxyz"]
    fem = FEM(StructuredGrid(nelx, nely, max(0, nelz)))
    regions = preset_format.parse_regions(params.get("Regions"))
    pct = np.asarray(params["Materials"].get("percent", [100] * n_mat), dtype=float)
    shares = pct / pct.sum()
    free = np.ones(fem.nel, dtype=bool)
    expected = np.zeros(n_mat, dtype=float)
    for m in range(n_mat):
        solid, void = fem.region_masks(regions, mat_idx=m)
        free &= ~(solid | void)
        expected[m] = float(solid.sum()) + 1e-6 * float(void.sum())
    for m in range(n_mat):
        expected[m] = (expected[m] + float(free.sum()) * volfrac * shares[m]) / nel
    return expected, free


@pytest.mark.filterwarnings("ignore:No support is defined")
@pytest.mark.parametrize("preset_name, preset_params", _load_presets())
def test_optimizers_with_presets(preset_name: str, preset_params: dict):
    """Unit Test: Runs the optimizer with a given preset."""
    is_3d = preset_params["Dimensions"]["nelxyz"][2] > 0
    n_mat = len(preset_params.get("Materials", {}).get("E", [1.0]))
    is_multi = n_mat > 1

    # Prepare the parameters for the optimizer function
    params = copy.deepcopy(preset_params)
    if "Displacement" in params:
        params.pop("Displacement", None)
    if "Materials" in params:
        params["Materials"].pop("color", None)
        if not is_multi:
            params["Materials"].pop("percent", None)

    # Run the entire optimization
    result, u_vec = optimizers.optimize(**params, multimaterial=is_multi)

    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u_vec is not None, "Displacement vector is None"

    # Check shape
    pd = params["Dimensions"]
    pf = params["Forces"]
    nel = pd["nelxyz"][0] * pd["nelxyz"][1] * (pd["nelxyz"][2] if is_3d else 1)
    volfrac = pd["volfrac"]
    is_rigid = not any(d != "-" for d in pf.get("fodir", []))

    if is_multi:
        assert result.shape == (
            n_mat,
            nel,
        ), f"Multi-material result shape is wrong, expected ({n_mat}, {nel})"
        assert np.all(result >= 0), "All densities should be >= 0"
        assert np.all(result <= 1.0 + 1e-6), "All densities should be <= 1"
        # Volume targets: volfrac applies to the designable elements; regions
        # are passive (solid for the targeted material, void for the others).
        expected, _ = _expected_volumes(params, n_mat, volfrac, nel)
        for i in range(n_mat):
            target_vf = expected[i]
            actual_vf = result[i].mean()
            # The volume constraint is an upper bound; it is active (met with
            # equality) for compliance problems.
            assert actual_vf <= target_vf + 1e-6, (
                f"Material {i} volume ({actual_vf:.4f}) exceeds the "
                f"constraint ({target_vf:.4f})"
            )
            if is_rigid:
                assert np.isclose(actual_vf, target_vf, atol=1e-6)
        # Partition of unity: no element holds more than one unit of material.
        assert np.all(result.sum(axis=0) <= 1.0 + 1e-6)
    else:
        assert result.shape == (nel,), f"Result shape is wrong, expected ({nel},)"
        expected, _ = _expected_volumes(params, 1, volfrac, nel)
        assert result.mean() <= expected[0] + 1e-6, (
            f"Final volume ({result.mean():.4f}) exceeds the constraint "
            f"({expected[0]:.4f})"
        )
        if is_rigid:
            assert np.isclose(result.mean(), expected[0], atol=1e-6)

    ndof = (
        (3 if is_3d else 2)
        * (pd["nelxyz"][0] + 1)
        * (pd["nelxyz"][1] + 1)
        * ((pd["nelxyz"][2] + 1) if is_3d else 1)
    )
    assert u_vec.shape == (ndof,), (
        "Displacement vector should be (ndof,) for the combined-load solve"
    )

    if not is_multi:
        # Combined-load solve: per-input displacement alignment is not
        # guaranteed (opposing inputs partially cancel). Instead verify the
        # combined input load does positive work on the combined displacement
        # (strain energy = f^T u > 0 for a positive-definite system).
        active_iforces_indices = [
            i for i in range(len(pf["fidir"])) if pf["fidir"][i] != "-"
        ]
        work = 0.0
        for i in active_iforces_indices:
            # Node index from StructuredGrid.node_index (x-major ordering).
            idx = (
                (
                    pf["fiz"][i] * (pd["nelxyz"][0] + 1) * (pd["nelxyz"][1] + 1)
                    if is_3d
                    else 0
                )
                + pf["fix"][i] * (pd["nelxyz"][1] + 1)
                + pf["fiy"][i]
            )
            idx = idx * (3 if is_3d else 2) + (
                0
                if pf["fidir"][i] == "X:\u2192" or pf["fidir"][i] == "X:\u2190"
                else (
                    1
                    if pf["fidir"][i] == "Y:\u2193" or pf["fidir"][i] == "Y:\u2191"
                    else 2
                )
            )
            direction_sign = (
                1
                if pf["fidir"][i] == "X:\u2192"
                or pf["fidir"][i] == "Y:\u2193"
                or pf["fidir"][i] == "Z:<"
                else -1
            )
            work += direction_sign * pf["finorm"][i] * u_vec[idx]
        assert work > 0, (
            "Combined input load should do positive work on the displacement"
        )

    # Compare with the reference solver output. Regenerate with
    # tests/references/regenerate_references.py after any intentional
    # numerical change, and document the change in the commit message.
    if params["Materials"]["init_type"] != 2:
        reference_path = (
            REFERENCES_DIR / f"test_optimizers_with_presets_{preset_name}.npz"
        )
        if reference_path.exists():
            with np.load(reference_path) as reference:
                np.testing.assert_allclose(
                    result,
                    reference["result"],
                    rtol=REFERENCE_RTOL,
                    atol=REFERENCE_ATOL,
                    err_msg=f"Result mismatch for preset {preset_name}",
                )
                np.testing.assert_allclose(
                    u_vec,
                    reference["u_vec"],
                    rtol=REFERENCE_RTOL,
                    atol=REFERENCE_ATOL,
                    err_msg=f"Displacement vector mismatch for preset {preset_name}",
                )


def test_initialize_materials():
    """Unit Test: Checks that initialize_materials returns correct shape and constraints."""
    percents = [40, 60]
    volfrac = 0.5
    nelx, nely, nelz = 10, 5, 0
    nel = nelx * nely
    all_x = np.array([0, 10])
    all_y = np.array([0, 5])
    all_z = np.array([0, 0])

    rho = initializers.initialize_materials(
        0, percents, volfrac, nelx, nely, nelz, all_x, all_y, all_z
    )

    # Shape check
    assert rho.shape == (2, nel), f"Expected shape (2, {nel}), got {rho.shape}"

    # All values in valid range
    assert np.all(rho >= 0), "All densities should be >= 0"
    assert np.all(rho <= 1.0 + 1e-6), "All densities should be <= 1"

    # Partition of unity: column sums close to volfrac
    col_sums = rho.sum(axis=0)
    assert np.allclose(col_sums, volfrac, atol=0.05), (
        f"Column sums should be ~{volfrac}, min={col_sums.min():.3f} max={col_sums.max():.3f}"
    )

    # Volume fractions roughly correct
    volfracs = [volfrac * p / 100.0 for p in percents]
    for i, vf in enumerate(volfracs):
        actual = rho[i].mean()
        assert np.isclose(actual, vf, atol=0.1), (
            f"Material {i} vol frac ({actual:.3f}) far from target ({vf:.3f})"
        )
