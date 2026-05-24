# tests/test_optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the optimizers.

import json
from pathlib import Path

import numpy as np
import pytest

from app.core import initializers, optimizers


def test_oc_update_rule():
    """Unit Test: Checks the Optimality Criteria function."""
    nel = 4
    x = np.array([0.5, 0.5, 0.5, 0.5])
    eta = 0.3
    max_change = 0.05
    dc = np.array([-1.0, -0.5, 0.0, 0.5])
    dv = np.array([1.0, 1.0, 1.0, 1.0])
    g = 0.0

    xnew, gt = optimizers._oc(nel, x, eta, max_change, dc, dv, g)

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

    # gt should be close to zero (KKT condition)
    assert abs(gt) < 1e-3, "KKT condition should be satisfied (gt close to zero)"


# Helper function to load the presets file for the test
def _load_presets():
    """Finds and loads the presets.json file."""
    # Go up two directories from this test file to find the project root
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r") as f:
        presets_data = json.load(f)

    # Return the presets as a list of tuples for pytest
    return presets_data.items()


def _is_multimaterial(preset_params: dict) -> bool:
    return len(preset_params.get("Materials", {}).get("E", [1.0])) > 1


@pytest.mark.parametrize("preset_name, preset_params", _load_presets())
def test_optimizers_with_presets(preset_name: str, preset_params: dict):
    """Unit Test: Runs the optimizer with a given preset."""
    is_3d = preset_params["Dimensions"]["nelxyz"][2] > 0
    is_multi = _is_multimaterial(preset_params)

    # Prepare the parameters for the optimizer function
    params = preset_params.copy()
    if "Displacement" in params:
        params.pop("Displacement", None)
    if "Materials" in params:
        params["Materials"].pop("color", None)
        if not is_multi:
            params["Materials"].pop("percent", None)

    # Run the entire optimization
    if is_multi:
        result, u_vec = optimizers.optimize_multimaterial(**params)
    else:
        result, u_vec = optimizers.optimize(**params)

    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u_vec is not None, "Displacement vector is None"

    # Check shape
    pd = params["Dimensions"]
    pf = params["Forces"]
    nel = pd["nelxyz"][0] * pd["nelxyz"][1] * (pd["nelxyz"][2] if is_3d else 1)

    if is_multi:
        n_mat = len(params["Materials"]["E"])
        assert result.shape == (
            n_mat,
            nel,
        ), f"Multi-material result shape is wrong, expected ({n_mat}, {nel})"
        assert np.all(result >= 0), "All densities should be >= 0"
        assert np.all(result <= 1.0 + 1e-6), "All densities should be <= 1"
        for i in range(n_mat):
            volfrac = params["Dimensions"]["volfrac"]
            target_vf = volfrac * params["Materials"]["percent"][i] / 100.0
            actual_vf = result[i].mean()
            assert np.isclose(actual_vf, target_vf, atol=0.15), (
                f"Material {i} volume ({actual_vf:.3f}) far from target ({target_vf:.3f})"
            )
    else:
        assert result.shape == (nel,), f"Result shape is wrong, expected ({nel},)"
        volfrac = preset_params["Dimensions"]["volfrac"]
        assert np.isclose(result.mean(), volfrac, atol=0.1), (
            f"Final volume ({result.mean():.3f}) is far to target ({volfrac:.3f})"
        )

    ndof = (
        (3 if is_3d else 2)
        * (pd["nelxyz"][0] + 1)
        * (pd["nelxyz"][1] + 1)
        * ((pd["nelxyz"][2] + 1) if is_3d else 1)
    )
    assert u_vec.size == ndof * sum(1 for x in pf["fidir"] if x != "-"), (
        "Displacement vector should be (ndof x nb_active_iforces)"
    )

    if not is_multi:
        # Check displacement direction at the input forces
        j = 0
        active_iforces_indices = [
            i for i in range(len(pf["fidir"])) if pf["fidir"][i] != "-"
        ]
        for i in active_iforces_indices:
            idx = (
                (pf["fiz"][i] * pd["nelxyz"][0] * pd["nelxyz"][1] if is_3d else 0)
                + pf["fix"][i] * pd["nelxyz"][1]
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
            assert u_vec[idx, j] * direction_sign > 0
            j += 1


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


def test_multimaterial_bridge_2d():
    """Unit Test: Runs a small 2-material bridge optimization."""
    params = {
        "Dimensions": {"nelxyz": [10, 5, 0], "volfrac": 0.5},
        "Supports": {
            "sdim": ["XYZ", "XYZ"],
            "sx": [0, 10],
            "sy": [0, 0],
            "sz": [0, 0],
            "sr": [0, 0],
        },
        "Forces": {
            "fix": [5, 0],
            "fiy": [0, 0],
            "fiz": [0, 0],
            "fidir": ["Y:\u2193", "-"],
            "finorm": [0.01, 0.0],
            "fox": [0, 0],
            "foy": [0, 0],
            "foz": [0, 0],
            "fodir": ["-", "-"],
            "fonorm": [0.0, 0.0],
        },
        "Materials": {
            "E": [1.0, 0.3],
            "nu": [0.3, 0.3],
            "percent": [40, 60],
            "init_type": 0,
        },
        "Optimizer": {
            "filter_type": "Sensitivity",
            "filter_radius_min": 1.3,
            "penal": 3.0,
            "eta": 0.3,
            "max_change": 0.05,
            "n_it": 3,
            "solver": "Auto",
        },
    }

    result, u_vec = optimizers.optimize_multimaterial(**params)
    nel = 10 * 5

    # Shape
    assert result.shape == (2, nel), f"Expected (2, {nel}), got {result.shape}"

    # All densities in range
    assert np.all(result >= 0)
    assert np.all(result <= 1.0 + 1e-6)

    # Displacement vector exists
    assert u_vec is not None
    assert u_vec.size > 0
