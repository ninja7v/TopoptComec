# tests/test_analyzers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the analyzers.

import json
from pathlib import Path

import numpy as np
import pytest

from topoptcomec.core import analyzers


# Helper function to load the presets file for the test
def _load_presets():
    """Finds and loads the presets.json file."""
    # Go up two directories from this test file to find the project root
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r") as f:
        presets_data = json.load(f)

    # Return the presets as a list of tuples for pytest
    return presets_data.items()


@pytest.mark.parametrize("preset_name, preset_params", _load_presets())
def test_analysis_with_presets(preset_name: str, preset_params: dict):
    """Unit Test: Runs the 2D/3D optimizer with a given preset."""
    # Prepare the parameters for the optimizer function
    disp_params = preset_params.copy()
    nelx, nely, nelz = disp_params["Dimensions"]["nelxyz"]
    is_3d = nelz > 0
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ["filter_type", "filter_radius_min", "max_change", "n_it"]
    if not is_3d:
        keys_to_remove = keys_to_remove + ["rz", "fz", "sz"]
    for key in keys_to_remove:
        disp_params.pop(key, None)

    # Generate a mock result and displacement vector
    nel = nelx * nely * (nelz if is_3d else 1)
    ndof = (3 if is_3d else 2) * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)
    p = (
        1 / preset_params["Dimensions"]["volfrac"] - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel)
    densities = x**p
    np.random.shuffle(densities)
    result = densities
    u = np.random.rand(ndof)

    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u is not None, "Displacement vector is None"

    # Test analyze function
    contains_checkerboard, is_watertight, is_thresholded, is_efficient = (
        analyzers.analyze(
            result, u, preset_params["Dimensions"], preset_params["Forces"]
        )
    )
    assert all(
        isinstance(v, bool)
        for v in (
            contains_checkerboard,
            is_watertight,
            is_thresholded,
            is_efficient,
        )
    )


def test_checkerboard():
    """Test checkerboard pattern analysis."""
    # Create a perfect 3x3 checkerboard pattern embedded in a larger grid
    x = np.zeros((5, 5))
    x[1, 1] = 1
    x[1, 3] = 1
    x[2, 2] = 1
    x[3, 1] = 1
    x[3, 3] = 1
    assert analyzers._checkerboard(x) is True, "Checkerboard should have been detected"
    x = np.ones((5, 5))
    assert analyzers._checkerboard(x) is False, (
        "Checkerboard shouldn't have been detected"
    )
    x = np.ones((4, 4, 4))
    assert analyzers._checkerboard(x) is False, (
        "Checkerboard shouldn't have been detected"
    )


def test_watertight():
    """Test watertight analysis."""
    x = np.zeros((5, 5))
    x[1:4, 1:4] = 1.0
    assert analyzers._watertight(x) is True, "Watertighteness wrongly detected"
    x = np.zeros((10, 10))
    x[0, 0] = 1.0
    x[9, 9] = 1.0
    assert analyzers._watertight(x) is False, "Watertighteness wrongly detected"


def test_thresholded():
    """Test thresholded returns True for a fully binarized field."""
    x = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 1.0])
    assert analyzers._thresholded(x) is True
    x = np.full(100, 0.5)
    assert analyzers._thresholded(x) is False


def test_efficient():
    """Test efficient() for rigid mechanisms (no output forces)."""
    # Rigid body
    Dimensions = {"nelxyz": [10, 10, 0]}
    Forces = {
        "fidir": ["X:→"],
        "fix": [5],
        "fiy": [5],
        "fiz": [0],
        "finorm": [0.01],
        "fodir": [],  # No output forces = rigid mechanism
    }
    ndof = 2 * 11 * 11
    u = np.zeros(ndof)
    # Small displacement at force location -> efficient
    idx = 5 * 11 + 5
    u[idx * 2] = 0.001
    result = analyzers._efficient(u, Dimensions, Forces)
    assert isinstance(result, bool)

    # Compliant mechanism
    Dimensions = {"nelxyz": [10, 10, 0]}
    Forces = {
        "fidir": ["X:→"],
        "fix": [0],
        "fiy": [5],
        "fiz": [0],
        "finorm": [0.01],
        "fodir": ["X:→"],
        "fox": [10],
        "foy": [5],
        "foz": [0],
        "fonorm": [0.01],
    }
    ndof = 2 * 11 * 11
    u = np.zeros(ndof)
    result = analyzers._efficient(u, Dimensions, Forces)
    assert isinstance(result, bool)


def test_analyze_with_progress_callback_cancel():
    """Test analyze() cancels early when progress_callback returns True at step 1."""
    nelx, nely = 5, 5
    xPhys = np.random.rand(nelx * nely)
    u = np.random.rand(2 * 6 * 6)
    Dimensions = {"nelxyz": [nelx, nely, 0]}
    Forces = {
        "fidir": ["X:→"],
        "fix": [0],
        "fiy": [2],
        "fiz": [0],
        "finorm": [0.01],
        "fodir": [],
        "fox": [],
        "foy": [],
        "foz": [],
        "fonorm": [],
    }

    def cancel_at_1(step: int) -> bool:
        return step >= 1

    result = analyzers.analyze(
        xPhys, u, Dimensions, Forces, progress_callback=cancel_at_1
    )
    # Should return early with False for watertight, thresholded, efficient
    assert len(result) == 4
    assert result[1] is False
    assert result[2] is False
    assert result[3] is False


def test_analyze_no_callback():
    """Test analyze() runs fully without a progress_callback."""
    nelx, nely = 5, 5
    xPhys = np.random.rand(nelx * nely)
    u = np.random.rand(2 * 6 * 6)
    Dimensions = {"nelxyz": [nelx, nely, 0]}
    Forces = {
        "fidir": ["X:→"],
        "fix": [0],
        "fiy": [2],
        "fiz": [0],
        "finorm": [0.01],
        "fodir": [],
        "fox": [],
        "foy": [],
        "foz": [],
        "fonorm": [],
    }

    result = analyzers.analyze(xPhys, u, Dimensions, Forces)
    assert len(result) == 4
    assert all(isinstance(v, bool) for v in result)


def test_efficiency_heuristic_is_unit_free():
    """Scaling forces and stiffnesses together must not change the verdict."""
    from topoptcomec.core import analyzers, optimizers

    dimensions = {"nelxyz": [20, 10, 0], "volfrac": 0.3}
    forces = {
        "fix": [20],
        "fiy": [5],
        "fiz": [0],
        "fidir": ["Y:\u2193"],
        "finorm": [0.01],
        "fox": [20],
        "foy": [0],
        "foz": [0],
        "fodir": ["Y:\u2191"],
        "fonorm": [0.01],
    }
    materials = {"E": [1.0], "nu": [0.3], "init_type": 0}
    optimizer = {
        "filter_type": "Sensitivity",
        "filter_radius_min": 1.3,
        "penal": 3.0,
        "eta": 0.3,
        "max_change": 0.05,
        "n_it": 3,
        "solver": "Auto",
    }
    supports = {"sx": [0], "sy": [0], "sz": [0], "sdim": ["XYZ"], "sr": [0]}
    _x, u = optimizers.optimize(
        dimensions, forces, materials, optimizer, supports, verbose=False
    )
    baseline = analyzers._efficient(u, dimensions, forces)

    scaled = {key: list(value) for key, value in forces.items()}
    scaled["finorm"] = [10.0]
    scaled["fonorm"] = [10.0]
    assert analyzers._efficient(u * 0.01, dimensions, scaled) == baseline


def test_checkerboard_small_grids_are_supported():
    """Analysis must not fail when a grid is smaller than the 3x3 mask."""
    assert analyzers._checkerboard(np.ones((2, 5))) is False
    assert analyzers._checkerboard(np.ones((1, 1))) is False
    assert analyzers._checkerboard(np.ones((5, 5, 2))) is False
    assert analyzers._checkerboard(np.ones((5, 5, 1))) is False


def test_checkerboard_still_detected_on_large_grids():
    """The small-grid guard must not disable the actual detection."""
    field = np.indices((5, 5)).sum(axis=0) % 2
    assert analyzers._checkerboard(field) is True
    field_3d = np.indices((5, 5, 5)).sum(axis=0) % 2
    assert analyzers._checkerboard(field_3d) is True
