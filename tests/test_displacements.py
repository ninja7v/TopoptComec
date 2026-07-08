# tests/test_displacements.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the displacements.

import json
import copy
from pathlib import Path

import numpy as np
import pytest

from topoptcomec.core import displacements


REFERENCES_DIR = Path(__file__).parent / "references"
# Loose enough to absorb BLAS/library differences across platforms.
REFERENCE_RTOL = 1e-3
REFERENCE_ATOL = 1e-6


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
def test_displacement_with_presets(preset_name: str, preset_params: dict):
    """Unit Test: Runs the 2D/3D optimizer with a given preset."""
    # Prepare the parameters for the optimizer function
    params = copy.deepcopy(preset_params)
    nelx, nely, nelz = params["Dimensions"]["nelxyz"]
    is_3d = nelz > 0
    # Remove all keys that are not part of the optimizer's function signature
    keys_to_remove = ["filter_type", "filter_radius_min", "max_change", "n_it"]
    if not is_3d:
        keys_to_remove = keys_to_remove + ["rz", "fz", "sz"]
    for key in keys_to_remove:
        params.pop(key, None)

    # Generate a mock result and displacement vector
    nel = nelx * nely * (nelz if is_3d else 1)
    ndof = (3 if is_3d else 2) * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)
    p = (
        1 / preset_params["Dimensions"]["volfrac"] - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel)
    # Don't use np.random.rand() here to ensure reproducibility across test runs
    densities = x**p
    result = densities[::-1]
    n_forces = sum(1 for fdir in params["Forces"]["fidir"] if fdir != "-")
    u_vec = np.linspace(0.0, 1.0, ndof * n_forces, dtype=float).reshape(ndof, n_forces)

    # Check if not empty
    assert result is not None, "Optimizer returned None"
    assert u_vec is not None, "Displacement vector is None"

    # Test linear displacement function
    if is_3d:
        X, Y, Z = displacements.single_linear_displacement(u_vec, nelx, nely, nelz, 1.0)
        assert not (X is None or Y is None or Z is None), (
            "Displacement function returned None arrays"
        )
        linear_arrays = (X, Y, Z)
    else:
        X, Y = displacements.single_linear_displacement(u_vec, nelx, nely, nelz, 1.0)
        assert not (X is None or Y is None), (
            "Displacement function returned None arrays"
        )
        linear_arrays = (X, Y)

    # Test iterative displacement function
    params["Displacement"]["disp_iterations"] = 2
    for frame in displacements.run_iterative_displacement(params, result):
        last_result_displaced = frame
    assert last_result_displaced is not None, (
        "Iterative displacement function returned None"
    )
    assert last_result_displaced.shape == np.array(result).shape, (
        "Iterative displacement function returned different shapes"
    )
    assert (
        np.max(last_result_displaced) <= 1.0 and np.min(last_result_displaced) >= 0.0
    ), "Displaced densities should remain within [0, 1]"
    assert np.isclose(
        last_result_displaced.mean(), preset_params["Dimensions"]["volfrac"], atol=0.15
    ), (
        f"Final volume ({last_result_displaced.mean():.3f}) is far to target ({preset_params['Dimensions']['volfrac']:.3f})"
    )

    params["Displacement"]["disp_factor"] = 0.0
    for frame in displacements.run_iterative_displacement(params, result):
        zero_factor_result_displaced = frame
    assert np.array_equal(zero_factor_result_displaced, result), (
        "Iterative displacement with factor 0 should return the same result"
    )
    # Compare with reference data if not random initialization.
    # Regenerate with tests/references/regenerate_references.py after any
    # intentional numerical change.
    if params["Materials"]["init_type"] != 2:
        reference_path = (
            REFERENCES_DIR / f"test_displacement_with_presets_{preset_name}.npz"
        )
        if reference_path.exists():
            with np.load(reference_path) as reference:
                for index, actual in enumerate(linear_arrays):
                    np.testing.assert_allclose(
                        actual, reference[f"linear_{index}"], rtol=1e-10, atol=1e-12
                    )
                np.testing.assert_allclose(
                    last_result_displaced,
                    reference["last_result_displaced"],
                    rtol=REFERENCE_RTOL,
                    atol=REFERENCE_ATOL,
                    err_msg=f"Displaced density mismatch for preset {preset_name}",
                )
                np.testing.assert_allclose(
                    zero_factor_result_displaced,
                    reference["zero_factor_result_displaced"],
                    rtol=REFERENCE_RTOL,
                    atol=REFERENCE_ATOL,
                    err_msg=f"Zero-factor displacement mismatch for preset {preset_name}",
                )


def test_embed_crop_3d_preserves_fem_ordering():
    """Embedding must use the FEM flat ordering.

    The historic implementation reshaped 3D fields as (nelx, nely, nelz)
    while the FEM flat ordering is z-major (nelz, nelx, nely), scrambling
    every 3D displacement simulation.
    """
    from topoptcomec.core.displacements import _crop_density, _embed_material
    from topoptcomec.core.grid import StructuredGrid

    small = StructuredGrid(4, 3, 2)
    large = StructuredGrid(6, 5, 4)
    mx, my, mz = 1, 1, 1

    x = np.arange(small.nel, dtype=float)
    embedded, n_mat, _ = _embed_material(x, False, small, large, mx, my, mz)
    assert n_mat == 1

    # The value of element (ex, ey, ez) must land at (ex+mx, ey+my, ez+mz).
    ex, ey, ez = 1, 2, 0
    src = small.element_index(ex, ey, ez)
    dst = large.element_index(ex + mx, ey + my, ez + mz)
    assert embedded[0, dst] == x[src]

    # Round trip must be the identity.
    cropped = _crop_density(embedded, 1, small, large, mx, my, mz)
    np.testing.assert_array_equal(cropped[0], x)
