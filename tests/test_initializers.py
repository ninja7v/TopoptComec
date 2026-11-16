# tests/test_initializers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the initializers.

import numpy as np
import pytest

from topoptcomec.core import initializers


def test_initialize_material():
    """Test material initialization (init_type=0)."""
    # Uniform initialization
    result = initializers.initialize_material(
        init_type=0,
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
    )
    assert result.shape == (100,)
    np.testing.assert_allclose(result, 0.3)

    # Uniform initialization
    result = initializers.initialize_material(
        init_type=1,
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([5]),
        all_y=np.array([5]),
        all_z=np.array([0]),
    )
    assert result.shape == (100,)
    assert np.mean(result) == pytest.approx(0.3, abs=0.01)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

    # Activity point initialization
    result = initializers.initialize_material(
        init_type=1,
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
    )
    assert result.shape == (100,)
    np.testing.assert_allclose(result, 0.3)
    result = initializers.initialize_material(
        init_type=1,
        volfrac=0.3,
        nelx=5,
        nely=5,
        nelz=5,
        all_x=np.array([2]),
        all_y=np.array([2]),
        all_z=np.array([2]),
    )
    assert result.shape == (125,)
    assert np.mean(result) == pytest.approx(0.3, abs=0.02)
    result = initializers.initialize_material(
        init_type=2,
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
    )
    assert result.shape == (100,)
    assert np.mean(result) == pytest.approx(0.3, abs=0.02)
    assert result.min() >= 0.0
    assert result.max() <= 1.0

    # Invalid type
    with pytest.raises(ValueError, match="Invalid init_type"):
        initializers.initialize_material(
            init_type=99,
            volfrac=0.3,
            nelx=10,
            nely=10,
            nelz=0,
            all_x=np.array([]),
            all_y=np.array([]),
            all_z=np.array([]),
        )

    # Random initialization
    result = initializers.initialize_materials(
        init_type=0,
        materials_percentage=[30, 40],
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
    )
    assert result is None


def test_initialize_materials_valid():
    """Test initialize_materials with valid percentages."""
    result = initializers.initialize_materials(
        init_type=0,
        materials_percentage=[60, 40],
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
    )
    assert result is not None
    assert result.shape == (2, 100)
    assert result.min() >= 1e-6


def test_rescale_densities():
    """Test _rescale_densities."""
    # Already at target
    d = np.full(100, 0.3)
    result = initializers._rescale_densities(d, 0.3)
    np.testing.assert_allclose(result, 0.3, atol=1e-3)

    # To be adjust
    d = np.random.rand(100)
    result = initializers._rescale_densities(d, 0.4)
    assert np.mean(result) == pytest.approx(0.4, abs=0.01)
    assert result.min() >= 0.0
    assert result.max() <= 1.0


def test_initialize_from_current_result():
    """Test initialization from a current result (init_type=3)."""
    current_x = np.random.rand(100)

    # 1. Single material matching shape (and matching volume fraction)
    volfrac_matching = np.mean(current_x)
    result = initializers.initialize_material(
        init_type=3,
        volfrac=volfrac_matching,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
        current_xPhys=current_x,
    )
    np.testing.assert_allclose(result, current_x)

    # 1b. Single material matching shape but mismatching volume fraction (should adjust)
    target_volfrac = 0.3
    result_adjusted = initializers.initialize_material(
        init_type=3,
        volfrac=target_volfrac,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
        current_xPhys=current_x,
    )
    assert np.mean(result_adjusted) == pytest.approx(target_volfrac, abs=0.01)

    # 2. Single material size mismatch fallback to uniform
    result_mismatch = initializers.initialize_material(
        init_type=3,
        volfrac=0.3,
        nelx=5,
        nely=5,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
        current_xPhys=current_x,
    )
    assert result_mismatch.shape == (25,)
    np.testing.assert_allclose(result_mismatch, 0.3)

    # 3. Multi-material matching shape: each row rescaled to its new target
    current_x_multi = np.random.rand(2, 100)
    result_multi = initializers.initialize_materials(
        init_type=3,
        materials_percentage=[60, 40],
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
        current_xPhys=current_x_multi,
    )
    assert result_multi.shape == (2, 100)
    # Per-material means hit the rescaled targets (0.18, 0.12).
    # Tolerance reflects _rescale_densities' accuracy for extreme targets.
    np.testing.assert_allclose(result_multi.mean(axis=1), [0.18, 0.12], atol=0.1)
    # Per-element totals may exceed 1 here (rows rescaled independently);
    # the optimizer clamps excess at runtime.

    # 4. Multi-material from single-material current: shared pattern split
    result_multi_from_single = initializers.initialize_materials(
        init_type=3,
        materials_percentage=[60, 40],
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
        current_xPhys=current_x,
    )
    assert result_multi_from_single.shape == (2, 100)
    # Per-material means hit the targets
    np.testing.assert_allclose(
        result_multi_from_single.mean(axis=1), [0.18, 0.12], atol=0.05
    )
    # Total density pattern has mean volfrac and stays <= 1
    col_sums = result_multi_from_single.sum(axis=0)
    np.testing.assert_allclose(np.mean(col_sums), 0.3, atol=0.01)
    assert col_sums.max() <= 1.0 + 1e-9


def test_rescale_densities_binary_field_reaches_target():
    """A thresholded field must be able to change its volume fraction."""
    binary = np.zeros(100)
    binary[:28] = 1.0
    result = initializers._rescale_densities(binary, 0.2)
    assert result.mean() == pytest.approx(0.2, abs=1e-3)
    assert result.min() >= 0.0 and result.max() <= 1.0
    assert result.sum() < binary.sum()


@pytest.mark.parametrize("multi", [False, True])
def test_distance_initializer_is_symmetric_about_a_centred_point(multi):
    """Distances are measured from the element centres, so a point in the
    middle of the grid gives a field symmetric in both directions."""
    nelx, nely = 8, 6
    args = (0.4, nelx, nely, 0, np.array([4.0]), np.array([3.0]), np.array([0.0]))
    if multi:
        field = initializers.initialize_materials(1, [60, 40], *args).sum(axis=0)
    else:
        field = initializers.initialize_material(1, *args)
    spatial = field.reshape(nelx, nely)
    np.testing.assert_allclose(spatial, spatial[::-1, :], atol=1e-12)
    np.testing.assert_allclose(spatial, spatial[:, ::-1], atol=1e-12)
    np.testing.assert_allclose(field.mean(), 0.4, atol=1e-3)
