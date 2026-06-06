# tests/test_initializers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the initializers.

import numpy as np
import pytest

from app.core import initializers


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

    # 1. Single material matching shape
    result = initializers.initialize_material(
        init_type=3,
        volfrac=0.3,
        nelx=10,
        nely=10,
        nelz=0,
        all_x=np.array([]),
        all_y=np.array([]),
        all_z=np.array([]),
        current_xPhys=current_x,
    )
    np.testing.assert_allclose(result, current_x)

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

    # 3. Multi-material matching shape
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
    np.testing.assert_allclose(result_multi, current_x_multi)

    # 4. Multi-material from single-material base (should compute complement/rescale)
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
    # Check partition of unity
    np.testing.assert_allclose(result_multi_from_single.sum(axis=0), 0.3, rtol=1e-3)
