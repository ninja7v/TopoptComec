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


def test_from_current_result_disabling_and_scaling(qt_app):
    """Test the disabling of 'From current result' and scaling behavior."""
    from unittest.mock import patch
    from topoptcomec.gui.main_window import MainWindow

    window = MainWindow()

    # 1. When last_successful_xPhys is None, option should be disabled
    window.last_successful_xPhys = None
    window.update_mat_init_type_state()
    assert window.last_successful_xPhys is None
    combo = window.materials_widget.mat_init_type
    assert not combo.model().item(3).isEnabled()

    # 2. Set last_successful_xPhys, option should be enabled
    params = window._gather_parameters()
    nx, ny, nz = params["Dimensions"]["nelxyz"]
    nel = nx * ny * (nz if nz > 0 else 1)
    dummy_x = np.ones(nel) * 0.5
    window.last_successful_xPhys = dummy_x
    window.update_mat_init_type_state()
    assert combo.model().item(3).isEnabled()

    # 3. Change dimensions while "From current result" is selected (index 3)
    combo.setCurrentIndex(3)
    # Change nx to 30
    window.dim_widget.nx.setValue(30)
    qt_app.processEvents()
    # The last_successful_xPhys should be scaled to size 30 * ny
    expected_size = 30 * ny * (nz if nz > 0 else 1)
    assert window.last_successful_xPhys.size == expected_size

    # 4. Change initialization option to "From current result" when size mismatches
    # First change dimensions to 10x10 while option is NOT "From current result" (index 0)
    combo.setCurrentIndex(0)
    window.dim_widget.nx.setValue(10)
    window.dim_widget.ny.setValue(10)
    qt_app.processEvents()
    # At this point, last_successful_xPhys size is still expected_size since option was index 0
    assert window.last_successful_xPhys.size == expected_size

    # Now change option to "From current result" (index 3)
    combo.setCurrentIndex(3)
    qt_app.processEvents()
    # It must have been scaled to the new dimensions 10x10 = 100
    assert window.last_successful_xPhys.size == 100

    # 5. Test starting a new optimization leaves last_successful_xPhys intact
    with patch("topoptcomec.gui.main_window.OptimizerWorker") as mock_worker:
        window._run_optimization()
        # last_successful_xPhys should not be reset to None
        assert window.last_successful_xPhys is not None
        assert combo.model().item(3).isEnabled()

        # Verify the worker was initialized with correct params
        called_args = mock_worker.call_args[0]
        worker_params = called_args[0]
        assert worker_params["Materials"]["init_type"] == 3
        assert worker_params["current_xPhys"] is not None

    # 6. Test reset when changing preset
    window.last_successful_xPhys = dummy_x
    window.update_mat_init_type_state()
    assert combo.model().item(3).isEnabled()
    # Trigger preset change
    window._on_preset_selected()
    assert window.last_successful_xPhys is None
    assert not combo.model().item(3).isEnabled()
    window.close()
