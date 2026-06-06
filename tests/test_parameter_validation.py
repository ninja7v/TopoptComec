# tests/test_parameter_validation.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for parameter validation and scaling logic.

import numpy as np
from unittest.mock import patch

from app.ui.main_window import MainWindow


# --- validate_parameters tests ---


def test_validate_invalid_dimensions(qt_app):
    """Test that validate_parameters rejects zero dimensions."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Dimensions"]["nelxyz"] = [0, 10, 0]
    err = window._validate_parameters(params)
    assert err is not None
    assert "positive" in err.lower() or "Nx" in err
    window.close()


def test_validate_negative_nelz(qt_app):
    """Test that validate_parameters rejects negative Nz."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Dimensions"]["nelxyz"] = [10, 10, -1]
    err = window._validate_parameters(params)
    assert err is not None
    window.close()


def test_validate_no_active_input_forces(qt_app):
    """Test that validate_parameters rejects no active input forces."""
    window = MainWindow()
    params = window._gather_parameters()
    # Set all input force directions to inactive
    params["Forces"]["fidir"] = ["-"] * len(params["Forces"]["fidir"])
    err = window._validate_parameters(params)
    assert err is not None
    assert "input force" in err.lower()
    window.close()


def test_validate_no_active_output_or_supports(qt_app):
    """Test that validate_parameters rejects no output forces and no supports."""
    window = MainWindow()
    params = window._gather_parameters()
    # Ensure at least one active input force
    params["Forces"]["fidir"] = ["X:→"]
    params["Forces"]["fix"] = [0]
    params["Forces"]["fiy"] = [5]
    params["Forces"]["fiz"] = [0]
    params["Forces"]["finorm"] = [0.01]
    # No active output forces
    params["Forces"]["fodir"] = ["-"]
    params["Forces"]["fox"] = [0]
    params["Forces"]["foy"] = [0]
    params["Forces"]["foz"] = [0]
    params["Forces"]["fonorm"] = [0.0]
    # No active supports
    params["Supports"] = {"sdim": ["-"], "sx": [0], "sy": [0], "sz": [0], "sr": [0]}
    err = window._validate_parameters(params)
    assert err is not None
    assert "output force" in err.lower() or "support" in err.lower()
    window.close()


def test_validate_duplicate_input_forces(qt_app):
    """Test that validate_parameters detects duplicate input forces."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Forces"]["fidir"] = ["X:→", "X:→"]
    params["Forces"]["fix"] = [5, 5]
    params["Forces"]["fiy"] = [5, 5]
    params["Forces"]["fiz"] = [0, 0]
    params["Forces"]["finorm"] = [0.01, 0.01]
    # Need some valid output/support
    params["Supports"] = {"sdim": ["Y"], "sx": [0], "sy": [0], "sz": [0], "sr": [0]}
    err = window._validate_parameters(params)
    assert err is not None
    assert "identical" in err.lower()
    window.close()


def test_validate_duplicate_output_forces(qt_app):
    """Test that validate_parameters detects duplicate output forces."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Forces"]["fodir"] = ["X:→", "X:→"]
    params["Forces"]["fox"] = [10, 10]
    params["Forces"]["foy"] = [5, 5]
    params["Forces"]["foz"] = [0, 0]
    params["Forces"]["fonorm"] = [0.01, 0.01]
    err = window._validate_parameters(params)
    assert err is not None
    assert "identical" in err.lower()
    window.close()


def test_validate_duplicate_supports(qt_app):
    """Test that validate_parameters detects duplicate supports."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Supports"] = {
        "sdim": ["Y", "Y"],
        "sx": [0, 0],
        "sy": [10, 10],
        "sz": [0, 0],
        "sr": [0, 0],
    }
    err = window._validate_parameters(params)
    assert err is not None
    assert "identical" in err.lower()
    window.close()


def test_validate_duplicate_materials(qt_app):
    """Test that validate_parameters detects duplicate materials."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Materials"]["E"] = [1.0, 1.0]
    params["Materials"]["nu"] = [0.3, 0.3]
    params["Materials"]["percent"] = [50, 50]
    err = window._validate_parameters(params)
    assert err is not None
    assert "identical" in err.lower()
    window.close()


def test_validate_materials_percent_not_100(qt_app):
    """Test that validate_parameters rejects material percentages not summing to 100."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Materials"]["E"] = [1.0, 2.0]
    params["Materials"]["nu"] = [0.3, 0.4]
    params["Materials"]["percent"] = [30, 40]  # Sum = 70, not 100
    err = window._validate_parameters(params)
    assert err is not None
    assert "100" in err
    window.close()


def test_validate_valid_params(qt_app):
    """Test that validate_parameters returns None for valid parameters."""
    window = MainWindow()
    params = window._gather_parameters()
    err = window._validate_parameters(params)
    assert err is None
    window.close()


# --- on_parameter_changed tests ---


def test_on_parameter_changed_resets_result(qt_app):
    """Test that on_parameter_changed clears xPhys when it exists."""
    window = MainWindow()
    # Simulate having a result with correct dimensions
    nelx, nely = window.last_params["Dimensions"]["nelxyz"][:2]
    nel = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    window.xPhys = np.ones(nel)
    result = window.xPhys
    window.u = np.ones((ndof, 1))
    window.is_displaying_deformation = True

    window.on_parameter_changed()

    assert not np.array_equal(window.xPhys, result)
    assert window.u is None
    assert window.is_displaying_deformation is False
    window.close()


def test_on_parameter_changed_without_result(qt_app):
    """Test that on_parameter_changed works when no result exists."""
    window = MainWindow()
    window.xPhys = None
    # Should not raise
    window.on_parameter_changed()
    window.close()


# --- scale_parameters tests ---


def test_scale_parameters_noop(qt_app):
    """Test that scale_parameters does nothing when scale is 1.0."""
    window = MainWindow()
    window.dim_widget.scale.setValue(1.0)
    initial_nx = window.dim_widget.nx.value()

    window._scale_parameters()

    assert window.dim_widget.nx.value() == initial_nx
    window.close()


def test_scale_parameters_upscale(qt_app):
    """Test that scale_parameters correctly scales up by 2x."""
    window = MainWindow()
    window.dim_widget.nx.setValue(10)
    window.dim_widget.ny.setValue(10)
    window.dim_widget.nz.setValue(0)
    window.dim_widget.scale.setValue(2.0)

    window._scale_parameters()

    assert window.dim_widget.nx.value() == 20
    assert window.dim_widget.ny.value() == 20
    window.close()


def test_scale_parameters_out_of_range(qt_app):
    """Test that scale_parameters rejects scaling that goes out of range."""
    window = MainWindow()
    window.dim_widget.nx.setValue(500)
    window.dim_widget.ny.setValue(500)
    window.dim_widget.scale.setValue(10.0)

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_critical:
        window._scale_parameters()

    # The QMessageBox.critical should have been called
    mock_critical.assert_called_once()
    window.close()


def test_scale_parameters_downscale(qt_app):
    """Test that scale_parameters correctly scales down by 0.5x."""
    window = MainWindow()
    window.dim_widget.nx.setValue(20)
    window.dim_widget.ny.setValue(20)
    window.dim_widget.nz.setValue(0)
    window.dim_widget.scale.setValue(0.5)

    window._scale_parameters()

    assert window.dim_widget.nx.value() == 10
    assert window.dim_widget.ny.value() == 10
    window.close()


# --- block_all_parameter_signals test ---


def test_block_all_parameter_signals(qt_app):
    """Test that block_all_parameter_signals runs without error."""
    window = MainWindow()
    # Should not raise
    window._block_all_parameter_signals(True)
    window._block_all_parameter_signals(False)
    window.close()


def test_validate_init_type_from_current_result(qt_app):
    """Test validation when init_type=3 (From current result)."""
    window = MainWindow()
    params = window._gather_parameters()
    params["Materials"]["init_type"] = 3

    # 1. No current result exists
    window.last_successful_xPhys = None
    err = window._validate_parameters(params)
    assert err == "No current result available to initialize from."

    # 2. Result exists but dimensions mismatch
    window.last_successful_xPhys = np.ones(
        50
    )  # default is 12x12=144 or 10x10=100 (let's use size match checks dynamically)
    err = window._validate_parameters(params)
    assert "does not match the active grid dimensions" in err

    # 3. Valid matching result
    nelx, nely, nelz = params["Dimensions"]["nelxyz"]
    nel = nelx * nely * (nelz if nelz > 0 else 1)
    window.last_successful_xPhys = np.ones(nel)
    err = window._validate_parameters(params)
    assert err is None
    window.close()
