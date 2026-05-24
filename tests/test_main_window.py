# tests/test_main_window.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the main window.

import pytest

from app.ui.main_window import MainWindow

from unittest.mock import patch

# --- Test Cases for the Intelligent Comparison ---
p_base_2d = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
        "fix": [0, 100],
        "fiy": [30, 40],
        "fiz": [0, 0],
        "fidir": ["X:\u2192", "Y:\u2193"],
        "finorm": [0.01, 0.01],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_inactive_support = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {
        "sdim": ["Y", "X", "-"],
        "sx": [0, 60, 0],
        "sy": [20, 20, 0],
        "sr": [0, 0, 0],
    },
    "Forces": {
        "fix": [0, 100],
        "fiy": [30, 40],
        "fiz": [0, 0],
        "fidir": ["X:\u2192", "Y:\u2193"],
        "finorm": [0.01, 0.01],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_support = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {
        "sdim": ["Y", "X", "X"],
        "sx": [0, 60, 0],
        "sy": [20, 20, 0],
        "sr": [0, 0, 0],
    },
    "Forces": {
        "fix": [0, 100],
        "fiy": [30, 40],
        "fiz": [0, 0],
        "fidir": ["X:\u2192", "Y:\u2193"],
        "finorm": [0.01, 0.01],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_inactive_force = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
        "fix": [0, 100, 0],
        "fiy": [30, 40, 0],
        "fiz": [0, 0, 0],
        "fidir": ["X:\u2192", "Y:\u2193", "-"],
        "finorm": [0.01, 0.01, 0.0],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_force = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {
        "sdim": ["Y", "X", "-"],
        "sx": [0, 60, 0],
        "sy": [20, 20, 0],
        "sr": [0, 0, 0],
    },
    "Forces": {
        "fix": [0, 100, 0],
        "fiy": [30, 40, 0],
        "fiz": [0, 0, 0],
        "fidir": ["X:\u2192", "Y:\u2193", "Y:\u2193"],
        "finorm": [0.01, 0.01, 0.0],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_inactive_region = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
        "fix": [0, 100],
        "fiy": [30, 40],
        "fiz": [0, 0],
        "fidir": ["X:\u2192", "Y:\u2193"],
        "finorm": [0.01, 0.01],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Regions": {
        "rshape": ["-"],
        "rstate": ["Void"],
        "rradius": [5],
        "rx": [30],
        "ry": [20],
        "rz": [0],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_region = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X", "-"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
        "fix": [0, 100],
        "fiy": [30, 40],
        "fiz": [0, 0],
        "fidir": ["X:\u2192", "Y:\u2193"],
        "finorm": [0.01, 0.01],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Regions": {
        "rshape": ["□"],
        "rstate": ["Filled"],
        "rradius": [5],
        "rx": [25],
        "ry": [20],
        "rz": [0],
    },
    "Materials": {
        "E": [1.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

p_different_material = {
    "Dimensions": {"nelxyz": [60, 40, 0]},
    "Supports": {"sdim": ["Y", "X"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
        "fix": [0, 100],
        "fiy": [30, 40],
        "fiz": [0, 0],
        "fidir": ["X:\u2192", "Y:\u2193"],
        "finorm": [0.01, 0.01],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [3.0],
        "nu": [0.4],
        "percent": [100],
        "color": ["#000000"],
        "init_type": 0,
    },
    "Regions": {},
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}
# A preset that is truly different
p_different = {
    "Dimensions": {"nelxyz": [80, 50, 0]},
    "Supports": {"sdim": ["X", "Y"], "sx": [0, 60], "sy": [20, 20], "sr": [0, 0]},
    "Forces": {
        "fix": [0, 100, 0],
        "fiy": [30, 40, 0],
        "fiz": [0, 0, 0],
        "fidir": ["X:\u2192", "Y:\u2193", "-"],
        "finorm": [0.01, 0.01, 0.0],
        "fox": [20],
        "foy": [20],
        "foz": [0],
        "fodir": ["X:\u2192"],
        "fonorm": [0.01],
    },
    "Materials": {
        "E": [2.0],
        "nu": [0.25],
        "percent": [100],
        "color": ["#000500"],
        "init_type": 0,
    },
    "Regions": {
        "rshape": ["□"],
        "rstate": ["Filled"],
        "rradius": [7],
        "rx": [20],
        "ry": [30],
        "rz": [0],
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}


@pytest.mark.parametrize(
    "p1, p2, expected",
    [
        (p_base_2d, p_base_2d, True),  # Should equal itself
        (
            p_base_2d,
            p_inactive_support,
            True,
        ),  # Should be equivalent despite extra inactive support
        (p_base_2d, p_different_support, False),  # Should be different
        (
            p_base_2d,
            p_inactive_force,
            True,
        ),  # Should be equivalent despite extra inactive force
        (p_base_2d, p_different_force, False),  # Should be different
        (
            p_base_2d,
            p_inactive_region,
            True,
        ),  # Should be equivalent despite extra inactive region
        (p_base_2d, p_different_region, False),  # Should be different
        (p_base_2d, p_different_material, False),  # Should be different
        (p_base_2d, p_different, False),  # Should be different
    ],
)
def test_are_parameters_equivalent(qt_app, p1: dict, p2: dict, expected: bool):
    """Unit Test: Tests the intelligent parameter comparison function."""
    # We need a MainWindow instance to get access to the method
    window = MainWindow()
    assert window._are_parameters_equivalent(p1, p2) == expected
    window.close()


def test_gather_and_apply_parameters(qt_app):
    """Unit Test: Checks if gathering and applying parameters works correctly."""
    window = MainWindow()

    # 1. Get the initial parameters from the UI
    initial_params = window._gather_parameters()

    # 2. Modify a known value in the dictionary
    modified_params = initial_params.copy()
    modified_params["Dimensions"]["nelxyz"] = [100, 80, 10]
    modified_params["Supports"]["sx"][0] = 50

    # 3. Add regions (simulate a preset with multiple regions)
    if "Regions" not in modified_params:
        modified_params["Regions"] = {}
    modified_params["Regions"]["rshape"] = ["□", "◯"]
    modified_params["Regions"]["rstate"] = ["Void", "Filled"]
    modified_params["Regions"]["rradius"] = [5, 10]
    modified_params["Regions"]["rx"] = [10, 20]
    modified_params["Regions"]["ry"] = [10, 20]
    modified_params["Regions"]["rz"] = [0, 0]

    # 4. Apply these modified parameters back to the UI
    window._apply_parameters(modified_params)

    # 5. Gather the parameters from the UI again
    new_params_from_ui = window._gather_parameters()

    # 6. Assert that the UI state now matches the modified parameters
    assert new_params_from_ui["Dimensions"]["nelxyz"] == [100, 80, 10]
    assert new_params_from_ui["Supports"]["sx"][0] == 50
    assert len(new_params_from_ui["Regions"]["rshape"]) == 2
    assert new_params_from_ui["Regions"]["rshape"][1] == "◯"
    assert new_params_from_ui["Regions"]["rradius"][1] == 10
    window.close()


def test_save_result(qt_app):
    """Unit Test: Checks if the save result function works without error."""
    window = MainWindow()

    # Mock result data
    window.xPhys = [0.5] * 100
    window.last_params["Dimensions"] = {"nelxyz": (10, 10, 0)}
    window.figure = type("Fig", (), {"savefig": lambda *a, **k: None})()

    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = ("results/test.png", "PNG")
        window._save_result_as("png")  # Should not raise


def test__on_visibility_toggled(qt_app):
    """Test visibility toggled ."""
    window = MainWindow()
    vis_btn = window.sections["Dimensions"].visibility_button
    vis_btn.setChecked(False)
    assert vis_btn.toolTip() == "Element is hidden. Click to show."
    vis_btn.setChecked(True)
    assert vis_btn.toolTip() == "Element is visible. Click to hide."
    window.close()


@patch("app.ui.main_window.np.savez_compressed")
@patch("app.ui.main_window.os.makedirs")
def test_handle_optimization_results(mock_makedirs, mock_savez, qt_app):
    """Test that _handle_optimization_results sets xPhys and enables buttons."""
    import numpy as np

    window = MainWindow()
    nelx, nely = window.last_params["Dimensions"]["nelxyz"][:2]
    nel = nelx * nely
    ndof = 2 * (nelx + 1) * (nely + 1)
    mock_xPhys = np.ones(nel)
    mock_u = np.ones((ndof, 1))

    window._handle_optimization_results((mock_xPhys, mock_u))

    np.testing.assert_array_equal(window.xPhys, mock_xPhys)
    np.testing.assert_array_equal(window.u, mock_u)
    assert window.footer.create_button.isEnabled()
    assert window.footer.binarize_button.isEnabled()
    assert window.footer.save_button.isEnabled()
    assert window.analysis_widget.run_analysis_button.isEnabled()
    assert window.displacement_widget.run_disp_button.isEnabled()
    window.close()


def test_handle_optimization_error(qt_app):
    """Test that _handle_optimization_error re-enables buttons and shows message."""
    window = MainWindow()

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
        window._handle_optimization_error("Something went wrong")

    mock_msg.assert_called_once()
    assert window.footer.create_button.isEnabled()
    window.close()


def test_toggle_theme(qt_app):
    """Test toggling the theme between dark and light."""
    window = MainWindow()
    initial_theme = window.current_theme
    window._toggle_theme()
    assert window.current_theme != initial_theme
    window._toggle_theme()
    assert window.current_theme == initial_theme
    window.close()


def test_run_optimization_validation_error(qt_app):
    """Test that _run_optimization shows error when validation fails."""
    window = MainWindow()
    # Force an invalid parameter: set all dimensions to zero
    window.last_params = window._gather_parameters()
    window.last_params["Dimensions"]["nelxyz"] = [0, 0, 0]

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
        window._run_optimization()

    mock_msg.assert_called_once()
    window.close()


def test_binarize(qt_app):
    """Test that binarize does nothing when no xPhys exists."""
    window = MainWindow()
    # No xPhys exists
    window.xPhys = None
    window._on_binarize_clicked()  # Should not raise

    # xPhys exists
    import numpy as np

    nelx, nely = window.last_params["Dimensions"]["nelxyz"][:2]
    nel = nelx * nely
    window.xPhys = np.linspace(0.1, 0.9, nel)
    window._on_binarize_clicked()
    assert set(np.unique(window.xPhys)).issubset({0.0, 1.0})
    window.close()


def test_stop_optimization_no_worker(qt_app):
    """Test that _stop_optimization does nothing when no worker exists."""
    window = MainWindow()
    window.worker = None
    # Should not raise
    window._stop_optimization()
    window.close()


def test_style_plot_default(qt_app):
    """Test that style_plot_default sets the plot background to white."""
    window = MainWindow()
    window._style_plot_default()
    assert window.figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    window.close()


def test_update_optimization_progress(qt_app):
    """Test that _update_optimization_progress sets progress bar value."""
    window = MainWindow()
    window.progress_bar.setRange(0, 100)
    window.progress_bar.setVisible(True)
    window._update_optimization_progress(42, 1.234, 0.001)
    assert window.progress_bar.value() == 42
    window.close()


def test_handle_analysis_finished(qt_app):
    """Test that _handle_analysis_finished updates the analysis widget."""
    window = MainWindow()
    results = (True, False, True, False)
    window._handle_analysis_finished(results)
    assert window.analysis_widget.checkerboard_result.text() == "yes"
    assert window.analysis_widget.watertight_result.text() == "no"
    assert window.analysis_widget.threshold_result.text() == "yes"
    assert window.analysis_widget.efficiency_result.text() == "no"
    assert window.footer.create_button.isEnabled()
    window.close()


def test_handle_analysis_error(qt_app):
    """Test that _handle_analysis_error re-enables buttons."""
    window = MainWindow()

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
        window._handle_analysis_error("Analysis failed badly")

    mock_msg.assert_called_once()
    assert window.analysis_widget.run_analysis_button.isEnabled()
    window.close()


def test_handle_displacement_finished(qt_app):
    """Test _handle_displacement_finished updates UI state."""
    window = MainWindow()
    window._handle_displacement_finished("Done")
    assert window.is_displaying_deformation is True
    assert window.footer.create_button.isEnabled()
    window.close()


def test_handle_displacement_error(qt_app):
    """Test handle_displacement_error re-enables buttons."""
    window = MainWindow()

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
        window._handle_displacement_error("Displacement crashed")

    mock_msg.assert_called_once()
    assert window.displacement_widget.run_disp_button.isEnabled()
    window.close()
