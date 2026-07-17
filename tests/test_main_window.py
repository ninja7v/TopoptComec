# tests/test_main_window.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the main window.

import pytest
import numpy as np
from topoptcomec.gui.main_window import MainWindow
from PySide6.QtWidgets import QCheckBox
from unittest.mock import patch


@pytest.fixture(autouse=True)
def _prevent_loss_csv_writes():
    """Keep GUI unit tests from overwriting checked-in result histories."""
    with patch(
        "topoptcomec.gui.main_window.exporters.save_loss", return_value=(True, None)
    ):
        yield


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
        "rstate": ["Material 1"],
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
        "rstate": ["Material 1"],
        "rradius": [7],
        "rx": [20],
        "ry": [30],
        "rz": [0],
    },
    "Displacement": {"disp_factor": 1.0, "disp_iterations": 1},
}

_optimizer_base = {
    "filter_type": "Sensitivity",
    "filter_radius_min": 1.3,
    "penal": 3.0,
    "eta": 0.3,
    "max_change": 0.05,
    "n_it": 30,
    "solver": "Auto",
}

p_with_optimizer = {**p_base_2d, "Optimizer": dict(_optimizer_base)}

# Same optimizer plus the UI-only save_frames key: must stay equivalent.
p_optimizer_ui_only = {
    **p_base_2d,
    "Optimizer": {**_optimizer_base, "save_frames": True},
}

# One optimizer value changed: must NOT be equivalent (regression: the
# Optimizer section was once ignored by the comparison, so editing it did
# not deselect the active preset).
p_different_optimizer = {
    **p_base_2d,
    "Optimizer": {**_optimizer_base, "n_it": 60},
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
        (p_with_optimizer, p_with_optimizer, True),  # Should equal itself
        (
            p_with_optimizer,
            p_optimizer_ui_only,
            True,
        ),  # Equivalent despite the UI-only save_frames key
        (
            p_with_optimizer,
            p_different_optimizer,
            False,
        ),  # Changing an optimizer value must deselect the preset
    ],
)
def test_are_parameters_equivalent(qt_app, p1: dict, p2: dict, expected: bool):
    """Unit Test: Tests the intelligent parameter comparison function."""
    # We need a MainWindow instance to get access to the method
    window: MainWindow = MainWindow()
    assert window._are_parameters_equivalent(p1, p2) == expected
    window.close()


def test_gather_and_apply_parameters(qt_app):
    """Unit Test: Checks if gathering and applying parameters works correctly."""
    window: MainWindow = MainWindow()

    # 1. Get the initial parameters from the UI
    initial_params: dict = window._gather_parameters()

    # 2. Modify a known value in the dictionary
    modified_params: dict = initial_params.copy()
    modified_params["Dimensions"]["nelxyz"] = [100, 80, 10]
    modified_params["Supports"]["sx"][0] = 50

    # 3. Add regions (simulate a preset with multiple regions)
    if "Regions" not in modified_params:
        modified_params["Regions"] = {}
    modified_params["Regions"]["rshape"] = ["□", "◯"]
    modified_params["Regions"]["rstate"] = ["Void", "Material 1"]
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
    window: MainWindow = MainWindow()

    # Mock result data
    window.xPhys: np.ndarray = np.array([0.5] * 100)
    window.last_params["Dimensions"]: dict = {"nelxyz": (10, 10, 0)}

    with patch("PySide6.QtWidgets.QFileDialog.getSaveFileName") as mock_dialog:
        mock_dialog.return_value = ("results/test.png", "PNG")
        with patch.object(window, "_save_screenshot"):
            window._save_result_as("png")  # Should not raise


def test__on_visibility_toggled(qt_app):
    """Test visibility toggled ."""
    window: MainWindow = MainWindow()
    vis_btn: QCheckBox = window.sections["Dimensions"].visibility_button
    vis_btn.setChecked(False)
    assert vis_btn.toolTip() == "Element is hidden. Click to show."
    vis_btn.setChecked(True)
    assert vis_btn.toolTip() == "Element is visible. Click to hide."
    window.close()


@patch("topoptcomec.gui.main_window.np.savez_compressed")
@patch("topoptcomec.gui.main_window.os.makedirs")
def test_handle_optimization_results(mock_makedirs, mock_savez, qt_app):
    """Test that _handle_optimization_results sets xPhys and enables buttons."""
    import numpy as np

    window: MainWindow = MainWindow()
    nelx, nely = window.last_params["Dimensions"]["nelxyz"][:2]
    nel: int = nelx * nely
    ndof: int = 2 * (nelx + 1) * (nely + 1)
    mock_xPhys: np.ndarray = np.ones(nel)
    mock_u: np.ndarray = np.ones((ndof, 1))

    window._handle_optimization_results((mock_xPhys, mock_u))

    np.testing.assert_array_equal(window.xPhys, mock_xPhys)
    np.testing.assert_array_equal(window.u, mock_u[:, 0])
    assert window.footer.create_button.isEnabled()
    assert window.footer.binarize_button.isEnabled()
    assert window.footer.save_button.isEnabled()
    assert window.analysis_widget.run_analysis_button.isEnabled()
    assert window.displacement_widget.run_disp_button.isEnabled()
    window.close()


def test_handle_optimization_error(qt_app):
    """Test that _handle_optimization_error re-enables buttons and shows message."""
    window: MainWindow = MainWindow()

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
        window._handle_optimization_error("Something went wrong")

    mock_msg.assert_called_once()
    assert window.footer.create_button.isEnabled()
    window.close()


def test_toggle_theme(qt_app):
    """Test toggling the theme between dark and light."""
    window: MainWindow = MainWindow()
    initial_theme: str = window.current_theme
    window._toggle_theme()
    assert window.current_theme != initial_theme
    window._toggle_theme()
    assert window.current_theme == initial_theme
    window.close()


def test_run_optimization_validation_error(qt_app):
    """Test that _run_optimization shows error when validation fails."""
    window: MainWindow = MainWindow()
    # Force an invalid parameter: set all dimensions to zero
    window.last_params: dict = window._gather_parameters()
    window.last_params["Dimensions"]["nelxyz"] = [0, 0, 0]

    with patch("PySide6.QtWidgets.QMessageBox.critical") as mock_msg:
        window._run_optimization()

    mock_msg.assert_called_once()
    window.close()


def test_binarize(qt_app):
    """Test that binarize does nothing when no xPhys exists."""
    window: MainWindow = MainWindow()
    # No xPhys exists
    window.xPhys = None
    window._on_binarize_clicked()  # Should not raise

    # xPhys exists
    import numpy as np

    nelx, nely = window.last_params["Dimensions"]["nelxyz"][:2]
    nel: int = nelx * nely
    window.xPhys = np.linspace(0.1, 0.9, nel)
    window._on_binarize_clicked()
    assert set(np.unique(window.xPhys)).issubset({0.0, 1.0})
    window.close()


def test_stop_optimization_no_worker(qt_app):
    """Test that _stop_optimization does nothing when no worker exists."""
    window: MainWindow = MainWindow()
    window.worker = None
    # Should not raise
    window._stop_optimization()
    window.close()


def test_style_plot_default(qt_app):
    """Test that style_plot_default sets the plot background to white."""
    window: MainWindow = MainWindow()
    window._style_plot_default()
    assert window.plotter.background_color.float_rgb == (1.0, 1.0, 1.0)
    window.close()


def test_update_camera_uses_non_rotating_2d_style(qt_app):
    """2D plots use image interaction while 3D plots retain trackball rotation."""
    window: MainWindow = MainWindow()
    window._camera_mode = None
    window._camera_dims = None

    with patch.object(window.plotter, "enable_image_style") as image_style:
        window._update_camera(False, (10, 8, 0))
    image_style.assert_called_once_with()

    with patch.object(window.plotter, "enable_trackball_style") as trackball_style:
        window._update_camera(True, (10, 8, 4))
    trackball_style.assert_called_once_with()
    window.close()


def test_update_optimization_progress(qt_app):
    """Test that _update_optimization_progress sets progress bar value."""
    window: MainWindow = MainWindow()
    window.progress_bar.setRange(0, 100)
    window.progress_bar.setVisible(True)
    window._update_optimization_progress(42, 1.234, 0.001)
    assert window.progress_bar.value() == 42
    assert window.loss_history == [(42, 1.234)]
    window.close()


def test_handle_analysis_finished(qt_app):
    """Test that _handle_analysis_finished updates the analysis widget."""
    window: MainWindow = MainWindow()
    results: tuple = (True, False, True, False)
    window._handle_analysis_finished(results)
    assert window.analysis_widget.checkerboard_result.text() == "yes"
    assert window.analysis_widget.watertight_result.text() == "no"
    assert window.analysis_widget.threshold_result.text() == "yes"
    assert window.analysis_widget.efficiency_result.text() == "no"
    assert window.footer.create_button.isEnabled()
    window.close()


def test_handle_analysis_error(qt_app):
    """Test that _handle_analysis_error re-enables buttons."""
    window: MainWindow = MainWindow()

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


def test_low_density_validation(qt_app):
    """Test that low density result is detected as invalid and blocks run buttons."""
    import numpy as np
    from unittest.mock import patch

    window: MainWindow = MainWindow()
    assert window.xPhys_valid == (window.xPhys is not None)

    # 1. Test handle results with low density (< 1%)
    nelx: int
    nely: int
    nelx, nely = window.last_params["Dimensions"]["nelxyz"][:2]
    nel: int = nelx * nely
    mock_xPhys_empty: np.ndarray = np.zeros(nel)
    mock_u: np.ndarray = np.ones((2 * (nelx + 1) * (nely + 1), 1))

    with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning:
        window._handle_optimization_results((mock_xPhys_empty, mock_u))

    assert window.xPhys_valid is False
    mock_warning.assert_called_once()
    assert not window.analysis_widget.run_analysis_button.isEnabled()
    assert not window.displacement_widget.run_disp_button.isEnabled()

    # 2. Test run buttons are blocked when invalid
    with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning_disp:
        window._run_displacement()
    mock_warning_disp.assert_called_once()

    with patch("PySide6.QtWidgets.QMessageBox.warning") as mock_warning_anal:
        window._run_analysis()
    mock_warning_anal.assert_called_once()

    # 3. Test handle results with valid density (>= 1%)
    mock_xPhys_valid = np.ones(nel)
    with patch("topoptcomec.gui.main_window.np.savez_compressed"):
        window._handle_optimization_results((mock_xPhys_valid, mock_u))

    assert window.xPhys_valid is True
    assert window.analysis_widget.run_analysis_button.isEnabled()
    assert window.displacement_widget.run_disp_button.isEnabled()

    # 4. Test binarization resulting in low density
    # Set xPhys to values just below threshold 0.5, so they binarize to 0.0
    window.xPhys: np.ndarray = np.ones(nel) * 0.4
    with patch("PySide6.QtWidgets.QMessageBox.warning"):
        window._on_binarize_clicked()

    assert window.xPhys_valid is False
    assert not window.analysis_widget.run_analysis_button.isEnabled()
    assert not window.displacement_widget.run_disp_button.isEnabled()

    window.close()


def test_optimizer_change_deselects_preset(qt_app):
    """Regression: editing any Optimizer widget must deselect the active
    preset (comparison must include the Optimizer section AND every
    optimizer widget must be wired to on_parameter_changed)."""
    window: MainWindow = MainWindow()
    combo = window.preset.presets_combo
    if combo.count() < 2:
        window.close()
        pytest.skip("No presets available")

    changes = [
        lambda: window.optimizer_widget.opt_n_it.setValue(
            window.optimizer_widget.opt_n_it.value() + 5
        ),
        lambda: window.optimizer_widget.opt_solver.setCurrentIndex(
            (window.optimizer_widget.opt_solver.currentIndex() + 1)
            % window.optimizer_widget.opt_solver.count()
        ),
        lambda: window.optimizer_widget.opt_ft.setCurrentIndex(
            (window.optimizer_widget.opt_ft.currentIndex() + 1)
            % window.optimizer_widget.opt_ft.count()
        ),
        lambda: window.optimizer_widget.opt_p.setValue(
            window.optimizer_widget.opt_p.value() + 0.5
        ),
    ]
    for i, change in enumerate(changes):
        combo.setCurrentIndex(1)
        window._on_preset_selected()
        preset_name = combo.currentText()
        assert preset_name != "Select a preset..."
        change()
        assert combo.currentText() == "Select a preset...", (
            f"Optimizer change #{i} did not deselect the preset"
        )
    window.close()
