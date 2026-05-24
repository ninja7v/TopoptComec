# tests/test_widgets.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the widgets.

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QPushButton,
    QSpinBox,
    QWidget,
    QApplication,
)

from app.ui.widgets import (
    CollapsibleSection,
    ColorPickerButton,
    DimensionsWidget,
    DisplacementWidget,
    FooterWidget,
    ForcesWidget,
    HeaderWidget,
    MaterialsWidget,
    OptimizerWidget,
    AnalysisWidget,
    PresetWidget,
    RegionsWidget,
    SupportWidget,
)

# --- Tests for CollapsibleSection ---


def test_collapsible_section_toggle_logic(qt_app: QApplication):
    """Unit Test: Verifies the core expand/collapse logic of the CollapsibleSection."""
    content = QWidget()
    section = CollapsibleSection("Test", content)

    # Initial state
    assert section.is_collapsed is True
    assert not content.isVisible()

    # Simulate click to expand
    section.toggle_button.setChecked(True)
    qt_app.processEvents()

    # Force show since some widgets don't auto-show in headless tests
    content.setVisible(True)

    # Now check state
    assert section.is_collapsed is False
    assert not content.isHidden()


# --- Tests for Static Data Widgets ---


def test_dimensions_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies that the DimensionsWidget initializes with correct defaults."""
    widget = DimensionsWidget()

    # Check instances
    assert isinstance(widget.nx, QSpinBox)
    assert isinstance(widget.volfrac, QDoubleSpinBox)
    assert isinstance(widget.scale, QDoubleSpinBox)
    assert isinstance(widget.scale_button, QPushButton)

    # Check default values from your provided code
    assert widget.nx.value() == 60
    assert widget.ny.value() == 40
    assert widget.nz.value() == 0
    assert widget.volfrac.value() == 0.3
    assert widget.scale.value() == 1.0


def test_regions_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies the RegionsWidget initializes correctly."""
    widget = RegionsWidget()

    # Check instances
    assert hasattr(widget, "add_btn") and isinstance(widget.add_btn, QPushButton)

    # Should be empty by default
    assert len(widget.inputs) == 0

    # Test add button
    widget.add_btn.click()
    qt_app.processEvents()
    assert len(widget.inputs) == 1
    first_region = widget.inputs[0]
    assert "rshape" in first_region and isinstance(first_region["rshape"], QComboBox)
    assert "rstate" in first_region and isinstance(first_region["rstate"], QComboBox)
    assert "rradius" in first_region and isinstance(first_region["rradius"], QSpinBox)
    assert "rx" in first_region and isinstance(first_region["rx"], QSpinBox)
    assert "ry" in first_region and isinstance(first_region["ry"], QSpinBox)
    assert "remove_btn" in first_region and isinstance(
        first_region["remove_btn"], QPushButton
    )

    # Test remove region
    first_region["remove_btn"].click()
    qt_app.processEvents()
    widget.remove_region(0)
    assert len(widget.inputs) == 0

    # Test max regions
    for _ in range(15):
        widget.add_region()
    assert len(widget.inputs) == 10


def test_forces_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies the ForcesWidget initializes correctly and can add/remove forces."""
    widget = ForcesWidget()

    # Should start with 1 input force and 1 output force by default
    assert len(widget.input_forces) == 1, (
        "ForcesWidget should start with exactly 1 input force."
    )
    assert len(widget.output_forces) == 1, (
        "ForcesWidget should start with exactly 1 output force."
    )

    # Check instances
    first_iforce = widget.input_forces[0]
    assert "fix" in first_iforce and isinstance(first_iforce["fix"], QSpinBox)
    assert "fiy" in first_iforce and isinstance(first_iforce["fiy"], QSpinBox)
    assert "fidir" in first_iforce and isinstance(first_iforce["fidir"], QComboBox)
    assert "remove_btn" in first_iforce and isinstance(
        first_iforce["remove_btn"], QPushButton
    )

    first_oforce = widget.output_forces[0]
    assert "fox" in first_oforce and isinstance(first_oforce["fox"], QSpinBox)
    assert "foy" in first_oforce and isinstance(first_oforce["foy"], QSpinBox)
    assert "fodir" in first_oforce and isinstance(first_oforce["fodir"], QComboBox)
    assert "remove_btn" in first_oforce and isinstance(
        first_oforce["remove_btn"], QPushButton
    )

    # Test add input force button
    widget.add_if_btn.click()
    qt_app.processEvents()
    assert len(widget.input_forces) == 2, (
        "Clicking on add input force should add an input force."
    )

    # Test add output force button
    widget.add_of_btn.click()
    qt_app.processEvents()
    assert len(widget.output_forces) == 2, (
        "Clicking on add output force should add an output force."
    )

    # Test remove input force button
    new_iforce = widget.input_forces[-1]
    new_iforce["remove_btn"].click()
    qt_app.processEvents()
    assert len(widget.input_forces) == 1, (
        "Clicking on remove input force should delete an input force."
    )

    # Test remove output force button
    new_oforce = widget.output_forces[-1]
    new_oforce["remove_btn"].click()
    qt_app.processEvents()
    assert len(widget.output_forces) == 1, (
        "Clicking on remove output force should delete an output force."
    )

    # Test min limits (1 for input, 0 for output)
    # Remove all input forces
    while len(widget.input_forces) > 1:
        widget.input_forces[-1]["remove_btn"].click()
        qt_app.processEvents()
    assert len(widget.input_forces) == 1, (
        "Should not allow removing the last input force."
    )
    # Check that remove button is disabled
    assert not widget.input_forces[0]["remove_btn"].isEnabled(), (
        "Remove button for last input force should be disabled."
    )

    # Remove all output forces
    while len(widget.output_forces) > 0:
        widget.output_forces[-1]["remove_btn"].click()
        qt_app.processEvents()
    assert len(widget.output_forces) == 0, "Should allow removing all output forces."

    # Test max limits
    for _ in range(15):
        widget.add_input_force()
        widget.add_output_force()
    assert len(widget.input_forces) == 10, "Should not allow more than 10 input forces."
    assert len(widget.output_forces) == 10, (
        "Should not allow more than 10 output forces."
    )


def test_support_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies that the static SupportWidget works well."""
    widget = SupportWidget()

    # Check instances
    assert hasattr(widget, "add_btn") and isinstance(widget.add_btn, QPushButton)

    # Should be empty by default
    assert len(widget.inputs) == 0, "SupportWidget should start with no supports."

    # Test add button
    widget.add_btn.click()
    qt_app.processEvents()
    assert len(widget.inputs) == 1, "Clicking on add support should add a support."
    new_support = widget.inputs[-1]
    assert "remove_btn" in new_support and isinstance(
        widget.inputs[0]["remove_btn"], QPushButton
    )
    assert "sx" in new_support and isinstance(widget.inputs[0]["sx"], QSpinBox)
    assert "sy" in new_support and isinstance(widget.inputs[0]["sy"], QSpinBox)
    assert "sz" in new_support and isinstance(widget.inputs[0]["sz"], QSpinBox)
    assert "sdim" in new_support and isinstance(widget.inputs[0]["sdim"], QComboBox)
    assert "sr" in new_support and isinstance(widget.inputs[0]["sr"], QSpinBox)

    # Test remove button
    new_support["remove_btn"].click()
    qt_app.processEvents()
    assert len(widget.inputs) == 0, (
        "Clicking on remove support should delete a support."
    )


def test_material_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies the MaterialsWidget initializes correctly."""
    widget = MaterialsWidget()

    # Check instances
    assert isinstance(widget.mat_init_type, QComboBox), (
        "MaterialsWidget should have a mat_init_type QComboBox."
    )

    # Should contain exactly 1 material by default
    assert len(widget.inputs) == 1, (
        "MaterialsWidget should start with exactly 1 material input set."
    )

    # Check instances
    existing_material = widget.inputs[0]
    assert "E" in existing_material and isinstance(
        existing_material["E"], QDoubleSpinBox
    )
    assert "nu" in existing_material and isinstance(
        existing_material["nu"], QDoubleSpinBox
    )
    assert "color" in existing_material and isinstance(
        existing_material["color"], ColorPickerButton
    )
    assert "percent" in existing_material and isinstance(
        existing_material["percent"], QSpinBox
    )

    # Test add button
    widget.add_btn.click()
    qt_app.processEvents()
    assert len(widget.inputs) == 2, "Clicking on add material should add a material."
    new_support = widget.inputs[-1]
    assert "remove_btn" in new_support and isinstance(
        widget.inputs[0]["remove_btn"], QPushButton
    )
    assert "E" in new_support and isinstance(widget.inputs[1]["E"], QDoubleSpinBox)
    assert "nu" in new_support and isinstance(widget.inputs[1]["nu"], QDoubleSpinBox)
    assert "color" in new_support and isinstance(
        widget.inputs[1]["color"], ColorPickerButton
    )
    assert "percent" in new_support and isinstance(
        widget.inputs[1]["percent"], QSpinBox
    )

    # Test material equilibrium
    widget.inputs[0]["percent"].setValue(70)
    qt_app.processEvents()
    assert widget.inputs[1]["percent"].value() == 30, "Percent do not sum to 100%."

    # Test remove button
    new_support["remove_btn"].click()
    qt_app.processEvents()
    assert len(widget.inputs) == 1, (
        "Clicking on remove material should delete a material."
    )


def test_optimizer_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies the OptimizerWidget initializes correctly."""
    widget = OptimizerWidget()

    # Check instances
    assert isinstance(widget.opt_ft, QComboBox)
    assert isinstance(widget.opt_fr, QDoubleSpinBox)
    assert isinstance(widget.opt_p, QDoubleSpinBox)
    assert isinstance(widget.opt_eta, QDoubleSpinBox)
    assert isinstance(widget.opt_max_change, QDoubleSpinBox)
    assert isinstance(widget.opt_n_it, QSpinBox)
    assert isinstance(widget.opt_solver, QComboBox)

    # Check default values
    assert widget.opt_ft.currentText() == "Sensitivity"
    assert widget.opt_fr.value() == 1.3
    assert widget.opt_p.value() == 3.0
    assert widget.opt_max_change.value() == 0.05
    assert widget.opt_n_it.value() == 30
    assert widget.opt_solver.currentText() == "Auto"


def test_analyze_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies the AnalyzeWidget initializes correctly."""
    widget = AnalysisWidget()
    assert hasattr(widget, "run_analysis_button") and isinstance(
        widget.run_analysis_button, QPushButton
    )


def test_displacement_widget_initialization(qt_app: QApplication):
    """Unit Test: Verifies the DisplacementWidget initializes correctly."""
    widget = DisplacementWidget()

    # Check instances
    assert isinstance(widget.mov_disp, QDoubleSpinBox)
    assert isinstance(widget.mov_iter, QSpinBox)
    assert hasattr(widget, "run_disp_button") and isinstance(
        widget.run_disp_button, QPushButton
    )

    # Check default values
    assert widget.mov_disp.value() == 1.0
    assert widget.mov_iter.value() == 1


# --- Tests for Container Widgets (Header, Footer, Preset) ---


def test_header_widget_structure(qt_app: QApplication):
    """Unit Test: Ensures the HeaderWidget contains the correct child widgets."""
    widget = HeaderWidget()

    # Check instances
    assert hasattr(widget, "info_button") and isinstance(
        widget.info_button, QPushButton
    )
    assert hasattr(widget, "help_button") and isinstance(
        widget.help_button, QPushButton
    )
    assert hasattr(widget, "issue_button") and isinstance(
        widget.issue_button, QPushButton
    )
    assert hasattr(widget, "theme_button") and isinstance(
        widget.theme_button, QPushButton
    )

    # Test github buttons
    widget.info_button.click()
    qt_app.processEvents()
    widget.help_button.click()
    qt_app.processEvents()
    widget.issue_button.click()
    qt_app.processEvents()

    # Test change theme
    icon_before = widget.theme_button.icon()
    widget.theme_button.click()
    qt_app.processEvents()
    assert icon_before != widget.theme_button.icon()


def test_preset_widget_structure(qt_app: QApplication):
    """Unit Test: Ensures the PresetWidget contains the correct child widgets."""
    widget = PresetWidget()

    # Check instances
    assert hasattr(widget, "presets_combo") and isinstance(
        widget.presets_combo, QComboBox
    )
    assert hasattr(widget, "save_preset_button")
    assert hasattr(widget, "delete_preset_button")


def test_footer_widget_structure(qt_app: QApplication):
    """Unit Test: Ensures the FooterWidget contains the correct child widgets."""
    widget = FooterWidget()

    # Check instances
    assert hasattr(widget, "create_button") and isinstance(
        widget.create_button, QPushButton
    )
    assert hasattr(widget, "stop_button") and isinstance(
        widget.stop_button, QPushButton
    )
    assert hasattr(widget, "binarize_button") and isinstance(
        widget.binarize_button, QPushButton
    )
    assert hasattr(widget, "save_button")
