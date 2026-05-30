# app/ui/widgets.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Custom PySide6 widgets for the TopoptComec UI.

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QAction, QColor, QFont
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGraphicsDropShadowEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QStackedLayout,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .icons import icons


# Make spinbox functions to avoid code repetition
def _make_spin(
    min_v: int, max_v: int, val: int, tip: str = "", width: int = 60
) -> QSpinBox:
    """
    Create a configured QSpinBox.

    Parameters
    ----------
    min_v : int
        Minimum value.
    max_v : int
        Maximum value.
    val : int
        Initial value.
    tip : str
        Tooltip text.
    width : int
        Maximum width in pixels.

    Returns
    -------
    QSpinBox
        Configured spin box widget.
    """
    s = QSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(val)
    if width:
        s.setMaximumWidth(width)
    s.setToolTip(tip)
    return s


def _make_dspin(
    min_v: float,
    max_v: float,
    val: float,
    step: float = 0.01,
    tip: str = "",
    width: int = 60,
) -> QDoubleSpinBox:
    """
    Create a configured QDoubleSpinBox.

    Parameters
    ----------
    min_v : float
        Minimum value.
    max_v : float
        Maximum value.
    val : float
        Initial value.
    step : float
        Step increment.
    tip : str
        Tooltip text.
    width : int
        Maximum width in pixels.

    Returns
    -------
    QDoubleSpinBox
        Configured double spin box widget.
    """
    s = QDoubleSpinBox()
    s.setRange(min_v, max_v)
    s.setValue(val)
    s.setSingleStep(step)
    if width:
        s.setMaximumWidth(width)
    s.setToolTip(tip)
    return s


def _make_combo(items: list = [], index: int = 0, tip: str = "") -> QComboBox:
    """
    Create a configured QComboBox.

    Parameters
    ----------
    items : list
        List of string items to add.
    index : int
        Initial selected index.
    tip : str
        Tooltip text.

    Returns
    -------
    QComboBox
        Configured combo box widget.
    """
    c = QComboBox()
    c.addItems(items)
    c.setCurrentIndex(index)
    c.setToolTip(tip)
    return c


class CollapsibleSection(QWidget):
    """A collapsible widget section with a title bar and a content area."""

    def __init__(self, title="Section", content_widget=None, parent=None):
        """
        Initialize the collapsible section.

        Parameters
        ----------
        title : str
            Title text for the section header.
        content_widget : QWidget, optional
            Widget to display in the content area.
        parent : QWidget, optional
            Parent widget.
        """
        super().__init__(parent)
        self.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
        )
        self.is_collapsed = True

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)

        # Title bar
        self.title_bar = QFrame()
        self.title_bar.setObjectName("collapsibleTitleBar")
        self.title_bar.setLayout(QHBoxLayout())
        self.title_bar.layout().setContentsMargins(5, 2, 5, 2)
        # Expand/Collapse Button
        self.toggle_button = QPushButton()
        self.toggle_button.setFixedSize(18, 18)
        self.toggle_button.setIcon(icons._get("arrow_right"))
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(False)
        self.title_bar.layout().addWidget(self.toggle_button)
        # Title
        self.title_label = QLabel(title)
        self.title_label.setObjectName("collapsibleTitleLabel")
        self.title_label.setStyleSheet("background: transparent;")
        self.title_bar.layout().addWidget(self.title_label)
        self.title_bar.layout().addStretch()
        # Visibility Toggle Button
        self.visibility_button = QPushButton()
        self.visibility_button.setIcon(icons._get("eye_open"))
        self.visibility_button.setCheckable(True)
        self.visibility_button.setChecked(True)
        self.visibility_button.setToolTip(
            "Toggle visibility of this element on the plot"
        )
        self.visibility_button.setVisible(False)
        self.visibility_button.setFixedSize(25, 18)
        self.title_bar.layout().addWidget(self.visibility_button)

        # Content area
        self.content_widget = content_widget if content_widget else QWidget()
        self.content_widget.setSizePolicy(
            QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Preferred
        )
        self.content_widget.setObjectName("collapsibleContent")
        self.content_widget.setVisible(not self.is_collapsed)

        self.main_layout.addWidget(self.title_bar)
        self.main_layout.addWidget(self.content_widget)

        self.title_bar.setFixedHeight(25)

        self.toggle_button.toggled.connect(self.toggle_collapse)
        self.title_bar.mousePressEvent = lambda event: self.toggle_button.toggle()

    def set_visibility_toggle(self, visible: bool):
        """
        Set the visibility of the visibility toggle button.

        Parameters
        ----------
        visible : bool
            Whether to show the visibility button.
        """
        self.visibility_button.setVisible(visible)

    def toggle_collapse(self, checked: bool):
        """
        Toggle the collapsed/expanded state of the section.

        Parameters
        ----------
        checked : bool
            Whether the toggle button is checked (expanded).
        """
        self.is_collapsed = not checked
        self.content_widget.setVisible(not self.is_collapsed)
        self.toggle_button.setIcon(
            icons._get("arrow_right" if self.is_collapsed else "arrow_down")
        )

    def collapse(self):
        """Collapse the section content."""
        self.toggle_button.setChecked(False)

    def expand(self):
        """Expand the section content."""
        self.toggle_button.setChecked(True)

    def update_all_icons(self):
        """Updates the collapsible arrow and visibility eye icons to match the current theme."""
        # Update the expand/collapse arrow
        self.toggle_button.setIcon(
            icons._get("arrow_right" if self.is_collapsed else "arrow_down")
        )

        # Update the visibility eye icon
        if self.visibility_button.isVisible():
            self.visibility_button.setIcon(
                icons._get(
                    "eye_open" if self.visibility_button.isChecked() else "eye_closed"
                )
            )


class ColorPickerButton(QPushButton):
    """A button that opens a color dialog and shows the selected color."""

    colorChanged = Signal(QColor)

    def __init__(self, initial_color: QColor = QColor("black")):
        """Initialize the ColorPickerButton."""
        super().__init__()
        self.setObjectName("ColorPickerButton")
        if isinstance(initial_color, str):
            initial_color = QColor(initial_color)

        self.color = initial_color
        self.update_color()
        self.clicked.connect(self.pick_color)
        self.setToolTip("Select material color")

    def pick_color(self):
        """
        Open a color dialog and update the selected color.

        Emits colorChanged signal if a valid color is selected.
        """
        new_color = QColorDialog.getColor(self.color, self, "Choose a color")
        if new_color.isValid():
            self.color = new_color
            self.update_color()
            self.colorChanged.emit(self.color)

    def update_color(self):
        """Update the button's background color to reflect the current color."""
        self.setStyleSheet(
            f"QPushButton#ColorPickerButton {{ background-color: {self.color.name()}; }}"
        )

    def get_color(self) -> str:
        """
        Get the current color as a hex string.

        Returns
        -------
        str
            Color in "#RRGGBB" format.
        """
        return self.color.name()

    def set_color(self, color: str | QColor):
        """
        Set the button's color.

        Parameters
        ----------
        color : str | QColor
            Color as hex string or QColor object.
        """
        if isinstance(color, str):
            color = QColor(color)

        if color != self.color:
            self.color = color
            self.update_color()
            self.colorChanged.emit(color)


class HeaderWidget(QWidget):
    """
    Header widget for the control panel with title and action buttons.

    Includes info, help, issue buttons and theme toggle.
    """

    def __init__(self):
        """Initialize the header widget with title and navigation buttons."""
        super().__init__()
        title_layout = QHBoxLayout(self)
        title_layout.setContentsMargins(0, 0, 0, 0)
        title_layout.addStretch()
        # Title
        title = QLabel("TopoptComec")
        title_font = QFont("JetBrains Mono", 20, QFont.Bold)
        title.setFont(title_font)
        title.setToolTip("Topology Optimization for Compliant Mechanisms")
        title.setAlignment(Qt.AlignCenter)
        title_layout.addWidget(title)
        # Info
        self.info_button = QPushButton()
        self.info_button.setIcon(icons._get("info"))
        self.info_button.setFixedSize(24, 24)
        self.info_button.setToolTip("Open the project's GitHub page")
        self.info_button.setFlat(True)
        title_layout.addWidget(self.info_button)
        # Wiki
        self.help_button = QPushButton()
        self.help_button.setIcon(icons._get("help"))
        self.help_button.setFixedSize(24, 24)
        self.help_button.setToolTip("Open the wiki")
        self.help_button.setFlat(True)
        title_layout.addWidget(self.help_button)
        # Issue
        self.issue_button = QPushButton("🪲")
        self.issue_button.setFixedSize(24, 24)
        self.issue_button.setToolTip("Report an issue")
        self.issue_button.setFlat(True)
        title_layout.addWidget(self.issue_button)
        # Theme Toggle
        self.theme_button = QPushButton()
        if icons.theme == "dark":
            self.theme_button.setIcon(icons._get("sun"))
            self.theme_button.setToolTip("Switch to light theme")
        else:
            self.theme_button.setIcon(icons._get("moon"))
            self.theme_button.setToolTip("Switch to dark theme")
        self.theme_button.setFixedSize(31, 31)
        self.theme_button.setCheckable(True)
        title_layout.addWidget(self.theme_button)


class PresetWidget(QFrame):
    """
    Widget for managing saved parameter presets.

    Provides UI for loading, saving, and deleting presets.
    """

    def __init__(self):
        """Initialize the preset widget."""
        super().__init__()
        # Frame
        self.setObjectName("presetFrame")
        preset_layout = QHBoxLayout(self)
        preset_layout.setContentsMargins(5, 5, 5, 5)
        # Combo Box
        preset_layout.addWidget(QLabel("Presets:"))
        self.presets_combo = _make_combo([], 0, "Load a saved set of parameters")
        preset_layout.addWidget(self.presets_combo, 1)  # Give combo box more stretch
        # Save
        self.save_preset_button = QPushButton()
        self.save_preset_button.setIcon(icons._get("save"))
        self.save_preset_button.setToolTip("Save current parameters as a new preset")
        preset_layout.addWidget(self.save_preset_button)
        # Delete
        self.delete_preset_button = QPushButton()
        self.delete_preset_button.setIcon(icons._get("delete"))
        self.delete_preset_button.setToolTip("Delete the selected preset")
        self.delete_preset_button.setEnabled(False)
        preset_layout.addWidget(self.delete_preset_button)


class DimensionsWidget(QWidget):
    """
    Widget for specifying problem dimensions and volume fraction.

    Includes size inputs (nx, ny, nz), volume fraction, and scaling controls.
    """

    def __init__(self):
        """Initialize the dimensions widget."""
        super().__init__()
        layout = QVBoxLayout(self)
        # Line 1: "Size:" [X] x [Y] x [Z]
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Size:"))
        self.nx = _make_spin(5, 1000, 60, "X")
        size_layout.addWidget(self.nx)
        size_layout.addWidget(QLabel("x"))
        self.ny = _make_spin(5, 1000, 40, "Y")
        size_layout.addWidget(self.ny)
        size_layout.addWidget(QLabel("x"))
        self.nz = _make_spin(0, 1000, 0, "Z → set to 0 for a 2D problem")
        size_layout.addWidget(self.nz)
        size_layout.addStretch()
        layout.addLayout(size_layout)
        # Line 2: "Vol. Frac:" [Spin]
        vol_layout = QHBoxLayout()
        vol_layout.addWidget(QLabel("Vol. Frac:"))
        self.volfrac = _make_dspin(0.05, 0.8, 0.3, 0.05, "Volume Fraction")
        vol_layout.addWidget(self.volfrac)
        vol_layout.addStretch()
        layout.addLayout(vol_layout)
        # Line 3: "Scale:" [Spin]   Scale Button
        scale_layout = QHBoxLayout()
        scale_layout.addWidget(QLabel("Scale:"))
        scale_layout.addSpacing(15)  # Align with line above
        self.scale = _make_dspin(
            0.5, 5, 1.0, 0.5, "Scale every component by this factor"
        )
        scale_layout.addWidget(self.scale)
        self.scale_button = QPushButton("Scale")
        self.scale_button.setIcon(icons._get("scale"))
        self.scale_button.setToolTip("Scale elements")
        scale_layout.addWidget(self.scale_button)
        scale_layout.addStretch(1)
        layout.addLayout(scale_layout)


class RegionsWidget(QWidget):
    """
    Widget for defining void/filled regions in the design domain.

    Supports adding and removing regions with shape, state, radius, and position.
    """

    nbRegionsChanged = Signal()

    def __init__(self):
        """Initialize the regions widget."""
        super().__init__()
        self.inputs = []  # This list will hold the input widgets so the MainWindow can access them

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        # Layout to hold the region containers
        self.regions_layout = QVBoxLayout()
        self.regions_layout.setSpacing(10)
        self.main_layout.addLayout(self.regions_layout)

        # Add button
        self.add_btn = QPushButton("+ Add Region")
        self.add_btn.clicked.connect(lambda: self.add_region())
        self.add_btn.setToolTip("Add a region")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)
        self.main_layout.addStretch()

    def add_region(
        self,
        rshape: str = "□",
        rstate: str = "Void",
        rradius: int = 5,
        pos: list = None,
        emit_signal: bool = True,
    ):
        """
        Add a new region to the widget.

        Parameters
        ----------
        rshape : str
            Shape type ("□" or "◯").
        rstate : str
            State ("Void" or "Filled").
        rradius : int
            Radius of the region.
        pos : list
            Position [x, y, z].
        emit_signal : bool
            Whether to emit nbRegionsChanged signal.
        """
        row = len(self.inputs)
        if row >= 10:  # safety
            return

        if pos is None:
            pos = [0, 0, 0]

        # Container Widget for this region
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5)

        # --- Line 1: [Minus] "Shape" [Combos] "Radius" [Spin] ---
        line1_layout = QHBoxLayout()

        # Remove button
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this region")
        remove_btn.clicked.connect(lambda: self.remove_region_by_widget(container))
        line1_layout.addWidget(remove_btn)

        # Shape
        line1_layout.addWidget(QLabel("Shape:"))
        rshape = _make_combo(["-", "□", "◯"], 0, "Shape of the region")
        line1_layout.addWidget(rshape)
        rstate = _make_combo(["Void", "Filled"], 0, "State of the region")
        line1_layout.addWidget(rstate)
        # Radius
        line1_layout.addWidget(QLabel("Radius:"))
        rradius = _make_spin(1, 100, rradius, "Radius")
        line1_layout.addWidget(rradius)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # --- Line 2: "Center" [X] [Y] [Z] ---
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)  # Align with line above
        line2_layout.addWidget(QLabel("Center:"))
        rx = _make_spin(0, 1000, pos[0], "X")
        line2_layout.addWidget(rx)
        ry = _make_spin(0, 1000, pos[1], "Y")
        line2_layout.addWidget(ry)
        rz = _make_spin(0, 1000, pos[2], "Z")
        line2_layout.addWidget(rz)
        line2_layout.addStretch()
        container_layout.addLayout(line2_layout)

        # Add to main layout (before the Add button)
        # self.main_layout has: grid (now empty/unused?), add_btn
        # We should insert the container into a specific layout for regions.
        # To avoid index mess, let's create a 'regions_layout' VBox

        self.regions_layout.addWidget(container)

        self.inputs.append(
            {
                "rshape": rshape,
                "rstate": rstate,
                "rradius": rradius,
                "rx": rx,
                "ry": ry,
                "rz": rz,
                "remove_btn": remove_btn,
                "container": container,
            }
        )

        if emit_signal:
            self.nbRegionsChanged.emit()
        self.update_remove_buttons()

    def remove_region_by_widget(self, container_widget: QWidget):
        """
        Remove a region by its container widget.

        Parameters
        ----------
        container_widget : QWidget
            The container widget of the region to remove.
        """
        # Find index
        idx = -1
        for i, region in enumerate(self.inputs):
            if region["container"] == container_widget:
                idx = i
                break

        if idx != -1:
            self.remove_region(idx)

    def remove_region(self, row: int, emit_signal: bool = True):
        """
        Remove a region by its row index.

        Parameters
        ----------
        row : int
            Index of the region to remove.
        emit_signal : bool
            Whether to emit nbRegionsChanged signal.
        """
        if row < 0 or row >= len(self.inputs):
            return

        # Remove widgets
        region = self.inputs.pop(row)
        # We just need to delete the container
        region["container"].deleteLater()
        region["container"].setParent(None)

        # No need to rebuild grid!
        if emit_signal:
            self.nbRegionsChanged.emit()
        self.update_remove_buttons()

    def update_remove_buttons(self):
        enabled = len(self.inputs) > 0
        for region in self.inputs:
            region["remove_btn"].setEnabled(enabled)


class ForcesWidget(QWidget):
    """
    Widget for defining input and output forces.

    Supports adding/removing forces with position, direction, and magnitude.
    """

    nbForcesChanged = Signal()

    def __init__(self):
        """Initialize the forces widget."""
        super().__init__()
        self.input_forces = []
        self.output_forces = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        self.arrows = ["-", "X:→", "X:←", "Y:↑", "Y:↓", "Z:<", "Z:>"]

        # Input forces section
        input_label = QLabel("Input")
        input_label.setStyleSheet("color: red;")
        self.main_layout.addWidget(input_label)

        self.input_layout = QVBoxLayout()
        self.input_layout.setSpacing(10)
        self.main_layout.addLayout(self.input_layout)

        self.add_if_btn = QPushButton("+ Add Input Force")
        self.add_if_btn.clicked.connect(lambda: self.add_input_force())
        self.add_if_btn.setToolTip("Add an input force")
        self.main_layout.addWidget(self.add_if_btn, alignment=Qt.AlignLeft)

        # Separator
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        self.main_layout.addWidget(line)

        # Output forces section
        output_label = QLabel("Output")
        output_label.setStyleSheet("color: blue;")
        self.main_layout.addWidget(output_label)

        self.output_layout = QVBoxLayout()
        self.output_layout.setSpacing(10)
        self.main_layout.addLayout(self.output_layout)

        self.add_of_btn = QPushButton("+ Add Output Force")
        self.add_of_btn.clicked.connect(lambda: self.add_output_force())
        self.add_of_btn.setToolTip("Add an output force")
        self.main_layout.addWidget(self.add_of_btn, alignment=Qt.AlignLeft)

        # Initialize default values
        self.add_input_force([30, 0, 0], 3, 0.01, emit_signal=False)
        self.add_output_force([30, 40, 0], 4, 0.01, emit_signal=False)

    @property
    def inputs(self) -> list:
        """
        Get all force widgets (input + output combined).

        Returns
        -------
        list
            Combined list of input and output force dictionaries.
        """
        return self.input_forces + self.output_forces

    def add_input_force(
        self,
        pos: list = None,
        arrow_idx: int = 1,
        norm: float = 0.01,
        emit_signal: bool = True,
    ):
        """
        Add a new input force.

        Parameters
        ----------
        pos : list
            Position [x, y, z].
        arrow_idx : int
            Index of the direction arrow.
        norm : float
            Force magnitude.
        emit_signal : bool
            Whether to emit nbForcesChanged signal.
        """
        if len(self.input_forces) >= 10:
            return

        if pos is None:
            pos = [0, 0, 0]

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5)

        # Line 1
        line1_layout = QHBoxLayout()
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this input force")
        remove_btn.clicked.connect(lambda: self.remove_force_by_widget(container, True))
        line1_layout.addWidget(remove_btn)

        line1_layout.addWidget(QLabel("Origin:"))
        fx = _make_spin(0, 1000, pos[0], "X")
        fy = _make_spin(0, 1000, pos[1], "Y")
        fz = _make_spin(0, 1000, pos[2], "Z")
        line1_layout.addWidget(fx)
        line1_layout.addWidget(fy)
        line1_layout.addWidget(fz)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # Line 2
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)
        line2_layout.addWidget(QLabel("Dir:"))
        fdir = _make_combo(self.arrows, arrow_idx, "Force direction")
        line2_layout.addWidget(fdir)
        line2_layout.addSpacing(20)
        line2_layout.addWidget(QLabel("Force (N):"))
        fnorm = _make_dspin(0, 10, norm, 0.01, "Force magnitude for input")
        line2_layout.addWidget(fnorm)
        line2_layout.addStretch()
        container_layout.addLayout(line2_layout)

        self.input_layout.addWidget(container)
        self.input_forces.append(
            {
                "fix": fx,
                "fiy": fy,
                "fiz": fz,
                "fidir": fdir,
                "finorm": fnorm,
                "container": container,
                "remove_btn": remove_btn,
            }
        )

        if emit_signal:
            self.nbForcesChanged.emit()
        self.update_ui_state()

    def add_output_force(
        self,
        pos: list = None,
        arrow_idx: int = 1,
        norm: float = 0.01,
        emit_signal: bool = True,
    ):
        """
        Add a new output force (spring).

        Parameters
        ----------
        pos : list
            Position [x, y, z].
        arrow_idx : int
            Index of the direction arrow.
        norm : float
            Spring stiffness magnitude.
        emit_signal : bool
            Whether to emit nbForcesChanged signal.
        """
        if len(self.output_forces) >= 10:
            return

        if pos is None:
            pos = [0, 0, 0]

        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5)

        # Line 1
        line1_layout = QHBoxLayout()
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this output force")
        remove_btn.clicked.connect(
            lambda: self.remove_force_by_widget(container, False)
        )
        line1_layout.addWidget(remove_btn)

        line1_layout.addWidget(QLabel("Origin:"))
        fx = _make_spin(0, 1000, pos[0], "X")
        fy = _make_spin(0, 1000, pos[1], "Y")
        fz = _make_spin(0, 1000, pos[2], "Z")
        line1_layout.addWidget(fx)
        line1_layout.addWidget(fy)
        line1_layout.addWidget(fz)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # Line 2
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)
        line2_layout.addWidget(QLabel("Dir:"))
        fdir = _make_combo(self.arrows, arrow_idx, "Force direction")
        line2_layout.addWidget(fdir)
        line2_layout.addSpacing(20)
        line2_layout.addWidget(QLabel("Spring (N/m):"))
        fnorm = _make_dspin(0, 10, norm, 0.01, "Spring stiffness for output")
        line2_layout.addWidget(fnorm)
        line2_layout.addStretch()
        container_layout.addLayout(line2_layout)

        self.output_layout.addWidget(container)
        self.output_forces.append(
            {
                "fox": fx,
                "foy": fy,
                "foz": fz,
                "fodir": fdir,
                "fonorm": fnorm,
                "container": container,
                "remove_btn": remove_btn,
            }
        )

        if emit_signal:
            self.nbForcesChanged.emit()
        self.update_ui_state()

    def remove_force_by_widget(self, container: QWidget, is_input: bool):
        """
        Remove a force by its container widget.

        Parameters
        ----------
        container : QWidget
            The container widget of the force to remove.
        is_input : bool
            Whether this is an input force (True) or output force (False).
        """
        target_list = self.input_forces if is_input else self.output_forces
        idx = -1
        for i, force in enumerate(target_list):
            if force["container"] == container:
                idx = i
                break

        if idx != -1:
            self.remove_force(idx, is_input)

    def remove_force(self, row: int, is_input: bool, emit_signal: bool = True):
        """
        Remove a force by its row index.

        Parameters
        ----------
        row : int
            Index of the force to remove.
        is_input : bool
            Whether this is an input force (True) or output force (False).
        emit_signal : bool
            Whether to emit nbForcesChanged signal.
        """
        target_list = self.input_forces if is_input else self.output_forces
        if row < 0 or row >= len(target_list):
            return

        force = target_list.pop(row)
        force["container"].deleteLater()
        force["container"].setParent(None)

        if emit_signal:
            self.nbForcesChanged.emit()
        self.update_ui_state()

    def update_ui_state(self):
        """
        Update the enabled/disabled state of add and remove buttons.

        Ensures constraints are met: 1-10 input forces, 0-10 output forces.
        """
        can_add_input = len(self.input_forces) < 10
        can_remove_input = len(self.input_forces) > 1
        self.add_if_btn.setEnabled(can_add_input)
        for force in self.input_forces:
            force["remove_btn"].setEnabled(can_remove_input)

        can_add_output = len(self.output_forces) < 10
        can_remove_output = len(self.output_forces) > 0  # allow 0 output forces
        self.add_of_btn.setEnabled(can_add_output)
        for force in self.output_forces:
            force["remove_btn"].setEnabled(can_remove_output)


class SupportWidget(QWidget):
    """
    Widget for defining support boundary conditions.

    Supports adding/removing supports with position, fixed dimensions, and radius.
    """

    nbSupportsChanged = (
        Signal()
    )  # Signal to update the parameters when the number of supports changes

    def __init__(self):
        """Initialize the support widget."""
        super().__init__()
        self.inputs = []  # This list will hold the input widgets so the MainWindow can access them

        self.main_layout = QVBoxLayout(self)

        # Container for the list of supports to ensure vertical stacking
        self.supports_container = QWidget()
        self.supports_list_layout = QVBoxLayout(self.supports_container)
        self.supports_list_layout.setContentsMargins(0, 0, 0, 0)
        self.supports_list_layout.setSpacing(10)
        self.main_layout.addWidget(self.supports_container)

        # Add button
        self.add_btn = QPushButton("+ Add Support")
        self.add_btn.clicked.connect(lambda: self.add_support())
        self.add_btn.setToolTip("Add a support")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)

        self.dims = ["-", "X", "Y", "Z", "XY", "XZ", "YZ", "XYZ"]

    def add_support(
        self,
        pos: list = None,
        dim: str = "XYZ",
        radius: int = 0,
        emit_signal: bool = True,
    ):
        """
        Add a new support.

        Parameters
        ----------
        pos : list
            Position [x, y, z].
        dim : str
            Fixed dimensions ("X", "Y", "Z", "XY", etc.).
        radius : int
            Support radius (0 for single node).
        emit_signal : bool
            Whether to emit nbSupportsChanged signal.
        """
        row = len(self.inputs)
        if row >= 10:  # safety
            return

        if pos is None:
            pos = [0, 0, 0]

        # Container Widget for this support
        container = QWidget()
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(0, 5, 0, 5)

        # Line 1: [Minus] "Center:" [X] [Y] [Z]
        line1_layout = QHBoxLayout()

        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this support")
        remove_btn.clicked.connect(lambda: self.remove_support_by_widget(container))
        line1_layout.addWidget(remove_btn)

        line1_layout.addWidget(QLabel("Center:"))
        sx = _make_spin(0, 1000, pos[0], "X")
        sy = _make_spin(0, 1000, pos[1], "Y")
        sz = _make_spin(0, 1000, pos[2], "Z")
        line1_layout.addWidget(sx)
        line1_layout.addWidget(sy)
        line1_layout.addWidget(sz)
        line1_layout.addStretch()
        container_layout.addLayout(line1_layout)

        # Line 2: "Fixed:" [Dim]   "Radius:" [Radius]
        line2_layout = QHBoxLayout()
        line2_layout.addSpacing(40)  # Align with line above
        line2_layout.addWidget(QLabel("Fixed:"))
        sdim = _make_combo(self.dims, self.dims.index(dim), "Fixed direction(s)")
        line2_layout.addWidget(sdim)
        line2_layout.addSpacing(20)
        line2_layout.addWidget(QLabel("Radius:"))
        sr = _make_spin(0, 10, radius, "Support's radius (0 = single node)")
        line2_layout.addWidget(sr)
        line2_layout.addStretch()
        container_layout.addLayout(line2_layout)

        self.supports_list_layout.addWidget(container)

        # Store the widgets in the public 'inputs' list
        data = {
            "sx": sx,
            "sy": sy,
            "sz": sz,
            "sdim": sdim,
            "sr": sr,
            "remove_btn": remove_btn,
            "container": container,
        }
        self.inputs.append(data)
        if emit_signal:
            self.nbSupportsChanged.emit()
        self.update_remove_buttons()

    def remove_support_by_widget(self, container_widget: QWidget):
        """
        Remove a support by its container widget.

        Parameters
        ----------
        container_widget : QWidget
            The container widget of the support to remove.
        """
        # Find index
        idx = -1
        for i, support in enumerate(self.inputs):
            if support["container"] == container_widget:
                idx = i
                break

        if idx != -1:
            self.remove_support(idx)

    def remove_support(self, row: int, emit_signal: bool = True):
        """
        Remove a support by its row index.

        Parameters
        ----------
        row : int
            Index of the support to remove.
        emit_signal : bool
            Whether to emit nbSupportsChanged signal.
        """
        if row < 0 or row >= len(self.inputs):
            return

        # Remove widgets
        support = self.inputs.pop(row)
        support["container"].deleteLater()
        support["container"].setParent(None)

        if emit_signal:
            self.nbSupportsChanged.emit()
        self.update_remove_buttons()

    def update_remove_buttons(self):
        enabled = len(self.inputs) > 0
        for support in self.inputs:
            support["remove_btn"].setEnabled(enabled)


class MaterialsWidget(QWidget):
    """
    Widget for defining material properties.

    Supports adding/removing materials with E, nu, percentage, and color.
    Maximum 2 materials supported.
    """

    nbMaterialsChanged = (
        Signal()
    )  # Signal to update the parameters when the number of material changes

    def __init__(self):
        """Initialize the materials widget."""
        super().__init__()
        self.inputs = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setSpacing(10)

        # Initialization Type (Global for all materials)
        init_layout = QHBoxLayout()
        init_layout.addWidget(QLabel("Initialization:"))
        self.mat_init_type = _make_combo(
            ["Uniform", "Around activity points", "Random"],
            0,
            "Materials distribution initialization type",
        )
        init_layout.addWidget(self.mat_init_type)
        self.main_layout.addLayout(init_layout)

        # Layout for material items
        self.materials_layout = QVBoxLayout()
        self.materials_layout.setSpacing(10)
        self.main_layout.addLayout(self.materials_layout)

        # Add button
        self.add_btn = QPushButton("+ Add Material")
        self.add_btn.clicked.connect(lambda: self.add_material())
        self.add_btn.setToolTip("Add a material (max 2)")
        self.main_layout.addWidget(self.add_btn, alignment=Qt.AlignLeft)

        self.main_layout.addStretch()

        # Add default material
        self.add_material(E=1.0, nu=0.3, percent=100, color="#000000")

    def add_material(
        self,
        E: float = 1.0,
        nu: float = 0.3,
        percent: int = 0,
        color: str = "#000000",
        emit_signal: bool = True,
    ):
        """
        Add a new material.

        Parameters
        ----------
        E : float
            Young's modulus.
        nu : float
            Poisson's ratio.
        percent : int
            Volume fraction percentage.
        color : str
            Hex color string.
        emit_signal : bool
            Whether to emit nbMaterialsChanged signal.
        """
        if len(self.inputs) > 1:
            return

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 5, 0, 5)

        # Line 1: [Minus] Color [Btn] % [Spin]
        line1 = QHBoxLayout()
        remove_btn = QPushButton("−")
        remove_btn.setFixedWidth(30)
        remove_btn.setToolTip("Remove this material")
        remove_btn.clicked.connect(lambda: self.remove_material_by_widget(container))
        line1.addWidget(remove_btn)
        # Color
        line1.addWidget(QLabel("Color:"))
        if isinstance(color, str):
            color = QColor(color)
        color_btn = ColorPickerButton(color)
        line1.addWidget(color_btn)
        line1.addSpacing(20)
        # Percentage
        line1.addWidget(QLabel("%:"))
        mat_percent = _make_spin(0, 100, percent, "Volume Fraction percentage")
        mat_percent.setSingleStep(5)
        mat_percent.valueChanged.connect(self._on_percent_changed)
        line1.addWidget(mat_percent)
        line1.addStretch()
        layout.addLayout(line1)

        # Line 2: E [Spin] nu [Spin]
        line2 = QHBoxLayout()
        line2.addSpacing(40)  # Align with line above
        # Young's Modulus
        line2.addWidget(QLabel("E:"))
        mat_E = _make_dspin(
            0.1, 100.0, E, 0.05, "Young's Modulus, material’s stiffness"
        )
        line2.addWidget(mat_E)
        line2.addSpacing(20)
        # Poisson's Ratio
        line2.addWidget(QLabel("ν:"))
        mat_nu = _make_dspin(
            0.0,
            0.49,
            nu,
            0.05,
            "Poisson's Ratio, material’s lateral shrinkage relative to its elongation",
        )
        line2.addWidget(mat_nu)
        line2.addStretch()
        layout.addLayout(line2)

        self.materials_layout.addWidget(container)

        self.inputs.append(
            {
                "E": mat_E,
                "nu": mat_nu,
                "percent": mat_percent,
                "color": color_btn,
                "remove_btn": remove_btn,
                "container": container,
            }
        )

        if emit_signal:
            self.nbMaterialsChanged.emit()
        self.update_ui_state()

    def remove_material_by_widget(self, container: QWidget):
        """
        Remove a material by its container widget.

        Parameters
        ----------
        container : QWidget
            The container widget of the material to remove.
        """
        idx = -1
        for i, mat in enumerate(self.inputs):
            if mat["container"] == container:
                idx = i
                break
        if idx != -1:
            self.remove_material(idx)

    def remove_material(self, row: int, emit_signal: bool = True):
        """
        Remove a material by its row index.

        Parameters
        ----------
        row : int
            Index of the material to remove.
        emit_signal : bool
            Whether to emit nbMaterialsChanged signal.
        """
        if row < 0 or row >= len(self.inputs):
            return

        mat = self.inputs.pop(row)
        mat["container"].deleteLater()
        mat["container"].setParent(None)

        self.inputs[0]["percent"].blockSignals(True)
        self.inputs[0]["percent"].setValue(100)
        self.inputs[0]["percent"].blockSignals(False)

        if emit_signal:
            self.nbMaterialsChanged.emit()
        self.update_ui_state()

    def _on_percent_changed(self, value: int):
        """When there are exactly 2 materials, auto-adjust the other to keep sum = 100."""
        if len(self.inputs) == 1:
            self.inputs[0]["percent"].setValue(100)
        elif len(self.inputs) == 2:
            sender = self.sender()
            if sender is self.inputs[0]["percent"]:
                other = self.inputs[1]["percent"]
            elif sender is self.inputs[1]["percent"]:
                other = self.inputs[0]["percent"]
            else:
                return
            other.blockSignals(True)
            other.setValue(100 - value)
            other.blockSignals(False)

    def update_ui_state(self):
        """
        Update the enabled/disabled state of add and remove buttons.

        Ensures constraints are met: 1-2 materials maximum.
        """
        # Update Add/Remove buttons state
        can_add = len(self.inputs) < 2
        can_remove = len(self.inputs) > 1

        self.add_btn.setEnabled(can_add)
        for mat in self.inputs:
            mat["remove_btn"].setEnabled(can_remove)


class OptimizerWidget(QWidget):
    """
    Widget for specifying optimizer parameters.

    Includes filter type, penalization, convergence criteria, and solver options.
    """

    def __init__(self):
        """Initialize the optimizer widget."""
        super().__init__()
        layout = QGridLayout(self)
        # Filter type
        tip = (
            "Regularization to avoid checkerboards and mesh dependency\n"
            "Sensitivity: Averages sensitivities to ensure stable and physically meaningful optimization gradients.\n"
            "Density: Smooths the material distribution to avoid checkerboard patterns and mesh dependency."
        )
        self.opt_ft = _make_combo(["Sensitivity", "Density", "None"], 0, tip)
        layout.addWidget(QLabel("Filter Type:"), 0, 0)
        layout.addWidget(self.opt_ft, 0, 1)
        # Filter Radius
        layout.addWidget(QLabel("Filter Radius:"), 1, 0)
        self.opt_fr = _make_dspin(
            0.1, 10.0, 1.3, 0.01, "Range of the filter coverage", None
        )
        layout.addWidget(self.opt_fr, 1, 1)
        # Penalization
        layout.addWidget(QLabel("Penalization:"), 2, 0)
        self.opt_p = _make_dspin(
            1.0,
            10.0,
            3.0,
            0.01,
            "Exponent in the SIMP method to penalize intermediate densities",
            None,
        )
        layout.addWidget(self.opt_p, 2, 1)
        # Eta
        layout.addWidget(QLabel("Eta:"), 3, 0)
        self.opt_eta = _make_dspin(
            0.05,
            1.0,
            0.3,
            0.05,
            "Damping factor in OC update rule to controls aggressiveness of density updates.\nLower eta: slower, more stable convergence; Higher eta: faster, but risk",
            None,
        )
        layout.addWidget(self.opt_eta, 3, 1)
        # Max change
        layout.addWidget(QLabel("Max change:"), 4, 0)
        self.opt_max_change = _make_dspin(
            0.01,
            0.5,
            0.05,
            0.05,
            "Bound the density change between two iterations to a maximum value",
            None,
        )
        layout.addWidget(self.opt_max_change, 4, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 5, 0)
        self.opt_n_it = _make_spin(
            1, 100, 30, "Number of optimization iterations", None
        )
        layout.addWidget(self.opt_n_it, 5, 1)
        # Solver
        layout.addWidget(QLabel("Solver:"), 6, 0)
        tip = (
            "Solver type for the linear system\n"
            "Auto: Chooses the best solver based on the problem size\n"
            "Direct: Uses LU factorization (spsolve)\n"
            "Iterative: Uses Conjugate Gradient (cg) with preconditioning\n"
        )
        self.opt_solver = _make_combo(["Auto", "Direct", "Iterative"], 0, tip)
        layout.addWidget(self.opt_solver, 6, 1)

        # Save frames
        self.save_frames_cb = QCheckBox("Save frames")
        self.save_frames_cb.setToolTip("Save intermediate optimization frames as PNG")
        layout.addWidget(self.save_frames_cb, 7, 0, 1, 2)


class AnalysisWidget(QWidget):
    """
    Widget for displaying mechanism analysis results.

    Shows checkerboard pattern detection, watertightness, thresholding, and efficiency.
    """

    def __init__(self):
        """Initialize the analysis widget."""
        super().__init__()
        layout = QGridLayout(self)
        # Checkerboard
        layout.addWidget(QLabel("Checkerboard:"), 0, 0)
        self.checkerboard_result = QLabel("-")
        self.checkerboard_result.setToolTip(
            "Check if the mechanism contains some checkerboard patterns"
        )
        layout.addWidget(self.checkerboard_result, 0, 1)
        # Watertight
        layout.addWidget(QLabel("Watertight:"), 1, 0)
        self.watertight_result = QLabel("-")
        self.watertight_result.setToolTip("Check if the mechanism is watertight")
        layout.addWidget(self.watertight_result, 1, 1)
        # Threshold
        layout.addWidget(QLabel("Thresholded:"), 2, 0)
        self.threshold_result = QLabel("-")
        self.threshold_result.setToolTip("Evaluate if gray zones are gone")
        layout.addWidget(self.threshold_result, 2, 1)
        # Efficiency
        layout.addWidget(QLabel("Efficient:"), 3, 0)
        self.efficiency_result = QLabel("-")
        self.efficiency_result.setToolTip(
            "Compare output displacement with input displacement"
        )
        layout.addWidget(self.efficiency_result, 3, 1)
        self.button_stack = QStackedLayout()
        # Analyze button
        self.run_analysis_button = QPushButton("🔍 Analyze")
        self.run_analysis_button.setToolTip("Start the analysis process")
        self.button_stack.addWidget(self.run_analysis_button)
        # Stop button
        self.stop_analysis_button = QPushButton(" Stop")
        self.stop_analysis_button.setObjectName("stop_analysis_button")
        self.stop_analysis_button.setIcon(icons._get("stop"))
        self.stop_analysis_button.setToolTip("Stop the analysis process")
        self.stop_analysis_button.setStyleSheet("background-color: #C0392B;")
        self.button_stack.addWidget(self.stop_analysis_button)
        self.button_stack_widget = QWidget()
        self.button_stack_widget.setLayout(self.button_stack)
        layout.addWidget(self.button_stack_widget, 4, 0, 1, 2)


class DisplacementWidget(QWidget):
    """
    Widget for displacement animation parameters.

    Includes displacement factor, iterations, and animation controls.
    """

    def __init__(self):
        """Initialize the displacement widget."""
        super().__init__()
        layout = QGridLayout(self)
        # Displacement
        layout.addWidget(QLabel("Displacement:"), 0, 0)
        self.mov_disp = _make_dspin(
            0.1,
            100.0,
            1.0,
            0.1,
            "Total scaling factor for the displacement animation",
            None,
        )
        layout.addWidget(self.mov_disp, 0, 1)
        # Iterations
        layout.addWidget(QLabel("Iterations:"), 1, 0)
        self.mov_iter = _make_spin(
            1,
            20,
            1,
            "Number of frames in the displacement animation.\nIf more than 1 iteration is set, a more complex function is used since it requires an domain enlargement and interpolations.",
            None,
        )
        layout.addWidget(self.mov_iter, 1, 1)
        # Bottom row with Checkbox and Button Stack
        row2_layout = QHBoxLayout()
        self.save_frames_cb = QCheckBox("Save frames")
        self.save_frames_cb.setToolTip("Save intermediate displacement frames as PNG")
        row2_layout.addWidget(self.save_frames_cb, 1)
        self.button_stack = QStackedLayout()
        # Move button
        self.run_disp_button = QPushButton("Move")
        self.run_disp_button.setIcon(icons._get("move"))
        self.run_disp_button.setToolTip("Start the displacement process")
        self.button_stack.addWidget(self.run_disp_button)
        # Stop button
        self.stop_disp_button = QPushButton(" Stop")
        self.stop_disp_button.setIcon(icons._get("stop"))
        self.stop_disp_button.setToolTip("Stop the displacement process")
        self.stop_disp_button.setStyleSheet("background-color: #C0392B;")
        self.button_stack.addWidget(self.stop_disp_button)
        # Reset button
        self.reset_disp_button = QPushButton("Reset View")
        self.reset_disp_button.setIcon(icons._get("reset"))
        self.reset_disp_button.setToolTip(
            "Reset the mechanism to its original position"
        )
        self.button_stack.addWidget(self.reset_disp_button)
        self.button_stack_widget = QWidget()
        self.button_stack_widget.setLayout(self.button_stack)
        row2_layout.addWidget(self.button_stack_widget, 1)

        layout.addLayout(row2_layout, 2, 0, 1, 2)


class FooterWidget(QWidget):
    """
    Footer widget with create, stop, binarize, and save buttons.

    Provides the main action controls for optimization.
    """

    def __init__(self):
        """Initialize the footer widget."""
        super().__init__()
        # Create button
        action_layout = QHBoxLayout(self)
        self.create_button = QPushButton(" Create")
        self.create_button.setIcon(icons._get("create"))
        self.create_button.setToolTip("Start the optimization process")
        self.create_button.setFont(QFont("JetBrains Mono", 14, QFont.Bold))
        self.start_create_button_effect()
        action_layout.addWidget(self.create_button)
        # Stop button
        self.stop_button = QPushButton(" Stop")
        self.stop_button.setIcon(icons._get("stop"))
        self.stop_button.setToolTip("Stop the optimization process")
        self.stop_button.setFont(QFont("JetBrains Mono", 14, QFont.Bold))
        self.stop_button.setStyleSheet("background-color: #C0392B;")
        self.stop_button.hide()  # Hidden by default
        action_layout.addWidget(self.stop_button)
        # Binarize button
        self.binarize_button = QPushButton()
        self.binarize_button.setIcon(icons._get("binarize"))
        self.binarize_button.setToolTip(
            "Apply threshold to make all colors solid (0 or 1)"
        )
        self.binarize_button.setFixedSize(29, 29)
        self.binarize_button.setEnabled(False)
        action_layout.addWidget(self.binarize_button)
        # Save button
        self.save_button = QToolButton()
        self.save_button.setIcon(icons._get("save"))
        self.save_button.setToolTip("Save the current result")
        self.save_button.setFixedSize(30, 30)  # A bit wider to accommodate the arrow
        self.save_button.setEnabled(False)
        self.save_button.setPopupMode(QToolButton.InstantPopup)

        save_menu = QMenu(self.save_button)

        self.save_png_action = QAction("Save as PNG Image...", self)
        save_menu.addAction(self.save_png_action)
        self.save_vti_action = QAction("Save as VTI (for ParaView)...", self)
        save_menu.addAction(self.save_vti_action)
        self.save_stl_action = QAction("Save as STL (for CAD)...", self)
        save_menu.addAction(self.save_stl_action)
        self.save_3mf_action = QAction("Save as 3MF (for 3D printing)...", self)
        save_menu.addAction(self.save_3mf_action)

        self.save_button.setMenu(save_menu)

        action_layout.addWidget(self.save_button)

    def start_create_button_effect(self, color_hex: str = "#F97316"):
        """
        Start the glow animation effect on the create button.

        Parameters
        ----------
        color_hex : str
            Hex color code for the glow effect.
        """
        # Shadow effect for glow
        create_button_effect = QGraphicsDropShadowEffect(self.create_button)
        create_button_effect.setBlurRadius(20)
        create_button_effect.setOffset(0)
        create_button_effect.setColor(QColor(color_hex))
        self.create_button.setGraphicsEffect(create_button_effect)
        # Animation for the shadow
        anim = QPropertyAnimation(create_button_effect, b"blurRadius", self)
        anim.setStartValue(5)
        anim.setEndValue(25)
        anim.setDuration(2000)
        anim.setEasingCurve(QEasingCurve.Type.CosineCurve)
        anim.setLoopCount(-1)
        anim.start()

    def stop_create_button_effect(self):
        """Stop the glow animation effect on the create button."""
        self.create_button.setGraphicsEffect(None)
