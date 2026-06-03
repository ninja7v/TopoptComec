# app/ui/main_window.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Main window for the TopoptComec application using PySide6.

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.colors import to_rgb
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QStatusBar,
    QScrollArea,
    QSplitter,
    QVBoxLayout,
    QWidget,
)
from typing import Tuple

# import mcubes
from app.ui import exporters

from .icons import icons
from .themes import DARK_THEME_STYLESHEET, LIGHT_THEME_STYLESHEET
from .widgets import (
    CollapsibleSection,
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
from .workers import AnalysisWorker, DisplacementWorker, OptimizerWorker
from .plotting import PlottingMixin
from .parameter_manager import ParameterManagerMixin

# Type aliases
FloatArray = npt.NDArray[np.float64]


class MainWindow(QMainWindow, PlottingMixin, ParameterManagerMixin):
    def __init__(self) -> None:
        """Initialize the main window, set up the UI, and connect signals."""
        super().__init__()
        self.setWindowTitle(
            "TopoptComec - Topology Optimization for Compliant Mechanisms"
        )
        self.setGeometry(100, 100, 1280, 720)

        self.xPhys: FloatArray = None
        self.u: FloatArray = None
        self.xPhys_valid: bool = False
        self.last_params: dict = {}
        self.current_theme: str = "dark"
        self.worker: AnalysisWorker | DisplacementWorker | OptimizerWorker | None = None

        self._set_theme(self.current_theme)

        self.presets: dict = {}
        self.presets_file: str = "presets.json"

        self.main_widget: QWidget = QWidget()
        self.setCentralWidget(self.main_widget)
        self.main_layout: QHBoxLayout = QHBoxLayout(self.main_widget)

        self.splitter: QSplitter = QSplitter(Qt.Horizontal)
        self.main_layout.addWidget(self.splitter)

        self.plot_frame: QFrame = QFrame()
        plot_layout: QVBoxLayout = QVBoxLayout(self.plot_frame)
        self.figure: plt.Figure = plt.figure()
        self.canvas: FigureCanvas = FigureCanvas(self.figure)
        plot_layout.addWidget(self.canvas)
        self.splitter.addWidget(self.plot_frame)

        self.is_displaying_deformation: bool = False
        self.last_displayed_frame_data: FloatArray = None

        self._create_control_panel()
        self.splitter.addWidget(self.control_panel_frame)
        self.splitter.setSizes([800, 480])

        self.status_bar: QStatusBar = self.statusBar()
        self.progress_bar: QProgressBar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self._load_presets()

        default_preset_name: str = "ForceInverter_2Sup_2D"
        if default_preset_name in self.presets:
            self.preset.presets_combo.setCurrentText(default_preset_name)
            self._on_preset_selected()  # This applies the preset and replots
        else:
            # Fallback in case the default preset isn't found
            print(f"Warning: Default preset '{default_preset_name}' not found.")
            self.last_params: dict = self._gather_parameters()
            self.replot()

    #################
    # CONTROL PANEL #
    #################

    def _create_control_panel(self) -> None:
        """Creates the right-hand side control panel with all settings."""
        self.control_panel_frame = QFrame()
        self.control_panel_frame.setFixedWidth(350)
        panel_layout = QVBoxLayout(self.control_panel_frame)

        # Header
        self.header = self._create_header()
        panel_layout.addWidget(self.header)

        # Preset
        self.preset = self._create_preset_section()
        panel_layout.addWidget(self.preset)

        # Parameters sections
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_widget = QWidget()
        self.sections_layout = QVBoxLayout(scroll_widget)
        self.sections_layout.setAlignment(Qt.AlignTop)
        scroll_widget.setLayout(self.sections_layout)
        scroll_area.setWidget(scroll_widget)
        panel_layout.addWidget(scroll_area)

        self.sections = {}
        self.sections["Dimensions"] = self._create_Dimensions_section()
        self.sections["Regions"] = self._create_regions_section()
        self.sections["Forces"] = self._create_forces_section()
        self.sections["Supports"] = self._create_supports_section()
        self.sections["Materials"] = self._create_materials_section()
        self.sections["Optimizer"] = self._create_optimizer_section()
        self.sections["Analysis"] = self._create_analysis_section()
        self.sections["Displacement"] = self._create_displacement_section()

        for section in self.sections.values():
            self.sections_layout.addWidget(section)

        # Footer
        self.footer = self._create_footer()
        panel_layout.addWidget(self.footer)

    ############
    # SECTIONS #
    ############

    def _create_header(self) -> HeaderWidget:
        """Creates the header widget and connects its signals."""
        header_widget = HeaderWidget()

        # Connect the signals from the widget's public buttons to MainWindow's handlers
        header_widget.info_button.clicked.connect(self._open_github_link)
        header_widget.help_button.clicked.connect(self._open_wiki_link)
        header_widget.issue_button.clicked.connect(self._open_issue_link)
        header_widget.theme_button.clicked.connect(self._toggle_theme)

        return header_widget

    def _create_preset_section(self) -> PresetWidget:
        """Creates the preset widget and connects its signals."""
        preset_widget = PresetWidget()
        preset_widget.presets_combo.activated.connect(self._on_preset_selected)
        preset_widget.save_preset_button.clicked.connect(self._save_new_preset)
        preset_widget.delete_preset_button.clicked.connect(self._delete_selected_preset)
        return preset_widget

    def _create_Dimensions_section(self) -> CollapsibleSection:
        """Creates the first section for Dimensions and volume fraction."""
        self.dim_widget = DimensionsWidget()
        section = CollapsibleSection("🔲 Dimensions", self.dim_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)
        self.dim_widget.nx.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.ny.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.nz.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.volfrac.valueChanged.connect(self.on_parameter_changed)
        self.dim_widget.scale_button.clicked.connect(self._scale_parameters)
        self.dim_widget.nx.valueChanged.connect(self._update_position_ranges)
        self.dim_widget.ny.valueChanged.connect(self._update_position_ranges)
        self.dim_widget.nz.valueChanged.connect(self._update_position_ranges)
        return section

    def _create_regions_section(self) -> CollapsibleSection:
        """Creates the second section for regions parameters."""
        self.regions_widget = RegionsWidget()
        section = CollapsibleSection("⚫ Regions", self.regions_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)
        self.regions_widget.add_btn.clicked.connect(self._connect_region_signals)
        self.regions_widget.add_btn.clicked.connect(self._update_position_ranges)
        self.regions_widget.nbRegionsChanged.connect(self.on_parameter_changed)
        return section

    def _create_forces_section(self) -> CollapsibleSection:
        """Creates the third section for forces parameters."""
        self.forces_widget = ForcesWidget()
        section = CollapsibleSection("💪 Forces", self.forces_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)

        self.forces_widget.add_if_btn.clicked.connect(self._connect_forces_signals)
        self.forces_widget.add_of_btn.clicked.connect(self._connect_forces_signals)
        self.forces_widget.add_if_btn.clicked.connect(self._update_position_ranges)
        self.forces_widget.add_of_btn.clicked.connect(self._update_position_ranges)
        self.forces_widget.nbForcesChanged.connect(self.on_parameter_changed)
        self._connect_forces_signals()

        return section

    def _create_supports_section(self) -> CollapsibleSection:
        """Creates the fourth section for supports parameters."""
        self.supports_widget = SupportWidget()
        section = CollapsibleSection("🔺 Supports", self.supports_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)
        self.supports_widget.add_btn.clicked.connect(self._connect_support_signals)
        self.supports_widget.add_btn.clicked.connect(self._update_position_ranges)
        self.supports_widget.nbSupportsChanged.connect(self.on_parameter_changed)

        return section

    def _create_materials_section(self) -> CollapsibleSection:
        """Creates the fifth section for material properties."""
        self.materials_widget = MaterialsWidget()
        section = CollapsibleSection("🧱 Materials", self.materials_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)
        self.materials_widget.add_btn.clicked.connect(self._connect_material_signals)
        self.materials_widget.nbMaterialsChanged.connect(self.on_parameter_changed)
        self.materials_widget.mat_init_type.currentIndexChanged.connect(
            self.on_parameter_changed
        )
        return section

    def _connect_material_signals(self) -> None:
        """(Re)connects on_parameter_changed signals to all current material widgets."""
        for mw in self.materials_widget.inputs:
            mw["color"].clicked.connect(self.replot)
            mw["E"].valueChanged.connect(self.on_parameter_changed)
            mw["nu"].valueChanged.connect(self.on_parameter_changed)
            mw["percent"].valueChanged.connect(self.on_parameter_changed)

    def _create_optimizer_section(self) -> CollapsibleSection:
        """Creates the sixth section for optimization parameters."""
        self.optimizer_widget = OptimizerWidget()
        section = CollapsibleSection("💻 Optimizer", self.optimizer_widget)
        self.optimizer_widget.opt_ft.currentIndexChanged.connect(
            self.on_parameter_changed
        )
        self.optimizer_widget.opt_fr.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_p.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_eta.valueChanged.connect(self.on_parameter_changed)
        self.optimizer_widget.opt_max_change.valueChanged.connect(
            self.on_parameter_changed
        )
        self.optimizer_widget.opt_n_it.valueChanged.connect(self.on_parameter_changed)
        return section

    def _create_analysis_section(self) -> CollapsibleSection:
        """Creates the seventh section for analysis parameters."""
        self.analysis_widget = AnalysisWidget()
        section = CollapsibleSection("🔍 Analysis", self.analysis_widget)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)
        self.analysis_widget.stop_analysis_button.clicked.connect(self._stop_analysis)
        self.analysis_widget.run_analysis_button.setEnabled(
            False
        )  # Disabled until a result is ready
        self.analysis_widget.run_analysis_button.clicked.connect(self._run_analysis)
        self.analysis_widget.stop_analysis_button.clicked.connect(self._stop_analysis)
        return section

    def _create_displacement_section(self) -> CollapsibleSection:
        """Creates the seventh section for displacement animation parameters."""
        self.displacement_widget = DisplacementWidget()
        section = CollapsibleSection("↔️ Displacement", self.displacement_widget)
        section.set_visibility_toggle(True)
        section.visibility_button.toggled.connect(self._on_visibility_toggled)
        section.visibility_button.setEnabled(False)  # Disabled until a result is ready
        self.displacement_widget.run_disp_button.setEnabled(
            False
        )  # Disabled until a result is ready
        section.visibility_button.setToolTip(
            "Preview displacement vectors on the main plot"
        )
        section.visibility_button.setChecked(False)

        section.visibility_button.toggled.connect(self.replot)
        self.displacement_widget.run_disp_button.clicked.connect(self._run_displacement)
        self.displacement_widget.stop_disp_button.clicked.connect(
            self._stop_displacement
        )
        self.displacement_widget.reset_disp_button.clicked.connect(
            self._reset_displacement_view
        )
        self.displacement_widget.mov_disp.valueChanged.connect(
            self._on_displacement_preview_changed
        )
        return section

    def _on_visibility_toggled(self, checked: bool) -> None:
        """
        Handles the toggling of any visibility button.

        Parameters
        ----------
        checked : bool
            The new checked state of the visibility button that was toggled.
        """
        button = self.sender()  # method gives the specific button that was clicked.
        if not button:
            return

        if checked:
            button.setIcon(icons._get("eye_open"))
            button.setToolTip("Element is visible. Click to hide.")
        else:
            button.setIcon(icons._get("eye_closed"))
            button.setToolTip("Element is hidden. Click to show.")

        self.replot()

    def _create_footer(self) -> FooterWidget:
        """Creates the footer widget and connects its signals."""
        footer_widget = FooterWidget()
        footer_widget.create_button.clicked.connect(self._run_optimization)
        footer_widget.stop_button.clicked.connect(self._stop_optimization)
        footer_widget.binarize_button.clicked.connect(self._on_binarize_clicked)
        footer_widget.save_png_action.triggered.connect(self._save_as_png)
        footer_widget.save_vti_action.triggered.connect(self._save_as_vti)
        footer_widget.save_stl_action.triggered.connect(self._save_as_stl)
        footer_widget.save_3mf_action.triggered.connect(self._save_as_3mf)
        return footer_widget

    ################
    # OPTIMIZATION #
    ################

    def _run_optimization(self) -> None:
        """
        Start the optimization process using the current UI parameters.

        This method validates parameters, prepares the UI (progress bar,
        disabling controls), and starts an `OptimizerWorker` thread. Live
        updates are received via connected signals which update the plot and
        progress indicators.
        """
        if self.worker is not None:
            QMessageBox.warning(
                self,
                "Analysis Warning",
                "A worker is already running.",
            )
            return
        error = self._validate_parameters(self.last_params)
        if error:
            QMessageBox.critical(self, "Input Error", error)
            return

        self.xPhys_valid = False

        if self.is_displaying_deformation or self.last_displayed_frame_data is not None:
            self._reset_displacement_view()

        # Stop animation
        self.footer.stop_create_button_effect()

        # This creates the gray box that will be updated live
        self.replot()
        QApplication.processEvents()  # Force the UI to draw the initial state

        self.preset.setEnabled(False)
        self.footer.create_button.setEnabled(False)
        self.footer.create_button.hide()
        self.footer.stop_button.setText(" Stop")
        self.footer.stop_button.setEnabled(True)
        self.footer.stop_button.show()
        self.footer.binarize_button.setEnabled(False)
        self.footer.save_button.setEnabled(False)
        self.status_bar.showMessage("Starting optimization...")
        self.progress_bar.setRange(0, self.last_params["Optimizer"]["n_it"])
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        self.worker = OptimizerWorker(self.last_params)
        self.worker.progress.connect(self._update_optimization_progress)
        self.worker.frameReady.connect(self._update_optimization_plot)
        self.worker.finished.connect(self._handle_optimization_results)
        self.worker.error.connect(self._handle_optimization_error)
        self.worker.start()

    def _stop_optimization(self) -> None:
        """Requests the running optimizer worker to stop."""
        if self.worker:
            self.status_bar.showMessage("Stopping optimization...")
            self.footer.stop_button.setText("Stopping...")
            self.footer.stop_button.setEnabled(False)  # Prevent multiple clicks
            self.worker.request_stop()

    def _update_optimization_progress(
        self, iteration: int, objective: float, change: float
    ) -> None:
        """
        Updates the progress bar and status message during optimization.

        Parameters
        ----------
        iteration : int
            Current iteration number of the optimizer.
        objective : float
            Current objective function value.
        change : float
            Current change in design variables (smaller than max density change).
        """
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(
            f"It: {iteration}, Obj: {objective:.4f}, Change: {change:.4f}"
        )

    def _update_optimization_plot(self, xPhys_frame: np.ndarray) -> None:
        """
        Update the main plot with an intermediate optimizer frame.

        Parameters
        ----------
        xPhys_frame : np.ndarray
            Flattened density frame produced by the optimizer. For
            multi-material problems the array may be 2D (n_mat, nel).
        """
        # Ensure a plot exist to update
        if not self.figure.get_axes():
            return

        ax = self.figure.get_axes()[0]

        # Get dimensions and update the image data
        is_3d = (
            self.last_params["Dimensions"]["nelxyz"][2] > 0
            if self.last_params
            else False
        )

        # Multi-material frames have shape (n_mat, nel) — use full replot
        is_multi = xPhys_frame.ndim == 2

        if is_3d or is_multi:
            self._plot_material(ax, is_3d=is_3d, xPhys_data=xPhys_frame)
            self._redraw_non_material_layers(ax, is_3d=is_3d)
        else:
            if not ax.images:
                return
            im = ax.images[0]  # The imshow object is the first image on the axes
            nelx, nely = self.last_params["Dimensions"]["nelxyz"][:2]
            im.set_array(xPhys_frame.reshape((nelx, nely)).T)

        self.canvas.draw()

        if self.optimizer_widget.save_frames_cb.isChecked():
            preset_name = self.preset.presets_combo.currentText()
            if preset_name and preset_name != "Select a preset...":
                folder = os.path.join("results", preset_name, f"{preset_name}_creation")
                os.makedirs(folder, exist_ok=True)
                iteration = self.progress_bar.value()
                filename = os.path.join(
                    folder, f"{preset_name}_creation_{iteration}.png"
                )
                self.figure.savefig(filename, dpi=300, bbox_inches="tight")

    def _handle_optimization_results(
        self, result: Tuple[np.ndarray, np.ndarray]
    ) -> None:
        """
        Finalize the UI after optimization completes and cache results.

        Parameters
        ----------
        result : tuple
            Tuple `(xPhys, u)` where `xPhys` is the density field and `u` is
            the displacement vector returned by the optimizer. If a preset
            is selected the result is saved to `results/<preset>/...`.
        """
        self.xPhys, self.u = result
        self.last_displayed_frame_data = None
        self.worker = None
        self.status_bar.showMessage("Optimization finished successfully.", 5000)
        self.preset.setEnabled(True)
        self.footer.stop_button.hide()
        self.footer.create_button.show()
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.footer.stop_create_button_effect()

        # Validate convergence (density > 1%)
        mean_density: float = (
            np.mean(self.xPhys.sum(axis=0))
            if self.xPhys.ndim == 2
            else np.mean(self.xPhys)
        )
        if mean_density < 0.01:
            self.xPhys_valid = False
            QMessageBox.warning(
                self,
                "Optimization Warning",
                "No valid structure was found. Please adjust your parameters and try again.",
            )
            self.footer.binarize_button.setEnabled(False)
            self.footer.save_button.setEnabled(False)
            self.analysis_widget.run_analysis_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.sections["Displacement"].visibility_button.setEnabled(False)
        else:
            self.xPhys_valid = True
            self.footer.binarize_button.setEnabled(True)
            self.footer.save_button.setEnabled(True)
            self.analysis_widget.run_analysis_button.setEnabled(True)
            self.displacement_widget.run_disp_button.setEnabled(True)
            self.sections["Displacement"].visibility_button.setEnabled(True)

        preset_name = self.preset.presets_combo.currentText()
        if self.xPhys_valid and preset_name in self.presets:
            # Detect if running from pytest and use tests/ instead of results/
            is_pytest = os.environ.get("PYTEST_CURRENT_TEST") is not None
            base_dir = "tests" if is_pytest else "results"

            os.makedirs(os.path.join(base_dir, preset_name), exist_ok=True)
            cache_file = os.path.join(
                base_dir, preset_name, f"{preset_name}_density_field.npz"
            )
            try:
                np.savez_compressed(cache_file, xPhys=self.xPhys, u=self.u)
            except Exception as e:
                print(f"Failed to save cache: {e}")

        self.replot()

    def _handle_optimization_error(self, error_msg: str) -> None:
        """
        Handles any errors that occur during optimization.

        Parameters
        ----------
        error_msg : str
            Message describing the error that occurred.
        """
        self.worker = None
        self.status_bar.showMessage("Optimization failed.", 5000)
        self.preset.setEnabled(True)
        self.footer.stop_button.hide()
        self.footer.create_button.show()
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.footer.binarize_button.setEnabled(False)
        self.footer.save_button.setEnabled(False)
        self.analysis_widget.run_analysis_button.setEnabled(False)
        self.displacement_widget.run_disp_button.setEnabled(False)
        self.sections["Displacement"].visibility_button.setEnabled(False)
        self.xPhys_valid = False
        QMessageBox.critical(self, "Runtime Error", error_msg)

    ################
    # DISPLACEMENT #
    ################

    def _run_displacement(self) -> None:
        """
        Start displacement computation/animation for the last optimization.

        If only a single displacement frame is requested it computes it
        synchronously; otherwise it launches a `DisplacementWorker` to
        compute frames in the background while updating the UI.
        """
        if not self.xPhys_valid or self.u is None:
            QMessageBox.warning(
                self,
                "Displacement Warning",
                "You must run a successful optimization with a valid result before analyzing movement.",
            )
            return
        if self.worker is not None:
            QMessageBox.warning(
                self,
                "Displacement Warning",
                "A worker is already running.",
            )
            return

        self.is_displaying_deformation = True
        self.last_displayed_frame_data = None
        self.displacement_widget.button_stack.setCurrentWidget(
            self.displacement_widget.stop_disp_button
        )
        self.footer.create_button.setEnabled(False)

        QApplication.processEvents()

        self.last_params = self._gather_parameters()
        if self.last_params["Displacement"]["disp_iterations"] == 1:
            # Run single-frame logic directly
            self.status_bar.showMessage("Calculating single displacement frame...")
            QApplication.processEvents()  # Update UI
            from app.core.displacements import single_linear_displacement

            nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
            self.last_displayed_frame_data = single_linear_displacement(
                self.u,
                nelx,
                nely,
                nelz,
                self.last_params["Displacement"]["disp_factor"],
            )
            self.replot()
            self._handle_displacement_finished("Single frame shown.")
            self.status_bar.showMessage("Single displacement plot shown.", 3000)
        else:
            self.footer.create_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.status_bar.showMessage("Starting displacement computation...")

            self.progress_bar.setRange(
                0, self.last_params["Displacement"]["disp_iterations"] + 1
            )
            self.progress_bar.setValue(0)
            self.progress_bar.setVisible(True)

            self.worker = DisplacementWorker(self.last_params, self.xPhys, self.u)
            self.worker.progress.connect(self._update_displacement_progress)
            self.worker.frameReady.connect(self._update_animation_frame)
            self.worker.finished.connect(self._handle_displacement_finished)
            self.worker.error.connect(self._handle_displacement_error)
            self.worker.start()

    def _stop_displacement(self) -> None:
        """Requests the running displacement worker to stop."""
        if self.worker:
            self.displacement_widget.stop_disp_button.setText("Stopping...")
            self.displacement_widget.stop_disp_button.setEnabled(False)
            self.worker.request_stop()

    def _reset_displacement_view(self) -> None:
        """Resets the plot to the original, undeformed optimizer result."""
        self.is_displaying_deformation = False
        self.last_displayed_frame_data = None
        self.replot()  # Redraw the original view
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.displacement_widget.button_stack.setCurrentWidget(
            self.displacement_widget.run_disp_button
        )

    def _update_displacement_progress(self, iteration: int) -> None:
        """Updates the progress bar and status message during displacement computation."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(
            f"Running non-linear displacement: step {iteration}..."
        )

    def _update_animation_frame(self, frame_data: np.ndarray) -> None:
        """
        Update the main plot with a displacement animation frame.

        Parameters
        ----------
        frame_data : np.ndarray
            Frame data containing either a single-material flattened density
            field or multi-material (n_mat, nel) array depending on the
            current result.
        """
        # Safety checks to ensure a plot exists and parameters are available
        if not self.figure.get_axes() or not self.last_params:
            return
        ax = self.figure.get_axes()[0]
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
        is_3d = nelz > 0
        self.last_displayed_frame_data = np.array(frame_data, copy=True)

        if is_3d:
            ax.clear()
            self._plot_material(ax, is_3d, frame_data)

        else:
            # we don't use _plot_material here because we use set_array instead of imshow
            if not ax.images:
                return
            im = ax.images[0]

            is_multi = hasattr(self.xPhys, "ndim") and self.xPhys.ndim > 1
            if is_multi:
                n_mat, nel = frame_data.shape
                rgb_image = np.ones((nel, 3))  # Start white

                for i in range(n_mat):
                    mat_rgb = np.array(
                        to_rgb(self.materials_widget.inputs[i]["color"].get_color())
                    )
                    # Blend: pixel = sum(rho_i * color_i)
                    rgb_image += frame_data[i, :, np.newaxis] * (mat_rgb - 1.0)

                rgb_image = np.clip(rgb_image, 0.0, 1.0)
                # Reshape to (nelx, nely, 3 -RGB-) and transpose spatial dimensions to (nely, nelx, 2)
                final_image = rgb_image.reshape((nelx, nely, 3)).transpose(1, 0, 2)

                im.set_array(final_image)
            else:
                im.set_array(frame_data.reshape((nelx, nely)).T)

            # Remove old layers like previous forces, regions, and especially the undeformed displacement preview
            # that were drawn prior to the animation starting.
            for coll in list(ax.collections):
                coll.remove()
            for patch in list(ax.patches):
                patch.remove()
        self._redraw_non_material_layers(ax, is_3d)

        # Redraw the canvas to show the changes
        self.canvas.draw()

        if self.last_params.get("Displacement", {}).get("save_frames", False):
            preset_name = self.preset.presets_combo.currentText()
            if preset_name and preset_name != "Select a preset...":
                folder = os.path.join(
                    "results", preset_name, f"{preset_name}_displacement"
                )
                os.makedirs(folder, exist_ok=True)
                iteration = self.progress_bar.value()
                filename = os.path.join(
                    folder, f"{preset_name}_displacement_{iteration}.png"
                )
                self.figure.savefig(filename, dpi=300, bbox_inches="tight")

    def _on_displacement_preview_changed(self) -> None:
        """Triggers a replot if the preview is active when displacement factor changes."""
        if self.sections["Displacement"].visibility_button.isChecked():
            self.replot()

    def _handle_displacement_finished(self, message: str) -> None:
        """Handles the results after displacement computation finishes successfully."""
        self.status_bar.showMessage(message, 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(False)
        self.displacement_widget.button_stack.setCurrentWidget(
            self.displacement_widget.reset_disp_button
        )
        self.displacement_widget.stop_disp_button.setText(
            " Stop"
        )  # Reset text for next run
        self.displacement_widget.stop_disp_button.setEnabled(True)
        self.is_displaying_deformation = True
        self.worker = None

    def _handle_displacement_error(self, error_msg: str) -> None:
        """Handles any errors that occur during displacement computation."""
        self.worker = None
        self.status_bar.showMessage("Displacements failed.", 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        QMessageBox.critical(self, "Displacement Runtime Error", error_msg)

    ############
    # Analysis #
    ############
    def _run_analysis(self) -> None:
        """Starts the analysis evaluation based on the last optimization result"""
        if not self.xPhys_valid or self.u is None:
            QMessageBox.warning(
                self,
                "Analysis Warning",
                "You must run a successful optimization with a valid result before analyzing results.",
            )
            return
        if self.worker is not None:
            QMessageBox.warning(
                self,
                "Analysis Warning",
                "A worker is already running.",
            )
            return

        aw = self.analysis_widget
        aw.run_analysis_button.setEnabled(False)
        aw.stop_analysis_button.setText(" Stop")
        aw.button_stack.setCurrentWidget(aw.stop_analysis_button)
        aw.stop_analysis_button.setEnabled(True)
        self.footer.create_button.setEnabled(False)
        self.progress_bar.setRange(0, 4)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.displacement_widget.run_disp_button.setEnabled(False)

        self.worker = AnalysisWorker(self.last_params, self.xPhys, self.u)
        self.worker.progress.connect(self._update_analysis_progress)
        self.worker.finished.connect(self._handle_analysis_finished)
        self.worker.error.connect(self._handle_analysis_error)
        self.worker.start()

    def _stop_analysis(self) -> None:
        """Requests the running analysis worker to stop."""
        if self.worker:
            self.analysis_widget.stop_analysis_button.setText(" Stopping...")
            self.analysis_widget.stop_analysis_button.setEnabled(False)
            self.worker.request_stop()

    def _update_analysis_progress(self, iteration: int) -> None:
        """Updates the progress bar and status message during analysis."""
        self.progress_bar.setValue(iteration)
        self.status_bar.showMessage(f"Running analysis: step {iteration}...")

    def _handle_analysis_finished(self, results: Tuple[bool, bool, bool, bool]) -> None:
        """Handles the results after analysis finishes successfully."""
        aw = self.analysis_widget
        aw.checkerboard_result.setText("yes" if results[0] else "no")
        aw.checkerboard_result.setStyleSheet(
            "color: green;" if not results[0] else "color: red;"
        )
        aw.watertight_result.setText("yes" if results[1] else "no")
        aw.watertight_result.setStyleSheet(
            "color: green;" if results[1] else "color: red;"
        )
        aw.threshold_result.setText("yes" if results[2] else "no")
        aw.threshold_result.setStyleSheet(
            "color: green;" if results[2] else "color: red;"
        )
        aw.efficiency_result.setText("yes" if results[3] else "no")
        aw.efficiency_result.setStyleSheet(
            "color: green;" if results[3] else "color: red;"
        )
        self.status_bar.showMessage("Analysis finished successfully.", 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        aw.stop_analysis_button.setEnabled(False)
        aw.stop_analysis_button.setText(" Stop")  # Reset text for next run
        aw.button_stack.setCurrentWidget(aw.run_analysis_button)
        aw.run_analysis_button.setEnabled(True)
        self.displacement_widget.run_disp_button.setEnabled(True)
        self.footer.create_button.setEnabled(True)
        self.worker = None

    def _handle_analysis_error(self, error_msg: str) -> None:
        """Handles any errors that occur during analysis."""
        self.worker = None
        self.status_bar.showMessage("Analysis failed.", 5000)
        self.progress_bar.setVisible(False)
        self.footer.create_button.setEnabled(True)
        self.analysis_widget.run_analysis_button.setEnabled(True)
        QMessageBox.critical(self, "Analysis Runtime Error", error_msg)

    ########
    # Save #
    ########

    def _save_result_as(self, file_type: str) -> None:
        """
        Save the current result to disk in the requested format.

        Parameters
        ----------
        file_type : str
            One of `"png"`, `"vti"`, `"stl"`, or `"3mf"`. The function
            will present a file dialog and call the appropriate exporter.
        """
        if self.xPhys is None:
            return

        os.makedirs("results", exist_ok=True)

        # Base filename depending on preset
        if self.preset.presets_combo.currentText() != "Select a preset...":
            base_name = self.preset.presets_combo.currentText()
        else:
            base_name = (
                "result_3d"
                if self.last_params["Dimensions"]["nelxyz"][2] > 0
                else "result_2d"
            )

        # File dialog config
        filters = {
            "png": ("Save as PNG", "Portable Network Graphics (*.png)"),
            "vti": ("Save as VTI", "VTK Image Data (*.vti)"),
            "stl": ("Save as STL", "STL File (*.stl)"),
            "3mf": ("Save as 3MF", "3D Manufacturing File (*.3mf)"),
        }

        window_title, extension_filter = filters[file_type]
        default_path = f"results/{base_name}.{file_type}"

        filepath, _ = QFileDialog.getSaveFileName(
            self, window_title, default_path, extension_filter
        )
        if not filepath:  # user canceled
            return

        try:
            colors = self.last_params.get("Materials", {}).get("color", [])

            if file_type == "png":
                self.figure.savefig(filepath, dpi=300, bbox_inches="tight")

            elif file_type == "vti":
                success, error_msg = exporters.save_as_vti(
                    self.xPhys,
                    self.last_params["Dimensions"]["nelxyz"],
                    filepath,
                )
                if not success:
                    raise Exception(error_msg)

            elif file_type == "stl":
                success, error_msg = exporters.save_as_stl(
                    self.xPhys,
                    self.last_params["Dimensions"]["nelxyz"],
                    filepath,
                )
                if not success:
                    raise Exception(error_msg)

            elif file_type == "3mf":
                success, error_msg = exporters.save_as_3mf(
                    self.xPhys,
                    self.last_params["Dimensions"]["nelxyz"],
                    filepath,
                    colors,
                )
                if not success:
                    raise Exception(error_msg)

            self.status_bar.showMessage(f"Result saved to {filepath}", 5000)

        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Could not save the file:\n{e}")

    def _save_as_png(self) -> None:
        """Function connected to the save as PNG button."""
        self._save_result_as("png")

    def _save_as_vti(self) -> None:
        """Function connected to the save as VTI button."""
        self._save_result_as("vti")

    def _save_as_stl(self) -> None:
        """Function connected to the save as STL button."""
        self._save_result_as("stl")

    def _save_as_3mf(self) -> None:
        """Function connected to the save as 3MF button."""
        self._save_result_as("3mf")

    ###########
    # Presets #
    ###########

    def _load_presets(self) -> None:
        """Loads presets from the JSON file and populates the combo box."""
        try:
            with open(self.presets_file, "r") as f:
                self.presets = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.presets = {}
            print("Presets file not found or invalid. Starting fresh.")

        self.preset.presets_combo.blockSignals(True)
        self.preset.presets_combo.clear()
        self.preset.presets_combo.addItem("Select a preset...")  # Index 0
        self.preset.presets_combo.addItems(sorted(self.presets.keys()))
        self.preset.presets_combo.blockSignals(False)
        self.preset.delete_preset_button.setEnabled(False)

    def _save_presets(self) -> None:
        """Saves the current presets dictionary to the JSON file."""
        with open(self.presets_file, "w") as f:
            json.dump(self.presets, f, indent=4)
        self.status_bar.showMessage("Presets saved.", 3000)

    def _save_new_preset(self) -> None:
        """Prompts the user for a name and saves the current parameters."""
        preset_name, ok = QInputDialog.getText(
            self, "Save Preset", "Enter a name for this preset:"
        )

        if ok and preset_name:
            if preset_name in self.presets:
                reply = QMessageBox.question(
                    self,
                    "Overwrite Preset",
                    f"A preset named '{preset_name}' already exists. Overwrite it?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                if reply == QMessageBox.StandardButton.No:
                    return  # User cancelled

            self.presets[preset_name] = self._gather_parameters()
            self._save_presets()
            self._load_presets()
            self.preset.presets_combo.setCurrentText(preset_name)
            self.preset.delete_preset_button.setEnabled(True)

    def _delete_selected_preset(self) -> None:
        """Deletes the currently selected preset after confirmation."""
        preset_name = self.preset.presets_combo.currentText()
        if preset_name not in self.presets:
            return

        reply = QMessageBox.warning(
            self,
            "Delete Preset",
            f"You are about to permanently delete the preset '{preset_name}'.\nContinue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )

        if reply == QMessageBox.StandardButton.Yes:
            del self.presets[preset_name]
            self._save_presets()
            self._load_presets()  # Reloads the list and disables the delete button
            self.status_bar.showMessage(f"Preset '{preset_name}' deleted.", 3000)

    def _on_preset_selected(self) -> None:
        """
        Apply the selected preset to the UI widgets and, if a cached result
        exists under `results/<preset>/`, load it into the viewer.
        """
        preset_name = self.preset.presets_combo.currentText()
        if preset_name in self.presets:
            self.footer.start_create_button_effect()
            self._apply_parameters(self.presets[preset_name])
            self.preset.delete_preset_button.setEnabled(True)
            cache_file = os.path.join(
                "results", preset_name, f"{preset_name}_density_field.npz"
            )
            if os.path.exists(cache_file):
                try:
                    data = np.load(cache_file)
                    self._handle_optimization_results((data["xPhys"], data["u"]))
                    self.status_bar.showMessage(
                        f"Loaded cached result for {preset_name}.", 5000
                    )
                except Exception as e:
                    print(f"Failed to load cache: {e}")

        else:
            self.preset.delete_preset_button.setEnabled(False)

    def _apply_parameters(self, params: dict) -> None:
        """Sets all UI widgets to the values from a given parameter dictionary."""
        self._block_all_parameter_signals(True)

        self._apply_dimensions_param(params)
        self._apply_regions_param(params)
        self._apply_forces_param(params)
        self._apply_supports_param(params)
        self._apply_materials_param(params)
        self._apply_optimizer_param(params)
        self._apply_displacement_param(params)

        # Unblock signals
        self._block_all_parameter_signals(False)

        # Manually trigger a single update
        self._update_position_ranges()
        self.on_parameter_changed()
        self.status_bar.showMessage(
            f"Loaded preset: {self.preset.presets_combo.currentText()}", 3000
        )

    def _apply_dimensions_param(self, params: dict) -> None:
        """Applies dimension parameters from a preset dictionary."""
        Dimensions = params.get("Dimensions", {})
        self.dim_widget.nx.setValue(Dimensions.get("nelxyz", [1, 1, 1])[0])
        self.dim_widget.ny.setValue(Dimensions.get("nelxyz", [1, 1, 1])[1])
        self.dim_widget.nz.setValue(Dimensions.get("nelxyz", [1, 1, 1])[2])
        self.dim_widget.volfrac.setValue(Dimensions.get("volfrac", 0.3))
        self._update_position_ranges()

    def _apply_regions_param(self, params: dict) -> None:
        """Applies region parameters from a preset dictionary."""
        pr = params.get("Regions", {})
        num_regions_in_preset = len(pr.get("rshape", []))
        current_num_regions = len(self.regions_widget.inputs)

        while current_num_regions > num_regions_in_preset:
            self.regions_widget.remove_region(current_num_regions - 1, False)
            current_num_regions -= 1
        while current_num_regions < num_regions_in_preset:
            self.regions_widget.add_region(emit_signal=False)
            current_num_regions += 1

        for i in range(num_regions_in_preset):
            rw = self.regions_widget.inputs[i]
            rw["rshape"].blockSignals(True)
            rw["rstate"].blockSignals(True)
            rw["rshape"].setCurrentText(pr["rshape"][i])
            rw["rstate"].setCurrentText(pr["rstate"][i])
            rw["rradius"].setValue(pr["rradius"][i])
            rw["rx"].setValue(pr["rx"][i])
            rw["ry"].setValue(pr["ry"][i])
            rw["rz"].setValue(pr["rz"][i])
        self._connect_region_signals()

    def _apply_forces_param(self, params: dict) -> None:
        """Applies force parameters from a preset dictionary."""
        pf = params.get("Forces", {})

        # Input forces
        nb_input_forces = len(pf.get("fix", [])) if "fix" in pf else 0
        current_input_num = len(self.forces_widget.input_forces)
        while current_input_num > nb_input_forces:
            last_btn = self.forces_widget.input_forces[-1]["remove_btn"]
            try:
                last_btn.clicked.disconnect(self.on_parameter_changed)
            except TypeError:
                pass
            self.forces_widget.remove_force(current_input_num - 1, True, False)
            current_input_num -= 1
        while current_input_num < nb_input_forces:
            self.forces_widget.add_input_force(emit_signal=False)
            current_input_num += 1

        for i in range(nb_input_forces):
            fw = self.forces_widget.input_forces[i]
            fw["fix"].setValue(pf["fix"][i])
            fw["fiy"].setValue(pf["fiy"][i])
            fw["fiz"].setValue(pf["fiz"][i])
            fw["fidir"].setCurrentText(pf["fidir"][i])
            fw["finorm"].setValue(pf["finorm"][i])

        # Output forces
        nb_output_forces = len(pf.get("fox", [])) if "fox" in pf else 0
        current_output_num = len(self.forces_widget.output_forces)
        while current_output_num > nb_output_forces:
            last_btn = self.forces_widget.output_forces[-1]["remove_btn"]
            try:
                last_btn.clicked.disconnect(self.on_parameter_changed)
            except TypeError:
                pass
            self.forces_widget.remove_force(current_output_num - 1, False, False)
            current_output_num -= 1
        while current_output_num < nb_output_forces:
            self.forces_widget.add_output_force(emit_signal=False)
            current_output_num += 1

        for i in range(nb_output_forces):
            fw = self.forces_widget.output_forces[i]
            fw["fox"].setValue(pf["fox"][i])
            fw["foy"].setValue(pf["foy"][i])
            fw["foz"].setValue(pf["foz"][i])
            fw["fodir"].setCurrentText(pf["fodir"][i])
            fw["fonorm"].setValue(pf["fonorm"][i])

        self._connect_forces_signals()

    def _apply_supports_param(self, params: dict) -> None:
        """Applies support parameters from a preset dictionary."""
        ps = params.get("Supports", {})
        num_supports_in_preset = len(ps.get("sx", []))
        current_num = len(self.supports_widget.inputs)
        while current_num > num_supports_in_preset:
            last_btn = self.supports_widget.inputs[-1]["remove_btn"]
            try:
                last_btn.clicked.disconnect(self.on_parameter_changed)
            except TypeError:
                pass
            self.supports_widget.remove_support(current_num - 1, False)
            current_num -= 1
        while current_num < num_supports_in_preset:
            self.supports_widget.add_support(None, "XYZ", 0, False)
            current_num += 1
        for i in range(num_supports_in_preset):
            sw = self.supports_widget.inputs[i]
            sw["sx"].setValue(ps["sx"][i])
            sw["sy"].setValue(ps["sy"][i])
            sw["sz"].setValue(ps["sz"][i])
            sw["sdim"].setCurrentText(ps["sdim"][i])
            sw["sr"].setValue(ps["sr"][i])
        self._connect_support_signals()

    def _apply_materials_param(self, params: dict) -> None:
        """Applies material parameters from a preset dictionary."""
        pm = params.get("Materials", {})
        num_material_in_preset = len(pm.get("E", []))
        current_num = len(self.materials_widget.inputs)
        while current_num > num_material_in_preset:
            last_btn = self.materials_widget.inputs[-1]["remove_btn"]
            try:
                last_btn.clicked.disconnect()
            except (TypeError, RuntimeError):
                pass
            self.materials_widget.remove_material(current_num - 1, False)
            current_num -= 1
        while current_num < num_material_in_preset:
            self.materials_widget.add_material(emit_signal=False)
            current_num += 1
        for i in range(num_material_in_preset):
            mw = self.materials_widget.inputs[i]
            mw["E"].setValue(pm["E"][i])
            mw["nu"].setValue(pm["nu"][i])
            # percent and color can be optional, skip if missing
            percents = pm.get(
                "percent", [int(100 / num_material_in_preset)] * num_material_in_preset
            )
            mw["percent"].setValue(percents[i])
            color = pm.get("color", ["#000000"] * num_material_in_preset)
            mw["color"].set_color(color[i])
        self.materials_widget.mat_init_type.setCurrentIndex(pm.get("init_type", 0))
        self._connect_material_signals()

    def _apply_optimizer_param(self, params: dict) -> None:
        """Applies optimizer parameters from a preset dictionary."""
        po = params.get("Optimizer", {})
        self.optimizer_widget.opt_ft.setCurrentIndex(
            0
            if po.get("filter_type", "Sensitivity") == "Sensitivity"
            else 1
            if po.get("filter_type", "Sensitivity") == "Density"
            else 2
        )
        self.optimizer_widget.opt_fr.setValue(po.get("filter_radius_min", 1.3))
        self.optimizer_widget.opt_p.setValue(po.get("penal", 3.0))
        self.optimizer_widget.opt_eta.setValue(po.get("eta", 0.3))
        self.optimizer_widget.opt_max_change.setValue(po.get("max_change", 0.05))
        self.optimizer_widget.opt_n_it.setValue(po.get("n_it", 30))
        self.optimizer_widget.opt_solver.setCurrentText(po.get("solver", "Auto"))
        self.optimizer_widget.save_frames_cb.setChecked(po.get("save_frames", False))

    def _apply_displacement_param(self, params: dict) -> None:
        """Applies displacement parameters from a preset dictionary."""
        Displacement = params.get("Displacement", {})
        self.displacement_widget.mov_disp.setValue(Displacement.get("disp_factor", 1.0))
        self.displacement_widget.mov_iter.setValue(
            Displacement.get("disp_iterations", 1)
        )
        self.displacement_widget.save_frames_cb.setChecked(
            Displacement.get("save_frames", False)
        )

    def _connect_forces_signals(self) -> None:
        """(Re)connects on_parameter_changed signals to all current forces widgets."""
        for force_group in self.forces_widget.input_forces:
            force_group["fix"].valueChanged.connect(self.on_parameter_changed)
            force_group["fiy"].valueChanged.connect(self.on_parameter_changed)
            force_group["fiz"].valueChanged.connect(self.on_parameter_changed)
            force_group["fidir"].currentIndexChanged.connect(self.on_parameter_changed)
            force_group["finorm"].valueChanged.connect(self.on_parameter_changed)
        for force_group in self.forces_widget.output_forces:
            force_group["fox"].valueChanged.connect(self.on_parameter_changed)
            force_group["foy"].valueChanged.connect(self.on_parameter_changed)
            force_group["foz"].valueChanged.connect(self.on_parameter_changed)
            force_group["fodir"].currentIndexChanged.connect(self.on_parameter_changed)
            force_group["fonorm"].valueChanged.connect(self.on_parameter_changed)

    def _connect_support_signals(self) -> None:
        """(Re)connects on_parameter_changed signals to all current support widgets."""
        for support_group in self.supports_widget.inputs:
            support_group["sx"].valueChanged.connect(self.on_parameter_changed)
            support_group["sy"].valueChanged.connect(self.on_parameter_changed)
            support_group["sz"].valueChanged.connect(self.on_parameter_changed)
            support_group["sdim"].currentIndexChanged.connect(self.on_parameter_changed)
            support_group["sr"].valueChanged.connect(self.on_parameter_changed)
            # there is a special "nbSupportsChanged" signal for the remove button

    def _connect_region_signals(self) -> None:
        """(Re)connects on_parameter_changed signals to all current region widgets."""
        for region_group in self.regions_widget.inputs:
            region_group["rshape"].currentIndexChanged.connect(
                self.on_parameter_changed
            )
            region_group["rstate"].currentIndexChanged.connect(
                self.on_parameter_changed
            )
            region_group["rradius"].valueChanged.connect(self.on_parameter_changed)
            region_group["rx"].valueChanged.connect(self.on_parameter_changed)
            region_group["ry"].valueChanged.connect(self.on_parameter_changed)
            region_group["rz"].valueChanged.connect(self.on_parameter_changed)
            # there is a special "nbRegionsChanged" signal for the remove button

    ############
    # BINARIZE #
    ############

    def _on_binarize_clicked(self) -> None:
        """Applies a 0-or-1 threshold to the displayed result."""
        if self.xPhys is None:
            return

        # Define a threshold. Values below this are considered "white".
        threshold = 0.5

        # Use NumPy's fast vectorized 'where' function
        self.xPhys = np.where(self.xPhys > threshold, 1.0, 0.0)

        self.replot()
        self.status_bar.showMessage("View binarized.", 3000)

        self.footer.binarize_button.setEnabled(False)

        # Validate density after binarization (> 1%)
        mean_density: float = (
            np.mean(self.xPhys.sum(axis=0))
            if self.xPhys.ndim == 2
            else np.mean(self.xPhys)
        )
        if mean_density < 0.01:
            self.xPhys_valid = False
            self.footer.save_button.setEnabled(False)
            self.analysis_widget.run_analysis_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.sections["Displacement"].visibility_button.setEnabled(False)
        else:
            self.xPhys_valid = True

    #########
    # THEME #
    #########

    def _set_theme(self, theme_name: str, initial_setup: bool = False) -> None:
        """
        Applies a theme stylesheet to the application GUI only.

        Parameters
        ----------
        theme_name : str
            Name of the theme to apply ("light" or "dark").
        initial_setup : bool
            Whether this is the initial setup of the application.
        """
        stylesheet = (
            LIGHT_THEME_STYLESHEET if theme_name == "light" else DARK_THEME_STYLESHEET
        )
        QApplication.instance().setStyleSheet(stylesheet)
        self.current_theme = theme_name
        icons._set_theme(theme_name)

        # On the very first run, we need to style the plot once.
        # Afterwards, the plot theme is never touched again.
        if initial_setup:
            self._style_plot_default()

    def _toggle_theme(self) -> None:
        """Switches between light and dark themes."""
        if self.current_theme == "light":
            self._set_theme("dark")
        else:
            self._set_theme("light")
        self._update_ui_icons()

    def _update_ui_icons(self) -> None:
        """
        Resets all icons in the UI to force them to be re-fetched from the
        now theme-aware IconProvider.
        """
        # Update the theme button itself
        if self.current_theme == "dark":
            self.header.theme_button.setIcon(icons._get("sun"))
            self.header.theme_button.setToolTip("Switch to light theme")
        else:
            self.header.theme_button.setIcon(icons._get("moon"))
            self.header.theme_button.setToolTip("Switch to dark theme")

        # Update header icons
        self.header.info_button.setIcon(icons._get("info"))
        self.header.help_button.setIcon(icons._get("help"))
        # Update presets icons
        self.preset.save_preset_button.setIcon(icons._get("save"))
        self.preset.delete_preset_button.setIcon(icons._get("delete"))
        # Update dimensions icons
        self.dim_widget.scale_button.setIcon(icons._get("scale"))
        # Update displacement icons
        self.displacement_widget.run_disp_button.setIcon(icons._get("move"))
        self.displacement_widget.stop_disp_button.setIcon(icons._get("stop"))
        self.displacement_widget.reset_disp_button.setIcon(icons._get("reset"))
        # Update footer icons
        self.footer.create_button.setIcon(icons._get("create"))
        self.footer.stop_button.setIcon(icons._get("stop"))
        self.footer.save_button.setIcon(icons._get("save"))

        # Update dynamic icons (like the visibility and collapsible arrows)
        for section in self.sections.values():
            section.update_all_icons()

    def _open_github_link(self) -> None:
        """Opens the specified URL in the user's default web browser."""
        url = QUrl("https://github.com/ninja7v/TopoptComec")
        QDesktopServices.openUrl(url)

    def _open_wiki_link(self) -> None:
        """Opens the specified URL in the user's default web browser."""
        url = QUrl("https://github.com/ninja7v/TopoptComec/wiki/TopoptComec-wiki")
        QDesktopServices.openUrl(url)

    def _open_issue_link(self) -> None:
        """Opens the specified URL in the user's default web browser."""
        url = QUrl("https://github.com/ninja7v/TopoptComec/issues")
        QDesktopServices.openUrl(url)

    def closeEvent(self, event: QCloseEvent) -> None:
        """Close figure when the app is closed"""
        # Needed for the tests
        if self.figure:
            plt.close(self.figure)
        super().closeEvent(event)
