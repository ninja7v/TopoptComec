# topoptcomec/ui/workers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# QThread worker for running optimizers and displacements in the background.

"""
Workers for running topology optimization, displacement simulation,
and mechanism analysis in background threads.
"""

from __future__ import annotations

import copy
from abc import abstractmethod

import numpy as np
import numpy.typing as npt
from PySide6.QtCore import QThread, Signal

from topoptcomec.core import analyzers, displacements, optimizers
from topoptcomec.update import (
    UpdateCheck,
    UpdateError,
    UpdatePlan,
    UpdateResult,
    build_plan,
    check_for_update,
    install_update,
)

# Type aliases
FloatArray = npt.NDArray[np.float64]


class Worker:
    """Abstract base class for workers."""

    @abstractmethod
    def request_stop(self) -> None:
        """Request the worker to stop."""

    @abstractmethod
    def run(self) -> None:
        """Execute the worker's main task."""


class OptimizerWorker(QThread, Worker):
    """Runs the topology optimization in a separate thread."""

    progress = Signal(int, float, float)
    frameReady = Signal(object)
    #: Emitted with the (xPhys, u) result tuple. Named resultsReady to avoid
    #: shadowing the built-in QThread.finished signal.
    resultsReady = Signal(object)
    error = Signal(str)

    def __init__(self, params: dict) -> None:
        """Initialize the optimizer worker.

        Parameters
        ----------
        params : dict
            Optimization parameters dictionary.
        """
        super().__init__()
        self.params: dict = params
        self.stop_requested: bool = False

    def request_stop(self) -> None:
        """Request the worker to stop."""
        print("Stop request received by worker.")
        self.stop_requested = True

    def run(self) -> None:
        """Execute the optimization based on the provided parameters."""
        try:
            optimizer_params: dict = copy.deepcopy(self.params)
            # Remove unneeded parameters for the optimizer
            if "Displacement" in optimizer_params:
                optimizer_params.pop("Displacement", None)

            is_multimaterial: bool = (
                len(optimizer_params.get("Materials", {}).get("E", [1.0])) > 1
            )

            if "Materials" in optimizer_params:
                optimizer_params["Materials"].pop("color", None)
                if not is_multimaterial:
                    optimizer_params["Materials"].pop("percent", None)

            def _progress_callback(
                iteration: int, objective: float, change: float, xPhys_frame: FloatArray
            ) -> bool:
                self.progress.emit(iteration, objective, change)
                self.frameReady.emit(xPhys_frame)
                return self.stop_requested

            optimizer_params["progress_callback"] = _progress_callback

            print("Dispatching to optimizer...")
            result: FloatArray
            u: FloatArray
            result, u = optimizers.optimize(
                **optimizer_params, multimaterial=is_multimaterial
            )

            self.resultsReady.emit((result, u))  # Emit the tuple (xPhys, u)
        except Exception as e:  # noqa: BLE001
            import traceback

            error_msg: str = f"An error occurred during optimization:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class DisplacementWorker(QThread, Worker):
    """Runs the displacement simulation in a separate thread."""

    progress = Signal(int)
    frameReady = Signal(np.ndarray)
    linearResultReady = Signal(object)
    #: Emitted when the simulation completes or is stopped (message, success).
    #: Named simulationFinished to avoid shadowing QThread.finished.
    simulationFinished = Signal(str, bool)
    error = Signal(str)

    def __init__(self, params: dict, xPhys: FloatArray, u: FloatArray) -> None:
        """Initialize the displacement worker.

        Parameters
        ----------
        params : dict
            Simulation parameters.
        xPhys : FloatArray
            Density field from optimization.
        u : FloatArray
            Displacement vector from optimization.
        """
        super().__init__()
        self.params: dict = params
        self.xPhys: FloatArray = xPhys
        self.u: FloatArray = u
        self._stop_requested: bool = False

    def request_stop(self) -> None:
        """Request the worker to stop."""
        print("Stop request received by worker.")
        self._stop_requested = True

    def run(self) -> None:
        """Execute the displacement simulation based on provided parameters."""
        try:

            def _progress_callback(iteration: int) -> bool:
                self.progress.emit(iteration)
                return self._stop_requested

            # The function is a generator, yielding each frame
            for frame_data in displacements.run_iterative_displacement(
                self.params, self.xPhys, _progress_callback
            ):
                self.frameReady.emit(frame_data)
                if self._stop_requested:
                    print("Displacement stopped by user.")
                    break

            self.simulationFinished.emit("Displacement finished or stopped.", True)
        except Exception as e:  # noqa: BLE001
            import traceback

            error_msg: str = f"An error occurred during displacement analysis:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class UpdateCheckWorker(QThread):
    """Runs the GitHub update check off the GUI thread."""

    checked = Signal(object)
    failed = Signal(str)

    def __init__(self, parent=None) -> None:
        """Initialize the worker."""
        super().__init__(parent)

    def run(self) -> None:
        """Check for updates and emit the check result and installation plan."""
        try:
            check: UpdateCheck = check_for_update()
            self.checked.emit((check, build_plan(check)))
        except UpdateError as error:
            self.failed.emit(str(error))
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(f"Unexpected update error: {error}")


class UpdateInstallWorker(QThread):
    """Downloads and installs an update off the GUI thread."""

    progressChanged = Signal(int, str)
    installed = Signal(object)
    failed = Signal(str)

    def __init__(self, check: UpdateCheck, plan: UpdatePlan, parent=None) -> None:
        """Initialize the worker with the checked release and its plan."""
        super().__init__(parent)
        self.check = check
        self.plan = plan

    def run(self) -> None:
        """Install the update and emit progress, success, or failure."""
        try:
            result: UpdateResult = install_update(
                self.check, self.plan, progress=self.progressChanged.emit
            )
            self.installed.emit(result)
        except UpdateError as error:
            self.failed.emit(str(error))
        except Exception as error:  # noqa: BLE001
            import traceback

            traceback.print_exc()
            self.failed.emit(f"Unexpected update error: {error}")


class AnalysisWorker(QThread, Worker):
    """Runs the mechanism analysis in a separate thread."""

    progress = Signal(int)
    frameReady = Signal(np.ndarray)
    #: Emitted with the 4-tuple of analysis booleans. Named analysisFinished
    #: to avoid shadowing QThread.finished.
    analysisFinished = Signal(object)
    error = Signal(str)

    def __init__(self, params: dict, xPhys: FloatArray, u: FloatArray) -> None:
        """Initialize the analysis worker.

        Parameters
        ----------
        params : dict
            Analysis parameters.
        xPhys : FloatArray
            Density field from optimization.
        u : FloatArray
            Displacement vector from optimization.
        """
        super().__init__()
        self.params: dict = params
        self.xPhys: FloatArray = xPhys
        self.u: FloatArray = u
        self._stop_requested: bool = False

    def request_stop(self) -> None:
        """Request the worker to stop."""
        print("Stop request received by worker.")
        self._stop_requested = True

    def run(self) -> None:
        """Execute the mechanism analysis based on provided parameters."""
        try:

            def _progress_callback(iteration: int) -> bool:
                self.progress.emit(iteration)
                return self._stop_requested

            # Pass exactly what analyze() expects (the params dict may carry
            # optimizer-only keys such as 'current_xPhys').
            checkerboard: bool
            watertight: bool
            thresholded: bool
            efficient: bool
            checkerboard, watertight, thresholded, efficient = analyzers.analyze(
                xPhys=self.xPhys,
                u=self.u,
                Dimensions=self.params["Dimensions"],
                Forces=self.params["Forces"],
                progress_callback=_progress_callback,
            )
            self.analysisFinished.emit(
                (checkerboard, watertight, thresholded, efficient)
            )

        except Exception as e:  # noqa: BLE001
            import traceback

            error_msg: str = (
                f"An error occurred during analysis:\n{e}\n\n{traceback.format_exc()}"
            )
            self.error.emit(error_msg)
