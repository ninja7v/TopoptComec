# app/ui/workers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# QThread worker for running optimizers and displacements in the background.

"""
Workers for running topology optimization, displacement simulation,
and mechanism analysis in background threads.
"""

from __future__ import annotations
import numpy as np
import numpy.typing as npt
import copy
from PySide6.QtCore import QThread, Signal
from abc import abstractmethod
from app.core import analyzers, displacements, optimizers

# Type aliases
FloatArray = npt.NDArray[np.float64]


class Worker:
    """Abstract base class for workers."""

    @abstractmethod
    def request_stop(self) -> None:
        """Request the worker to stop."""
        pass

    @abstractmethod
    def run(self) -> None:
        """Execute the worker's main task."""
        pass


class OptimizerWorker(QThread, Worker):
    """Runs the topology optimization in a separate thread."""

    progress = Signal(int, float, float)
    frameReady = Signal(object)
    finished = Signal(np.ndarray)
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

            if is_multimaterial:
                print("Dispatching to multi-material optimizer...")
                result: FloatArray
                u: FloatArray
                result, u = optimizers.optimize_multimaterial(**optimizer_params)
            else:
                print("Dispatching to optimizer...")
                result, u = optimizers.optimize(**optimizer_params)

            self.finished.emit((result, u))  # Emit the tuple (xPhys, u)
        except Exception as e:
            import traceback

            error_msg: str = f"An error occurred during optimization:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class DisplacementWorker(QThread, Worker):
    """Runs the displacement simulation in a separate thread."""

    progress = Signal(int)
    frameReady = Signal(np.ndarray)
    linearResultReady = Signal(object)
    finished = Signal(str, bool)
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

            self.finished.emit("Displacement finished or stopped.", True)
        except Exception as e:
            import traceback

            error_msg: str = f"An error occurred during displacement analysis:\n{e}\n\n{traceback.format_exc()}"
            self.error.emit(error_msg)


class AnalysisWorker(QThread, Worker):
    """Runs the mechanism analysis in a separate thread."""

    progress = Signal(int)
    frameReady = Signal(np.ndarray)
    analysis_finished = Signal(object)
    finished = Signal(np.ndarray)
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
            analysis_params: dict = self.params.copy()
            analysis_params["xPhys"] = self.xPhys
            analysis_params["u"] = self.u

            # Remove unneeded parameters for the analysis
            if "Displacement" in analysis_params:
                analysis_params.pop("Displacement", None)
            if "Materials" in analysis_params:
                analysis_params.pop("Materials", None)

            if "Optimizer" in analysis_params:
                analysis_params.pop("Optimizer", None)
            if "Supports" in analysis_params:
                analysis_params.pop("Supports", None)
            if "Regions" in analysis_params:
                analysis_params.pop("Regions", None)

            def _progress_callback(iteration: int) -> bool:
                self.progress.emit(iteration)
                return self._stop_requested

            analysis_params["progress_callback"] = _progress_callback

            # The function is a generator, yielding each frame
            checkerboard: bool
            watertight: bool
            thresholded: bool
            efficient: bool
            checkerboard, watertight, thresholded, efficient = analyzers.analyze(
                **analysis_params
            )
            self.finished.emit((checkerboard, watertight, thresholded, efficient))

        except Exception as e:
            import traceback

            error_msg: str = (
                f"An error occurred during analysis:\n{e}\n\n{traceback.format_exc()}"
            )
            self.error.emit(error_msg)
