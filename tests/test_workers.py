# tests/test_workers.py
# MIT License - Copyright (c) 2025 Luc Prevost
# Tests for the workers.

import numpy as np

from topoptcomec.gui.workers import OptimizerWorker, DisplacementWorker, AnalysisWorker


def test_optimizer_worker(monkeypatch):
    recorded = {"progress": [], "frame": None, "finished": None, "error": None}

    def dummy_optimize(**kwargs):
        progress_cb = kwargs.get("progress_callback")
        # call progress callback once with a dummy frame
        progress_cb(1, 3.14, 0.01, np.array([[42]]))
        return np.array([0.5]), np.array([0.1])

    monkeypatch.setattr("topoptcomec.gui.workers.optimizers.optimize", dummy_optimize)

    worker = OptimizerWorker({"some_param": 1})

    worker.progress.connect(
        lambda it, obj, ch: recorded["progress"].append((it, obj, ch))
    )
    worker.frameReady.connect(lambda frame: recorded.update({"frame": frame}))
    worker.resultsReady.connect(lambda res: recorded.update({"finished": res}))
    worker.error.connect(lambda e: recorded.update({"error": e}))

    # Run synchronously in the test thread
    worker.run()

    assert recorded["progress"] == [(1, 3.14, 0.01)]
    assert isinstance(recorded["frame"], np.ndarray)
    assert recorded["frame"].shape == (1, 1) or recorded["frame"].size == 1
    np.testing.assert_array_equal(recorded["finished"][0], np.array([0.5]))
    np.testing.assert_array_equal(recorded["finished"][1], np.array([0.1]))
    assert isinstance(recorded["finished"], tuple)
    assert recorded["error"] is None


def test_displacement_worker(monkeypatch):
    frames = [np.array([1]), np.array([2])]

    def dummy_generator(params, xPhys, progress_cb):
        for i, f in enumerate(frames):
            stop = progress_cb(i)
            yield f
            if stop:
                break

    monkeypatch.setattr(
        "topoptcomec.gui.workers.displacements.run_iterative_displacement",
        dummy_generator,
    )

    recorded = {"progress": [], "frames": [], "finished": None, "error": None}

    worker = DisplacementWorker({"p": 1}, np.array([0.0]), np.array([0.0]))

    worker.progress.connect(lambda it: recorded["progress"].append(it))
    worker.frameReady.connect(lambda f: recorded["frames"].append(np.array(f)))
    worker.simulationFinished.connect(
        lambda msg, flag: recorded.update({"finished": (msg, flag)})
    )
    worker.error.connect(lambda e: recorded.update({"error": e}))

    # Run synchronously in the test thread
    worker.run()

    assert recorded["progress"] == [0, 1]
    assert len(recorded["frames"]) == 2
    np.testing.assert_array_equal(recorded["frames"][0], frames[0])
    np.testing.assert_array_equal(recorded["frames"][1], frames[1])
    assert recorded["finished"] is not None
    assert recorded["error"] is None


def test_optimizer_worker_error(monkeypatch):
    """Test that OptimizerWorker emits error signal on exception."""

    def failing_optimize(**kwargs):
        raise RuntimeError("Optimization exploded")

    monkeypatch.setattr("topoptcomec.gui.workers.optimizers.optimize", failing_optimize)

    recorded = {"error": None}
    worker = OptimizerWorker({"some_param": 1})
    worker.error.connect(lambda e: recorded.update({"error": e}))

    worker.run()

    assert recorded["error"] is not None
    assert "Optimization exploded" in recorded["error"]


def test_optimizer_worker_multimaterial(monkeypatch):
    """Test that OptimizerWorker dispatches to optimize with multimaterial=True."""
    recorded = {"called": None}

    def dummy_optimize(**kwargs):
        assert kwargs.get("multimaterial") is True
        recorded["called"] = "multi"
        cb = kwargs.get("progress_callback")
        cb(1, 1.0, 0.01, np.array([0.5]))
        return np.array([0.5]), np.array([0.1])

    monkeypatch.setattr(
        "topoptcomec.gui.workers.optimizers.optimize",
        dummy_optimize,
    )

    params = {
        "Materials": {
            "E": [1.0, 2.0],
            "nu": [0.3, 0.3],
            "percent": [50, 50],
            "color": ["#000", "#fff"],
        },
    }
    worker = OptimizerWorker(params)
    worker.resultsReady.connect(lambda res: None)
    worker.progress.connect(lambda *a: None)
    worker.frameReady.connect(lambda f: None)

    worker.run()

    assert recorded["called"] == "multi"


def test_optimizer_worker_request_stop():
    """Test that request_stop sets the flag."""
    worker = OptimizerWorker({"p": 1})
    assert worker.stop_requested is False
    worker.request_stop()
    assert worker.stop_requested is True


def test_displacement_worker_stop(monkeypatch):
    """Test that DisplacementWorker stops when stop is requested mid-iteration."""
    frames = [np.array([1]), np.array([2]), np.array([3])]

    def dummy_generator(params, xPhys, progress_cb):
        for i, f in enumerate(frames):
            progress_cb(i)
            yield f

    monkeypatch.setattr(
        "topoptcomec.gui.workers.displacements.run_iterative_displacement",
        dummy_generator,
    )

    recorded = {"frames": [], "finished": None}
    worker = DisplacementWorker({"p": 1}, np.array([0.0]), np.array([0.0]))
    worker.frameReady.connect(lambda f: recorded["frames"].append(np.array(f)))
    worker.simulationFinished.connect(
        lambda msg, flag: recorded.update({"finished": (msg, flag)})
    )
    worker.progress.connect(lambda it: None)
    worker.error.connect(lambda e: None)

    # Request stop before running — should stop after first frame
    worker.request_stop()
    worker.run()

    assert len(recorded["frames"]) == 1
    assert recorded["finished"] is not None


def test_displacement_worker_error(monkeypatch):
    """Test that DisplacementWorker emits error signal on exception."""

    def failing_generator(params, xPhys, progress_cb):
        raise RuntimeError("Displacement failed")
        yield  # pragma: no cover

    monkeypatch.setattr(
        "topoptcomec.gui.workers.displacements.run_iterative_displacement",
        failing_generator,
    )

    recorded = {"error": None}
    worker = DisplacementWorker({"p": 1}, np.array([0.0]), np.array([0.0]))
    worker.error.connect(lambda e: recorded.update({"error": e}))
    worker.progress.connect(lambda it: None)
    worker.frameReady.connect(lambda f: None)
    worker.simulationFinished.connect(lambda msg, flag: None)

    worker.run()

    assert recorded["error"] is not None
    assert "Displacement failed" in recorded["error"]


def test_analysis_worker(monkeypatch):
    """Test that AnalysisWorker runs analysis and emits results."""

    def dummy_analyze(**kwargs):
        cb = kwargs.get("progress_callback")
        if cb:
            cb(1)
        return True, False, True, False

    monkeypatch.setattr("topoptcomec.gui.workers.analyzers.analyze", dummy_analyze)

    recorded = {"finished": None, "error": None}
    params = {
        "Dimensions": {"nelxyz": [10, 10, 0]},
        "Forces": {
            "fidir": ["X:→"],
            "fix": [0],
            "fiy": [5],
            "fiz": [0],
            "finorm": [0.01],
        },
        "Displacement": {"disp_factor": 1.0},
        "Materials": {"E": [1.0]},
        "Supports": {"sdim": ["Y"]},
        "Optimizer": {"solver": "Direct"},
        "Regions": {},
        # Optimizer-only key that must not leak into analyze()
        "current_xPhys": np.ones(100) * 0.5,
    }
    xPhys = np.ones(100) * 0.5
    u = np.ones((242, 1)) * 0.01

    worker = AnalysisWorker(params, xPhys, u)
    worker.analysisFinished.connect(lambda res: recorded.update({"finished": res}))
    worker.error.connect(lambda e: recorded.update({"error": e}))
    worker.progress.connect(lambda it: None)
    worker.frameReady.connect(lambda f: None)

    worker.run()

    assert recorded["error"] is None
    assert recorded["finished"] is not None
    assert recorded["finished"][0] is True  # checkerboard
    assert recorded["finished"][1] is False  # watertight
    assert recorded["finished"][2] is True  # thresholded
    assert recorded["finished"][3] is False  # efficient


def test_analysis_worker_error(monkeypatch):
    """Test that AnalysisWorker emits error signal on exception."""

    def failing_analyze(**kwargs):
        raise RuntimeError("Analysis exploded")

    monkeypatch.setattr("topoptcomec.gui.workers.analyzers.analyze", failing_analyze)

    recorded = {"error": None}
    params = {
        "Dimensions": {"nelxyz": [10, 10, 0]},
        "Forces": {"fidir": ["X:→"]},
    }
    worker = AnalysisWorker(params, np.array([0.5]), np.array([0.1]))
    worker.error.connect(lambda e: recorded.update({"error": e}))
    worker.progress.connect(lambda it: None)
    worker.frameReady.connect(lambda f: None)
    worker.analysisFinished.connect(lambda res: None)

    worker.run()

    assert recorded["error"] is not None
    assert "Analysis exploded" in recorded["error"]


def test_analysis_worker_request_stop():
    """Test that AnalysisWorker.request_stop sets the flag."""
    worker = AnalysisWorker({"p": 1}, np.array([0.5]), np.array([0.1]))
    assert worker._stop_requested is False
    worker.request_stop()
    assert worker._stop_requested is True
