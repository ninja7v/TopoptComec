# tests/conftest.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Fixture for the tests.

import os
import warnings

import pytest

# You can also move your qt_app fixture here to make it available to all test files
from PySide6.QtWidgets import QApplication


@pytest.fixture(autouse=True)
def _suppress_vtk_numpy_deprecation():
    """Suppress vtk's NumPy 2.5 deprecation warning for the duration of each test.

    vtk 9.x sets array shapes in-place (``arr.shape = shape``) which is
    deprecated in NumPy 2.5.  The warning fires inside vtkmodules and
    cannot be fixed from our code.  When CI runs ``pytest -W error`` this
    would turn into a test failure, so we wrap every test in a
    ``catch_warnings`` block that ignores this specific third-party
    deprecation.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Setting the shape on a NumPy array has been deprecated.*",
            category=DeprecationWarning,
        )
        yield


@pytest.fixture(autouse=True)
def _isolated_user_presets(tmp_path, monkeypatch):
    """Keep tests from reading or writing ~/.topoptcomec/presets.json."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))  # Windows


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """Fixture to create a QApplication instance for the test session."""
    # Force Qt to use offscreen platform to avoid crashes in CI
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
    # Unset DISPLAY so VTK falls back to EGL headless rendering instead
    # of trying to use an X server that the offscreen Qt platform can't
    # provide a valid window handle for.
    os.environ.pop("DISPLAY", None)

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
