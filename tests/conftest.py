# tests/conftest.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Fixture for the tests.

import os

import matplotlib
import pytest

# Use a non-interactive backend for tests to prevent warnings and errors
matplotlib.use("Agg")

# You can also move your qt_app fixture here to make it available to all test files
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="session")
def qt_app() -> QApplication:
    """Fixture to create a QApplication instance for the test session."""
    # Force Qt to use offscreen platform to avoid crashes in CI
    if "QT_QPA_PLATFORM" not in os.environ:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app
