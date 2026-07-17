# tests/test_main.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for application startup configuration.

from topoptcomec.__main__ import _configure_qt_platform


def test_configure_qt_platform_uses_xcb_on_wayland():
    """VTK's X11 render window must receive an XWayland window ID."""
    environment = {"WAYLAND_DISPLAY": "wayland-0", "DISPLAY": ":0"}

    _configure_qt_platform(environment)

    assert environment["QT_QPA_PLATFORM"] == "xcb"


def test_configure_qt_platform_preserves_offscreen():
    """Explicit non-Wayland platforms remain untouched."""
    environment = {
        "WAYLAND_DISPLAY": "wayland-0",
        "DISPLAY": ":0",
        "QT_QPA_PLATFORM": "offscreen",
    }

    _configure_qt_platform(environment)

    assert environment["QT_QPA_PLATFORM"] == "offscreen"
