# topoptcomec/__main__.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Entry point for TopoptComec when invoked as a module or via console script.

from __future__ import annotations
import os
import sys
from collections.abc import MutableMapping
from pathlib import Path
from topoptcomec.gui.resource_path_finder import resource_path


def _configure_qt_platform(
    environ: MutableMapping[str, str] | None = None,
) -> None:
    """Use XWayland for VTK-backed Qt widgets on Linux Wayland sessions."""
    env = os.environ if environ is None else environ
    qt_platform = env.get("QT_QPA_PLATFORM", "")
    if (
        sys.platform.startswith("linux")
        and env.get("WAYLAND_DISPLAY")
        and env.get("DISPLAY")
        and (not qt_platform or qt_platform.startswith("wayland"))
    ):
        # VTK's XOpenGL render window interprets Wayland surface IDs as X11
        # window IDs, causing a fatal BadWindow error in QtInteractor.
        env["QT_QPA_PLATFORM"] = "xcb"


def main() -> None:
    """Initializes and runs the Qt application."""
    if len(sys.argv) > 1:
        # CLI mode
        from topoptcomec.cli.cli import run_cli

        run_cli()
    else:
        # GUI mode
        _configure_qt_platform()

        from PySide6.QtGui import QIcon
        from PySide6.QtSvg import QSvgRenderer  # noqa: F401
        from PySide6.QtWidgets import QApplication, QStyle

        app: QApplication = QApplication(sys.argv)

        icon_path: Path = resource_path("icons") / "window_icon.svg"
        if icon_path.exists():
            app_icon: QIcon = QIcon(str(icon_path))
            if not app_icon.isNull():
                app.setWindowIcon(app_icon)
        else:
            print(
                f"Warning: Window icon not found at {icon_path}. Using a built-in icon."
            )
            fallback_icon: QIcon = app.style().standardIcon(
                QStyle.StandardPixmap.SP_ComputerIcon
            )
            app.setWindowIcon(fallback_icon)

        # Now that the app exists, import the main window
        from topoptcomec.gui.main_window import MainWindow

        window: MainWindow = MainWindow()
        window.show()

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
