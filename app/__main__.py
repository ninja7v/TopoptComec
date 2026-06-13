# app/__main__.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Entry point for TopoptComec when invoked as a module or via console script.

from __future__ import annotations
import sys
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle
from PySide6.QtSvg import (  # noqa: F401
    QSvgRenderer,
)  # make sure SVG support is available
from app.ui.resource_path_finder import resource_path


def main() -> None:
    """Initializes and runs the Qt application."""
    if len(sys.argv) > 1:
        # CLI mode
        from app.cli import run_cli

        run_cli()
    else:
        # GUI mode
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
        from app.ui.main_window import MainWindow

        window: MainWindow = MainWindow()
        window.show()

        sys.exit(app.exec())


if __name__ == "__main__":
    main()
