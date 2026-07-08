# topoptcomec/ui/icons.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Icon management.

from __future__ import annotations
from pathlib import Path
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QStyle
from .resource_path_finder import resource_path


class IconProvider:
    """
    Provides icons for the UI with theme support.

    Tries to load custom themed icons, falls back to Qt standard icons.
    """

    def __init__(self) -> None:
        """
        Initialize the icon provider.

        Sets the default style and light theme.
        """
        self.style: QStyle = QApplication.style()
        self.theme: str = "light"

    def _set_theme(self, theme_name: str) -> None:
        """
        Sets the current theme ('light' or 'dark'). Called by the MainWindow.

        Parameters
        ----------
        theme_name : str
            The theme name to set ('light' or 'dark').
        """
        self.theme = theme_name

    def _get(self, icon_name: str) -> QIcon:
        """
        Retrieves a QIcon by its name. Tries to load a custom icon from assets first,
        falls back to standard Qt icons if not found.

        Parameters
        ----------
        icon_name : str
            The name of the icon to retrieve.

        Returns
        -------
        QIcon
            The requested icon, or an empty QIcon if not found.
        """
        # 1. Try to find a themed icon file
        extensions: list[str] = [
            "svg",
            "png",
            "jpg",
        ]  # Try .svg first, then .png, then .jpg
        icon_dir: Path = resource_path("icons")
        for ext in extensions:
            themed_path: Path = icon_dir / f"{icon_name}_{self.theme}.{ext}"
            if themed_path.is_file():
                icon: QIcon = QIcon(str(themed_path))
                if not icon.isNull():
                    return icon

        # 2. If not found, try to find a generic (non-themed) icon file
        for ext in extensions:
            generic_path: Path = icon_dir / f"{icon_name}.{ext}"
            if generic_path.is_file():
                icon = QIcon(str(generic_path))
                if not icon.isNull():
                    return icon

        # 3. If no file is found, fall back to built-in Qt icons
        if self.style is None:
            # Ensure the style is initialized, especially for tests
            self.style = QApplication.instance().style()

        icon_map: dict[str, QStyle.StandardPixmap] = {
            "save": QStyle.StandardPixmap.SP_DialogSaveButton,
            "delete": QStyle.StandardPixmap.SP_TrashIcon,
            "eye_open": QStyle.StandardPixmap.SP_DialogYesButton,
            "eye_closed": QStyle.StandardPixmap.SP_DialogNoButton,
            "arrow_right": QStyle.StandardPixmap.SP_TitleBarShadeButton,
            "arrow_down": QStyle.StandardPixmap.SP_TitleBarUnshadeButton,
            "create": QStyle.StandardPixmap.SP_MediaPlay,
            "folder": QStyle.StandardPixmap.SP_DirOpenIcon,
            "color": QStyle.StandardPixmap.SP_CustomBase,
            "window": QStyle.StandardPixmap.SP_ComputerIcon,
            "sun": QStyle.StandardPixmap.SP_TitleBarMaxButton,
            "moon": QStyle.StandardPixmap.SP_TitleBarMaxButton,
            "info": QStyle.StandardPixmap.SP_MessageBoxInformation,
            "help": QStyle.StandardPixmap.SP_MessageBoxQuestion,
            "binarize": QStyle.StandardPixmap.SP_DialogApplyButton,
            "stop": QStyle.StandardPixmap.SP_MediaStop,
            "move": QStyle.StandardPixmap.SP_ArrowRight,
            "reset": QStyle.StandardPixmap.SP_BrowserReload,
            "scale": QStyle.StandardPixmap.SP_ArrowRight,
        }
        pixmap: QStyle.StandardPixmap | None = icon_map.get(icon_name)
        if pixmap:
            return self.style.standardIcon(pixmap)

        print(f"Warning: Icon '{icon_name}' not found as custom or built-in.")
        return QIcon()  # Return empty icon if not found


# Global instance for easy access
icons: IconProvider = IconProvider()
