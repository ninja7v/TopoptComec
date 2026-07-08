# topoptcomec/ui/resource_path_finder.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Resource path finder.

from __future__ import annotations
import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """
    Get the absolute path to a resource, working for both development and PyInstaller.

    Parameters
    ----------
    relative_path : str
        The relative path to the resource (e.g., "icons", "assets/logo.png").

    Returns
    -------
    Path
        The absolute path to the resource.
    """
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller frozen exe
        base_path: Path = Path(sys._MEIPASS)
    else:
        # Normal run: resources (icons/, presets.json) ship inside the
        # topoptcomec package, one level above this gui/ subpackage.
        base_path = Path(__file__).resolve().parent.parent

    return base_path / relative_path
