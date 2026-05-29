# app/ui/resource_path_finder.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Resource path finder.

from __future__ import annotations
import sys
from pathlib import Path


def resource_path(relative_path: str) -> Path:
    """Get the absolute path to a resource, working for both development and PyInstaller."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        # PyInstaller frozen exe
        base_path: Path = Path(sys._MEIPASS)
    else:
        # Normal development run
        # Try to find the project root by looking for main.py or known file
        current: Path = Path(__file__).resolve()
        for parent in [
            current.parent,
            current.parent.parent,
            current.parent.parent.parent,
        ]:
            if (parent / "main.py").is_file() or (
                parent / "requirements.txt"
            ).is_file():
                base_path = parent
                break
        else:
            # Fallback: directory of this file
            base_path = current.parent

    return base_path / relative_path
