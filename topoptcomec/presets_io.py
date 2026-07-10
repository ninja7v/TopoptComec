# topoptcomec/presets_io.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Locating and seeding the presets file for both the CLI and the GUI.

from __future__ import annotations
import shutil
from pathlib import Path


def packaged_presets_path() -> Path:
    """Path of the read-only default presets shipped inside the package."""
    return Path(__file__).resolve().parent / "presets.json"


def user_presets_path() -> Path:
    """Per-user writable presets file (seeded from the packaged defaults)."""
    return Path.home() / ".topoptcomec" / "presets.json"


def resolve_presets_file(explicit: str | None = None, writable: bool = False) -> Path:
    """
    Locate the presets file to use.

    Resolution order:

    1. An explicitly provided path (used as-is).
    2. ``presets.json`` in the current working directory (developer workflow).
    3. The per-user presets file, seeded from the packaged defaults when
       ``writable`` is requested; otherwise the packaged defaults directly.

    Parameters
    ----------
    explicit : str or None
        Path given by the user (e.g. via ``--presets``). Returned unchanged.
    writable : bool
        Whether the caller intends to save presets. When True, a writable
        per-user copy is created if no local file exists.

    Returns
    -------
    Path
        Path to the presets file (may not exist if ``explicit`` is wrong).
    """
    if explicit is not None:
        return Path(explicit)

    cwd_presets = Path.cwd() / "topoptcomec" / "presets.json"
    if cwd_presets.is_file():
        return cwd_presets

    user_path = user_presets_path()
    if user_path.is_file():
        return user_path

    packaged = packaged_presets_path()
    if writable:
        user_path.parent.mkdir(parents=True, exist_ok=True)
        if packaged.is_file():
            shutil.copyfile(packaged, user_path)
        else:
            user_path.write_text("{}")
        return user_path
    return packaged
