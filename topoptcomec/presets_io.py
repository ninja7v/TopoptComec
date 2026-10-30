# topoptcomec/presets_io.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Locating, layering and saving presets for both the CLI and the GUI.
#
# Presets come from up to three layers, merged in increasing precedence:
#
# 1. the built-in presets: ``./topoptcomec/presets.json`` when running from a
#    source checkout, otherwise the presets shipped inside the package;
# 2. the per-user file ``~/.topoptcomec/presets.json`` (presets saved in the
#    GUI);
# 3. ``./presets.json`` in the current working directory.
#
# An explicit file (CLI ``--presets``) replaces all layers. Only the per-user
# layer is ever written, so the application never modifies the source
# checkout (which would block the git updater) and new built-in presets of a
# release are always visible.

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


def packaged_presets_path() -> Path:
    """Path of the read-only default presets shipped inside the package."""
    return Path(__file__).resolve().parent / "presets.json"


def user_presets_path() -> Path:
    """Per-user writable presets file."""
    return Path.home() / ".topoptcomec" / "presets.json"


def local_presets_path() -> Path:
    """Project presets file in the current working directory."""
    return Path.cwd() / "presets.json"


def builtin_presets_path() -> Path:
    """Built-in presets: the checkout copy when running from source."""
    checkout = Path.cwd() / "topoptcomec" / "presets.json"
    return checkout if checkout.is_file() else packaged_presets_path()


def resolve_presets_file(explicit: str | None = None, writable: bool = False) -> Path:
    """
    Return a single presets file path.

    Parameters
    ----------
    explicit : str or None
        Path given by the user (e.g. via ``--presets``). Returned unchanged.
    writable : bool
        When True, return the per-user file (the only file that is written);
        its directory is created if needed.

    Returns
    -------
    Path
        The explicit path, the per-user file (``writable``), or the file with
        the highest precedence that exists (``./presets.json``, the per-user
        file, then the built-in presets).
    """
    if explicit is not None:
        return Path(explicit)
    if writable:
        user_path = user_presets_path()
        user_path.parent.mkdir(parents=True, exist_ok=True)
        return user_path
    for candidate in (local_presets_path(), user_presets_path()):
        if candidate.is_file():
            return candidate
    return builtin_presets_path()


def _read(path: Path) -> dict:
    """Read one presets file; a missing file is an empty layer."""
    if not path.is_file():
        return {}
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} does not contain a JSON object of presets.")
    return data


@dataclass
class PresetLayers:
    """Presets split by origin.

    Attributes
    ----------
    builtin : dict
        Presets that are not owned by the user (built-in and ``./presets.json``).
    user : dict
        Presets stored in the per-user file.
    merged : dict
        What the application shows: ``builtin`` overridden by ``user``, then
        by ``./presets.json``.
    """

    builtin: dict = field(default_factory=dict)
    user: dict = field(default_factory=dict)
    merged: dict = field(default_factory=dict)


def load_preset_layers(explicit: str | None = None) -> PresetLayers:
    """
    Load and merge the preset layers.

    Parameters
    ----------
    explicit : str or None
        Explicit presets file; when given, it is the only layer.

    Returns
    -------
    PresetLayers
        The individual layers and their merge.

    Raises
    ------
    json.JSONDecodeError
        If a presets file is not valid JSON.
    TypeError
        If a presets file does not contain a JSON object.
    FileNotFoundError
        If ``explicit`` does not exist.
    """
    if explicit is not None:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        data = _read(path)
        return PresetLayers(builtin=dict(data), user={}, merged=dict(data))

    builtin = _read(builtin_presets_path())
    user = _read(user_presets_path())
    local = _read(local_presets_path())
    merged = {**builtin, **user, **local}
    not_owned = {**builtin, **local}
    return PresetLayers(builtin=not_owned, user=user, merged=merged)


def load_presets(explicit: str | None = None) -> dict:
    """Merged presets (see :func:`load_preset_layers`)."""
    return load_preset_layers(explicit).merged


def save_user_presets(presets: dict, builtin: dict) -> Path:
    """
    Write the presets owned by the user to the per-user file.

    A preset is owned by the user when it is not a built-in preset or when it
    differs from the built-in preset of the same name.

    Parameters
    ----------
    presets : dict
        All presets currently shown by the application.
    builtin : dict
        The presets that are not owned by the user.

    Returns
    -------
    Path
        The per-user presets file.
    """
    owned = {
        name: value
        for name, value in presets.items()
        if name not in builtin or builtin[name] != value
    }
    path = resolve_presets_file(writable=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(owned, handle, indent=4)
    return path
