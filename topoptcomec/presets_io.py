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
import math
from dataclasses import dataclass, field
from pathlib import Path

#: Direction glyphs accepted in force direction strings (``"X:→"``, ...).
DIRECTION_GLYPHS = frozenset({"\u2190", "\u2192", "\u2191", "\u2193", "<", ">"})

#: Region shape glyphs accepted in presets (``"□"`` and ``"◯"``).
REGION_SHAPES = frozenset({"\u25a1", "\u25ef", "-"})


def validate_preset_name(name: object) -> str | None:
    """
    Validate a preset name that will be used as a filesystem path component.

    Parameters
    ----------
    name : object
        Preset name as stored in a presets file or typed in the GUI.

    Returns
    -------
    str or None
        An error message when the name is unsafe, otherwise ``None``.
    """
    if not isinstance(name, str) or not name.strip():
        return "Preset names must be non-empty strings."
    if name in (".", ".."):
        return f"Preset name {name!r} is not allowed."
    if "\x00" in name:
        return "Preset names must not contain NUL characters."
    if "/" in name or "\\" in name or Path(name).name != name:
        return f"Preset name {name!r} must not contain path separators."
    if Path(name).is_absolute():
        return f"Preset name {name!r} must not be an absolute path."
    return None


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _finite(value: object) -> bool:
    return _is_number(value) and math.isfinite(float(value))


def _numbers(values: object) -> bool:
    return isinstance(values, (list, tuple)) and all(_finite(value) for value in values)


def _strings(values: object) -> bool:
    return isinstance(values, (list, tuple)) and all(
        isinstance(value, str) for value in values
    )


def _check_section(
    params: dict, key: str, *, required: bool = False
) -> tuple[dict | None, str | None]:
    """Return one preset section, reporting a wrong container type."""
    if key not in params:
        if required:
            return None, f"Missing required section {key!r}."
        return None, None
    value = params[key]
    if not isinstance(value, dict):
        return None, f"Section {key!r} must be a JSON object."
    return value, None


def check_directions(directions: list[str], label: str) -> str | None:
    """Validate direction strings such as ``"X:→"`` or ``"XYZ"``."""
    for index, direction in enumerate(directions):
        if direction == "-":
            continue
        if not any(axis in direction for axis in ("X", "Y", "Z")):
            return (
                f"{label} {index + 1} has unknown direction {direction!r} "
                "(expected X, Y or Z)."
            )
        if ":" in direction:
            glyph = direction.split(":", 1)[1]
            if glyph not in DIRECTION_GLYPHS:
                return f"{label} {index + 1} has unknown direction glyph {glyph!r}."
    return None


def validate_preset_structure(params: object) -> str | None:
    """
    Validate the structure of a preset before it reaches the solvers.

    The check is intentionally about types and required keys only: numerical
    and physical constraints stay in :class:`topoptcomec.parameter_check.ParameterCheck`.
    It guarantees that malformed JSON cannot raise ``KeyError``/``TypeError``
    later in the CLI or the GUI.

    Parameters
    ----------
    params : object
        Preset value as loaded from JSON.

    Returns
    -------
    str or None
        An error message when the preset is malformed, otherwise ``None``.
    """
    if not isinstance(params, dict) or not params:
        return "A preset must be a non-empty JSON object."

    dimensions, error = _check_section(params, "Dimensions", required=True)
    if error:
        return error
    nelxyz = dimensions.get("nelxyz")
    if not isinstance(nelxyz, (list, tuple)) or len(nelxyz) != 3:
        return "Dimensions.nelxyz must be a list of three integers [nx, ny, nz]."
    if not all(_is_int(value) for value in nelxyz):
        return "Dimensions.nelxyz entries must be integers."
    if not _finite(dimensions.get("volfrac")):
        return "Dimensions.volfrac must be a finite number."

    forces, error = _check_section(params, "Forces", required=True)
    if error:
        return error
    if "fidir" not in forces:
        return "Forces.fidir is required."
    for key in ("fidir", "fodir"):
        if key in forces and not _strings(forces[key]):
            return f"Forces.{key} must be a list of direction strings."
    for key in ("fidir", "fodir"):
        if key in forces:
            error = check_directions(list(forces[key]), f"Forces.{key}")
            if error:
                return error
    for prefix in ("fi", "fo"):
        for key in (f"{prefix}x", f"{prefix}y", f"{prefix}z", f"{prefix}norm"):
            if key in forces and not _numbers(forces[key]):
                return f"Forces.{key} must be a list of numbers."

    supports, error = _check_section(params, "Supports")
    if error:
        return error
    if supports is not None:
        if "sdim" in supports and not _strings(supports["sdim"]):
            return "Supports.sdim must be a list of direction strings."
        for key in ("sx", "sy", "sz", "sr"):
            if key in supports and not _numbers(supports[key]):
                return f"Supports.{key} must be a list of numbers."

    regions, error = _check_section(params, "Regions")
    if error:
        return error
    if regions is not None:
        rshape = regions.get("rshape", [])
        if not _strings(rshape):
            return "Regions.rshape must be a list of shape glyphs."
        unknown = [shape for shape in rshape if shape not in REGION_SHAPES]
        if unknown:
            return f"Regions.rshape contains unknown shape {unknown[0]!r}."
        for key in ("rx", "ry", "rz", "rradius"):
            if key in regions and not _numbers(regions[key]):
                return f"Regions.{key} must be a list of numbers."
        if "rstate" in regions and not _strings(regions["rstate"]):
            return "Regions.rstate must be a list of material names."

    materials, error = _check_section(params, "Materials", required=True)
    if error:
        return error
    if not _numbers(materials.get("E")) or not _numbers(materials.get("nu")):
        return "Materials.E and Materials.nu must be lists of numbers."
    if len(materials["E"]) != len(materials["nu"]):
        return "Materials.E and Materials.nu must have the same length."
    if "percent" in materials and not _numbers(materials["percent"]):
        return "Materials.percent must be a list of numbers."
    init_type = materials.get("init_type", 0)
    if not _is_int(init_type) or init_type not in (0, 1, 2, 3):
        return "Materials.init_type must be 0, 1, 2 or 3."

    optimizer, error = _check_section(params, "Optimizer", required=True)
    if error:
        return error
    if optimizer is not None:
        for key in (
            "filter_radius_min",
            "eta",
            "penal",
            "max_change",
            "n_it",
        ):
            if key in optimizer and not _finite(optimizer[key]):
                return f"Optimizer.{key} must be a finite number."

    displacement, error = _check_section(params, "Displacement")
    if error:
        return error
    if displacement is not None:
        for key in ("disp_factor", "disp_iterations"):
            if key in displacement and not _finite(displacement[key]):
                return f"Displacement.{key} must be a finite number."

    return None


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
