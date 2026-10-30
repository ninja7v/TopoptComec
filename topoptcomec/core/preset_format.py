# topoptcomec/core/preset_format.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Adapter between the legacy preset/GUI dictionary format and the typed
# problem model (app.core.model). All glyph parsing ("←", "↑", "◯", "-")
# is quarantined here: no other core module may interpret UI symbols.

from __future__ import annotations

from collections.abc import Sequence

from topoptcomec.core.model import AXIS_X, AXIS_Y, AXIS_Z, Load, Region, Support

#: Direction glyphs meaning "negative direction" per axis.
_NEGATIVE_GLYPHS = {"\u2190", "\u2191", ">"}  # ← (x), ↑ (y), > (z)

#: Marker for inactive entries in preset lists.
INACTIVE = "-"


def parse_loads(
    forces: dict,
    kx: str,
    ky: str,
    kz: str,
    kdir: str,
    knorm: str,
    is_3d: bool,
) -> list[Load]:
    """
    Convert one legacy force family (input or output) to a list of Loads.

    Parameters
    ----------
    forces : dict
        Legacy Forces dictionary.
    kx, ky, kz, kdir, knorm : str
        Keys for coordinates, direction glyph, and magnitude.
    is_3d : bool
        Whether z coordinates are meaningful.

    Returns
    -------
    list[Load]
        One entry per active force, in preset order.
    """
    fx = forces.get(kx, [])
    fy = forces.get(ky, [])
    fz = forces.get(kz, []) if is_3d else []
    fdir = forces.get(kdir, [])
    fnorm = forces.get(knorm, [])

    def _at(values: Sequence, index: int, default: float = 0.0) -> float:
        """Entry ``index`` of ``values``, or ``default`` when absent."""
        return values[index] if index < len(values) else default

    loads: list[Load] = []
    for i, direction in enumerate(fdir):
        if direction == INACTIVE:
            continue
        if "X" in direction:
            axis = AXIS_X
        elif "Y" in direction:
            axis = AXIS_Y
        elif is_3d and "Z" in direction:
            axis = AXIS_Z
        else:
            continue
        sign = -1 if any(g in direction for g in _NEGATIVE_GLYPHS) else 1
        loads.append(
            Load(
                x=int(_at(fx, i)),
                y=int(_at(fy, i)),
                z=int(_at(fz, i)) if is_3d else 0,
                axis=axis,
                sign=sign,
                magnitude=float(_at(fnorm, i, 1.0)),
            )
        )
    return loads


def parse_supports(supports: dict | None, is_3d: bool) -> list[Support]:
    """Convert the legacy Supports dictionary to a list of Supports."""
    if not supports:
        return []
    sx = supports.get("sx", [])
    sy = supports.get("sy", [])
    sz = supports.get("sz", [])
    sr = supports.get("sr", [])
    sdim = supports.get("sdim", [])

    result: list[Support] = []
    for i, dim in enumerate(sdim):
        if dim == INACTIVE:
            continue
        result.append(
            Support(
                x=int(sx[i]),
                y=int(sy[i]),
                z=int(sz[i]) if is_3d else 0,
                radius=float(sr[i]) if i < len(sr) else 0.0,
                fix_x="X" in dim,
                fix_y="Y" in dim,
                fix_z=is_3d and "Z" in dim,
            )
        )
    return result


def region_material_index(state: str) -> int | None:
    """
    0-based material index of a region state, or None when not a material.

    ``"Material N"`` maps to ``N - 1`` for any ``N >= 1``; ``"Void"`` and
    unknown strings map to None.
    """
    prefix = "Material "
    if not isinstance(state, str) or not state.startswith(prefix):
        return None
    try:
        number = int(state[len(prefix) :])
    except ValueError:
        return None
    return number - 1 if number >= 1 else None


def _region_material(states: Sequence, index: int) -> int:
    """Material index of region ``index`` (``-1`` for void or missing state)."""
    state = states[index] if index < len(states) else "Void"
    material = region_material_index(state)
    return -1 if material is None else material


def parse_regions(regions: dict | None) -> list[Region]:
    """Convert the Regions dictionary to a list of Regions."""
    if not regions:
        return []
    rshape = regions.get("rshape", [])
    rx = regions.get("rx", [])
    ry = regions.get("ry", [])
    rz = regions.get("rz", [])
    rradius = regions.get("rradius", [])
    rstate = regions.get("rstate", [])

    result: list[Region] = []
    for i, shape in enumerate(rshape):
        if shape == INACTIVE:
            continue
        result.append(
            Region(
                shape="sphere" if shape == "\u25ef" else "box",  # ◯
                x=float(rx[i]),
                y=float(ry[i]),
                z=float(rz[i]) if i < len(rz) else 0.0,
                radius=float(rradius[i]),
                material=_region_material(rstate, i),
            )
        )
    return result
