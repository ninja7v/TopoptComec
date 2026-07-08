# topoptcomec/core/preset_format.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Adapter between the legacy preset/GUI dictionary format and the typed
# problem model (app.core.model). All glyph parsing ("←", "↑", "◯", "-")
# is quarantined here: no other core module may interpret UI symbols.

from __future__ import annotations

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
                x=int(fx[i]),
                y=int(fy[i]),
                z=int(fz[i]) if is_3d else 0,
                axis=axis,
                sign=sign,
                magnitude=float(fnorm[i]),
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


def parse_regions(regions: dict | None) -> list[Region]:
    """Convert the legacy Regions dictionary to a list of Regions."""
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
                solid=rstate[i] != "Void",
            )
        )
    return result
