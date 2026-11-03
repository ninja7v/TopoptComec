# topoptcomec/preset_normalization.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Normalize preset dictionaries before comparing them or hashing them for the
# result cache. The GUI and the CLI must agree on this normalization, otherwise
# a result produced by one cannot be reused by the other.

from __future__ import annotations

import copy


def normalized_params(params: dict) -> dict:
    """Return a normalized deep copy of ``params``.

    Colors are removed because they only affect the display, never the
    optimized field.
    """
    normalized = copy.deepcopy(params)
    normalize_params(normalized)
    materials = normalized.get("Materials")
    if isinstance(materials, dict):
        materials.pop("color", None)
    return normalized


def normalize_params(p: dict) -> None:
    """
    Normalize parameters for comparison by removing irrelevant keys.

    Parameters
    ----------
    p : dict
        Parameters dictionary (modified in place).
    """
    # Tolerate malformed presets: this runs while comparing the current
    # parameters against every known preset, so a bad entry must not raise.
    pd = p.get("Dimensions")
    is_2d: bool = False
    if isinstance(pd, dict) and isinstance(pd.get("nelxyz"), (list, tuple)):
        nelxyz = pd["nelxyz"]
        is_2d = len(nelxyz) < 3 or nelxyz[2] == 0.0
        if is_2d:
            pd["nelxyz"] = list(nelxyz)[:2]

    _normalize_regions(p, is_2d)
    _normalize_supports(p, is_2d)
    _normalize_forces(p, is_2d)
    _normalize_materials(p)
    _normalize_optimizer(p)

    # Remove irrelevant keys (post-processing only, or optimizer-only state)
    p.pop("Displacement", None)
    p.pop("current_xPhys", None)


def _normalize_optimizer(p: dict) -> None:
    """Drop UI-only optimizer keys."""
    po = p.get("Optimizer")
    if isinstance(po, dict):
        po.pop("save_frames", None)  # UI-only, not stored in presets


def _normalize_regions(p: dict, is_2d: bool) -> None:
    """Drop inactive regions and 2D z coordinates."""
    pr = p.get("Regions")
    if isinstance(pr, dict):
        if "rshape" in pr:
            rx = pr.get("rx", [])
            rz = pr.get("rz", [])
            if is_2d or not rz:
                rz = [0] * len(rx)
            zipped = zip(
                pr.get("rshape", []),
                pr.get("rstate", []),
                pr.get("rradius", []),
                rx,
                pr.get("ry", []),
                rz,
            )
            active = [r for r in zipped if r[0] != "-"]
            if active:
                (
                    pr["rshape"],
                    pr["rstate"],
                    pr["rradius"],
                    pr["rx"],
                    pr["ry"],
                    pr["rz"],
                ) = map(list, zip(*active))
            else:
                p.pop("Regions", None)
                return

            if is_2d and "rz" in pr:
                pr.pop("rz")
        else:
            p.pop("Regions", None)


def _normalize_supports(p: dict, is_2d: bool) -> None:
    """Drop inactive supports and 2D z coordinates."""
    ps = p.get("Supports")
    if isinstance(ps, dict):
        if "sdim" in ps:
            sx = ps.get("sx", [])
            sz = ps.get("sz", [])
            if is_2d or not sz:
                sz = [0] * len(sx)
            zipped = zip(
                sx,
                ps.get("sy", []),
                sz,
                ps.get("sdim", []),
                ps.get("sr", []),
            )
            active = [s for s in zipped if s[3] != "-"]
            if active:
                (ps["sx"], ps["sy"], ps["sz"], ps["sdim"], ps["sr"]) = map(
                    list, zip(*active)
                )
            else:
                p.pop("Supports", None)
                return

            if is_2d and "sz" in ps:
                ps.pop("sz")
        else:
            p.pop("Supports", None)


def _normalize_forces(p: dict, is_2d: bool) -> None:
    """Drop inactive forces and 2D z coordinates."""
    pf = p.get("Forces")
    if not isinstance(pf, dict):
        return
    for prefix in ["fi", "fo"]:
        dir_key: str = f"{prefix}dir"
        if dir_key in pf:
            keys = [
                f"{prefix}x",
                f"{prefix}y",
                f"{prefix}z",
                f"{prefix}dir",
                f"{prefix}norm",
            ]
            fx = pf.get(f"{prefix}x", [])
            fz = pf.get(f"{prefix}z", [])
            if is_2d or not fz:
                fz = [0] * len(fx)
            vals: list = [
                fx,
                pf.get(f"{prefix}y", []),
                fz,
                pf.get(f"{prefix}dir", []),
                pf.get(f"{prefix}norm", []),
            ]
            zipped: tuple = zip(*vals)
            active: list = [item for item in zipped if item[3] != "-"]

            if active:
                unzipped: list = list(zip(*active))
                for i, k in enumerate(keys):
                    pf[k] = list(unzipped[i])
            else:
                for k in keys:
                    pf.pop(k, None)

            if is_2d and f"{prefix}z" in pf:
                pf.pop(f"{prefix}z")


def _normalize_materials(p: dict) -> None:
    """Normalize single-material defaults and hex color case."""
    pm = p.get("Materials")
    if isinstance(pm, dict):
        if isinstance(pm.get("E"), (list, tuple)) and len(pm["E"]) == 1:
            # If only one material, ignore the percentage
            pm["percent"] = [100]
            if "color" not in pm or pm["color"] == [""]:
                pm["color"] = ["#000000"]  # Default black color
            if "init_type" not in pm:
                pm["init_type"] = 0
        # Normalize hex color case: presets may store uppercase (#30EF54)
        # while the GUI color picker returns lowercase (#30ef54).
        if "color" in pm:
            pm["color"] = [c.lower() for c in pm["color"]]


__all__ = ["normalize_params", "normalized_params"]
