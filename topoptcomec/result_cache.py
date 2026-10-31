# topoptcomec/result_cache.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Result cache shared by the CLI and the GUI.
#
# A cached density field is only reused when it was produced with the exact
# same parameters (hash) and matches the current grid. Anything else is
# treated as stale and ignored, so the CLI and the GUI can never display a
# result that belongs to a different problem.

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import numpy.typing as npt

from topoptcomec.presets_io import validate_preset_name

# Type aliases
FloatArray = npt.NDArray[np.float64]

#: Keys that belong to a single run and must not invalidate a cache.
_VOLATILE_KEYS = ("current_xPhys", "progress_callback", "multimaterial")


def params_hash(params: dict) -> str:
    """
    Stable hash of a parameter dictionary, used to invalidate cached results.

    Parameters
    ----------
    params : dict
        Full parameter configuration dictionary.

    Returns
    -------
    str
        Hex digest identifying this exact parameter set.
    """
    payload = {k: v for k, v in params.items() if k not in _VOLATILE_KEYS}
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(base_dir: str | Path, preset_name: str) -> Path:
    """Path of the density-field cache of one preset.

    Raises
    ------
    ValueError
        If ``preset_name`` is not a safe single path component. Presets are
        shareable files, so a hostile name must never escape ``base_dir``.
    """
    error = validate_preset_name(preset_name)
    if error:
        raise ValueError(f"Invalid preset name {preset_name!r}: {error}")
    return Path(base_dir) / preset_name / f"{preset_name}_density_field.npz"


def ndof_of(nelxyz: list[int]) -> int:
    """Number of degrees of freedom for a ``[nelx, nely, nelz]`` grid."""
    nelx, nely, nelz = (int(v) for v in nelxyz)
    is_3d = nelz > 0
    dim = 3 if is_3d else 2
    return dim * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)


def nel_of(nelxyz: list[int]) -> int:
    """Number of elements for a ``[nelx, nely, nelz]`` grid."""
    nelx, nely, nelz = (int(v) for v in nelxyz)
    return nelx * nely * (nelz if nelz > 0 else 1)


def load_density_cache(path: Path) -> dict | None:
    """
    Load a cache file, returning ``None`` when it is missing or unreadable.

    Parameters
    ----------
    path : Path
        Cache file path.

    Returns
    -------
    dict or None
        Dictionary with at least ``xPhys`` and ``u``, or ``None``.
    """
    if not path.is_file():
        return None
    try:
        with np.load(path) as data:
            if "xPhys" not in data or "u" not in data:
                return None
            return {key: data[key] for key in data.files}
    except Exception:  # noqa: BLE001 - corrupt cache is simply ignored
        return None


def cache_matches_grid(data: dict, nelxyz: list[int]) -> bool:
    """
    Whether a cached result was produced for the given grid.

    Parameters
    ----------
    data : dict
        Dictionary returned by :func:`load_density_cache`.
    nelxyz : list[int]
        Current ``[nelx, nely, nelz]``.

    Returns
    -------
    bool
        ``True`` when the density and displacement shapes fit the grid.
    """
    x_phys = np.asarray(data["xPhys"])
    u = np.asarray(data["u"])
    nel = nel_of(nelxyz)
    return x_phys.shape[-1] == nel and u.shape[0] == ndof_of(nelxyz)


def validate_density_field(data: dict) -> str | None:
    """
    Check that a loaded density field is usable as a restart point.

    Parameters
    ----------
    data : dict
        Dictionary returned by :func:`load_density_cache`.

    Returns
    -------
    str or None
        An error message when the field contains non-finite or out-of-range
        values, otherwise ``None``.
    """
    if "xPhys" not in data:
        return "The restart file does not contain a density field."
    field = np.asarray(data["xPhys"], dtype=np.float64)
    if not np.all(np.isfinite(field)):
        return "The restart density field contains NaN or infinite values."
    if field.size and (field.min() < 0.0 or field.max() > 1.0):
        return "The restart density field must contain values between 0 and 1."
    return None


def save_density_cache(
    path: Path,
    xPhys: FloatArray,
    u: FloatArray,
    params: dict,
    nelxyz: list[int],
    converged: bool,
) -> None:
    """
    Write a density-field cache with the metadata needed to validate it.

    Parameters
    ----------
    path : Path
        Cache file path.
    xPhys : FloatArray
        Density field, shape (nel,) or (n_mat, nel).
    u : FloatArray
        Combined displacement vector, shape (ndof,).
    params : dict
        Parameters used for the run (hashed for invalidation).
    nelxyz : list[int]
        Grid dimensions of the run.
    converged : bool
        Whether the optimizer reached its convergence criterion.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        xPhys=xPhys,
        u=u,
        params_hash=params_hash(params),
        nelxyz=np.asarray(nelxyz, dtype=np.int64),
        converged=np.asarray(bool(converged)),
    )


__all__ = [
    "cache_matches_grid",
    "cache_path",
    "load_density_cache",
    "ndof_of",
    "nel_of",
    "params_hash",
    "save_density_cache",
    "validate_density_field",
]
