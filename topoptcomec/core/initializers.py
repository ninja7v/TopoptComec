# topoptcomec/core/initializers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Material initializers.

from __future__ import annotations

from collections.abc import Callable
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import cdist

# Type aliases
FloatArray = npt.NDArray[np.float64]


class InitType(IntEnum):
    UNIFORM = 0
    DISTANCE = 1
    RANDOM = 2
    CURRENT = 3


def _bisect_alpha(
    mean_fn: Callable[[float], float],
    lo: float,
    hi: float,
    target: float,
    iterations: int = 80,
) -> float:
    """Bisect a monotone non-decreasing ``mean_fn`` towards ``target``."""
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if mean_fn(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _rescale_densities(d: FloatArray, volfrac: float) -> FloatArray:
    """
    Rescale densities so their mean equals volfrac while keeping values in [0,1].

    The routine first searches for a scalar :math:`\\alpha` in a smooth
    perturbation:

    .. math::

        \\rho_e(\\alpha) =
        d_e + \\alpha(-4d_e^2 + 4d_e)

    so that:

    .. math::

        \\frac{1}{n_e}\\sum_e \\rho_e(\\alpha) = V^*

    This map is flat for binary inputs (where :math:`4d(1-d) = 0`), so when it
    cannot reach the target - typically when restarting from a thresholded
    result - a monotone shift is used instead:

    .. math::

        \\rho_e(\\alpha) = \\operatorname{clip}(d_e - \\alpha, 0, 1)

    The shift removes the lowest-density elements first, which is the natural
    way to lower the volume fraction of a binary design.

    Parameters
    ----------
    d : FloatArray
        Input density field.
    volfrac : float
        Target volume fraction (mean value).

    Returns
    -------
    FloatArray
        Rescaled density field with mean approximately equal to volfrac.
    """
    d = np.clip(np.asarray(d, dtype=np.float64), 0.0, 1.0)
    target = float(np.clip(volfrac, 0.0, 1.0))
    tol: float = 1e-4
    if abs(float(np.mean(d)) - target) < tol:
        return d

    def quad_mean(alpha: float) -> float:
        return float(np.mean(np.clip(d + (-4 * d**2 + 4 * d) * alpha, 0.0, 1.0)))

    lo_q, hi_q = quad_mean(-2.0), quad_mean(2.0)
    if lo_q - tol <= target <= hi_q + tol:
        alpha = _bisect_alpha(quad_mean, -2.0, 2.0, target)
        return np.clip(d + (-4 * d**2 + 4 * d) * alpha, 0.0, 1.0)

    def shift_mean(alpha: float) -> float:
        return float(np.mean(np.clip(d + alpha, 0.0, 1.0)))

    alpha = _bisect_alpha(shift_mean, -1.0, 1.0, target)
    return np.clip(d + alpha, 0.0, 1.0)


def initialize_material(
    init_type: int,
    volfrac: float,
    nelx: int,
    nely: int,
    nelz: int,
    all_x: FloatArray,
    all_y: FloatArray,
    all_z: FloatArray,
    current_xPhys: FloatArray | None = None,
) -> FloatArray:
    """
    Initialize a single-material density field according to a chosen strategy.

    Uniform initialization starts from:

    .. math::

        \\rho_e = V^*

    Distance initialization maps element-center distance to active supports or
    loads into a density seed:

    .. math::

        \\rho_e^{raw} =
        \\frac{d_{max} - \\min_i \\lVert \\mathbf{x}_e -
        \\mathbf{x}_i \\rVert}{d_{max}}

    and then rescales it to satisfy the requested volume fraction.

    Parameters
    ----------
    init_type : int
        Type of initializer: 0 = uniform, 1 = distance-field from active
        coordinates, 2 = random seed, 3 = current result.
    volfrac : float
        Target volume fraction (mean density between 0 and 1).
    nelx, nely, nelz : int
        Number of elements along each axis. If `nelz` > 0, a 3D field is
        assumed.
    all_x, all_y, all_z : FloatArray
        Arrays of active coordinate locations used by the distance-field
        initializer (typically forces and supports coordinates). For 2D
        problems `all_z` may be an array of zeros.
    current_xPhys : FloatArray, optional
        Current density field to initialize from if init_type is 3.

    Returns
    -------
    FloatArray
        Flattened density field of length `nelx * nely * (nelz or 1)` with
        values clipped to [0, 1] and mean approximately equal to `volfrac`.

    Raises
    ------
    ValueError
        If `init_type` is not one of the supported integers (0, 1, 2, 3).
    """
    is_3d: bool = nelz > 0
    nel: int = nelx * nely * (nelz if is_3d else 1)

    match init_type:
        case InitType.UNIFORM:
            return np.full(nel, volfrac, dtype=np.float64)

        case InitType.DISTANCE:
            points: FloatArray = np.column_stack(
                [all_x, all_y, all_z] if is_3d else [all_x, all_y]
            )

            if len(points) == 0:
                return np.full(nel, volfrac, dtype=np.float64)

            # Generate element center coordinates matching FEM loop order:
            if is_3d:
                Z: FloatArray = np.repeat(np.arange(nelz), nelx * nely)
                X: FloatArray = np.tile(np.repeat(np.arange(nelx), nely), nelz)
                Y: FloatArray = np.tile(np.arange(nely), nelx * nelz)
                coords: FloatArray = np.column_stack((X, Y, Z))
            else:
                X = np.repeat(np.arange(nelx), nely)
                Y = np.tile(np.arange(nely), nelx)
                coords = np.column_stack((X, Y))

            dists: FloatArray = cdist(
                coords, points, metric="euclidean"
            )  # Shape: (nel, n_points)
            min_dist: FloatArray = dists.min(axis=1)

            # Invert distance: Near = 1.0, Far = 0.0
            distance_max: float = np.sqrt(nelx**2 + nely**2 + (nelz**2 if is_3d else 0))

            raw: FloatArray = (distance_max - min_dist) / distance_max

            return _rescale_densities(d=raw, volfrac=volfrac)

        case InitType.RANDOM:
            rng = np.random.default_rng(42)
            raw: FloatArray = rng.random(nel)
            return _rescale_densities(d=raw, volfrac=volfrac)

        case InitType.CURRENT:
            if current_xPhys is not None:
                res = None
                if current_xPhys.ndim == 2:
                    if current_xPhys.shape[1] == nel:
                        res = current_xPhys[0].copy()

                elif current_xPhys.ndim == 1 and current_xPhys.size == nel:
                    res = current_xPhys.copy()

                if res is not None:
                    if not np.isclose(np.mean(res), volfrac, atol=1e-4):
                        res = _rescale_densities(res, volfrac)
                    return res

            return np.full(nel, volfrac, dtype=np.float64)

        case _:
            raise ValueError(f"Invalid init_type: {init_type}")


def initialize_materials(
    init_type: int,
    materials_percentage: list[int],
    volfrac: float,
    nelx: int,
    nely: int,
    nelz: int,
    all_x: FloatArray,
    all_y: FloatArray,
    all_z: FloatArray,
    current_xPhys: FloatArray | None = None,
) -> FloatArray | None:
    """Initialize multi-material density fields.

    A single shared spatial pattern :math:`p(\\mathbf{x}_e)\\in[0,1]` with
    :math:`\\bar p = V^*` is built from the requested strategy, then split
    across materials proportionally to their volume shares:

    .. math::

        \\rho_{m,e} = \\frac{p_m}{100}\\,p_e,
        \\qquad \\sum_m \\frac{p_m}{100} = 1

    so that per-material targets and the element-wise total budget are met
    simultaneously:

    .. math::

        \\frac{1}{n_e}\\sum_e \\rho_{m,e} = V^*\\frac{p_m}{100} = V_m^*,
        \\qquad \\sum_m \\rho_{m,e} = p_e \\le 1.

    For ``init_type=3`` (current result), each material row is rescaled
    independently to its new target fraction :math:`V_m^*`, so a previous
    field can be reused with a different volume split.

    Args:
        init_type: Initialization strategy (0=Uniform, 1=Distance, 2=Random, 3=Current).
        materials_percentage: List of percentage of each material (sum to 100).
        volfrac: Total target volume fraction.
        nelx, nely, nelz: Grid dimensions.
        all_x, all_y, all_z: Active coordinate arrays for distance-based init.
        current_xPhys: Current density field to initialize from if init_type is 3.

    Returns:
        Array of shape (n_mat, nel) with per-material densities. Each row's
        mean approximates the corresponding volume fraction and per-element
        column sums stay within [0, 1].
    """
    if sum(materials_percentage) != 100:
        return None

    n_mat: int = len(materials_percentage)
    shares: FloatArray = np.array(materials_percentage, dtype=np.float64) / 100.0
    materials_frac: FloatArray = volfrac * shares
    is_3d: bool = nelz > 0
    nel: int = nelx * nely * (nelz if is_3d else 1)
    rho: FloatArray | None = None
    pattern: FloatArray

    match init_type:
        case InitType.UNIFORM:
            pattern = np.full(nel, volfrac, dtype=np.float64)

        case InitType.DISTANCE:
            points: FloatArray = np.column_stack(
                [all_x, all_y, all_z] if is_3d else [all_x, all_y]
            )
            if len(points) == 0:
                pattern = np.full(nel, volfrac, dtype=np.float64)
            else:
                if is_3d:
                    Z: FloatArray = np.repeat(np.arange(nelz), nelx * nely)
                    X: FloatArray = np.tile(np.repeat(np.arange(nelx), nely), nelz)
                    Y: FloatArray = np.tile(np.arange(nely), nelx * nelz)
                    coords: FloatArray = np.column_stack((X, Y, Z))
                else:
                    X = np.repeat(np.arange(nelx), nely)
                    Y = np.tile(np.arange(nely), nelx)
                    coords = np.column_stack((X, Y))

                dists: FloatArray = cdist(coords, points, metric="euclidean")
                min_dist: FloatArray = dists.min(axis=1)
                distance_max: float = float(
                    np.sqrt(nelx**2 + nely**2 + (nelz**2 if is_3d else 0))
                )
                raw: FloatArray = (distance_max - min_dist) / distance_max
                pattern = _rescale_densities(d=raw, volfrac=volfrac)

        case InitType.RANDOM:
            rng = np.random.default_rng(42)
            raw = rng.random(nel)
            pattern = _rescale_densities(d=raw, volfrac=volfrac)

        case InitType.CURRENT:
            if (
                current_xPhys is not None
                and current_xPhys.ndim == 2
                and current_xPhys.shape == (n_mat, nel)
            ):
                # Reuse a full per-material field: rescale each row to its
                # new target, preserving per-material spatial structure.
                rho = current_xPhys.copy()
                for i in range(n_mat):
                    rho[i] = _rescale_densities(d=rho[i], volfrac=materials_frac[i])
            else:
                # Reuse a single-material field as the shared pattern.
                if (
                    current_xPhys is not None
                    and current_xPhys.ndim == 1
                    and current_xPhys.size == nel
                ):
                    pattern = _rescale_densities(d=current_xPhys, volfrac=volfrac)
                else:  # fallback to uniform pattern
                    pattern = np.full(nel, volfrac, dtype=np.float64)

        case _:
            raise ValueError(f"Invalid init_type: {init_type}")

    # Shared-pattern cases: split the spatial pattern across materials.
    if rho is None:
        rho = shares[:, None] * pattern[None, :]

    return np.clip(rho, 1e-6, 1.0)
