# app/core/post_processing.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Post-processing utilities for topology optimization results.

from __future__ import annotations
import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def scale_axis_1d(
    arr: FloatArray, axis: int, target_size: int, a: int = 3
) -> FloatArray:
    """
    Scale a multi-dimensional numpy array along a given axis using Lanczos interpolation.

    Parameters
    ----------
    arr : FloatArray
        Input multi-dimensional array.
    axis : int
        Axis along which to scale.
    target_size : int
        Target size for the scaled axis.
    a : int, optional
        Lanczos kernel support radius parameter, by default 3.

    Returns
    -------
    FloatArray
        Scaled array.
    """
    n = arr.shape[axis]
    m = target_size
    if n == m:
        return arr.copy()

    scale = m / n
    if scale < 1.0:
        kernel_scale = scale
        r = a / scale
    else:
        kernel_scale = 1.0
        r = float(a)

    j = np.arange(m)
    x_centers = (j + 0.5) / scale - 0.5

    i_indices = np.arange(n)[np.newaxis, :]  # (1, n)
    centers = x_centers[:, np.newaxis]  # (m, 1)

    diffs = i_indices - centers  # (m, n)
    x = diffs * kernel_scale

    mask = np.abs(diffs) < r
    weights = np.zeros_like(diffs)

    x_masked = x[mask]
    w_vals = np.zeros_like(x_masked)

    # sinc(x) = sin(pi * x) / (pi * x)
    zero_mask = x_masked == 0
    w_vals[zero_mask] = 1.0

    nz_mask = ~zero_mask
    x_nz = x_masked[nz_mask]
    pi_x = np.pi * x_nz
    w_vals[nz_mask] = (np.sin(pi_x) / pi_x) * (np.sin(pi_x / a) / (pi_x / a))

    weights[mask] = w_vals

    row_sums = weights.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    weights /= row_sums

    arr_rolled = np.moveaxis(arr, axis, -1)  # shape: (..., n)
    res_rolled = np.dot(arr_rolled, weights.T)
    res = np.moveaxis(res_rolled, -1, axis)
    return res


def rescale_density_field(
    xPhys: FloatArray | None,
    old_res: tuple[int, int, int],
    new_res: tuple[int, int, int],
) -> FloatArray | None:
    """
    Rescale the density field xPhys from old_res to new_res using Lanczos interpolation.

    Parameters
    ----------
    xPhys : FloatArray or None
        Density field. Can be 1D of shape (nel,) or 2D of shape (n_mat, nel).
    old_res : tuple of (int, int, int)
        Old grid dimensions (nelx, nely, nelz). nelz is 0 for 2D.
    new_res : tuple of (int, int, int)
        New grid dimensions (nelx, nely, nelz). nelz is 0 for 2D.

    Returns
    -------
    FloatArray or None
        Rescaled density field of the same shape format (1D or 2D).
    """
    if xPhys is None:
        return None

    nelx_old, nely_old, nelz_old = old_res
    nelx_new, nely_new, nelz_new = new_res

    is_3d = nelz_old > 0 or nelz_new > 0
    if old_res == new_res:
        return xPhys.copy()

    is_multi = xPhys.ndim == 2

    if is_3d:
        if is_multi:
            arr = xPhys.reshape((xPhys.shape[0], nelz_old, nelx_old, nely_old))
            arr = scale_axis_1d(arr, 1, nelz_new)
            arr = scale_axis_1d(arr, 2, nelx_new)
            arr = scale_axis_1d(arr, 3, nely_new)
            res = arr.reshape((xPhys.shape[0], -1))
        else:
            arr = xPhys.reshape((nelz_old, nelx_old, nely_old))
            arr = scale_axis_1d(arr, 0, nelz_new)
            arr = scale_axis_1d(arr, 1, nelx_new)
            arr = scale_axis_1d(arr, 2, nely_new)
            res = arr.flatten()
    else:
        if is_multi:
            arr = xPhys.reshape((xPhys.shape[0], nelx_old, nely_old))
            arr = scale_axis_1d(arr, 1, nelx_new)
            arr = scale_axis_1d(arr, 2, nely_new)
            res = arr.reshape((xPhys.shape[0], -1))
        else:
            arr = xPhys.reshape((nelx_old, nely_old))
            arr = scale_axis_1d(arr, 0, nelx_new)
            arr = scale_axis_1d(arr, 1, nely_new)
            res = arr.flatten()

    return np.clip(res, 0.0, 1.0)


def rescale_displacement_field(
    u: np.ndarray | None,
    old_dim: tuple[int, int, int],
    new_dim: tuple[int, int, int],
) -> np.ndarray | None:
    """
    Approximate displacement vectors at a new resolution using spatial rescaling.

    Parameters
    ----------
    u : np.ndarray or None
        Displacement vector(s) from a FEM result, shape (ndof,) or (ndof, n_load_cases).
    old_res : tuple[int, int, int]
        Original element grid dimensions (nelx, nely, nelz).
    new_res : tuple[int, int, int]
        Target element grid dimensions (nelx, nely, nelz).

    Returns
    -------
    np.ndarray or None
        Rescaled displacement array with the same layout as `u`.
    """
    if u is None:
        return None

    if old_dim == new_dim:
        return np.asarray(u, dtype=np.float64).copy()

    old_nelx, old_nely, old_nelz = old_dim
    new_nelx, new_nely, new_nelz = new_dim
    if bool(old_nelz > 0) != bool(new_nelz > 0):
        raise ValueError("Cannot rescale displacement between 2D and 3D grids.")

    is_3d = old_nelz > 0
    comp_count = 3 if is_3d else 2
    u_arr = np.asarray(u, dtype=np.float64)
    squeeze = False
    if u_arr.ndim == 1:
        u_arr = u_arr[:, None]
        squeeze = True
    if u_arr.ndim != 2:
        raise ValueError("Displacement field must be 1D or 2D.")

    n_nodes_old = (
        (old_nelx + 1) * (old_nely + 1) * (old_nelz + 1)
        if is_3d
        else (old_nelx + 1) * (old_nely + 1)
    )
    if u_arr.shape[0] != n_nodes_old * comp_count:
        raise ValueError("Displacement field size does not match old resolution.")

    n_load_cases: int = u_arr.shape[1]
    components: list[np.ndarray] = []
    for comp_idx in range(comp_count):
        comp = u_arr[comp_idx::comp_count]
        # Preserve original value range to avoid interpolation ringing
        comp_min = float(np.min(comp))
        comp_max = float(np.max(comp))
        if is_3d:
            comp = comp.reshape(
                (old_nelz + 1, old_nelx + 1, old_nely + 1, n_load_cases),
                order="C",
            )
            comp = scale_axis_1d(comp, axis=0, target_size=new_nelz + 1)
            comp = scale_axis_1d(comp, axis=1, target_size=new_nelx + 1)
            comp = scale_axis_1d(comp, axis=2, target_size=new_nely + 1)
        else:
            comp = comp.reshape((old_nelx + 1, old_nely + 1, n_load_cases), order="C")
            comp = scale_axis_1d(comp, axis=0, target_size=new_nelx + 1)
            comp = scale_axis_1d(comp, axis=1, target_size=new_nely + 1)

        # Clamp to original range to prevent overshoot from interpolation
        comp = np.clip(comp, comp_min, comp_max)

        components.append(comp.reshape(-1, n_load_cases, order="C"))

    new_ndof: int = (
        (new_nelx + 1) * (new_nely + 1) * (new_nelz + 1)
        if is_3d
        else (new_nelx + 1) * (new_nely + 1)
    ) * comp_count
    res = np.empty((new_ndof, n_load_cases), dtype=np.float64)
    for comp_idx, comp_flat in enumerate(components):
        res[comp_idx::comp_count, :] = comp_flat

    return res[:, 0] if squeeze else res
