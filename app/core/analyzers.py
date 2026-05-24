# app/core/analyzers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Analyze a mechanism.

from typing import Dict, Callable, Optional, Tuple
import numpy as np


def _checkerboard(x: np.ndarray) -> bool:
    """Check if the mechanism contains a checkerboard pattern."""
    # Apply a mask [[0, 1], [1, 0]] to the xPhys array with a tolerance to detect checkerboard patterns
    xbin = (x > 0.5).astype(int)
    if xbin.ndim == 2:
        mask1 = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=bool)
        mask2 = ~mask1
        h, w = xbin.shape
        for i in range(h - 2):
            for j in range(w - 2):
                block = xbin[i : i + 3, j : j + 3]
                if np.array_equal(block, mask1) or np.array_equal(block, mask2):
                    return True
    else:
        mask1 = np.array(
            [
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            ],
            dtype=bool,
        )
        mask2 = ~mask1
        h, w, d = xbin.shape
        for i in range(h - 2):
            for j in range(w - 2):
                for k in range(d - 2):
                    block = xbin[i : i + 3, j : j + 3, k : k + 3]
                    if np.array_equal(block, mask1) or np.array_equal(block, mask2):
                        return True
    return False


def _watertight(x: np.ndarray) -> bool:
    """Check if the mechanism is watertight."""
    # Binarize xPhys with a threshold of 0.5 to get a binary image
    xbin = (x > 0.5).astype(int)
    from scipy.ndimage import label, generate_binary_structure

    # Define a structure to consider diagonal connections as touching
    structure = generate_binary_structure(rank=xbin.ndim, connectivity=xbin.ndim)
    res = label(xbin, structure)
    if isinstance(res, tuple):
        _, n = res
    else:
        n = res
    return n == 1  # If there is only one connected component (connex), it is watertight


def _thresholded(xPhys: np.ndarray) -> bool:
    """Check if the mechanism is thresholded."""
    # Check if np.mean(np.minimum(x, 1 - x)) is close to 0 (worst case is 0.5 where all elements are at 0.5)
    mean = np.mean(np.minimum(xPhys, 1 - xPhys))
    return bool(mean < 0.1)


def _efficient(u: np.ndarray, Dimensions: Dict, Forces: Dict) -> bool:
    """Check if the mechanism is efficient."""
    nelx, nely, nelz = Dimensions["nelxyz"]
    is_3d = nelz > 0
    dim_mul = 3 if is_3d else 2

    active_iforces_indices = [
        i for i, fdir in enumerate(Forces.get("fidir", [])) if fdir != "-"
    ]
    nbInputForces = len(active_iforces_indices)
    if nbInputForces == 0:
        return False

    def get_disp(x: int, y: int, z: int, fdir: str, col_idx: int) -> float:
        node = (z * (nelx + 1) * (nely + 1) if is_3d else 0) + x * (nely + 1) + y
        dof_base = node * dim_mul
        if "X" in fdir:
            dof = dof_base
        elif "Y" in fdir:
            dof = dof_base + 1
        else:
            dof = dof_base + 2

        sign = -1 if "\u2190" in fdir or "\u2191" in fdir or "<" in fdir else 1
        return u[dof, col_idx] * sign

    effectiveness = 0.0
    active_oforces_indices = [
        i for i, fdir in enumerate(Forces.get("fodir", [])) if fdir != "-"
    ]

    if active_oforces_indices:
        # Compliant mechanism: compare total input travel to total output geometric travel
        total_u_in = 0.0
        total_u_out = 0.0

        nbOutputForces = len(active_oforces_indices)

        for col_idx, i in enumerate(active_iforces_indices):
            total_u_in += abs(
                get_disp(
                    Forces["fix"][i],
                    Forces["fiy"][i],
                    Forces["fiz"][i] if is_3d else 0,
                    Forces["fidir"][i],
                    col_idx,
                )
            )

        for col_idx, oi in enumerate(active_oforces_indices):
            actual_col = col_idx if col_idx < nbInputForces else 0
            u_out_val = get_disp(
                Forces["fox"][oi],
                Forces["foy"][oi],
                Forces["foz"][oi] if is_3d else 0,
                Forces["fodir"][oi],
                actual_col,
            )
            # Only reward positive movement in the intended direction
            if u_out_val > 0:
                total_u_out += u_out_val

        effectiveness = total_u_in / max(total_u_out, 1e-9)
        return bool(effectiveness < 1 * nbOutputForces)
    else:
        # Rigid mechanism: displacement at input location must remain small
        for col_idx, i in enumerate(active_iforces_indices):
            u_in = get_disp(
                Forces["fix"][i],
                Forces["fiy"][i],
                Forces["fiz"][i] if is_3d else 0,
                Forces["fidir"][i],
                col_idx,
            )
            effectiveness += abs(u_in) / max(Forces["finorm"][i], 1e-9)

        return bool(effectiveness < 500 * nbInputForces)


def analyze(
    xPhys: np.ndarray,
    u: np.ndarray,
    Dimensions: Dict,
    Forces: Dict,
    progress_callback: Optional[Callable] = None,
) -> Tuple[bool, bool, bool, bool]:
    """Analyze the mechanism."""
    xPhys_copy = xPhys.copy()
    if xPhys.ndim == 2:
        xPhys_copy = np.clip(xPhys_copy.sum(axis=0, keepdims=True), 0.0, 1.0)
    x = (
        xPhys_copy.reshape(
            Dimensions["nelxyz"][2], Dimensions["nelxyz"][0], Dimensions["nelxyz"][1]
        )
        if Dimensions["nelxyz"][2] > 0
        else xPhys_copy.reshape(Dimensions["nelxyz"][0], Dimensions["nelxyz"][1])
    )

    contains_checkerboard = _checkerboard(x)
    if progress_callback and progress_callback(1):
        print("Optimization stopped by user.")
        return contains_checkerboard, False, False, False

    is_watertight = _watertight(x)
    if progress_callback and progress_callback(2):
        print("Optimization stopped by user.")
        return contains_checkerboard, is_watertight, False, False

    is_thresholded = _thresholded(xPhys)
    if progress_callback and progress_callback(3):
        print("Optimization stopped by user.")
        return contains_checkerboard, is_watertight, is_thresholded, False

    is_efficient = _efficient(u, Dimensions, Forces)
    if progress_callback and progress_callback(4):
        print("Optimization stopped by user.")

    return contains_checkerboard, is_watertight, is_thresholded, is_efficient
