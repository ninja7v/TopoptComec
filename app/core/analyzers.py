# app/core/analyzers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Analyze a mechanism.

from __future__ import annotations
from collections.abc import Callable
from functools import partial
import numpy as np
import numpy.typing as npt

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _checkerboard(x: FloatArray) -> bool:
    """
    Detect checkerboard patterns in a binaryized density field.

    Parameters
    ----------
    x : FloatArray
        Binary or continuous spatial field. Values are interpreted by
        thresholding at 0.5.

    Returns
    -------
    bool
        True if a checkerboard-like pattern is detected, False otherwise.
    """
    xbin = (x > 0.5).astype(np.int8)

    if xbin.ndim == 2:
        mask1 = np.array(
            [
                [0, 1, 0],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=bool,
        )
    elif xbin.ndim == 3:
        mask1 = np.array(
            [
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
                [[1, 0, 1], [0, 1, 0], [1, 0, 1]],
                [[0, 1, 0], [1, 0, 1], [0, 1, 0]],
            ],
            dtype=bool,
        )
    else:
        raise ValueError("Input must be 2D or 3D.")

    mask2 = ~mask1

    windows = np.lib.stride_tricks.sliding_window_view(xbin, mask1.shape)

    return bool(
        np.any(np.all(windows == mask1, axis=tuple(range(-xbin.ndim, 0))))
        or np.any(np.all(windows == mask2, axis=tuple(range(-xbin.ndim, 0))))
    )


def _watertight(x: FloatArray) -> bool:
    """
    Test whether the solid region is a single connected component.

    Parameters
    ----------
    x : FloatArray
        Spatial density field (binary or continuous). Values are thresholded
        at 0.5 prior to connectivity analysis.

    Returns
    -------
    bool
        True when exactly one connected component exists, False otherwise.
    """
    xbin = (x > 0.5).astype(int)
    from scipy.ndimage import label, generate_binary_structure

    # Define a structure to consider diagonal connections as touching
    structure = generate_binary_structure(rank=xbin.ndim, connectivity=xbin.ndim)
    _, n = label(xbin, structure=structure)
    return n == 1  # If there is only one connected component (connex), it is watertight


def _thresholded(xPhys: FloatArray) -> bool:
    """
    Determine whether the density field is mostly near binary values.

    Parameters
    ----------
    xPhys : FloatArray
        Density field (can be multi-material where values are in [0,1]).

    Returns
    -------
    bool
        True if the field appears thresholded, False otherwise.
    """
    # This should be close to 0 (worst case is 0.5 where all elements are at 0.5)
    mean: float = np.mean(np.minimum(xPhys, 1.0 - xPhys))
    return bool(mean < 0.1)


def _efficient(u: FloatArray, Dimensions: dict, Forces: dict) -> bool:
    """
    Heuristic check whether the mechanism achieves useful movement.

    For compliant mechanisms (with output forces) the routine compares the
    total input travel to the achieved output travel and expects a favourable
    ratio. For rigid mechanisms (no output forces) it checks that input
    displacements remain acceptably small. The thresholds are heuristic and
    chosen to provide a quick indicator.

    Parameters
    ----------
    u : FloatArray
        Displacement matrix returned by the solver (DOFs x columns).
    Dimensions : dict
        Dimensions dictionary containing `nelxyz` for mesh sizes.
    Forces : dict
        Forces dictionary with input/output force definitions and norms.

    Returns
    -------
    bool
        True if the mechanism is considered efficient by the heuristic,
        False otherwise.
    """
    nelx: int = Dimensions["nelxyz"][0]
    nely: int = Dimensions["nelxyz"][1]
    nelz: int = Dimensions["nelxyz"][2]
    is_3d: bool = nelz > 0
    dim_mul: int = 3 if is_3d else 2

    active_iforces_indices: list[int] = [
        i for i, fdir in enumerate(Forces.get("fidir", [])) if fdir != "-"
    ]
    nbInputForces: int = len(active_iforces_indices)
    if nbInputForces == 0:
        return False

    def get_disp(x: int, y: int, z: int, fdir: str, col_idx: int) -> float:
        node: int = (z * (nelx + 1) * (nely + 1) if is_3d else 0) + x * (nely + 1) + y
        dof_base: int = node * dim_mul
        if "X" in fdir:
            dof: int = dof_base
        elif "Y" in fdir:
            dof = dof_base + 1
        else:
            dof = dof_base + 2

        sign: float = (
            -1.0 if "\u2190" in fdir or "\u2191" in fdir or "<" in fdir else 1.0
        )
        return u[dof, col_idx] * sign

    effectiveness: float = 0.0
    active_oforces_indices: list[int] = [
        i for i, fdir in enumerate(Forces.get("fodir", [])) if fdir != "-"
    ]

    if active_oforces_indices:
        # Compliant mechanism: compare total input travel to total output geometric travel
        total_u_in: float = 0.0
        total_u_out: float = 0.0

        nbOutputForces: int = len(active_oforces_indices)

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
            actual_col: int = col_idx if col_idx < nbInputForces else 0
            u_out_val: float = get_disp(
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
            u_in: float = get_disp(
                Forces["fix"][i],
                Forces["fiy"][i],
                Forces["fiz"][i] if is_3d else 0,
                Forces["fidir"][i],
                col_idx,
            )
            effectiveness += abs(u_in) / max(Forces["finorm"][i], 1e-9)

        return bool(effectiveness < 500 * nbInputForces)


def analyze(
    xPhys: FloatArray,
    u: FloatArray,
    Dimensions: dict,
    Forces: dict,
    progress_callback: Callable[[int], bool] | None = None,
) -> tuple[bool, bool, bool, bool]:
    """
    Run a series of quick, independent analyses on a design.

    The function computes four boolean indicators:
    - contains_checkerboard: detection of local checkerboard artifacts
    - is_watertight: whether the solid region is a single connected component
    - is_thresholded: whether the density field is near-binary
    - is_efficient: heuristic effectiveness of the mechanism movement

    The `progress_callback` can be used to interrupt long-running analysis
    (it will be called with increasing step indices and should return True
    to request cancellation).

    Parameters
    ----------
    xPhys : FloatArray
        Density field produced by the optimizer (can be multi-material).
    u : FloatArray
        Displacement field produced by the solver.
    Dimensions : dict
        Dimensions dictionary (contains `nelxyz`).
    Forces : dict
        Forces dictionary used to evaluate efficiency.
    progress_callback : callable or None
        Optional callback called with step index; if it returns True the
        analysis will stop early and the partial results will be returned.

    Returns
    -------
    tuple(bool, bool, bool, bool)
        (contains_checkerboard, is_watertight, is_thresholded, is_efficient)
    """
    xPhys_copy: FloatArray = xPhys.copy()
    if xPhys.ndim == 2:
        xPhys_copy = np.clip(xPhys_copy.sum(axis=0, keepdims=True), 0.0, 1.0)
    x: FloatArray = (
        xPhys_copy.reshape(
            Dimensions["nelxyz"][2], Dimensions["nelxyz"][0], Dimensions["nelxyz"][1]
        )
        if Dimensions["nelxyz"][2] > 0
        else xPhys_copy.reshape(Dimensions["nelxyz"][0], Dimensions["nelxyz"][1])
    )

    checks = [
        partial(_checkerboard, x),
        partial(_watertight, x),
        partial(_thresholded, xPhys),
        partial(
            _efficient,
            u=u,
            Dimensions=Dimensions,
            Forces=Forces,
        ),
    ]

    results: list[bool] = []

    for step, check in enumerate(checks, start=1):
        results.append(check())
        if progress_callback and progress_callback(step):
            print("Optimization stopped by user.")
            # Fill remaining checks with False
            results.extend([False] * (len(checks) - len(results)))
            break

    return tuple(results)
