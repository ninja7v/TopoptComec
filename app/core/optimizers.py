# app/core/optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Topology Optimizers.

from __future__ import annotations
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

from app.core import initializers
from app.core.fem import FEM

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def _oc(
    nel: int,
    x: FloatArray,
    eta: float,
    max_change: float,
    dc: FloatArray,
    dv: FloatArray,
    g: float,
) -> tuple[FloatArray, float]:
    """
    Optimality Criterion (OC) update scheme.

    Args:
        nel: Total number of elements.
        x: Current design variables (densities).
        max_change: Maximum allowed change in design variables per iteration.
        dc: Sensitivities of the objective function.
        dv: Sensitivities of the volume constraint.
        g: Lagrangian multiplier for the volume constraint.

    Returns:
        A tuple containing the new design variables (xnew) and the updated gt value.
    """
    l1: float = 0.0
    l2: float = 1e9
    rhomin: float = 1e-6
    xnew: FloatArray = np.zeros(nel, dtype=np.float64)

    while (l2 - l1) / (l1 + l2) > 1e-4 and l2 > 1e-40:
        lmid: float = 0.5 * (l2 + l1)
        # Bisection method to find the Lagrange multiplier
        # This is the OC update rule with move limits
        x_update: FloatArray = x * np.maximum(0.1, -dc / dv / lmid) ** eta
        xnew[:] = np.maximum(
            rhomin,
            np.maximum(
                x - max_change, np.minimum(1.0, np.minimum(x + max_change, x_update))
            ),
        )

        gt: float = g + np.sum(
            dv * (xnew - x)
        )  # Should be near zero for the volume constraint
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, gt


def optimize(
    Dimensions: dict,
    Forces: dict,
    Materials: dict,
    Optimizer: dict,
    Supports: dict | None = None,
    Regions: dict | None = None,
    progress_callback: Callable[[int, float, float, FloatArray], bool] | None = None,
    verbose: bool = True,
    current_xPhys: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """
    Topology optimization

    Args:
        All parameters split per section.
        progress_callback: A function to call with (iteration, objective, change) for UI updates.

    Returns:
        xPhys: Final physical densities after optimization.
        ui: Associated displacement vector.
    """
    if verbose:
        print("Optimizer starting...")
    Supports = Supports or {}
    Regions = Regions or {}

    # Initialize FEM Environment
    fem: FEM = FEM(Dimensions, Materials, Optimizer)
    fem.setup_boundary_conditions(Forces, Supports)

    # Initialize Material
    x: FloatArray = initializers.initialize_material(
        init_type=Materials.get("init_type", 0),
        volfrac=Dimensions.get("volfrac", 0.5),
        nelx=fem.nelx,
        nely=fem.nely,
        nelz=fem.nelz,
        all_x=_get_active_coords(Supports, Forces, fem.is_3d)[0],
        all_y=_get_active_coords(Supports, Forces, fem.is_3d)[1],
        all_z=_get_active_coords(Supports, Forces, fem.is_3d)[2],
        current_xPhys=current_xPhys,
    )
    x = fem.apply_regions(x, Regions)
    xPhys: FloatArray = x.copy()
    g: float = 0.0

    # Optimization Params
    eta: float = Optimizer.get("eta", 1.0)
    max_change: float = Optimizer.get("max_change", 0.1)
    n_it: int = Optimizer.get("n_it", 30)

    if verbose:
        print("   Preparation done -> Optimization loop starting...")
    loop: int = 0
    change: float = 1.0

    # Emit frame 0 (initial density)
    if progress_callback and progress_callback(0, 0.0, 1.0, xPhys.copy()):
        pass

    while change > 0.01 and loop < n_it:
        loop += 1
        xold: FloatArray = x.copy()

        # Finite element analysis
        ui, uo = fem.solve(xPhys)

        # Optional: Compute Objective Value for Console Output (can also be computed inside compute_sensitivities for efficiency)
        obj_val: float = fem.compute_objective(xPhys, ui, uo)

        # Compute Sensitivities & Filter
        dc: FloatArray
        dv: FloatArray
        (dc, dv) = fem.compute_sensitivities(xPhys, ui, uo)

        # Update Design Variables
        x, g = _oc(fem.nel, x, eta, max_change, dc, dv, g)
        xPhys = fem.update_xPhys(x)

        # Apply regions
        xPhys = fem.apply_regions(xPhys, Regions)

        # Check Convergence
        change = np.linalg.norm(x - xold, np.inf)
        if verbose:
            print(
                f"It.: {loop:3d}, Obj.: {obj_val:.4f}, Vol.: {xPhys.mean():.3f}, Ch.: {change:.3f}"
            )

        if progress_callback and progress_callback(loop, obj_val, change, xPhys):
            if verbose:
                print("Optimization stopped by user.")
            break

    if verbose:
        print("Optimizer finished.")
    return xPhys, ui


def _get_active_coords(
    supports: dict, forces: dict, is_3d: bool
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    Extract active element coordinates from supports and forces dictionaries.

    Parameters
    ----------
    supports : dict
        Supports section from parameters, expected keys like `sx`, `sy`, `sz`,
        and `sdim` that indicate active supports.
    forces : dict
        Forces section from parameters, expected keys like `fix`, `fiy`, `fiz`,
        `fox`, `foy`, `foz`, and direction keys `fidir`, `fodir`.
    is_3d : bool
        Whether the problem is 3D. If False, a placeholder z-array of zeros is
        returned as the third element.

    Returns
    -------
    tuple of three FloatArray
        `all_x`, `all_y`, `all_z` arrays containing the x/y/z coordinates of
        active points concatenated from input forces, output forces and
        supports. For 2D problems `all_z` is an array of zeros matching the
        length of `all_x`.
    """

    # This extracts the logic previously doing np.concatenate on active indices
    def get_act(d: dict, k_dim: str, k_flag: str) -> FloatArray:
        return np.array(d.get(k_dim, []), dtype=np.float64)[
            [i for i, v in enumerate(d.get(k_flag, [])) if v != "-"]
        ]

    sx: FloatArray
    sy: FloatArray
    fix: FloatArray
    fiy: FloatArray
    fox: FloatArray
    foy: FloatArray
    sx, sy = get_act(supports, "sx", "sdim"), get_act(supports, "sy", "sdim")
    fix, fiy = get_act(forces, "fix", "fidir"), get_act(forces, "fiy", "fidir")
    fox, foy = get_act(forces, "fox", "fodir"), get_act(forces, "foy", "fodir")

    all_x: FloatArray = np.concatenate([fix, fox, sx])
    all_y: FloatArray = np.concatenate([fiy, foy, sy])

    if is_3d:
        sz: FloatArray = get_act(supports, "sz", "sdim")
        fiz: FloatArray = get_act(forces, "fiz", "fidir")
        foz: FloatArray = get_act(forces, "foz", "fodir")
        return all_x, all_y, np.concatenate([fiz, foz, sz])
    return all_x, all_y, np.array([0] * len(all_x), dtype=np.float64)


def optimize_multimaterial(
    Dimensions: dict,
    Forces: dict,
    Materials: dict,
    Optimizer: dict,
    Supports: dict | None = None,
    Regions: dict | None = None,
    progress_callback: Callable[[int, float, float, FloatArray], bool] | None = None,
    verbose: bool = True,
    current_xPhys: FloatArray | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Multi-material topology optimization (max 2 materials).

    Uses per-material density fields. Each material is updated via OC
    with its own volume constraint, and then projected to ensure the
    sum of densities does not exceed 1 in any element.

    Args:
        all parameters splited per section.
        progress_callback: A function called with (iteration, objective, change, xPhys_multi).

    Returns:
        xPhys_multi: Final densities, shape (n_mat, nel).
        ui: Displacement vector from the final solve.
    """
    if verbose:
        print("Multi-material optimizer starting...")
    Supports = Supports or {}
    Regions = Regions or {}

    # Initialize FEM Environment
    fem: FEM = FEM(Dimensions, Materials, Optimizer)
    fem.setup_boundary_conditions(Forces, Supports)

    E_list: list[float] = Materials.get("E", [1.0])
    percents: list[int] = Materials.get("percent", [100])
    n_mat: int = len(E_list)

    volfrac: float = Dimensions.get("volfrac", 0.5)

    # Initialize Material
    x: FloatArray = initializers.initialize_materials(
        init_type=Materials.get("init_type", 0),
        materials_percentage=percents,
        volfrac=volfrac,
        nelx=fem.nelx,
        nely=fem.nely,
        nelz=fem.nelz,
        all_x=_get_active_coords(Supports, Forces, fem.is_3d)[0],
        all_y=_get_active_coords(Supports, Forces, fem.is_3d)[1],
        all_z=_get_active_coords(Supports, Forces, fem.is_3d)[2],
        current_xPhys=current_xPhys,
    )
    for i in range(n_mat):
        x[i] = fem.apply_regions(x[i], Regions)
    xPhys: FloatArray = x.copy()
    g: FloatArray = np.zeros(n_mat, dtype=np.float64)

    # Optimization Params
    eta: float = Optimizer.get("eta", 1.0)
    max_change: float = Optimizer.get("max_change", 0.1)
    n_it: int = Optimizer.get("n_it", 30)

    if verbose:
        print("   Preparation done -> Optimization loop starting...")
    loop: int = 0
    change: float = 1.0

    # Emit frame 0 (initial density)
    if progress_callback and progress_callback(0, 0.0, 1.0, xPhys.copy()):
        pass

    while change > 0.01 and loop < n_it:
        loop += 1
        xold: FloatArray = x.copy()

        # Finite element analysis
        ui, uo = fem.solve(xPhys)

        # Optional: Compute Objective Value for Console Output (can also be computed inside compute_sensitivities for efficiency)
        obj_val: float = fem.compute_objective(xPhys, ui, uo)

        # Per-material sensitivity & OC update
        for i in range(n_mat):
            # Compute Sensitivities & Filter
            dc_i: FloatArray
            dv_i: FloatArray
            (dc_i, dv_i) = fem.compute_sensitivities(xPhys[i], ui, uo)

            # Chain-rule: Scale the sensitivity by the material's stiffness.
            dc_i *= E_list[i]

            # Update Design Variables
            x[i], g[i] = _oc(fem.nel, x[i], eta, max_change, dc_i, dv_i, g[i])
            xPhys[i] = fem.update_xPhys(x[i])

        # Apply regions
        for i in range(n_mat):
            xPhys[i] = fem.apply_regions(xPhys[i], Regions)

        # Partition-of-unity constraint: ensure sum of densities <= 1 per element
        col_sums: FloatArray = xPhys.sum(axis=0)
        excess: np.ndarray = col_sums > 1.0
        if np.any(excess):
            xPhys[:, excess] /= col_sums[excess]

        # Check Convergence
        change = float(np.max(np.abs(x - xold)))
        if verbose:
            print(
                f"It.: {loop:3d}, Obj.: {obj_val:.4f}, "
                + ", ".join(f"V{i}: {xPhys[i].mean():.3f}" for i in range(n_mat))
                + f", Ch.: {change:.3f}"
            )

        if progress_callback and progress_callback(loop, obj_val, change, xPhys):
            if verbose:
                print("Optimization stopped by user.")
            break

    if verbose:
        print("Multi-material optimizer finished.")
    return xPhys, ui
