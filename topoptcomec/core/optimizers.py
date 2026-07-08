# topoptcomec/core/optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Topology Optimizers.

from __future__ import annotations
from collections.abc import Callable
import numpy as np
import numpy.typing as npt

from topoptcomec.core import initializers, preset_format
from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Region, Support

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

    The update approximates the KKT stationarity condition through a bisection
    search on the Lagrange multiplier:

    .. math::

        x_e^{new} =
        \\operatorname{clip}_{[x_e-m, x_e+m]}
        \\left(
            x_e\\left(-\\frac{\\partial C / \\partial x_e}
            {\\lambda\\,\\partial V / \\partial x_e}\\right)^\\eta
        \\right)

    followed by the density bounds:

    .. math::

        \\rho_{\\min} \\leq x_e^{new} \\leq 1

    Args:
        nel: Total number of elements.
        x: Current design variables (densities).
        eta: Damping exponent of the OC update.
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


def _translate_problem(
    Dimensions: dict,
    Forces: dict,
    Supports: dict | None,
    Regions: dict | None,
) -> tuple[StructuredGrid, list[Load], list[Load], list[Support], list[Region]]:
    """
    Translate the legacy preset dictionaries into the typed problem model.

    Parameters
    ----------
    Dimensions, Forces, Supports, Regions : dict
        Legacy preset sections.

    Returns
    -------
    tuple
        (grid, loads_in, loads_out, supports, regions)
    """
    nelx, nely, nelz = (int(v) for v in Dimensions.get("nelxyz", [1, 1, 0]))
    grid = StructuredGrid(nelx, nely, max(0, nelz))
    loads_in = preset_format.parse_loads(
        Forces, "fix", "fiy", "fiz", "fidir", "finorm", grid.is_3d
    )
    loads_out = preset_format.parse_loads(
        Forces, "fox", "foy", "foz", "fodir", "fonorm", grid.is_3d
    )
    supports = preset_format.parse_supports(Supports, grid.is_3d)
    regions = preset_format.parse_regions(Regions)
    return grid, loads_in, loads_out, supports, regions


def build_fem(
    Dimensions: dict,
    Forces: dict,
    Materials: dict,
    Optimizer: dict,
    Supports: dict | None = None,
) -> tuple[FEM, list[Region]]:
    """
    Build a configured FEM solver from legacy preset dictionaries.

    Returns
    -------
    tuple[FEM, list[Region]]
        FEM instance with boundary conditions applied, and parsed regions.
    """
    grid, loads_in, loads_out, supports, regions = _translate_problem(
        Dimensions, Forces, Supports, None
    )
    fem = FEM(
        grid,
        E=Materials.get("E", [1.0]),
        nu=Materials.get("nu", [0.3]),
        penal=float(Optimizer.get("penal", 3.0)),
        solver=Optimizer.get("solver", "Auto"),
        filter_type=Optimizer.get("filter_type", "Sensitivity"),
        filter_radius=float(Optimizer.get("filter_radius_min", 0.0)),
    )
    fem.setup_boundary_conditions(loads_in, loads_out, supports)
    return fem, regions


def _active_coords(
    loads_in: list[Load], loads_out: list[Load], supports: list[Support]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Coordinates of all active loads and supports (for the distance initializer)."""
    pts = [(load.x, load.y, load.z) for load in loads_in + loads_out]
    pts += [(sup.x, sup.y, sup.z) for sup in supports]
    if not pts:
        empty = np.array([], dtype=np.float64)
        return empty, empty.copy(), empty.copy()
    arr = np.asarray(pts, dtype=np.float64)
    return arr[:, 0], arr[:, 1], arr[:, 2]


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
    Single-material topology optimization.

    The loop alternates finite element analysis, sensitivity
    analysis, filtering, and OC updates for:

    .. math::

        \\min_{\\boldsymbol{\\rho}} C(\\boldsymbol{\\rho})
        \\quad \\text{s.t.} \\quad
        \\bar{\\rho} = \\frac{1}{n_e}\\sum_e \\rho_e \\leq V^*

    Args:
        All parameters split per section (legacy preset dictionaries).
        progress_callback: A function called with (iteration, objective,
            change, xPhys) for UI updates; returning True stops the run.

    Returns:
        xPhys: Final physical densities after optimization.
        ui: Associated displacement vector.
    """
    xPhys, ui = _optimize_common(
        Dimensions,
        Forces,
        Materials,
        Optimizer,
        Supports,
        Regions,
        progress_callback,
        verbose,
        current_xPhys,
        multimaterial=False,
    )
    return xPhys[0], ui


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
    """Multi-material topology optimization.

    Uses per-material density fields. Each material is updated via OC
    with its own volume constraint, and then projected to ensure the
    sum of densities does not exceed 1 in any element.

    Multi-material densities satisfy a local partition constraint:

    .. math::

        \\sum_{m=1}^{n_m}\\rho_{m,e} \\leq 1,
        \\qquad 0 \\leq \\rho_{m,e} \\leq 1

    The effective stiffness is the additive SIMP interpolation:

    .. math::

        E_e = \\sum_{m=1}^{n_m}\\rho_{m,e}^{p}E_{0,m}

    Args:
        All parameters split per section (legacy preset dictionaries).
        progress_callback: A function called with (iteration, objective, change, xPhys_multi).

    Returns:
        xPhys_multi: Final densities, shape (n_mat, nel).
        ui: Displacement vector from the final solve.
    """
    return _optimize_common(
        Dimensions,
        Forces,
        Materials,
        Optimizer,
        Supports,
        Regions,
        progress_callback,
        verbose,
        current_xPhys,
        multimaterial=True,
    )


def _optimize_common(
    Dimensions: dict,
    Forces: dict,
    Materials: dict,
    Optimizer: dict,
    Supports: dict | None,
    Regions: dict | None,
    progress_callback: Callable[[int, float, float, FloatArray], bool] | None,
    verbose: bool,
    current_xPhys: FloatArray | None,
    multimaterial: bool,
) -> tuple[FloatArray, FloatArray]:
    """Shared optimization loop for single- and multi-material problems."""
    if verbose:
        print("Optimizer starting...")

    grid, loads_in, loads_out, supports, regions = _translate_problem(
        Dimensions, Forces, Supports, Regions
    )
    fem = FEM(
        grid,
        E=Materials.get("E", [1.0]),
        nu=Materials.get("nu", [0.3]),
        penal=float(Optimizer.get("penal", 3.0)),
        solver=Optimizer.get("solver", "Auto"),
        filter_type=Optimizer.get("filter_type", "Sensitivity"),
        filter_radius=float(Optimizer.get("filter_radius_min", 0.0)),
    )
    fem.setup_boundary_conditions(loads_in, loads_out, supports)

    n_mat: int = fem.nb_mat
    volfrac: float = Dimensions.get("volfrac", 0.5)
    all_x, all_y, all_z = _active_coords(loads_in, loads_out, supports)

    # Initialize Material
    if multimaterial:
        percents: list[int] = Materials.get("percent", [100])
        x = initializers.initialize_materials(
            init_type=Materials.get("init_type", 0),
            materials_percentage=percents,
            volfrac=volfrac,
            nelx=fem.nelx,
            nely=fem.nely,
            nelz=fem.nelz,
            all_x=all_x,
            all_y=all_y,
            all_z=all_z,
            current_xPhys=current_xPhys,
        )
    else:
        x = initializers.initialize_material(
            init_type=Materials.get("init_type", 0),
            volfrac=volfrac,
            nelx=fem.nelx,
            nely=fem.nely,
            nelz=fem.nelz,
            all_x=all_x,
            all_y=all_y,
            all_z=all_z,
            current_xPhys=current_xPhys,
        )[np.newaxis, :]

    for i in range(n_mat):
        x[i] = fem.apply_regions(x[i], regions)
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
    ui: FloatArray = np.zeros((fem.ndof, len(loads_in)), dtype=np.float64)

    def _emit(iteration: int, obj_val: float, chg: float) -> bool:
        """Send a *copy* of the current densities to the callback (C7)."""
        if progress_callback is None:
            return False
        frame = xPhys.copy() if multimaterial else xPhys[0].copy()
        return progress_callback(iteration, obj_val, chg, frame)

    # Emit frame 0 (initial density)
    _emit(0, 0.0, 1.0)

    while change > 0.01 and loop < n_it:
        loop += 1
        xold: FloatArray = x.copy()

        # Finite element analysis
        ui, uo = fem.solve(xPhys if multimaterial else xPhys[0])

        # Objective value (for logging/convergence display)
        obj_val: float = fem.compute_objective(
            xPhys if multimaterial else xPhys[0], ui, uo
        )

        # Per-material sensitivity, filtering and OC update
        for i in range(n_mat):
            dc_i, dv_i = fem.compute_sensitivities(xPhys[i], ui, uo, mat_idx=i)
            x[i], g[i] = _oc(fem.nel, x[i], eta, max_change, dc_i, dv_i, g[i])
            xPhys[i] = fem.update_xPhys(x[i])
            xPhys[i] = fem.apply_regions(xPhys[i], regions)

        if multimaterial:
            # Partition-of-unity constraint: sum of densities <= 1 per element
            col_sums: FloatArray = xPhys.sum(axis=0)
            excess = col_sums > 1.0
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

        if _emit(loop, obj_val, change):
            if verbose:
                print("Optimization stopped by user.")
            break

    if verbose:
        print("Optimizer finished.")
    return xPhys, ui
