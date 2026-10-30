# topoptcomec/core/optimizers.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Topology Optimizers.

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import numpy.typing as npt

from topoptcomec.core import initializers, preset_format
from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Region, Support

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


#: Lower bound of the design variables.
RHO_MIN: float = 1e-6
#: Convergence threshold on the maximum design-variable change.
CONVERGENCE_TOLERANCE: float = 0.01

VolumeFunction = Callable[[FloatArray], float]


def _oc(
    nel: int,
    x: FloatArray,
    eta: float,
    max_change: float,
    dc: FloatArray,
    dv: FloatArray,
    target: float,
    free: npt.NDArray[np.bool_] | None = None,
    upper: FloatArray | float = 1.0,
    volume: VolumeFunction | None = None,
) -> tuple[FloatArray, float]:
    """
    Optimality Criterion (OC) update scheme.

    The update approximates the KKT stationarity condition through a bisection
    search on the Lagrange multiplier:

    .. math::

        x_e^{new} =
        \\operatorname{clip}_{[\\max(\\rho_{\\min}, x_e-m),\\ \\min(u_e, x_e+m)]}
        \\left(
            x_e\\left(-\\frac{\\partial C / \\partial x_e}
            {\\lambda\\,\\partial V / \\partial x_e}\\right)^\\eta
        \\right)

    The multiplier is chosen so that ``volume(x_new)`` equals ``target``: the
    constraint is enforced exactly, not through an accumulated linearization.
    Passive elements (``free == False``) keep their value. The per-element
    upper bound ``u_e`` enforces the multi-material partition of unity.

    Args:
        nel: Total number of elements.
        x: Current design variables (densities).
        eta: Damping exponent of the OC update.
        max_change: Maximum allowed change in design variables per iteration.
        dc: Sensitivities of the objective function.
        dv: Sensitivities of the volume constraint.
        target: Required value of ``volume(x_new)``.
        free: Boolean mask of designable elements (all elements when omitted).
        upper: Upper bound of the design variables (scalar or per element).
        volume: Volume measure; defaults to the sum of the free design
            variables. Must be non-decreasing in every variable.

    Returns:
        A tuple containing the new design variables (xnew) and the residual
        ``volume(xnew) - target`` of the volume constraint.
    """
    if free is None:
        free = np.ones(nel, dtype=bool)
    if volume is None:

        def volume(values: FloatArray) -> float:
            return float(np.sum(values[free]))

    dv_safe: FloatArray = np.maximum(dv, 1e-12)
    lower: FloatArray = np.maximum(RHO_MIN, x - max_change)
    high: FloatArray = np.minimum(upper, x + max_change)
    lower = np.minimum(lower, high)

    def _candidate(lmid: float) -> FloatArray:
        ratio = np.maximum(0.1, -dc / dv_safe / lmid) ** eta
        candidate = np.clip(x * ratio, lower, high)
        candidate[~free] = x[~free]
        return candidate

    l1: float = 0.0
    l2: float = 1e9
    xnew: FloatArray = x.copy()
    gt: float = 0.0
    for _ in range(200):
        if not ((l2 - l1) / (l1 + l2) > 1e-6 and l2 > 1e-40):
            break
        lmid: float = 0.5 * (l2 + l1)
        xnew = _candidate(lmid)
        gt = volume(xnew) - target
        if gt > 0:
            l1 = lmid
        else:
            l2 = lmid
    return xnew, gt


def _project_volume(
    x: FloatArray,
    free: npt.NDArray[np.bool_],
    target: float,
    upper: FloatArray | float,
    volume: VolumeFunction,
) -> FloatArray:
    """
    Scale the free design variables so that ``volume(x)`` equals ``target``.

    Used once on the initial design so that the very first finite element
    analysis already satisfies the volume constraint.

    Parameters
    ----------
    x : FloatArray
        Design variables of one material, shape (nel,).
    free : npt.NDArray[np.bool_]
        Mask of designable elements.
    target : float
        Required value of ``volume``.
    upper : FloatArray | float
        Upper bound of the design variables.
    volume : VolumeFunction
        Non-decreasing volume measure.

    Returns
    -------
    FloatArray
        Projected design variables (a new array).
    """
    base = np.maximum(np.asarray(x, dtype=np.float64), RHO_MIN)
    upper_arr = np.broadcast_to(np.asarray(upper, dtype=np.float64), base.shape)

    def candidate(alpha: float) -> FloatArray:
        out = base.copy()
        out[free] = np.clip(alpha * base[free], RHO_MIN, upper_arr[free])
        return out

    lo, hi = 0.0, 1.0
    while volume(candidate(hi)) < target and hi < 1e12:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if volume(candidate(mid)) < target:
            lo = mid
        else:
            hi = mid
    return candidate(0.5 * (lo + hi))


def _material_shares(percent: Sequence[float] | None, n_mat: int) -> FloatArray:
    """Normalized per-material volume shares (sum to one)."""
    if percent is None or len(percent) != n_mat:
        return np.full(n_mat, 1.0 / n_mat, dtype=np.float64)
    shares = np.asarray([float(p) for p in percent], dtype=np.float64)
    total = float(shares.sum())
    if total <= 0.0:
        return np.full(n_mat, 1.0 / n_mat, dtype=np.float64)
    return shares / total


def _partition_upper(x: FloatArray, mat_idx: int) -> FloatArray:
    """Upper bound keeping the per-element material sum at most one."""
    others = np.delete(x, mat_idx, axis=0).sum(axis=0) if x.shape[0] > 1 else 0.0
    return np.maximum(RHO_MIN, 1.0 - others)


class _DesignSpace:
    """Physical field and volume measure shared by the OC update steps.

    Parameters
    ----------
    fem : FEM
        Configured solver (filter and grid).
    regions : list[Region]
        Passive geometric constraints.
    free : npt.NDArray[np.bool_]
        Mask of designable elements.
    """

    def __init__(
        self, fem: FEM, regions: list[Region], free: npt.NDArray[np.bool_]
    ) -> None:
        self.fem = fem
        self.regions = regions
        self.free = free

    def physical(self, x_m: FloatArray, mat_idx: int) -> FloatArray:
        """Physical density of one material (filter, then passive regions)."""
        return self.fem.apply_regions(
            self.fem.update_xPhys(x_m), self.regions, mat_idx=mat_idx
        )

    def volume_function(self, mat_idx: int) -> VolumeFunction:
        """Sum of the physical density of one material over the free elements."""

        def volume(x_m: FloatArray) -> float:
            return float(np.sum(self.physical(x_m, mat_idx)[self.free]))

        return volume

    def physical_field(self, x: FloatArray) -> FloatArray:
        """Physical density of every material, shape (n_mat, nel)."""
        return np.vstack([self.physical(x[m], m) for m in range(x.shape[0])])


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


def _volume_targets(
    fem: FEM, regions: list[Region], Materials: dict, volfrac: float
) -> tuple[npt.NDArray[np.bool_], FloatArray]:
    """
    Designable elements and per-material volume bound.

    A region makes the covered elements non-designable for *every* material
    (solid for the targeted material, void for the others). Passive elements
    are excluded from the volume budget: ``volfrac`` bounds the mean density
    of the *designable* elements.

    Parameters
    ----------
    fem : FEM
        Configured solver.
    regions : list[Region]
        Passive geometric constraints.
    Materials : dict
        Legacy materials section (``percent`` shares).
    volfrac : float
        Volume fraction bound.

    Returns
    -------
    tuple
        ``(free, targets)``: mask of designable elements and the bound on the
        sum of the physical densities over them, per material.
    """
    n_mat = fem.nb_mat
    passive = np.zeros(fem.nel, dtype=bool)
    for m in range(n_mat):
        solid, void = fem.region_masks(regions, mat_idx=m)
        passive |= solid | void
    free: npt.NDArray[np.bool_] = ~passive
    shares = _material_shares(Materials.get("percent"), n_mat)
    return free, volfrac * shares * int(np.count_nonzero(free))


def _initialize_design(
    fem: FEM,
    regions: list[Region],
    Materials: dict,
    volfrac: float,
    free: npt.NDArray[np.bool_],
    targets: FloatArray,
    current_xPhys: FloatArray | None,
    multimaterial: bool,
    all_x: FloatArray,
    all_y: FloatArray,
    all_z: FloatArray,
) -> tuple[FloatArray, FloatArray]:
    """
    Build the initial design variables and physical density field.

    Regions are applied, every material is projected onto its target volume
    (over the designable elements) and the physical field is computed.

    Parameters
    ----------
    fem : FEM
        Configured solver (grid, materials, filter).
    regions : list[Region]
        Passive geometric constraints.
    Materials : dict
        Legacy materials section (``init_type``, ``percent``).
    volfrac : float
        Target mean density of the designable elements.
    free : npt.NDArray[np.bool_]
        Mask of designable elements.
    targets : FloatArray
        Target physical volume (sum over free elements) per material.
    current_xPhys : FloatArray | None
        Starting field for ``init_type == 3``.
    multimaterial : bool
        Whether several materials are optimized.
    all_x, all_y, all_z : FloatArray
        Active coordinates for the distance initializer.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Design variables and physical density, both shape (n_mat, nel).
    """
    n_mat = fem.nb_mat
    if multimaterial:
        x = initializers.initialize_materials(
            init_type=Materials.get("init_type", 0),
            materials_percentage=Materials.get("percent", [100]),
            volfrac=volfrac,
            nelx=fem.nelx,
            nely=fem.nely,
            nelz=fem.nelz,
            all_x=all_x,
            all_y=all_y,
            all_z=all_z,
            current_xPhys=current_xPhys,
        )
        if x is None:
            raise ValueError("Material percentages must sum up to 100.")
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

    x = np.maximum(np.asarray(x, dtype=np.float64), RHO_MIN)
    for i in range(n_mat):
        x[i] = fem.apply_regions(x[i], regions, mat_idx=i)
    # Partition of unity in design space, then project every material onto its
    # target physical volume so the first analysis already satisfies it.
    column_sums = x.sum(axis=0)
    over = column_sums > 1.0
    if np.any(over):
        x[:, over] /= column_sums[over]
    space = _DesignSpace(fem, regions, free)
    for i in range(n_mat):
        x[i] = _project_volume(
            x[i],
            free,
            float(targets[i]),
            _partition_upper(x, i),
            space.volume_function(i),
        )
    return x, space.physical_field(x)


def _update_design(
    fem: FEM,
    x: FloatArray,
    xPhys: FloatArray,
    ui: FloatArray,
    uo: FloatArray,
    space: _DesignSpace,
    targets: FloatArray,
    eta: float,
    max_change: float,
) -> tuple[FloatArray, FloatArray]:
    """
    One OC update of every material plus the physical field update.

    Materials are updated one after the other; each update is bounded by the
    material left free by the others, so the per-element sum of the design
    variables (and therefore of the filtered physical densities) never
    exceeds one, while every material meets its own physical volume target.

    Parameters
    ----------
    fem : FEM
        Configured solver.
    x : FloatArray
        Current design variables, shape (n_mat, nel).
    xPhys : FloatArray
        Physical densities of the analysed design, shape (n_mat, nel). The
        sensitivities are evaluated at these densities.
    ui, uo : FloatArray
        Displacement solutions of the current design.
    space : _DesignSpace
        Physical field and volume measure.
    targets : FloatArray
        Required physical volume (sum over free elements) per material.
    eta, max_change : float
        OC damping exponent and move limit.

    Returns
    -------
    tuple[FloatArray, FloatArray]
        Updated design variables and physical density field.
    """
    x = x.copy()
    sensitivities = [
        fem.compute_sensitivities(xPhys[i], ui, uo, mat_idx=i)
        for i in range(fem.nb_mat)
    ]
    for i, (dc_i, dv_i) in enumerate(sensitivities):
        x[i], _ = _oc(
            fem.nel,
            x[i],
            eta,
            max_change,
            dc_i,
            dv_i,
            float(targets[i]),
            space.free,
            upper=_partition_upper(x, i),
            volume=space.volume_function(i),
        )
    return x, space.physical_field(x)


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
    multimaterial: bool = False,
) -> tuple[FloatArray, FloatArray]:
    """
    Run single- or multi-material topology optimization.

    Performs finite element analysis, sensitivity analysis, filtering,
    and Optimality Criteria (OC) updates until convergence.

    Args:
        Dimensions: Geometry and mesh parameters.
        Forces: Applied loads.
        Materials: Material properties.
        Optimizer: Optimization parameters.
        Supports: Boundary conditions.
        Regions: Passive/active regions.
        progress_callback: Called with (iteration, objective, change,
            xPhys); return True to stop early.
        verbose: Print optimization progress.
        current_xPhys: Optional initial density field.
        multimaterial: Run the multi-material optimizer.

    Returns:
        xPhys: Final physical density field.
        ui: Final displacement vector.
    """
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
    volfrac: float = float(Dimensions.get("volfrac", 0.5))
    all_x, all_y, all_z = _active_coords(loads_in, loads_out, supports)

    free, targets = _volume_targets(fem, regions, Materials, volfrac)

    x, xPhys = _initialize_design(
        fem,
        regions,
        Materials,
        volfrac,
        free,
        targets,
        current_xPhys,
        multimaterial,
        all_x,
        all_y,
        all_z,
    )

    space = _DesignSpace(fem, regions, free)

    # Optimization Params
    eta: float = Optimizer.get("eta", 1.0)
    max_change: float = Optimizer.get("max_change", 0.1)
    n_it: int = int(Optimizer.get("n_it", 30))

    loop: int = 0
    change: float = 1.0
    ui: FloatArray = np.zeros(fem.ndof, dtype=np.float64)

    def _density_argument() -> FloatArray:
        """Density field in the shape expected by the FEM."""
        return xPhys if multimaterial else xPhys[0]

    def _emit(iteration: int, obj_val: float, chg: float) -> bool:
        """Send a *copy* of the current densities to the callback (C7)."""
        if progress_callback is None:
            return False
        frame = xPhys.copy() if multimaterial else xPhys[0].copy()
        return progress_callback(iteration, obj_val, chg, frame)

    # Emit frame 0 (initial density)
    _emit(0, 0.0, 1.0)

    stopped: bool = False
    while change > CONVERGENCE_TOLERANCE and loop < n_it:
        loop += 1
        xold: FloatArray = x.copy()

        # Finite element analysis
        ui, uo = fem.solve(_density_argument())

        # Objective value (for logging/convergence display)
        obj_val: float = fem.compute_objective(_density_argument(), ui, uo)

        x, xPhys = _update_design(
            fem, x, xPhys, ui, uo, space, targets, eta, max_change
        )

        # Check Convergence
        change = float(np.max(np.abs(x - xold)))
        if verbose:
            print(
                f"It.: {loop:3d}, Obj.: {obj_val:.4f}, "
                + ", ".join(f"V{i}: {xPhys[i].mean():.3f}" for i in range(n_mat))
                + f", Ch.: {change:.3f}"
            )

        if _emit(loop, obj_val, change):
            stopped = True
            if verbose:
                print("Optimization stopped by user.")
            break

    # Re-solve on the *final* design so the returned displacement matches the
    # returned density field (the loop solves before updating xPhys).
    ui, uo = fem.solve(_density_argument())
    final_obj: float = fem.compute_objective(_density_argument(), ui, uo)
    converged: bool = change <= CONVERGENCE_TOLERANCE
    if not converged and not stopped:
        message = (
            f"Optimization stopped at the iteration limit (n_it={n_it}) without "
            f"converging (last change {change:.3f} > 0.010)."
        )
        print(f"Warning: {message} Increase n_it for a converged result.")
    if verbose:
        print(f"Optimizer finished. Final objective: {final_obj:.4f}")
    # Emit the final design (same iteration index; consumers replace the
    # previous entry for that iteration when logging the objective).
    if not stopped:
        _emit(loop, final_obj, change)
    return (xPhys if multimaterial else xPhys[0]), ui
