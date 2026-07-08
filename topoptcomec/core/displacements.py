# topoptcomec/core/displacements.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Linear displacement computation.

from __future__ import annotations
import dataclasses
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata

from topoptcomec.core import preset_format
from topoptcomec.core.fem import FEM
from topoptcomec.core.grid import StructuredGrid
from topoptcomec.core.model import Load, Support

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def single_linear_displacement(
    u: FloatArray, nelx: int, nely: int, nelz: int, disp_factor: float
) -> tuple[FloatArray, ...]:
    """
    Computes the deformed mesh grid for a single-frame plot.

    .. math::

        \\mathbf{x}'_i = \\mathbf{x}_i + s\\mathbf{u}_i

    where :math:`s` is `disp_factor`.

    Parameters
    ----------
    u : FloatArray
        Displacement vector from FEM solve, shape (ndof,) or (ndof, n_forces).
    nelx, nely, nelz : int
        Number of elements in each dimension.
    disp_factor : float
        Scaling factor to amplify displacements for visualization.

    Returns
    -------
    tuple[FloatArray, ...]
        X, Y, Z meshgrid arrays for plotting deformed geometry.
    """
    is_3d: bool = nelz > 0
    nodes_flat: FloatArray = np.arange(
        (nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1)
    )

    # Generate grid coordinates
    if is_3d:
        slice_size = (nelx + 1) * (nely + 1)
        z_coords: FloatArray = nodes_flat // slice_size
        rem: FloatArray = nodes_flat % slice_size
        x_coords: FloatArray = rem // (nely + 1)
        y_coords: FloatArray = rem % (nely + 1)
    else:
        x_coords: FloatArray = nodes_flat // (nely + 1)
        y_coords: FloatArray = nodes_flat % (nely + 1)

    # Calculate average displacement
    # u shape is (ndof, n_forces) or (ndof,)
    if u.ndim > 1:
        u_vec: FloatArray = np.mean(u, axis=1)
    else:
        u_vec: FloatArray = u

    # Map DOF indices to Node indices
    elemndof: int = 3 if is_3d else 2

    ux_avg: FloatArray = u_vec[elemndof * nodes_flat]
    uy_avg: FloatArray = u_vec[elemndof * nodes_flat + 1]

    X_flat: FloatArray = x_coords + ux_avg * disp_factor
    Y_flat: FloatArray = (
        y_coords - uy_avg * disp_factor
    )  # Minus for Y flip in plotting usually

    if is_3d:
        uz_avg: FloatArray = u_vec[elemndof * nodes_flat + 2]
        Z_flat: FloatArray = z_coords + uz_avg * disp_factor

        # Reshape to (Z, X, Y) then transpose to (X, Y, Z) for standard plotting tools
        shape: tuple[int, int, int] = (nelz + 1, nelx + 1, nely + 1)
        X: FloatArray = X_flat.reshape(shape).transpose(1, 2, 0)
        Y: FloatArray = Y_flat.reshape(shape).transpose(1, 2, 0)
        Z: FloatArray = Z_flat.reshape(shape).transpose(1, 2, 0)
        return X, Y, Z
    else:
        shape: tuple[int, int] = (nelx + 1, nely + 1)
        X: FloatArray = X_flat.reshape(shape)
        Y: FloatArray = Y_flat.reshape(shape)
        return X, Y


def _embed_material(
    xPhys_initial: FloatArray,
    is_multi: bool,
    small: StructuredGrid,
    large: StructuredGrid,
    mx: int,
    my: int,
    mz: int,
) -> tuple[FloatArray, int, FloatArray]:
    """
    Embed initial density into the expanded domain.

    .. math::

        \\rho^{large}_{m,e} =
        \\begin{cases}
            \\rho_{m,e}, & e \\in \\Omega_{original} \\\\
            0, & e \\in \\Omega_{padding}
        \\end{cases}

    Parameters
    ----------
    xPhys_initial : FloatArray
        Initial density field, shape (nel,) or (n_mat, nel).
    is_multi : bool
        Whether this is a multi-material optimization.
    small : StructuredGrid
        Original grid.
    large : StructuredGrid
        Enlarged (padded) grid.
    mx, my, mz : int
        Padding margins added to each side.

    Returns
    -------
    tuple[FloatArray, int, FloatArray]
        (xPhys, n_mat, volfrac) - Expanded density, material count, target volume fractions.
    """
    _x_init: FloatArray = xPhys_initial if is_multi else xPhys_initial[np.newaxis, :]
    n_mat: int = _x_init.shape[0]

    xPhys_large: FloatArray = np.zeros((n_mat, *large.spatial_shape))
    for i in range(n_mat):
        spatial = small.to_spatial(_x_init[i])
        if small.is_3d:
            # Spatial shape is (nelz, nelx, nely) - the FEM flat ordering.
            xPhys_large[
                i, mz : mz + small.nelz, mx : mx + small.nelx, my : my + small.nely
            ] = spatial
        else:
            xPhys_large[i, mx : mx + small.nelx, my : my + small.nely] = spatial

    xPhys: FloatArray = xPhys_large.reshape(n_mat, -1)
    volfrac: FloatArray = np.mean(xPhys, axis=1)
    return xPhys, n_mat, volfrac


def _crop_density(
    xPhys: FloatArray,
    n_mat: int,
    small: StructuredGrid,
    large: StructuredGrid,
    mx: int,
    my: int,
    mz: int,
) -> FloatArray:
    """
    Crop the expanded density field back to the original domain size.

    .. math::

        \\rho = \\rho^{large}\\rvert_{\\Omega_{original}}

    Parameters
    ----------
    xPhys : FloatArray
        Expanded density field, shape (n_mat, large.nel).
    n_mat : int
        Number of materials.
    small, large : StructuredGrid
        Original and enlarged grids.
    mx, my, mz : int
        Padding margins to remove.

    Returns
    -------
    FloatArray
        Cropped density field of shape (n_mat, nel).
    """
    cropped: FloatArray = np.zeros((n_mat, small.nel))
    for i in range(n_mat):
        spatial = large.to_spatial(xPhys[i])
        if small.is_3d:
            c = spatial[
                mz : mz + small.nelz, mx : mx + small.nelx, my : my + small.nely
            ]
        else:
            c = spatial[mx : mx + small.nelx, my : my + small.nely]
        cropped[i] = small.from_spatial(c)
    return cropped


def _reposition_loads(
    fem: FEM,
    loads_in: list[Load],
    loads_out: list[Load],
    supports: list[Support],
    grid: StructuredGrid,
    u_curr: FloatArray,
    delta_disp: float,
) -> list[Load]:
    """
    Move input load application points following the nodal displacement.

    .. math::

        \\mathbf{x}_{f}^{k+1} =
        \\Pi_{\\Omega}\\left(\\mathbf{x}_{f}^{k} +
        \\operatorname{round}(\\Delta s\\,\\mathbf{u}_f^k)\\right)

    where :math:`\\Pi_{\\Omega}` clips the point to the simulation domain.

    Parameters
    ----------
    fem : FEM
        FEM solver instance (boundary conditions are refreshed in place).
    loads_in, loads_out : list[Load]
        Current loads; a new input list is returned.
    supports : list[Support]
        Supports (unchanged, needed to refresh the BCs).
    grid : StructuredGrid
        Simulation (enlarged) grid used for clipping.
    u_curr : FloatArray
        Current displacement vector.
    delta_disp : float
        Incremental displacement per iteration.

    Returns
    -------
    list[Load]
        Updated input loads.
    """
    moved = False
    new_loads: list[Load] = []
    for i, load in enumerate(loads_in):
        base = fem.in_dofs[i] - load.axis  # x-DOF of the loaded node
        dx = round(u_curr[base] * delta_disp)
        dy = -round(u_curr[base + 1] * delta_disp)
        dz = round(u_curr[base + 2] * delta_disp) if grid.is_3d else 0
        if dx == 0 and dy == 0 and dz == 0:
            new_loads.append(load)
            continue
        moved = True
        new_loads.append(
            dataclasses.replace(
                load,
                x=max(0, min(grid.nelx, load.x + dx)),
                y=max(0, min(grid.nely, load.y + dy)),
                z=max(0, min(grid.nelz, load.z + dz)) if grid.is_3d else load.z,
            )
        )
    if moved:
        fem.setup_boundary_conditions(new_loads, loads_out, supports)
    return new_loads


def _warp_density(
    xPhys: FloatArray,
    n_mat: int,
    volfrac: FloatArray,
    points: FloatArray,
    points_interp: FloatArray,
    is_multi: bool,
    nel: int,
    k: float,
    nominator: float,
    c_val: float,
) -> None:
    """
    Interpolate density onto the warped grid and renormalize.

    Densities are interpolated from displaced element centers and sharpened
    with a smooth Heaviside-like sigmoid:

    .. math::

        \\tilde{\\rho}_e =
        \\frac{a}{1 + \\exp(-k(\\rho_e - 0.5))} + c

    A volume correction then enforces:

    .. math::

        \\frac{1}{n_e}\\sum_e \\tilde{\\rho}_{m,e} = V_m^*

    Parameters
    ----------
    xPhys : FloatArray
        Density field to warp (modified in place).
    n_mat : int
        Number of materials.
    volfrac : FloatArray
        Target volume fractions per material.
    points : FloatArray
        Warped grid points (current positions).
    points_interp : FloatArray
        Original grid points for interpolation.
    is_multi : bool
        Whether this is multi-material optimization.
    nel : int
        Number of elements in the simulation grid.
    k, nominator, c_val : float
        Sigmoid filter parameters.
    """
    for i in range(n_mat):
        # Interpolate density from old points to new (warped) points
        xPhys[i] = np.nan_to_num(
            griddata(points, xPhys[i], points_interp, method="linear"), nan=0.0
        )

        # Threshold & Normalize to prevent material spread (Sigmoid filter)
        xPhys[i] = nominator / (1 + np.exp(-k * (xPhys[i] - 0.5))) + c_val
        curr_sum: float = np.sum(xPhys[i])
        if curr_sum > 0:
            xPhys[i] = volfrac[i] * xPhys[i] / (curr_sum / nel)
        xPhys[i] = np.clip(xPhys[i], 0.0, 1.0)

    # Partition-of-unity constraint for multi-material
    if is_multi:
        col_sums: FloatArray = xPhys.sum(axis=0)
        excess: np.ndarray = col_sums > 1.0
        if np.any(excess):
            xPhys[:, excess] /= col_sums[excess]


def _translated_bcs(
    params: dict, is_3d: bool, mx: int, my: int, mz: int
) -> tuple[list[Load], list[Load], list[Support]]:
    """Parse boundary conditions and offset them into the enlarged domain."""
    loads_in = preset_format.parse_loads(
        params["Forces"], "fix", "fiy", "fiz", "fidir", "finorm", is_3d
    )
    loads_out = preset_format.parse_loads(
        params["Forces"], "fox", "foy", "foz", "fodir", "fonorm", is_3d
    )
    supports = preset_format.parse_supports(params.get("Supports"), is_3d)
    loads_in = [
        dataclasses.replace(ld, x=ld.x + mx, y=ld.y + my, z=ld.z + mz)
        for ld in loads_in
    ]
    loads_out = [
        dataclasses.replace(ld, x=ld.x + mx, y=ld.y + my, z=ld.z + mz)
        for ld in loads_out
    ]
    supports = [
        dataclasses.replace(s, x=s.x + mx, y=s.y + my, z=s.z + mz) for s in supports
    ]
    return loads_in, loads_out, supports


def run_iterative_displacement(
    params: dict,
    xPhys_initial: FloatArray,
    progress_callback: Callable[[int], bool] | None = None,
) -> FloatArray:
    """
    Performs iterative FE analysis to simulate displacement.

    Each iteration solves the linear state equation, moves element centers by
    average nodal displacement, and transports density:

    .. math::

        \\mathbf{x}_e^{k+1} =
        \\mathbf{x}_e^k + \\Delta s\\,
        \\frac{1}{n_n}\\sum_{i \\in e}\\mathbf{u}_i^k

    The density is then interpolated back onto the Eulerian grid and clipped to
    preserve admissible topology variables.

    Parameters
    ----------
    params : dict
        Simulation parameters (Dimensions, Forces, Materials, etc.).
    xPhys_initial : FloatArray
        Initial density field.
    progress_callback : Callable[[int], bool] | None
        Optional callback to report progress and allow cancellation.

    Yields
    ------
    FloatArray
        Cropped density field for each iteration.
    """
    dims: dict = params["Dimensions"]
    nelx, nely, nelz = (int(v) for v in dims["nelxyz"])
    small = StructuredGrid(nelx, nely, max(0, nelz))
    is_3d: bool = small.is_3d
    is_multi: bool = hasattr(xPhys_initial, "ndim") and xPhys_initial.ndim > 1

    # Enlarge domain (approx 20% total padding)
    mx: int = nelx // 5
    my: int = nely // 5
    mz: int = nelz // 5 if is_3d else 0
    large = StructuredGrid(nelx + 2 * mx, nely + 2 * my, nelz + 2 * mz)

    # Translate boundary conditions and offset them into the enlarged domain
    loads_in, loads_out, supports = _translated_bcs(params, is_3d, mx, my, mz)

    # Initialize FEM on the enlarged grid
    materials: dict = params["Materials"]
    optimizer: dict = params["Optimizer"]
    fem: FEM = FEM(
        large,
        E=materials.get("E", [1.0]),
        nu=materials.get("nu", [0.3]),
        penal=float(optimizer.get("penal", 3.0)),
        solver=optimizer.get("solver", "Auto"),
        filter_type=optimizer.get("filter_type", "Sensitivity"),
        filter_radius=float(optimizer.get("filter_radius_min", 0.0)),
    )
    fem.setup_boundary_conditions(loads_in, loads_out, supports)

    # Embed Material into Expanded Domain
    xPhys, n_mat, volfrac = _embed_material(
        xPhys_initial, is_multi, small, large, mx, my, mz
    )

    # Simulation Parameters
    pd: dict = params["Displacement"]
    delta_disp: float = pd["disp_factor"] / max(1, pd["disp_iterations"])

    # Points for interpolation (Eulerian grid: element centers)
    points_interp: FloatArray = large.element_centers()

    # Initial Yield
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)

    n_nodes_per_el: int = 8 if is_3d else 4

    # Sigmoid consts (Calculated outside loop for efficiency)
    k: float = 4  # Steepness
    nominator: float = (
        (1 + np.exp(-k / 2)) * (1 + np.exp(k / 2)) / (np.exp(k / 2) - np.exp(-k / 2))
    )
    c_val: float = -nominator / (1 + np.exp(k / 2))

    # Iterative Loop
    for it in range(pd["disp_iterations"]):
        # Solve FEM to get the deformation
        ui, _ = fem.solve(xPhys if is_multi else xPhys[0])

        # Collapse multiple load cases to average if necessary
        u_curr: FloatArray = (
            np.mean(ui, axis=1) if ui.shape[1] > 0 else np.zeros(fem.ndof)
        )

        # Move the input loads with the deforming structure
        loads_in = _reposition_loads(
            fem, loads_in, loads_out, supports, large, u_curr, delta_disp
        )

        # Get nodal displacements for every element
        u_elem: FloatArray = u_curr[fem.edofMat].reshape(
            fem.nel, n_nodes_per_el, fem.elemndof
        )

        # Displace points
        u_avg: FloatArray = np.mean(u_elem, axis=1)  # (nel, dim)
        points: FloatArray = points_interp.copy()
        points[:, 0] += u_avg[:, 0] * delta_disp
        points[:, 1] -= u_avg[:, 1] * delta_disp
        if is_3d:
            points[:, 2] += u_avg[:, 2] * delta_disp
        moved: bool = not np.allclose(points, points_interp, atol=1e-14)
        if moved:
            _warp_density(
                xPhys,
                n_mat,
                volfrac,
                points,
                points_interp,
                is_multi,
                fem.nel,
                k,
                nominator,
                c_val,
            )

        # Crop and Yield
        cropped: FloatArray = _crop_density(xPhys, n_mat, small, large, mx, my, mz)

        # Strip the extra dimension off if it was just a single material
        yield cropped if is_multi else cropped[0]
        if progress_callback:
            progress_callback(it + 2)
