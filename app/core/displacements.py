# app/core/displacements.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Linear displacement computation.

from __future__ import annotations
import copy
from collections.abc import Callable
import numpy as np
import numpy.typing as npt
from scipy.interpolate import griddata
from app.core.fem import FEM

# Type aliases
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]


def single_linear_displacement(
    u: np.ndarray, nelx: int, nely: int, nelz: int, disp_factor: float
) -> tuple[np.ndarray, ...]:
    """
    Computes the deformed mesh grid for a single-frame plot.

    Parameters
    ----------
    u : np.ndarray
        Displacement vector from FEM solve, shape (ndof,) or (ndof, n_forces).
    nelx, nely, nelz : int
        Number of elements in each dimension.
    disp_factor : float
        Scaling factor to amplify displacements for visualization.

    Returns
    -------
    tuple[np.ndarray, ...]
        X, Y, Z meshgrid arrays for plotting deformed geometry.
    """
    is_3d: bool = nelz > 0
    nodes_flat: np.ndarray = np.arange(
        (nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1)
    )

    # Generate grid coordinates
    if is_3d:
        slice_size = (nelx + 1) * (nely + 1)
        z_coords: np.ndarray = nodes_flat // slice_size
        rem: np.ndarray = nodes_flat % slice_size
        x_coords: np.ndarray = rem // (nely + 1)
        y_coords: np.ndarray = rem % (nely + 1)
    else:
        x_coords = nodes_flat // (nely + 1)
        y_coords = nodes_flat % (nely + 1)

    # Calculate average displacement
    # u shape is (ndof, n_forces) or (ndof,)
    if u.ndim > 1:
        u_vec: np.ndarray = np.mean(u, axis=1)
    else:
        u_vec = u

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
        X: np.ndarray = X_flat.reshape(shape).transpose(1, 2, 0)
        Y: np.ndarray = Y_flat.reshape(shape).transpose(1, 2, 0)
        Z: np.ndarray = Z_flat.reshape(shape).transpose(1, 2, 0)
        return X, Y, Z
    else:
        shape = (nelx + 1, nely + 1)
        X = X_flat.reshape(shape)
        Y = Y_flat.reshape(shape)
        return X, Y


def _embed_material(
    xPhys_initial: FloatArray,
    is_multi: bool,
    is_3d: bool,
    nelx: int,
    nely: int,
    nelz: int,
    mx: int,
    my: int,
    mz: int,
    fem: FEM,
) -> tuple[FloatArray, int, FloatArray]:
    """
    Embed initial density into the expanded domain.

    Parameters
    ----------
    xPhys_initial : FloatArray
        Initial density field, shape (nel,) or (n_mat, nel).
    is_multi : bool
        Whether this is a multi-material optimization.
    is_3d : bool
        Whether this is a 3D problem.
    nelx, nely, nelz : int
        Original element counts in each dimension.
    mx, my, mz : int
        Padding margins added to each side.
    fem : FEM
        FEM solver instance with enlarged domain.

    Returns
    -------
    tuple[FloatArray, int, FloatArray]
        (xPhys, n_mat, volfrac) - Expanded density, material count, target volume fractions.
    """
    _x_init: FloatArray = xPhys_initial if is_multi else xPhys_initial[np.newaxis, :]
    n_mat: int = _x_init.shape[0]

    full_shape: tuple[int, int, int] = (
        (fem.nelx, fem.nely, fem.nelz) if is_3d else (fem.nelx, fem.nely)
    )
    xPhys_large: FloatArray = np.zeros((n_mat, *full_shape))

    for i in range(n_mat):
        if is_3d:
            xPhys_large[i, mx : mx + nelx, my : my + nely, mz : mz + nelz] = _x_init[
                i
            ].reshape((nelx, nely, nelz), order="C")
        else:
            xPhys_large[i, mx : mx + nelx, my : my + nely] = _x_init[i].reshape(
                (nelx, nely), order="C"
            )

    xPhys: FloatArray = xPhys_large.reshape(n_mat, -1, order="C")
    volfrac: FloatArray = np.mean(xPhys, axis=1)
    return xPhys, n_mat, volfrac


def _reposition_forces(
    fem: FEM, sim_params: dict, u_curr: FloatArray, delta_disp: float, is_3d: bool
) -> None:
    """
    Move force application points following the full nodal displacement.

    Parameters
    ----------
    fem : FEM
        FEM solver instance.
    sim_params : dict
        Simulation parameters to update with new force positions.
    u_curr : FloatArray
        Current displacement vector.
    delta_disp : float
        Incremental displacement per iteration.
    is_3d : bool
        Whether this is a 3D problem.
    """
    snelx: int = sim_params["Dimensions"]["nelxyz"][0]
    snely: int = sim_params["Dimensions"]["nelxyz"][1]
    snelz: int = sim_params["Dimensions"]["nelxyz"][2]
    forces_moved: bool = False
    for i, f_idx in enumerate(fem.fi_indices):
        dof: int = fem.di_indices[i]
        fdir: str = sim_params["Forces"]["fidir"][f_idx]
        if "X" in fdir:
            dx: int = round(u_curr[dof] * delta_disp)
            dy: int = -round(u_curr[dof + 1] * delta_disp)
            dz: int = round(u_curr[dof + 2] * delta_disp) if is_3d else 0
        elif "Y" in fdir:
            dx = round(u_curr[dof - 1] * delta_disp)
            dy = -round(u_curr[dof] * delta_disp)
            dz = round(u_curr[dof + 1] * delta_disp) if is_3d else 0
        elif "Z" in fdir:
            dx = round(u_curr[dof - 2] * delta_disp)
            dy = -round(u_curr[dof - 1] * delta_disp)
            dz = round(u_curr[dof] * delta_disp)
        else:
            continue
        if dx != 0 or dy != 0 or dz != 0:
            sim_params["Forces"]["fix"][f_idx] = max(
                0, min(snelx, sim_params["Forces"]["fix"][f_idx] + dx)
            )
            sim_params["Forces"]["fiy"][f_idx] = max(
                0, min(snely, sim_params["Forces"]["fiy"][f_idx] + dy)
            )
            if is_3d:
                sim_params["Forces"]["fiz"][f_idx] = max(
                    0, min(snelz, sim_params["Forces"]["fiz"][f_idx] + dz)
                )
            forces_moved = True
    if forces_moved:
        fem.setup_boundary_conditions(sim_params["Forces"], sim_params.get("Supports"))


def _warp_density(
    xPhys: FloatArray,
    n_mat: int,
    volfrac: FloatArray,
    points: FloatArray,
    points_interp: FloatArray,
    is_multi: bool,
    fem: FEM,
    k: float,
    nominator: float,
    c_val: float,
) -> None:
    """
    Interpolate density onto the warped grid and renormalize.

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
    fem : FEM
        FEM solver instance.
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
            xPhys[i] = volfrac[i] * xPhys[i] / (curr_sum / fem.nel)
        xPhys[i] = np.clip(xPhys[i], 0.0, 1.0)

    # Partition-of-unity constraint for multi-material
    if is_multi:
        col_sums: FloatArray = xPhys.sum(axis=0)
        excess: np.ndarray = col_sums > 1.0
        if np.any(excess):
            xPhys[:, excess] /= col_sums[excess]


def _crop_density(
    xPhys: FloatArray,
    n_mat: int,
    is_3d: bool,
    fem: FEM,
    nelx: int,
    nely: int,
    nelz: int,
    mx: int,
    my: int,
    mz: int,
) -> FloatArray:
    """
    Crop the expanded density field back to the original domain size.

    Parameters
    ----------
    xPhys : FloatArray
        Expanded density field.
    n_mat : int
        Number of materials.
    is_3d : bool
        Whether this is a 3D problem.
    fem : FEM
        FEM solver instance with expanded domain.
    nelx, nely, nelz : int
        Original element counts.
    mx, my, mz : int
        Padding margins to remove.

    Returns
    -------
    FloatArray
        Cropped density field of shape (n_mat, nel) or (nel,).
    """
    cropped: FloatArray = np.zeros((n_mat, nelx * nely * (nelz if is_3d else 1)))
    for i in range(n_mat):
        if is_3d:
            c: np.ndarray = xPhys[i].reshape((fem.nelx, fem.nely, fem.nelz), order="C")[
                mx : mx + nelx, my : my + nely, mz : mz + nelz
            ]
        else:
            c = xPhys[i].reshape((fem.nelx, fem.nely), order="C")[
                mx : mx + nelx, my : my + nely
            ]
        cropped[i] = c.flatten(order="C")
    return cropped


def run_iterative_displacement(
    params: dict,
    xPhys_initial: FloatArray,
    progress_callback: Callable[[int], bool] | None = None,
) -> FloatArray:
    """
    Performs iterative FE analysis to simulate displacement.

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
    nelx: int = dims["nelxyz"][0]
    nely: int = dims["nelxyz"][1]
    nelz: int = dims["nelxyz"][2]
    is_3d: bool = nelz > 0
    is_multi: bool = hasattr(xPhys_initial, "ndim") and xPhys_initial.ndim > 1

    # Enlarge domain (approx 20% total padding)
    mx: int = nelx // 5
    my: int = nely // 5
    mz: int = nelz // 5 if is_3d else 0
    sim_params: dict = copy.deepcopy(params)
    sim_params["Dimensions"]["nelxyz"] = [nelx + 2 * mx, nely + 2 * my, nelz + 2 * mz]

    # Offset Supports and Forces coordinates to center the part in the new domain
    def offset_coords(container: dict, keys: list[str]) -> None:
        for k, o in zip(keys, [mx, my, mz]):
            if k in container:
                container[k] = [val + o for val in container[k]]

    if "Supports" in sim_params:
        offset_coords(sim_params["Supports"], ["sx", "sy", "sz"])
    offset_coords(sim_params["Forces"], ["fix", "fiy", "fiz"])

    # Initialize FEM
    fem: FEM = FEM(
        sim_params["Dimensions"], sim_params["Materials"], sim_params["Optimizer"]
    )
    fem.setup_boundary_conditions(sim_params["Forces"], sim_params.get("Supports"))

    # Embed Material into Expanded Domain
    xPhys: FloatArray
    n_mat: int
    volfrac: FloatArray
    xPhys, n_mat, volfrac = _embed_material(
        xPhys_initial, is_multi, is_3d, nelx, nely, nelz, mx, my, mz, fem
    )

    # Simulation Parameters
    pd: dict = params["Displacement"]
    delta_disp: float = pd["disp_factor"] / max(1, pd["disp_iterations"])

    # Points for interpolation (Eulerian grid)
    shape: tuple[int, int, int] = (
        (fem.nelz, fem.nelx, fem.nely) if is_3d else (fem.nelx, fem.nely)
    )
    unrvld: tuple[np.ndarray, ...] = np.unravel_index(np.arange(fem.nel), shape)
    points_interp: FloatArray = (
        np.column_stack((unrvld[1], unrvld[2], unrvld[0]) if is_3d else unrvld) + 0.5
    )

    # Initial Yield
    yield xPhys_initial
    if progress_callback:
        progress_callback(1)

    node_ids: list[int] = [2, 1, 6, 5, 3, 0, 7, 4] if is_3d else [2, 1, 3, 0]

    # Sigmoid consts (Calculated outside loop for efficiency)
    k: float = 4  # Steepness
    nominator: float = (
        (1 + np.exp(-k / 2)) * (1 + np.exp(k / 2)) / (np.exp(k / 2) - np.exp(-k / 2))
    )
    c_val: float = -nominator / (1 + np.exp(k / 2))

    # Iterative Loop
    for it in range(pd["disp_iterations"]):
        # Solve FEM to get the deformation
        ui, _ = fem.solve(xPhys)

        # Collapse multiple load cases to average if necessary
        u_curr: FloatArray = (
            np.mean(ui, axis=1) if ui.shape[1] > 0 else np.zeros(fem.ndof)
        )

        # Replace the Forces
        _reposition_forces(fem, sim_params, u_curr, delta_disp, is_3d)

        # Get nodal displacements for every element
        u_elem: FloatArray = u_curr[fem.edofMat].reshape(
            fem.nel, len(node_ids), fem.elemndof
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
                fem,
                k,
                nominator,
                c_val,
            )

        # Crop and Yield
        cropped: FloatArray = _crop_density(
            xPhys, n_mat, is_3d, fem, nelx, nely, nelz, mx, my, mz
        )

        # Strip the extra dimension off if it was just a single material
        yield cropped if is_multi else cropped[0]
        if progress_callback:
            progress_callback(it + 2)
