# topoptcomec/ui/exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Export result to various file formats.

from __future__ import annotations

import csv
import ctypes
import warnings
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pyvista as pv
import vtk
from stl import mesh
from vtk.util.numpy_support import get_vtk_array_type, numpy_to_vtk

from topoptcomec.core.grid import StructuredGrid

try:  # lib3mf is optional: PNG/STL/VTI export must work without it.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning, module="lib3mf")
        import lib3mf
except ImportError:  # pragma: no cover - platform without a lib3mf wheel
    lib3mf = None  # type: ignore[assignment]

# Type aliases
FloatArray = npt.NDArray[np.float64]

# pv.ImageData replaced pv.UniformGrid in PyVista 0.44
_ImageData = pv.ImageData if hasattr(pv, "ImageData") else pv.UniformGrid

_VOXEL_OPACITY_MIN = 0.4
_VOXEL_OPACITY_MAX = 0.95


def _hex_to_rgb_int(color: str) -> tuple[int, int, int]:
    """
    Convert a ``#RRGGBB`` hex string to an ``(r, g, b)`` tuple in 0-255.

    Parameters
    ----------
    color : str
        Hex color string (``"#FF0000"`` or ``"FF0000"``).

    Returns
    -------
    tuple[int, int, int]
        RGB components as integers in the 0-255 range.
    """
    c = color.lstrip("#")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


def _compute_element_colors(
    data: FloatArray,
    is_multi: bool,
    colors: list[str] | None,
) -> np.ndarray:
    """
    Compute per-element RGB colors (uint8) pre-blended against white.

    Parameters
    ----------
    data : FloatArray
        Density data of shape ``(nel,)`` or ``(n_mat, nel)``.
    is_multi : bool
        Whether this is multi-material data.
    colors : list[str] | None
        Hex color strings, one per material. ``None`` defaults to black.
    Returns
    -------
    np.ndarray
        uint8 RGB array of shape ``(nel, 3)``.
    """
    if colors is None or len(colors) == 0:
        colors = ["#000000"] * (data.shape[0] if is_multi else 1)

    if is_multi:
        n_mat, nel = data.shape
        rgb = np.ones((nel, 3))
        for i in range(n_mat):
            c = colors[i] if i < len(colors) else "#000000"
            mat_rgb = np.array(_hex_to_rgb_int(c), dtype=np.float64) / 255.0
            rgb += data[i, :, np.newaxis] * (mat_rgb - 1.0)
        rgb = np.clip(rgb, 0.0, 1.0)
    else:
        c = colors[0] if colors else "#000000"
        mat_rgb = np.array(_hex_to_rgb_int(c), dtype=np.float64) / 255.0
        rgb = np.clip(1.0 + data[:, np.newaxis] * (mat_rgb - 1.0), 0.0, 1.0)

    return (rgb * 255).astype(np.uint8)


def _volume_field(
    xPhys: FloatArray, nelxyz: list[int]
) -> tuple[FloatArray, int, int, int, bool]:
    """
    Build the 3-D density volume used by the mesh exporters.

    The returned array is indexed ``[ix, iy, iz]`` so that design x maps to
    world x, design y to world y, and a 2D design is extruded along z (the
    thickness). This keeps STL and 3MF files oriented like the GUI plot.

    Parameters
    ----------
    xPhys : FloatArray
        Element densities, shape (nel,) or (n_mat, nel).
    nelxyz : list[int]
        Number of elements in [x, y, z] directions.

    Returns
    -------
    tuple
        (field, nx, ny, nz, is_2d) where ``field`` has shape (nx, ny, nz).
    """
    nx, ny, nz = nelxyz
    is_multi = xPhys.ndim == 2
    field = xPhys.sum(axis=0) if is_multi else xPhys
    if nz > 0:
        volume = field.reshape((nz, nx, ny)).transpose(1, 2, 0)
        return volume, nx, ny, nz, False
    return field.reshape((nx, ny, 1)), nx, ny, 1, True


def _marching_cubes_surface(
    field: FloatArray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract the 0.5 iso-surface of a density volume.

    One voxel of zero density is padded around the volume so that material
    touching the border still produces a closed surface. The returned
    coordinates are expressed in element units: element ``i`` spans
    ``[i, i + 1]``.

    Parameters
    ----------
    field : FloatArray
        Density volume indexed ``[ix, iy, iz]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Vertex coordinates (n, 3) and triangle vertex indices (m, 3).
    """
    padded = np.pad(field, pad_width=1, mode="constant", constant_values=0.0)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*has been deprecated.*", category=DeprecationWarning
        )
        # Padded point i sits at element coordinate i - 0.5, so the outer zero
        # layers lie at -0.5 and n - 0.5 and the surface of a solid block
        # spans exactly [0, n] in element units.
        grid = pv.ImageData(dimensions=padded.shape)
        grid.origin = (-0.5, -0.5, -0.5)
        # The complement (void) field is contoured: VTK's case table resolves
        # ambiguous cells by separating the field that is above the level, so
        # contouring the void keeps diagonally touching solid voxels connected
        # (otherwise a design that is one piece is exported as many shells).
        # This matches the classic marching cubes of PyMCubes exactly.
        grid.point_data["void"] = 1.0 - padded.ravel(order="F")
        surface = grid.contour(isosurfaces=[0.5], scalars="void")
    if surface.n_points == 0:
        return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)
    surface = surface.triangulate()
    vertices = np.asarray(surface.points, dtype=np.float64)
    faces = np.asarray(surface.faces).reshape(-1, 4)
    # Reverse the winding: normals of the void surface point into the solid.
    triangles = faces[:, [3, 2, 1]].astype(np.int64)
    return vertices, triangles


def _triangle_material_indices(
    xPhys: FloatArray,
    vertices: np.ndarray,
    triangles: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    is_2d: bool,
) -> np.ndarray:
    """
    Assign each triangle to the material with the highest density at its centre.

    Parameters
    ----------
    xPhys : FloatArray
        Multi-material densities, shape (n_mat, nel).
    vertices : np.ndarray
        Surface vertices (n, 3) in element coordinates.
    triangles : np.ndarray
        Triangle vertex indices (m, 3).
    nx, ny, nz : int
        Element counts.
    is_2d : bool
        Whether the design is two-dimensional (z is the thickness axis).

    Returns
    -------
    np.ndarray
        Material index per triangle, shape (m,).
    """
    centres = vertices[triangles].mean(axis=1)
    # Element coordinates are in [0, n]: floor gives the element containing the
    # centroid, which is more accurate at material interfaces than rounding.
    ix = np.clip(np.floor(centres[:, 0]).astype(np.int64), 0, nx - 1)
    iy = np.clip(np.floor(centres[:, 1]).astype(np.int64), 0, ny - 1)
    if is_2d:
        idx = ix * ny + iy
    else:
        iz = np.clip(np.floor(centres[:, 2]).astype(np.int64), 0, nz - 1)
        idx = iz * (nx * ny) + ix * ny + iy
    return np.argmax(xPhys[:, idx], axis=0)


def save_as_png(
    xPhys: FloatArray, nelxyz: list[int], filename: str, colors: list[str] | None = None
) -> tuple[bool, str | None]:
    """
    Saves the density field as a .png image using PyVista off-screen rendering.

    Parameters
    ----------
    xPhys : FloatArray
        Element densities, shape (nel,) or (n_mat, nel).
    nelxyz : list[int]
        Number of elements in [x, y, z] directions.
    filename : str
        Output file path.
    colors : list[str] | None
        Optional color list for multi-material visualization.

    Returns
    -------
    tuple[bool, str | None]
        (success, error_message) - True if successful, error string otherwise.
    """
    plotter: pv.Plotter | None = None
    try:
        nx, ny, nz = nelxyz
        is_3d = nz > 0
        is_multi = xPhys.ndim == 2

        plotter = pv.Plotter(off_screen=True)
        plotter.set_background("white")

        if is_3d:
            eff_density = xPhys.sum(axis=0) if is_multi else xPhys
            elem_colors = _compute_element_colors(xPhys, is_multi, colors)
            opacity = np.interp(
                np.clip(eff_density, 0.0, 1.0),
                (0.0, 1.0),
                (_VOXEL_OPACITY_MIN, _VOXEL_OPACITY_MAX),
            )
            elem_colors = np.column_stack(
                (elem_colors, np.round(opacity * 255).astype(np.uint8))
            )

            structured_grid = StructuredGrid(nx, ny, nz)
            grid = _ImageData(dimensions=(nx + 1, ny + 1, nz + 1))
            grid.cell_data["density"] = structured_grid.to_vtk_cell_order(eff_density)
            grid.cell_data["colors"] = structured_grid.to_vtk_cell_order(elem_colors)
            visible = grid.threshold(0.01, scalars="density")
            if visible.n_cells > 0:
                plotter.add_mesh(
                    visible,
                    scalars="colors",
                    rgb=True,
                    lighting=False,
                )
            plotter.view_isometric()
        else:
            elem_colors = _compute_element_colors(xPhys, is_multi, colors)
            # Element ordering: app uses idx = y + x*nely, VTK uses x-fastest
            vtk_colors = (
                elem_colors.reshape((nx, ny, 3)).transpose(1, 0, 2).reshape(-1, 3)
            )

            grid = _ImageData(dimensions=(nx + 1, ny + 1, 1))
            grid.cell_data["colors"] = vtk_colors
            plotter.add_mesh(grid, scalars="colors", rgb=True, lighting=False)
            plotter.view_xy()
            plotter.enable_parallel_projection()

        plotter.screenshot(filename, scale=2)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)
    finally:
        if plotter is not None:
            plotter.close()


def save_as_vti(
    xPhys: FloatArray, nelxyz: list[int], filename: str
) -> tuple[bool, str | None]:
    """
    Saves the density field as a .vti file for ParaView.

    Parameters
    ----------
    xPhys : FloatArray
        Element densities, shape (nel,) or (n_mat, nel).
    nelxyz : list[int]
        Number of elements in [x, y, z] directions.
    filename : str
        Output file path.

    Returns
    -------
    tuple[bool, str | None]
        (success, error_message) - True if successful, error string otherwise.
    """
    try:
        nx, ny, nz = nelxyz
        is_multi = xPhys.ndim == 2
        field = xPhys.sum(axis=0) if is_multi else xPhys
        if nz > 0:
            field = field.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            field = field.reshape((nx, ny))[np.newaxis, :, :]

        # VTK requires data to be flattened in Fortran order ('F')
        vtk_array = numpy_to_vtk(
            num_array=field.flatten("F"),
            deep=True,
            array_type=get_vtk_array_type(field.dtype),
        )

        image_data = vtk.vtkImageData()
        image_data.SetOrigin([0, 0, 0])
        image_data.SetSpacing([1, 1, 1])
        image_data.SetDimensions([nx, ny, nz])
        image_data.GetPointData().SetScalars(vtk_array)
        image_data.GetPointData().GetScalars().SetName("Density")

        if is_multi:
            for i in range(xPhys.shape[0]):
                mat_xPhys = xPhys[i]
                if nz > 0:
                    mat_field = mat_xPhys.reshape((nz, nx, ny)).transpose(1, 2, 0)
                else:
                    mat_field = mat_xPhys.reshape((nx, ny))[np.newaxis, :, :]

                mat_vtk_array = numpy_to_vtk(
                    num_array=mat_field.flatten("F"),
                    deep=True,
                    array_type=get_vtk_array_type(mat_field.dtype),
                )
                mat_vtk_array.SetName(f"Material_{i + 1}")
                image_data.GetPointData().AddArray(mat_vtk_array)

        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        writer.SetInputData(image_data)
        writer.Write()
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def save_as_stl(
    xPhys: FloatArray, nelxyz: list[int], filename: str
) -> tuple[bool, str | None]:
    """
    Saves the result as a .stl file using marching cubes.

    Parameters
    ----------
    xPhys : FloatArray
        Element densities, shape (nel,) or (n_mat, nel).
    nelxyz : list[int]
        Number of elements in [x, y, z] directions.
    filename : str
        Output file path.

    Returns
    -------
    tuple[bool, str | None]
        (success, error_message) - True if successful, error string otherwise.
    """
    try:
        field, _nx, _ny, _nz, _is_2d = _volume_field(xPhys, nelxyz)
        vertices, triangles = _marching_cubes_surface(field)
        if triangles.shape[0] == 0:
            return False, "The density field has no surface at the 0.5 level."
        # Build STL mesh
        stl_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = vertices[triangles]
        stl_mesh.save(filename)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def save_as_3mf(
    xPhys: FloatArray, nelxyz: list[int], filename: str, colors: list[str] | None = None
) -> tuple[bool, str | None]:
    """
    Saves the result as a .3mf file using Lib3MF.

    Parameters
    ----------
    xPhys : FloatArray
        Element densities, shape (nel,) or (n_mat, nel).
    nelxyz : list[int]
        Number of elements in [x, y, z] directions.
    filename : str
        Output file path.
    colors : list[str] | None
        Optional color list for multi-material visualization.

    Returns
    -------
    tuple[bool, str | None]
        (success, error_message) - True if successful, error string otherwise.
    """
    if lib3mf is None:
        return False, (
            "3MF export requires the optional 'lib3mf' package; install it "
            "with 'pip install lib3mf'."
        )
    try:
        field, nx, ny, nz, is_2d = _volume_field(xPhys, nelxyz)
        vertices, triangles = _marching_cubes_surface(field)
        if triangles.shape[0] == 0:
            return False, "The density field has no surface at the 0.5 level."
        is_multi = xPhys.ndim == 2

        wrapper = lib3mf.get_wrapper()
        model = wrapper.CreateModel()
        model.SetUnit(lib3mf.ModelUnit.MilliMeter)

        mesh = model.AddMeshObject()

        # Pre-allocate ctypes types to prevent memory corruption/crashes
        c_float_3 = ctypes.c_float * 3
        c_uint_3 = ctypes.c_uint32 * 3

        # Safely assign elements to lib3mf Position and Triangle structs
        positions = []
        for v in vertices:
            pos = lib3mf.Position()
            pos.Coordinates = c_float_3(float(v[0]), float(v[1]), float(v[2]))
            positions.append(pos)

        tris = []
        for t in triangles:
            tri = lib3mf.Triangle()
            tri.Indices = c_uint_3(int(t[0]), int(t[1]), int(t[2]))
            tris.append(tri)

        mesh.SetGeometry(positions, tris)

        if colors:
            colorgroup = model.AddColorGroup()

            color_ids = []
            for c in colors:
                r, g, b = _hex_to_rgb_int(c)
                # Explicitly populate the color structure fields
                col = lib3mf.Color()
                col.Red = r
                col.Green = g
                col.Blue = b
                col.Alpha = 255

                cid = colorgroup.AddColor(col)
                color_ids.append(cid)

            # Fallback object-level property
            mesh.SetObjectLevelProperty(colorgroup.GetResourceID(), color_ids[0])

            if is_multi:
                # Pre-compute ctypes arrays for each material's color PropertyIDs
                prop_arrays = [c_uint_3(cid, cid, cid) for cid in color_ids]
                res_id = colorgroup.GetResourceID()
                materials = _triangle_material_indices(
                    xPhys, vertices, triangles, nx, ny, nz, is_2d
                )

                for tri_index, mat in enumerate(materials):
                    # Build the strict TriangleProperties object required by
                    # the C++ backend
                    prop = lib3mf.TriangleProperties()
                    prop.ResourceID = res_id
                    prop.PropertyIDs = prop_arrays[int(mat)]

                    mesh.SetTriangleProperties(tri_index, prop)

        model.AddBuildItem(mesh, wrapper.GetIdentityTransform())
        writer = model.QueryWriter("3mf")
        writer.WriteToFile(filename)

        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def save_loss(
    loss: list[tuple[int, float]],
    output_dir: str,
    preset_name: str,
) -> tuple[bool, str | None]:
    """
    Save the loss (objective) history to a CSV file.

    The CSV is written to ``<output_dir>/<preset_name>/<preset_name>_loss_function.csv``
    with the columns ``iteration,objective``.

    Parameters
    ----------
    loss : list[tuple[int, float]]
        List of ``(iteration, objective)`` tuples recorded during optimization.
    output_dir : str
        Base output directory under which the preset's folder lives.
    preset_name : str
        Preset identifier used to name the CSV file.

    Returns
    -------
    tuple[bool, str | None]
        A tuple containing a boolean indicating success or failure, and an
        error message if applicable.
    """
    folder: Path = Path(output_dir) / preset_name
    folder.mkdir(parents=True, exist_ok=True)
    csv_file: Path = folder / f"{preset_name}_loss_function.csv"
    try:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["iteration", "objective"])
            writer.writerows(loss)
        return True, None
    except Exception as e:  # noqa: BLE001
        return False, str(e)
