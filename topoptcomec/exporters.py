# topoptcomec/ui/exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Export result to various file formats.

from __future__ import annotations
import csv
import ctypes
import warnings
from pathlib import Path
import mcubes
import numpy as np
import numpy.typing as npt
import pyvista as pv
import vtk
from stl import mesh
from vtk.util.numpy_support import get_vtk_array_type, numpy_to_vtk

from topoptcomec.core.grid import StructuredGrid

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lib3mf")
    import lib3mf

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
    except Exception as e:
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
    except Exception as e:
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
        nx, ny, nz = nelxyz
        is_multi = xPhys.ndim == 2
        field = xPhys.sum(axis=0) if is_multi else xPhys
        if nz > 0:
            field = field.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            field = field.reshape((nx, ny)).T[np.newaxis, :, :]

        # Add 1-voxel padding to avoid border loss in marching cubes
        field = np.pad(field, pad_width=1, mode="constant", constant_values=0)
        # Run marching cubes; mcubes can emit a NumPy 2.5 deprecation warning
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="Setting the shape on a NumPy array has been deprecated in NumPy 2.5.*",
            )
            vertices, triangles = mcubes.marching_cubes(volume=field, isovalue=0.5)
        # Build STL mesh
        stl_mesh = mesh.Mesh(np.zeros(triangles.shape[0], dtype=mesh.Mesh.dtype))
        stl_mesh.vectors = vertices[triangles]
        stl_mesh.save(filename)
        return True, None
    except Exception as e:
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
    try:
        nx, ny, nz = nelxyz
        is_multi = xPhys.ndim == 2
        field = xPhys.sum(axis=0) if is_multi else xPhys
        if nz > 0:
            field = field.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            field = field.reshape((nx, ny)).T[np.newaxis, :, :]

        # Add 1-voxel padding to avoid border loss in marching cubes
        field = np.pad(field, pad_width=1, mode="constant", constant_values=0)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=DeprecationWarning,
                message="Setting the shape on a NumPy array has been deprecated in NumPy 2.5.*",
            )
            vertices, triangles = mcubes.marching_cubes(volume=field, isovalue=0.5)

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

                for tri_index, tri in enumerate(triangles):
                    v = vertices[tri].mean(axis=0)

                    ix = int(np.clip(round(v[0] - 1), 0, nx - 1))
                    iy = int(np.clip(round(v[1] - 1), 0, ny - 1))
                    iz = int(np.clip(round(v[2] - 1), 0, nz - 1))

                    idx = iz * (nx * ny) + ix * ny + iy
                    mat = np.argmax(xPhys[:, idx])

                    # Build the strict TriangleProperties object required by the C++ backend
                    prop = lib3mf.TriangleProperties()
                    prop.ResourceID = res_id
                    prop.PropertyIDs = prop_arrays[mat]

                    mesh.SetTriangleProperties(tri_index, prop)

        model.AddBuildItem(mesh, wrapper.GetIdentityTransform())
        writer = model.QueryWriter("3mf")
        writer.WriteToFile(filename)

        return True, None
    except Exception as e:
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
    except Exception as e:
        return False, str(e)
