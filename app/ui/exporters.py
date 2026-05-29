# app/ui/exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Export result to various file formats.

from __future__ import annotations
import ctypes
import warnings
import mcubes
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import vtk
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from stl import mesh
from vtk.util.numpy_support import get_vtk_array_type, numpy_to_vtk

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=DeprecationWarning, module="lib3mf")
    import lib3mf

# Type aliases
FloatArray = npt.NDArray[np.float64]


def save_as_png(
    xPhys: FloatArray, nelxyz: list[int], filename: str, colors: list[str] | None = None
) -> tuple[bool, str | None]:
    """Saves the density field as a .png image."""
    try:
        nelx, nely, nelz = nelxyz
        is_3d = nelz > 0
        is_multi = xPhys.ndim == 2

        fig = plt.figure()
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
            ax.set_facecolor("white")
            # Avoids plotting fully transparent points.
            if is_multi:
                eff_density = xPhys.sum(axis=0)
                visible_elements_mask = eff_density > 0.01
                visible_indices = np.where(visible_elements_mask)[0]

                z = visible_indices // (nelx * nely)
                x = (visible_indices % (nelx * nely)) // nely
                y = visible_indices % nely

                n_mat = xPhys.shape[0]
                voxel_colors = np.ones((len(visible_indices), 4))
                for i in range(n_mat):
                    mat_rgb = np.array(
                        to_rgb(colors[i] if colors and i < len(colors) else "black")
                    )
                    rho_vis = xPhys[i, visible_indices]
                    voxel_colors[:, :3] += rho_vis[:, np.newaxis] * (mat_rgb - 1.0)
                voxel_colors[:, :3] = np.clip(voxel_colors[:, :3], 0.0, 1.0)
                voxel_colors[:, 3] = np.clip(eff_density[visible_indices], 0.0, 1.0)
            else:
                visible_elements_mask = xPhys > 0.01
                visible_indices = np.where(visible_elements_mask)[0]
                densities = xPhys[visible_indices]

                z = visible_indices // (nelx * nely)
                x = (visible_indices % (nelx * nely)) // nely
                y = visible_indices % nely

                voxel_colors = np.zeros((len(densities), 4))
                base_color_rgb = to_rgb(colors[0] if colors else "black")
                voxel_colors[:, :3] = base_color_rgb
                voxel_colors[:, 3] = densities

            ax.scatter(
                x + 0.5,
                y + 0.5,
                z + 0.5,
                s=6000 / max(nelx, nely, nelz),
                marker="s",
                c=voxel_colors,
                alpha=None,
            )
            ax.set_box_aspect([nelx, nely, nelz])
            # Hide axes for cleaner output
            ax.set_axis_off()
        else:
            ax = fig.add_subplot(111)
            ax.set_facecolor("white")

            if is_multi:
                n_mat, nel = xPhys.shape
                rgb_image = np.ones((nel, 3))
                for i in range(n_mat):
                    mat_rgb = np.array(
                        to_rgb(colors[i] if colors and i < len(colors) else "black")
                    )
                    rgb_image += xPhys[i, :, np.newaxis] * (mat_rgb - 1.0)
                rgb_image = np.clip(rgb_image, 0.0, 1.0)
                rgb_image = rgb_image.reshape((nelx, nely, 3)).transpose(1, 0, 2)

                ax.imshow(
                    rgb_image,
                    interpolation="nearest",
                    origin="lower",
                )
            else:
                mat_color = colors[0] if colors else "black"
                cmap = LinearSegmentedColormap.from_list(
                    "custom_cmap", ["white", mat_color]
                )
                ax.imshow(
                    xPhys.reshape((nelx, nely)).T,
                    cmap=cmap,
                    interpolation="nearest",
                    origin="lower",
                    norm=plt.Normalize(0, 1),
                )
            ax.set_aspect("equal", "box")
            ax.axis("off")

        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return True, None
    except Exception as e:
        return False, str(e)


def save_as_vti(
    xPhys: FloatArray, nelxyz: list[int], filename: str
) -> tuple[bool, str | None]:
    """Saves the density field as a .vti file for ParaView."""
    try:
        nx, ny, nz = nelxyz
        is_multi = xPhys.ndim == 2

        eff_xPhys = xPhys.sum(axis=0) if is_multi else xPhys

        if nz > 0:
            density_field = eff_xPhys.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            density_field = eff_xPhys.reshape((nx, ny))[np.newaxis, :, :]

        # VTK requires data to be flattened in Fortran order ('F')
        vtk_array = numpy_to_vtk(
            num_array=density_field.flatten("F"),
            deep=True,
            array_type=get_vtk_array_type(density_field.dtype),
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
    """Saves the result as a .stl file."""
    try:
        nx, ny, nz = nelxyz
        is_multi = xPhys.ndim == 2
        eff_xPhys = xPhys.sum(axis=0) if is_multi else xPhys

        if nz > 0:
            density_field = eff_xPhys.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            density_field = eff_xPhys.reshape((nx, ny)).T[np.newaxis, :, :]

        # Add 1-voxel padding to avoid border loss in marching cubes
        density_field = np.pad(
            density_field, pad_width=1, mode="constant", constant_values=0
        )
        # Run marching cubes
        vertices, triangles = mcubes.marching_cubes(density_field, 0.5)
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
    """Saves the result as a .3mf file using Lib3MF."""
    try:
        nx, ny, nz = nelxyz
        is_multi = xPhys.ndim == 2
        eff = xPhys.sum(axis=0) if is_multi else xPhys
        if nz > 0:
            field = eff.reshape((nz, nx, ny)).transpose(1, 2, 0)
        else:
            nz = 1  # Extrude to a single layer
            # Reshape 2D data and add a new axis for the Z dimension
            field = eff.reshape((nx, ny)).T[np.newaxis, :, :]

        # Add 1-voxel padding to avoid border loss in marching cubes
        field = np.pad(field, pad_width=1, mode="constant", constant_values=0)
        vertices, triangles = mcubes.marching_cubes(field, 0.5)

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
                r, g, b = [int(v * 255) for v in to_rgb(c)]
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
