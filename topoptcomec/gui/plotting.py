# topoptcomec/ui/plotting.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Plotting class (PyVista backend).

import os

import numpy as np
import pyvista as pv
from PySide6.QtGui import QColor

from topoptcomec.core.grid import StructuredGrid

# pv.ImageData replaced pv.UniformGrid in PyVista 0.44
ImageData = pv.ImageData if hasattr(pv, "ImageData") else pv.UniformGrid

# Small z-offset keeping 2D overlay lines above the flat material mesh
_OVERLAY_Z = 0.05

_VOXEL_OPACITY_MIN = 0.25
_VOXEL_OPACITY_MAX = 0.95


class PlottingMixin:
    """Mixin for MainWindow to handle all plotting operations with PyVista."""

    def _init_plotting_state(self) -> None:
        """Initialize the scene bookkeeping (called once from MainWindow.__init__)."""
        self._material_actor = None
        self._material_grid = None
        self._overlay_actors = []
        self._message_actor = None
        self._camera_mode = None  # "2d" or "3d"
        self._camera_dims = None  # (nelx, nely, nelz) the camera was fitted on

    def _style_plot_default(self) -> None:
        """
        Apply the default white theme styling to the plot.

        Sets the plot background to white; text and axes are drawn in black.
        """
        self.plotter.set_background("white")
        self._render()

    @staticmethod
    def _color_to_rgb(color: str) -> np.ndarray:
        """
        Convert a color string to an RGB float array in [0, 1].

        Parameters
        ----------
        color : str
            Color as "#RRGGBB" (or any string accepted by QColor).

        Returns
        -------
        np.ndarray
            RGB components as floats in [0, 1].
        """
        r, g, b, _ = QColor(color).getRgbF()
        return np.array([r, g, b])

    def _material_colors(self, data: np.ndarray, is_multi: bool) -> np.ndarray:
        """
        Compute per-element RGB colors for a density field.

        Colors are pre-blended against a white background according to each
        element's material density.

        Parameters
        ----------
        data : np.ndarray
            Density data of shape (nel,) or (n_mat, nel).
        is_multi : bool
            Whether this is multi-material data.
        Returns
        -------
        np.ndarray
            uint8 RGB array of shape (nel, 3).
        """
        if is_multi:
            n_mat, nel = data.shape
            rgb = np.ones((nel, 3))  # Start white
            for i in range(n_mat):
                mat_rgb = self._color_to_rgb(
                    self.materials_widget.inputs[i]["color"].get_color()
                )
                # Blend: pixel = sum(rho_i * color_i)
                rgb += data[i, :, np.newaxis] * (mat_rgb - 1.0)
            rgb = np.clip(rgb, 0.0, 1.0)
        else:
            mat_rgb = self._color_to_rgb(
                self.materials_widget.inputs[0]["color"].get_color()
            )
            # white -> material color gradient
            rgb = np.clip(1.0 + data[:, np.newaxis] * (mat_rgb - 1.0), 0.0, 1.0)

        return (rgb * 255).astype(np.uint8)

    def _plot_deformation(self, is_3d: bool, nelx: int, nely: int, nelz: int):
        """
        Plot the deformed shape based on displacement results.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D problem.
        nelx, nely, nelz : int
            Number of elements in each dimension.
        """
        if (
            self.last_params["Displacement"]["disp_iterations"] == 1
        ):  # Single-frame grid plot
            if self.xPhys is None:
                return
            if is_3d:
                # Compute original element centers
                nel: int = nelx * nely * nelz
                visible_indices: np.ndarray = np.arange(nel)  # all of them
                z_idx: np.ndarray = visible_indices // (nelx * nely)
                x_idx: np.ndarray = (visible_indices % (nelx * nely)) // nely
                y_idx: np.ndarray = visible_indices % nely

                # Compute displaced centers from node positions
                X, Y, Z = self.last_displayed_frame_data  # displaced node coords

                # Take mean of the 8 node positions for each voxel center
                cx: np.ndarray = (
                    X[x_idx, y_idx, z_idx]
                    + X[x_idx + 1, y_idx, z_idx]
                    + X[x_idx, y_idx + 1, z_idx]
                    + X[x_idx + 1, y_idx + 1, z_idx]
                    + X[x_idx, y_idx, z_idx + 1]
                    + X[x_idx + 1, y_idx, z_idx + 1]
                    + X[x_idx, y_idx + 1, z_idx + 1]
                    + X[x_idx + 1, y_idx + 1, z_idx + 1]
                ) / 8.0

                cy: np.ndarray = (
                    Y[x_idx, y_idx, z_idx]
                    + Y[x_idx + 1, y_idx, z_idx]
                    + Y[x_idx, y_idx + 1, z_idx]
                    + Y[x_idx + 1, y_idx + 1, z_idx]
                    + Y[x_idx, y_idx, z_idx + 1]
                    + Y[x_idx + 1, y_idx, z_idx + 1]
                    + Y[x_idx, y_idx + 1, z_idx + 1]
                    + Y[x_idx + 1, y_idx + 1, z_idx + 1]
                ) / 8.0

                cz: np.ndarray = (
                    Z[x_idx, y_idx, z_idx]
                    + Z[x_idx + 1, y_idx, z_idx]
                    + Z[x_idx, y_idx + 1, z_idx]
                    + Z[x_idx + 1, y_idx + 1, z_idx]
                    + Z[x_idx, y_idx, z_idx + 1]
                    + Z[x_idx + 1, y_idx, z_idx + 1]
                    + Z[x_idx, y_idx + 1, z_idx + 1]
                    + Z[x_idx + 1, y_idx + 1, z_idx + 1]
                ) / 8.0

                # Colors pre-blended against white with alpha = density
                is_multi_3d: bool = hasattr(self.xPhys, "ndim") and self.xPhys.ndim > 1
                eff_density: np.ndarray = (
                    self.xPhys.sum(axis=0) if is_multi_3d else self.xPhys
                )
                mask: np.ndarray = eff_density > 0.01
                if not np.any(mask):
                    return
                colors: np.ndarray = self._material_colors(self.xPhys, is_multi_3d)[
                    mask
                ]
                opacity: np.ndarray = np.interp(
                    np.clip(eff_density[mask], 0.0, 1.0),
                    (0.0, 1.0),
                    (_VOXEL_OPACITY_MIN, _VOXEL_OPACITY_MAX),
                )
                if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
                    colors = np.column_stack(
                        (colors, np.round(opacity * 255).astype(np.uint8))
                    )

                # One cube glyph per visible displaced voxel center
                points: np.ndarray = np.column_stack([cx[mask], cy[mask], cz[mask]])
                poly = pv.PolyData(points)
                poly.point_data["colors"] = colors
                cube = pv.Cube(x_length=1.0, y_length=1.0, z_length=1.0)
                glyphs = poly.glyph(geom=cube, orient=False, scale=False, factor=1.0)

                self._remove_material_actor()
                self._material_actor = self.plotter.add_mesh(
                    glyphs,
                    scalars="colors",
                    rgb=True,
                    reset_camera=False,
                    render=False,
                )
                self._material_grid = None
            else:
                X, Y = self.last_displayed_frame_data

                is_multi: bool = hasattr(self.xPhys, "ndim") and self.xPhys.ndim > 1
                colors: np.ndarray = self._material_colors(self.xPhys, is_multi)

                # Structured grid of displaced nodes, colored per element
                grid = pv.StructuredGrid(X, Y, np.zeros_like(X))
                # Element ordering: app uses idx = y + x*nely, VTK uses x-fastest
                grid.cell_data["colors"] = (
                    colors.reshape((nelx, nely, 3)).transpose(1, 0, 2).reshape(-1, 3)
                )

                self._remove_material_actor()
                self._material_actor = self.plotter.add_mesh(
                    grid,
                    scalars="colors",
                    rgb=True,
                    lighting=False,
                    reset_camera=False,
                    render=False,
                )
                self._material_grid = None
        else:
            if self.sections["Materials"].visibility_button.isChecked():
                self._plot_material(
                    is_3d=is_3d,
                    xPhys_data=self.last_displayed_frame_data,
                )
        # Multi-iteration displacement handled in _update_animation_frame

    def _initialize_xphys(self, nelx: int, nely: int, nelz: int, is_3d: bool) -> None:
        """
        Initialize the material density field based on current parameters.

        Parameters
        ----------
        nelx, nely, nelz : int
            Number of elements in each dimension.
        is_3d : bool
            Whether this is a 3D problem.
        """
        from topoptcomec.core import initializers  # Import here to avoid circular

        pm: dict = self.last_params["Materials"]
        pd: dict = self.last_params["Dimensions"]
        pf: dict = self.last_params["Forces"]
        ps: dict = (
            self.last_params["Supports"] if "Supports" in self.last_params else None
        )

        active_iforces_indices = [
            i for i in range(len(pf["fidir"])) if np.array(pf["fidir"])[i] != "-"
        ]
        active_oforces_indices = [
            i for i in range(len(pf["fodir"])) if np.array(pf["fodir"])[i] != "-"
        ]
        active_supports_indices = (
            [i for i in range(len(ps["sdim"])) if np.array(ps["sdim"])[i] != "-"]
            if ps is not None
            else []
        )

        fix_active: np.ndarray = np.array(pf["fix"])[active_iforces_indices]
        fiy_active: np.ndarray = np.array(pf["fiy"])[active_iforces_indices]
        fox_active: np.ndarray = np.array(pf["fox"])[active_oforces_indices]
        foy_active: np.ndarray = np.array(pf["foy"])[active_oforces_indices]
        sx_active: np.ndarray = (
            np.array(ps["sx"])[active_supports_indices]
            if ps is not None
            else np.array([])
        )
        sy_active: np.ndarray = (
            np.array(ps["sy"])[active_supports_indices]
            if ps is not None
            else np.array([])
        )
        all_x: np.ndarray = np.concatenate([fix_active, fox_active, sx_active])
        all_y: np.ndarray = np.concatenate([fiy_active, foy_active, sy_active])

        if is_3d:
            fiz_active: np.ndarray = np.array(pf["fiz"])[active_iforces_indices]
            foz_active: np.ndarray = np.array(pf["foz"])[active_oforces_indices]
            sz_active: np.ndarray = np.array(ps["sz"])[active_supports_indices]
        all_z: np.ndarray = (
            np.concatenate([fiz_active, foz_active, sz_active])
            if is_3d
            else np.array([0] * len(all_x))
        )

        current_x: np.ndarray = getattr(self, "last_successful_xPhys", None)
        if len(pm["E"]) == 1:
            self.xPhys = initializers.initialize_material(
                pm["init_type"],
                pd["volfrac"],
                nelx,
                nely,
                nelz,
                all_x,
                all_y,
                all_z,
                current_xPhys=current_x,
            )
        else:
            self.xPhys = initializers.initialize_materials(
                pm["init_type"],
                pm["percent"],
                pd["volfrac"],
                nelx,
                nely,
                nelz,
                all_x,
                all_y,
                all_z,
                current_xPhys=current_x,
            )

        if self.xPhys is not None and "Regions" in self.last_params:
            self._apply_regions(nelx, nely, nelz, is_3d)

    def _apply_regions(self, nelx: int, nely: int, nelz: int, is_3d: bool) -> None:
        """
        Apply solid/void region overrides to the current density field.

        Parameters
        ----------
        nelx, nely, nelz : int
            Number of elements in each dimension.
        is_3d : bool
            Whether this is a 3D problem.
        """
        pr: dict = self.last_params["Regions"]
        for i, shape in enumerate(pr["rshape"]):
            if shape == "-":
                continue

            x_min: int = max(0, int(pr["rx"][i] - pr["rradius"][i]))
            x_max: int = min(nelx, int(pr["rx"][i] + pr["rradius"][i]))
            y_min: int = max(0, int(pr["ry"][i] - pr["rradius"][i]))
            y_max: int = min(nely, int(pr["ry"][i] + pr["rradius"][i]))
            if is_3d:
                z_min: int = max(0, int(pr["rz"][i] - pr["rradius"][i]))
                z_max: int = min(nelz, int(pr["rz"][i] + pr["rradius"][i]))

            idx_x: np.ndarray = np.arange(x_min, x_max)
            idx_y: np.ndarray = np.arange(y_min, y_max)
            if is_3d:
                idx_z: np.ndarray = np.arange(z_min, z_max)

            indices: np.ndarray | None = None

            if pr["rshape"][i] == "□":  # Square/Cube
                if len(idx_x) > 0 and len(idx_y) > 0:
                    if is_3d and len(idx_z) > 0:
                        xx, yy, zz = np.meshgrid(idx_x, idx_y, idx_z, indexing="ij")
                        indices = zz + yy * nelz + xx * nely * nelz
                    elif not is_3d:
                        xx, yy = np.meshgrid(idx_x, idx_y, indexing="ij")
                        indices = yy + xx * nely

            elif pr["rshape"][i] == "◯":  # Circle/Sphere
                if len(idx_x) > 0 and len(idx_y) > 0:
                    if is_3d and len(idx_z) > 0:
                        i_grid, j_grid, k_grid = np.meshgrid(
                            idx_x, idx_y, idx_z, indexing="ij"
                        )
                        mask: np.ndarray = (i_grid - pr["rx"][i]) ** 2 + (
                            j_grid - pr["ry"][i]
                        ) ** 2 + (k_grid - pr["rz"][i]) ** 2 <= pr["rradius"][i] ** 2
                        ii: np.ndarray = i_grid[mask]
                        jj: np.ndarray = j_grid[mask]
                        kk: np.ndarray = k_grid[mask]
                        indices: np.ndarray = kk + jj * nelz + ii * nely * nelz
                    elif not is_3d:
                        i_grid, j_grid = np.meshgrid(idx_x, idx_y, indexing="ij")
                        mask = (i_grid - pr["rx"][i]) ** 2 + (
                            j_grid - pr["ry"][i]
                        ) ** 2 <= pr["rradius"][i] ** 2
                        ii = i_grid[mask]
                        jj = j_grid[mask]
                        indices = jj + ii * nely

            if indices is not None:
                if pr["rstate"][i] == "Material 1":
                    mat = 0
                elif pr["rstate"][i] == "Material 2":
                    mat = 1
                else:  # "Void"
                    mat = -1
                flat_idx: np.ndarray = indices.flatten()
                if self.xPhys.ndim == 1:
                    self.xPhys[flat_idx] = 1.0 if mat == 0 else 1e-6
                else:
                    self.xPhys[:, flat_idx] = 1e-6
                    if mat >= 0:
                        self.xPhys[mat, flat_idx] = 1.0

    def _show_initial_message(self, is_3d: bool) -> None:
        """
        Display placeholder message on the plot before optimization results exist.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        """
        self._message_actor = None
        if self.footer.create_button.graphicsEffect() is not None:
            init_message: str = 'Configure parameters and press "Create"'
            try:
                self._message_actor = self.plotter.add_text(
                    init_message,
                    position=(0.5, 0.5),
                    viewport=True,
                    font_size=8,  # PyVista doubles this internally (≈16 pt)
                    color="black",
                )
                text_prop = self._message_actor.GetTextProperty()
                text_prop.SetJustificationToCentered()
                text_prop.SetVerticalJustificationToCentered()
            except Exception:
                self._message_actor = None

    def replot(self) -> None:
        """
        Redraw the entire plot including all visible layers.

        Determines which layers to show based on visibility button states
        and whether displaying deformation or static results.
        """
        if not self.last_params:
            return  # Do nothing if triggerd in sections initialization
        self.plotter.clear()
        self._material_actor = None
        self._material_grid = None
        self._overlay_actors = []
        self._message_actor = None
        self.plotter.set_background("white")

        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
        is_3d: bool = nelz > 0

        # Layer 1: The Main Result (Material)
        if (
            self.is_displaying_deformation
            and self.last_displayed_frame_data is not None
        ):
            self._plot_deformation(is_3d, nelx, nely, nelz)
        else:
            if self.sections["Materials"].visibility_button.isChecked():
                if self.xPhys is None:
                    self._initialize_xphys(nelx, nely, nelz, is_3d)
                self._plot_material(is_3d=is_3d)

        # Layer 2: Overlays
        self._redraw_non_material_layers(is_3d)

        # Show initial message if xPhys is not a result (even partial) of optimization
        if not self.is_displaying_deformation:
            self._show_initial_message(is_3d)

        self._update_camera(is_3d, (nelx, nely, nelz))
        self._render()

    def _update_camera(self, is_3d: bool, dims: tuple) -> None:
        """
        Set the default camera for the current problem type.

        The camera is only repositioned when switching between 2D and 3D or
        when the design-space dimensions change, so manual camera changes are
        preserved across replots. 2D interaction permits pan and zoom only.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        dims : tuple
            (nelx, nely, nelz) of the design space.
        """
        mode = "3d" if is_3d else "2d"
        dims = tuple(dims)
        if self._camera_mode == mode and self._camera_dims == dims:
            return
        self._camera_mode = mode
        self._camera_dims = dims
        if is_3d:
            self.plotter.enable_trackball_style()
            self.plotter.disable_parallel_projection()
            self.plotter.view_isometric()
            bounds = (0, dims[0], 0, dims[1], 0, dims[2])
        else:
            self.plotter.enable_image_style()
            self.plotter.view_xy()
            self.plotter.enable_parallel_projection()
            bounds = (0, dims[0], 0, dims[1], -_OVERLAY_Z, _OVERLAY_Z)
        self.plotter.reset_camera(bounds=bounds)
        self.plotter.camera.zoom(1.08)

    def _save_screenshot(self, filename: str) -> None:
        """
        Save the current view as a PNG image (2x resolution).

        Parameters
        ----------
        filename : str
            Target PNG file path.
        """
        self._fix_bounds_axes_ranges()
        try:
            self.plotter.screenshot(filename, scale=2)
        except (TypeError, ValueError):
            self.plotter.screenshot(filename)

    def _remove_actor(self, actor) -> None:
        """Remove an actor from the plotter, ignoring stale references."""
        if actor is None:
            return
        try:
            self.plotter.remove_actor(actor)
        except (ValueError, KeyError, RuntimeError):
            pass

    def _remove_material_actor(self) -> None:
        """Remove the material actor and forget its grid."""
        self._remove_actor(self._material_actor)
        self._material_actor = None
        self._material_grid = None

    def _plot_material(self, is_3d: bool, xPhys_data: np.ndarray | None = None) -> None:
        """
        Plot the material density field.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        xPhys_data : np.ndarray, optional
            Density data to plot. Uses self.xPhys if None.
        """
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
        data_to_plot: np.ndarray = self.xPhys if xPhys_data is None else xPhys_data
        if data_to_plot is None:
            return
        if not self.sections["Materials"].visibility_button.isChecked():
            return

        # Detect multi-material: shape (n_mat, nel)
        is_multi: bool = data_to_plot.ndim == 2

        if is_3d:
            self._plot_material_3d(data_to_plot, nelx, nely, nelz, is_multi)
        else:
            self._plot_material_2d(data_to_plot, nelx, nely, is_multi)

    def _plot_material_2d(self, data: np.ndarray, nelx: int, nely: int, is_multi: bool):
        """
        Plot 2D material density field as a flat colored grid.

        When a grid with matching dimensions already exists, only its cell
        colors are updated in place (fast path used for live frames).

        Parameters
        ----------
        data : np.ndarray
            Density data of shape (nel,) or (n_mat, nel).
        nelx, nely : int
            Number of elements in x and y dimensions.
        is_multi : bool
            Whether this is multi-material data.
        """
        colors: np.ndarray = self._material_colors(data, is_multi)
        # Element ordering: app uses idx = y + x*nely, VTK uses x-fastest
        vtk_colors: np.ndarray = (
            colors.reshape((nelx, nely, 3)).transpose(1, 0, 2).reshape(-1, 3)
        )

        grid = self._material_grid
        if (
            grid is not None
            and self._material_actor is not None
            and self._material_actor in self.plotter.actors.values()
            and tuple(grid.dimensions) == (nelx + 1, nely + 1, 1)
        ):
            # Fast in-place update of the cell colors (replaces im.set_array)
            grid.cell_data["colors"] = vtk_colors
            return

        self._remove_material_actor()
        grid = ImageData(dimensions=(nelx + 1, nely + 1, 1))
        grid.cell_data["colors"] = vtk_colors
        self._material_actor = self.plotter.add_mesh(
            grid,
            scalars="colors",
            rgb=True,
            lighting=False,
            name="material",
            reset_camera=False,
            render=False,
        )
        self._material_grid = grid

    def _plot_material_3d(
        self,
        data: np.ndarray,
        nelx: int,
        nely: int,
        nelz: int,
        is_multi: bool,
    ):
        """
        Plot 3D material density field as voxels.

        Elements denser than 0.01 are extracted with a threshold and shown
        as solid voxels colored by material (faded by effective density).

        Parameters
        ----------
        data : np.ndarray
            Density data of shape (nel,) or (n_mat, nel).
        nelx, nely, nelz : int
            Number of elements in each dimension.
        is_multi : bool
            Whether this is multi-material data.
        """
        self._remove_material_actor()

        eff_density: np.ndarray = data.sum(axis=0) if is_multi else data
        colors: np.ndarray = self._material_colors(data, is_multi)
        opacity: np.ndarray = np.interp(
            np.clip(eff_density, 0.0, 1.0),
            (0.0, 1.0),
            (_VOXEL_OPACITY_MIN, _VOXEL_OPACITY_MAX),
        )
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            colors = np.column_stack((colors, np.round(opacity * 255).astype(np.uint8)))

        structured_grid = StructuredGrid(nelx, nely, nelz)
        grid = ImageData(dimensions=(nelx + 1, nely + 1, nelz + 1))
        grid.cell_data["density"] = structured_grid.to_vtk_cell_order(eff_density)
        grid.cell_data["colors"] = structured_grid.to_vtk_cell_order(colors)
        visible = grid.threshold(0.01, scalars="density")
        if visible.n_cells == 0:
            return
        self._material_actor = self.plotter.add_mesh(
            visible,
            scalars="colors",
            rgb=True,
            name="material",
            reset_camera=False,
            render=False,
        )
        self._material_grid = None  # no in-place fast path for 3D

    def _redraw_non_material_layers(self, is_3d: bool) -> None:
        """
        Redraw overlay layers (forces, supports, regions, dimensions, displacement preview).

        Previously drawn overlay actors are removed first so layers never
        accumulate on the scene.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        """
        for actor in self._overlay_actors:
            self._remove_actor(actor)
        self._overlay_actors = []
        try:
            self.plotter.remove_bounds_axes()
        except (AttributeError, RuntimeError):
            pass

        self._plot_forces(is_3d=is_3d)
        self._plot_supports(is_3d=is_3d)
        self._plot_regions(is_3d=is_3d)
        self._plot_dimensions_frame(is_3d=is_3d)
        self._plot_displacement_preview(is_3d=is_3d)

    @staticmethod
    def _set_dotted(actor) -> None:
        """Best-effort dotted line style (ignored if the backend lacks support)."""
        try:
            actor.prop.SetLineStipplePattern(0x00FF)
            actor.prop.SetLineStippleRepeatFactor(1)
        except (AttributeError, RuntimeError):
            pass

    def _plot_dimensions_frame(self, is_3d: bool) -> None:
        """
        Draw a dotted frame around the design space with axis labels.
        Controlled by the Dimensions section's visibility button.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        """
        if not self.sections["Dimensions"].visibility_button.isChecked():
            return

        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]

        if is_3d:
            box = pv.Box(bounds=(0, nelx, 0, nely, 0, nelz))
            actor = self.plotter.add_mesh(
                box,
                style="wireframe",
                color="gray",
                line_width=1.5,
                reset_camera=False,
                render=False,
            )
            self._set_dotted(actor)
            self._overlay_actors.append(actor)
            self.plotter.show_bounds(
                bounds=(0, nelx, 0, nely, 0, nelz),
                xtitle="X",
                ytitle="Y",
                ztitle="Z",
                color="black",
                font_size=10,
                bold=False,
            )
        else:
            corners = np.array(
                [
                    [0, 0, _OVERLAY_Z],
                    [nelx, 0, _OVERLAY_Z],
                    [nelx, nely, _OVERLAY_Z],
                    [0, nely, _OVERLAY_Z],
                ],
                dtype=float,
            )
            rect = pv.lines_from_points(corners, close=True)
            actor = self.plotter.add_mesh(
                rect, color="gray", line_width=1.5, reset_camera=False, render=False
            )
            self._set_dotted(actor)
            self._overlay_actors.append(actor)
            self.plotter.show_bounds(
                bounds=(0, nelx, 0, nely, 0, 0),
                xtitle="X",
                ytitle="Y",
                color="black",
                font_size=10,
                bold=False,
            )

    def _fix_bounds_axes_ranges(self) -> None:
        """
        Force the bounds-axes box and ticks to span the design space exactly.

        PyVista resets the axis label ranges to the (padded) scene bounds
        every time an actor is added, so this must be re-applied after the
        last actor of a render cycle has been added (see _render). The
        physical extent (bounds) is also pinned to the design space so that
        overlay actors like force arrows extending beyond the mechanism
        cannot stretch the axis box.
        """
        if not self.last_params or "Dimensions" not in self.last_params:
            return
        try:
            cube_axes = self.plotter.renderer.cube_axes_actor
            if cube_axes is None:
                return
            nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
            # Pin the physical extent of the axis box to the design space.
            # Use z extent of 0 in 2D so the box stays flat on the xy plane.
            if nelz > 0:
                cube_axes.bounds = (
                    0.0,
                    float(nelx),
                    0.0,
                    float(nely),
                    0.0,
                    float(nelz),
                )
            else:
                cube_axes.bounds = (0.0, float(nelx), 0.0, float(nely), 0.0, 0.0)
            # Use the property setters (they also regenerate the labels)
            cube_axes.x_axis_range = (0.0, float(nelx))
            cube_axes.y_axis_range = (0.0, float(nely))
            if nelz > 0:
                cube_axes.z_axis_range = (0.0, float(nelz))
        except (AttributeError, RuntimeError):
            pass

    def _render(self) -> None:
        """Render the scene after re-applying the exact axis tick ranges."""
        self._fix_bounds_axes_ranges()
        self.plotter.render()

    def _plot_forces(self, is_3d: bool) -> None:
        """
        Plot force vectors on the scene.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        """
        if not self.sections["Forces"].visibility_button.isChecked():
            return
        if not self.last_params or "Forces" not in self.last_params:
            return

        pf: dict = self.last_params["Forces"]
        pd: dict = self.last_params["Dimensions"]
        length: float = np.mean(pd["nelxyz"][:2]) / 6.0

        if (
            self.is_displaying_deformation
            and self.u is not None
            and self.displacement_widget.mov_iter.value() == 1
        ):
            self._plot_deformed_forces(is_3d, pd, length)
        else:
            self._plot_initial_forces(is_3d, pf, length)

    def _arrow_vectors(
        self, dirs: list, length: float, is_3d: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Convert force direction strings to direction vectors.

        Parameters
        ----------
        dirs : list
            List of direction strings (e.g., "X:→", "Y:↑").
        length : float
            Length scaling factor.
        is_3d : bool
            Whether this is a 3D problem.

        Returns
        -------
        tuple
            (dx, dy, dz) - Direction vector components.
        """
        dx: np.ndarray = np.zeros(len(dirs))
        dy: np.ndarray = np.zeros(len(dirs))
        dz: np.ndarray | None = np.zeros(len(dirs)) if is_3d else None

        for i, d in enumerate(dirs):
            c: str = d.split(":")[1]
            if c == "→":
                dx[i] = length
            elif c == "←":
                dx[i] = -length
            elif c == "↑":
                dy[i] = length
            elif c == "↓":
                dy[i] = -length
            elif is_3d and c == "<":
                dz[i] = length
            elif is_3d and c == ">":
                dz[i] = -length

        return dx, dy, dz

    def _add_arrows(
        self,
        starts: np.ndarray,
        directions: np.ndarray,
        color: str,
        length: float,
    ):
        """
        Draw arrow glyphs of uniform length.

        Parameters
        ----------
        starts : np.ndarray
            (N, 3) arrow tail positions.
        directions : np.ndarray
            (N, 3) vectors giving the arrow orientations.
        color : str
            Arrow color.
        length : float
            Arrow length in scene units.

        Returns
        -------
        The created actor (also tracked as an overlay actor).
        """
        starts = np.atleast_2d(np.asarray(starts, dtype=float))
        directions = np.atleast_2d(np.asarray(directions, dtype=float))
        if len(starts) == 0:
            return None
        poly = pv.PolyData(starts)
        poly.point_data["direction"] = directions
        glyphs = poly.glyph(orient="direction", scale=False, factor=length)
        actor = self.plotter.add_mesh(
            glyphs, color=color, reset_camera=False, render=False
        )
        self._overlay_actors.append(actor)
        return actor

    def _plot_initial_forces(self, is_3d: bool, pf: dict, length: float) -> None:
        """
        Plot force vectors at their initial positions.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        pf : dict
            Forces parameter dictionary.
        length : float
            Arrow length scaling factor.
        """
        for prefix, color in [("fi", "red"), ("fo", "blue")]:
            dirs: np.ndarray = np.array(pf[f"{prefix}dir"])
            active: np.ndarray = dirs != "-"
            if not np.any(active):
                continue

            x: np.ndarray = np.array(pf[f"{prefix}x"])[active]
            y: np.ndarray = np.array(pf[f"{prefix}y"])[active]
            z: np.ndarray = (
                np.array(pf.get(f"{prefix}z", []))[active]
                if is_3d
                else np.zeros_like(x)
            )

            dx: np.ndarray
            dy: np.ndarray
            dz: np.ndarray | None
            dx, dy, dz = self._arrow_vectors(dirs[active], length, is_3d)

            starts = np.column_stack([x, y, z])
            vectors = np.column_stack([dx, dy, dz if is_3d else np.zeros_like(dx)])
            self._add_arrows(starts, vectors, color, length)

    def _plot_deformed_forces(self, is_3d: bool, pd: dict, length: float) -> None:
        """
        Plot force vectors at their displaced positions.

        Parameters
        ----------
        is_3d : bool
            Whether this is a 3D plot.
        pd : dict
            Dimensions parameter dictionary.
        length : float
            Arrow length scaling factor.
        """
        nely: int = pd["nelxyz"][1]
        disp_factor: float = self.displacement_widget.mov_disp.value()

        for xk, yk, zk, dk, color in [
            ("fix", "fiy", "fiz", "fidir", "red"),
            ("fox", "foy", "foz", "fodir", "blue"),
        ]:
            active: list = [
                g
                for g in self.forces_widget.inputs
                if dk in g and g[dk].currentText() != "-"
            ]
            if not active:
                continue

            fx: np.ndarray = np.array([g[xk].value() for g in active])
            fy: np.ndarray = np.array([g[yk].value() for g in active])
            fz: np.ndarray | None = (
                np.array([g[zk].value() for g in active]) if is_3d else None
            )
            dirs: list = [g[dk].currentText() for g in active]

            idx: np.ndarray = (
                (fz * (fx + 1) * (nely + 1) + fx * (nely + 1) + fy)
                if is_3d
                else (fx * (nely + 1) + fy)
            )

            dof: int = 3 if is_3d else 2
            ux: np.ndarray = self.u[dof * idx] * disp_factor
            uy: np.ndarray = self.u[dof * idx + 1] * disp_factor
            uz: np.ndarray | None = (
                self.u[dof * idx + 2] * disp_factor if is_3d else None
            )

            fx: np.ndarray = fx + ux  # using += will give an error
            fy: np.ndarray = fy + uy if is_3d else fy - uy
            if is_3d:
                fz: np.ndarray = fz + uz

            dx: np.ndarray
            dy: np.ndarray
            dz: np.ndarray | None
            dx, dy, dz = self._arrow_vectors(dirs, length, is_3d)

            starts = np.column_stack([fx, fy, fz if is_3d else np.zeros_like(fx)])
            vectors = np.column_stack([dx, dy, dz if is_3d else np.zeros_like(dx)])
            self._add_arrows(starts, vectors, color, length)

    def _plot_supports(self, is_3d: bool) -> None:
        """
        Plot support markers on the scene.

        Supports are drawn as black cones.

        Parameters
        ----------
        is_3d : bool
            Whether the scene represents a 3D plot. Controls marker placement.
        """
        if not self.sections["Supports"].visibility_button.isChecked():
            return
        if not self.last_params or "Supports" not in self.last_params:
            return
        # No need to consider the case is_displaying_deformation since the supports don't move
        ps: dict = self.last_params["Supports"]
        max_dimension = max(self.last_params["Dimensions"]["nelxyz"])
        base_scale = max_dimension / 30.0
        positions: list = []
        scales: list = []
        for i, d in enumerate(ps["sdim"]):
            if d == "-":
                continue
            positions.append([ps["sx"][i], ps["sy"][i], ps["sz"][i] if is_3d else 0.0])
            scales.append(base_scale * np.sqrt(1.0 + ps["sr"][i] ** 2))
        if not positions:
            return

        poly = pv.PolyData(np.array(positions, dtype=float))
        poly.point_data["marker_scale"] = np.asarray(scales)
        cone = pv.Cone(
            center=(0, 0, 0),
            direction=(0, 0, 1) if is_3d else (0, 1, 0),
            height=1.0,
            radius=0.5,
            resolution=24,
        )
        glyphs = poly.glyph(geom=cone, orient=False, scale="marker_scale", factor=1.0)
        actor = self.plotter.add_mesh(
            glyphs, color="black", reset_camera=False, render=False
        )
        self._overlay_actors.append(actor)

    def _plot_regions(self, is_3d: bool) -> None:
        """
        Draw region outlines (square/cube or circle/sphere) on the scene.

        Regions are drawn with a green, dashed outline. In 2D a line loop is
        used, while in 3D the function draws a wireframe box or sphere.

        Parameters
        ----------
        is_3d : bool
            Whether the plot is 3D.
        """
        if not self.sections["Regions"].visibility_button.isChecked():
            return
        if self.is_displaying_deformation:
            return  # Region are not relevant in deformation view
        if not self.last_params or "Regions" not in self.last_params:
            return
        pr: dict = self.last_params["Regions"]
        for i, shape in enumerate(pr["rshape"]):
            if shape == "-":
                continue

            r: int = pr["rradius"][i]
            rx: int = pr["rx"][i]
            ry: int = pr["ry"][i]

            if is_3d:
                rz: int = pr["rz"][i]
                if shape == "□":  # Square/Cube
                    mesh = pv.Box(
                        bounds=(rx - r, rx + r, ry - r, ry + r, rz - r, rz + r)
                    )
                else:  # "◯" Circle/Sphere
                    mesh = pv.Sphere(
                        radius=r,
                        center=(rx, ry, rz),
                        theta_resolution=20,
                        phi_resolution=20,
                    )
                actor = self.plotter.add_mesh(
                    mesh,
                    style="wireframe",
                    color="green",
                    line_width=1.0,
                    reset_camera=False,
                    render=False,
                )
                self._set_dotted(actor)
                self._overlay_actors.append(actor)

            else:
                if shape == "□":  # Square/Cube
                    corners = np.array(
                        [
                            [rx - r, ry - r, _OVERLAY_Z],
                            [rx + r, ry - r, _OVERLAY_Z],
                            [rx + r, ry + r, _OVERLAY_Z],
                            [rx - r, ry + r, _OVERLAY_Z],
                        ],
                        dtype=float,
                    )
                    mesh = pv.lines_from_points(corners, close=True)
                else:  # "◯" Circle/Sphere
                    theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
                    points = np.column_stack(
                        [
                            rx + r * np.cos(theta),
                            ry + r * np.sin(theta),
                            np.full(theta.size, _OVERLAY_Z),
                        ]
                    )
                    mesh = pv.lines_from_points(points, close=True)
                actor = self.plotter.add_mesh(
                    mesh,
                    color="green",
                    line_width=1.5,
                    reset_camera=False,
                    render=False,
                )
                self._set_dotted(actor)
                self._overlay_actors.append(actor)

    def _plot_displacement_preview(self, is_3d: bool) -> None:
        """
        Overlay displacement preview arrows on the scene.

        The function samples a subset of elements/nodes and draws arrow
        glyphs representing the displacement vector at those locations. Only
        elements considered to contain material are shown. Scaling and arrow
        density depend on the current UI controls.

        Parameters
        ----------
        is_3d : bool
            Whether to draw 3D arrows or 2D arrows.
        """
        if not self.sections["Displacement"].visibility_button.isChecked():
            return
        if self.is_displaying_deformation:
            return  # The displacement vector doesn't match the deformed shape
        if self.u is None or self.xPhys is None:
            return
        pf: dict = self.last_params["Forces"]
        disp_factor: float = self.displacement_widget.mov_disp.value()
        mean_force: float = np.mean(pf["finorm"][0])
        factor: float = disp_factor / mean_force if mean_force != 0 else disp_factor

        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
        step: int = max(
            1, int(max(nelx, nely, nelz) / 10)
        )  # number of elements to skip between 2 arrows
        if is_3d:
            x_coords, y_coords, z_coords = np.meshgrid(
                np.arange(0, nelx, step),
                np.arange(0, nely, step),
                np.arange(0, nelz, step),
                indexing="xy",
            )

            el_indices: np.ndarray = (
                z_coords * (nelx * nely) + x_coords * nely + y_coords
            ).flatten()
            node_indices: np.ndarray = (
                z_coords * ((nelx + 1) * (nely + 1)) + x_coords * (nely + 1) + y_coords
            ).flatten()
            material_mask: np.ndarray = (
                self.xPhys[el_indices] > 0.5
                if self.xPhys.ndim == 1
                else np.any(self.xPhys[:, el_indices] > 0.5, axis=0)
            )  # Only show arrows in material regions

            # Get the coordinates and displacement vectors at elements center for the valid points
            x_valid: np.ndarray = x_coords.flatten()[material_mask] + 0.5
            y_valid: np.ndarray = y_coords.flatten()[material_mask] + 0.5
            z_valid: np.ndarray = z_coords.flatten()[material_mask] + 0.5
            node_valid: np.ndarray = node_indices[material_mask]

            ux: np.ndarray = self.u[3 * node_valid] * factor
            uy: np.ndarray = -self.u[3 * node_valid + 1] * factor
            uz: np.ndarray = self.u[3 * node_valid + 2] * factor

            starts = np.column_stack([x_valid, y_valid, z_valid])
            vectors = np.column_stack([ux, uy, uz])
            # Skip near-zero vectors (undefined glyph orientation)
            keep = np.linalg.norm(vectors, axis=1) > 1e-12
            if not np.any(keep):
                return
            self._add_arrows(
                starts[keep], vectors[keep], "red", length=disp_factor / 4.0
            )
        else:
            x_coords, y_coords = np.meshgrid(
                np.arange(0, nelx, step), np.arange(0, nely, step), indexing="xy"
            )

            el_indices = (x_coords * nely + y_coords).flatten()
            node_indices = (x_coords * (nely + 1) + y_coords).flatten()

            material_mask = (
                self.xPhys[el_indices] > 0.5
                if self.xPhys.ndim == 1
                else np.any(self.xPhys[:, el_indices] > 0.5, axis=0)
            )  # Only show arrows in material regions

            # Get the coordinates and displacement vectors for the valid points
            x_valid = x_coords.flatten()[material_mask]
            y_valid = y_coords.flatten()[material_mask]
            node_valid = node_indices[material_mask]

            ux = self.u[2 * node_valid] * factor
            uy = -self.u[2 * node_valid + 1] * factor

            starts = np.column_stack([x_valid, y_valid, np.zeros_like(x_valid)])
            vectors = np.column_stack([ux, uy, np.zeros_like(ux)])
            magnitudes = np.linalg.norm(vectors, axis=1)
            max_mag = magnitudes.max() if magnitudes.size else 0.0
            if max_mag <= 0:
                return

            # Scale arrows so the longest one spans one sampling step
            poly = pv.PolyData(starts)
            poly.point_data["direction"] = vectors
            poly.point_data["magnitude"] = magnitudes
            glyphs = poly.glyph(
                orient="direction", scale="magnitude", factor=step / max_mag
            )
            actor = self.plotter.add_mesh(
                glyphs, color="red", reset_camera=False, render=False
            )
            self._overlay_actors.append(actor)
