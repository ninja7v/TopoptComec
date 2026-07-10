# topoptcomec/ui/plotting.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Plotting class.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb
from matplotlib.patches import Rectangle


class PlottingMixin:
    """Mixin for MainWindow to handle all plotting operations."""

    def _style_plot_default(self) -> None:
        """
        Apply the default white theme styling to the plot.

        Sets figure and axes background to white with black labels and spines.
        """
        self.figure.patch.set_facecolor("white")
        if self.figure.get_axes():
            ax = self.figure.get_axes()[0]
            ax.set_facecolor("white")
            ax.xaxis.label.set_color("black")
            ax.yaxis.label.set_color("black")
            ax.tick_params(axis="x", colors="black")
            ax.tick_params(axis="y", colors="black")
            for spine in ax.spines.values():
                spine.set_edgecolor("black")
        self.canvas.draw()

    def _plot_deformation(
        self, ax: plt.Axes, is_3d: bool, nelx: int, nely: int, nelz: int
    ):
        """
        Plot the deformed shape based on displacement results.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
        is_3d : bool
            Whether this is a 3D problem.
        nelx, nely, nelz : int
            Number of elements in each dimension.
        """
        if (
            self.last_params["Displacement"]["disp_iterations"] == 1
        ):  # Single-frame grid plot
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

                # Colors with alpha = density
                is_multi_3d: bool = hasattr(self.xPhys, "ndim") and self.xPhys.ndim > 1
                if is_multi_3d:
                    n_mat, _ = self.xPhys.shape
                    rgb_3d: np.ndarray = np.ones((nel, 3))
                    for i in range(n_mat):
                        mat_rgb: np.ndarray = np.array(
                            to_rgb(self.materials_widget.inputs[i]["color"].get_color())
                        )
                        rgb_3d += self.xPhys[i, :, np.newaxis] * (mat_rgb - 1.0)
                    rgb_3d = np.clip(rgb_3d, 0.0, 1.0)
                    alpha_3d: np.ndarray = self.xPhys.sum(axis=0).clip(0.0, 1.0)
                    colors: np.ndarray = np.zeros((nel, 4))
                    colors[:, :3] = rgb_3d
                    colors[:, 3] = alpha_3d
                else:
                    colors = np.zeros((nel, 4))
                    colors[:, :3] = to_rgb(
                        self.materials_widget.inputs[0]["color"].get_color()
                    )
                    colors[:, 3] = self.xPhys

                # Scatter plot of displaced centers
                ax.scatter(
                    cx,
                    cy,
                    cz,
                    s=6000 / max(nelx, nely, nelz),
                    marker="s",
                    c=colors,
                    alpha=None,
                )

                ax.set_box_aspect([nelx, nely, nelz])
            else:
                X, Y = self.last_displayed_frame_data

                is_multi: bool = hasattr(self.xPhys, "ndim") and self.xPhys.ndim > 1
                if is_multi:
                    n_mat, nel = self.xPhys.shape
                    rgb_image: np.ndarray = np.ones((nel, 3))  # Start white
                    for i in range(n_mat):
                        mat_rgb: np.ndarray = np.array(
                            to_rgb(self.materials_widget.inputs[i]["color"].get_color())
                        )
                        # Blend: pixel = sum(rho_i * color_i)
                        rgb_image += self.xPhys[i, :, np.newaxis] * (mat_rgb - 1.0)
                    rgb_image = np.clip(rgb_image, 0.0, 1.0)
                    # Matplotlib's pcolormesh natively accepts 3D RGB arrays for the C parameter.
                    # We reshape the (nel, 3) list into the 2D grid shape (nelx, nely, 3).
                    ax.pcolormesh(
                        X,
                        Y,
                        rgb_image.reshape((nelx, nely, 3)),
                        shading="auto",
                    )
                else:
                    # Single-material logic
                    hex_color: str = to_hex(
                        self.materials_widget.inputs[0]["color"].get_color()
                    )
                    color_cmap: LinearSegmentedColormap = (
                        LinearSegmentedColormap.from_list(
                            "material_shades",
                            [hex_color, "#ffffff"],  # selected material color → white
                        )
                    )
                    ax.pcolormesh(
                        X,
                        Y,
                        -self.xPhys.reshape((nelx, nely)),
                        cmap=color_cmap,
                        shading="auto",
                    )
        else:
            if self.sections["Materials"].visibility_button.isChecked():
                self._plot_material(
                    ax,
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

    def _show_initial_message(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Display placeholder message on the plot before optimization results exist.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
        is_3d : bool
            Whether this is a 3D plot.
        """
        if self.footer.create_button.graphicsEffect() is not None:
            init_message: str = 'Configure parameters and press "Create"'
            if is_3d:
                ax.text2D(
                    0.5,
                    0.5,
                    init_message,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=16,
                    alpha=0.5,
                    color="black",
                )
            else:
                ax.text(
                    0.5,
                    0.5,
                    s=init_message,
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=16,
                    alpha=0.5,
                    color="black",
                )

    def replot(self) -> None:
        """
        Redraw the entire plot including all visible layers.

        Determines which layers to show based on visibility button states
        and whether displaying deformation or static results.
        """
        if not self.last_params:
            return  # Do nothing if triggerd in sections initialization
        self.figure.clear()
        self.figure.patch.set_facecolor("white")
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]
        is_3d: bool = nelz > 0
        if is_3d:
            ax = self.figure.add_subplot(111, projection="3d", facecolor="white")
        else:
            ax = self.figure.add_subplot(111, facecolor="white")

        # Layer 1: The Main Result (Material)
        if (
            self.is_displaying_deformation
            and self.last_displayed_frame_data is not None
        ):
            self._plot_deformation(ax, is_3d, nelx, nely, nelz)
        else:
            if self.sections["Materials"].visibility_button.isChecked():
                if self.xPhys is None:
                    self._initialize_xphys(nelx, nely, nelz, is_3d)
                self._plot_material(ax, is_3d=is_3d)
            # Show initial message if xPhys is not a result (even partial) of optimization
            self._show_initial_message(ax, is_3d)

        self._redraw_non_material_layers(ax, is_3d)
        if not is_3d:
            ax.set_aspect("equal", "box")
        ax.autoscale(tight=True)
        self.canvas.draw()

    def _plot_material(
        self, ax: plt.Axes, is_3d: bool, xPhys_data: np.ndarray | None = None
    ) -> None:
        """
        Plot the material density field.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
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

        # Detect multi-material: shape (n_mat, nel)
        is_multi: bool = data_to_plot.ndim == 2

        ax.clear()
        if is_3d:
            self._plot_material_3d(ax, data_to_plot, nelx, nely, nelz, is_multi)
        else:
            self._plot_material_2d(ax, data_to_plot, nelx, nely, is_multi)

    def _plot_material_2d(
        self, ax: plt.Axes, data: np.ndarray, nelx: int, nely: int, is_multi: bool
    ):
        """
        Plot 2D material density field.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
        data : np.ndarray
            Density data of shape (nel,) or (n_mat, nel).
        nelx, nely : int
            Number of elements in x and y dimensions.
        is_multi : bool
            Whether this is multi-material data.
        """
        if is_multi:
            n_mat: int
            nel: int
            n_mat, nel = data.shape
            rgb_image: np.ndarray = np.ones((nel, 3))  # Start white
            for i in range(n_mat):
                mat_rgb: np.ndarray = np.array(
                    to_rgb(self.materials_widget.inputs[i]["color"].get_color())
                )
                # Blend: pixel = sum(rho_i * color_i)
                rgb_image += data[i, :, np.newaxis] * (mat_rgb - 1.0)
            rgb_image = np.clip(rgb_image, 0.0, 1.0)
            rgb_image = rgb_image.reshape((nelx, nely, 3)).transpose(1, 0, 2)

            ax.imshow(
                rgb_image,
                interpolation="nearest",
                origin="lower",
                extent=[0, nelx, 0, nely],
            )
        else:
            mat_color = self.materials_widget.inputs[0]["color"].get_color()
            cmap = LinearSegmentedColormap.from_list(
                "custom_cmap", ["white", mat_color]
            )
            image = data.reshape((nelx, nely)).T
            ax.imshow(
                image,
                cmap=cmap,
                interpolation="nearest",
                origin="lower",
                norm=plt.Normalize(0, 1),
                extent=[0, nelx, 0, nely],
            )

    def _plot_material_3d(
        self,
        ax: plt.Axes,
        data: np.ndarray,
        nelx: int,
        nely: int,
        nelz: int,
        is_multi: bool,
    ):
        """
        Plot 3D material density field as a scatter plot.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib 3D axes object.
        data : np.ndarray
            Density data of shape (nel,) or (n_mat, nel).
        nelx, nely, nelz : int
            Number of elements in each dimension.
        is_multi : bool
            Whether this is multi-material data.
        """
        eff_density: np.ndarray = data.sum(axis=0) if is_multi else data
        visible_mask: np.ndarray = eff_density > 0.01
        visible_idx: np.ndarray = np.where(visible_mask)[0]
        if len(visible_idx) == 0:
            return

        z: np.ndarray = visible_idx // (nelx * nely)
        x: np.ndarray = (visible_idx % (nelx * nely)) // nely
        y: np.ndarray = visible_idx % nely
        colors: np.ndarray = np.ones((len(visible_idx), 4))  # RGBA, start white
        if is_multi:
            n_mat: int = data.shape[0]
            for i in range(n_mat):
                mat_rgb: np.ndarray = np.array(
                    to_rgb(self.materials_widget.inputs[i]["color"].get_color())
                )
                rho_vis: np.ndarray = data[i, visible_idx]
                colors[:, :3] += rho_vis[:, np.newaxis] * (mat_rgb - 1.0)
            colors[:, :3] = np.clip(colors[:, :3], 0.0, 1.0)
            colors[:, 3] = np.clip(eff_density[visible_idx], 0.0, 1.0)
        else:
            base_color_rgb: np.ndarray = to_rgb(
                self.materials_widget.inputs[0]["color"].get_color()
            )
            colors[:, :3] = base_color_rgb
            colors[:, 3] = eff_density[visible_idx]

        ax.scatter(
            x + 0.5,
            y + 0.5,
            z + 0.5,
            s=6000 / max(nelx, nely, nelz),
            marker="s",
            c=colors,
            alpha=None,
        )
        ax.set_box_aspect([nelx, nely, nelz])

    def _redraw_non_material_layers(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Redraw overlay layers (forces, supports, regions, dimensions, displacement preview).

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
        is_3d : bool
            Whether this is a 3D plot.
        """
        # Layer 2: Overlays
        self._plot_forces(ax, is_3d=is_3d)
        self._plot_supports(ax, is_3d=is_3d)
        self._plot_regions(ax, is_3d=is_3d)
        self._plot_dimensions_frame(ax, is_3d=is_3d)
        self._plot_displacement_preview(ax, is_3d=is_3d)

    def _plot_dimensions_frame(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Draw a dotted frame around the design space.
        Controlled by the Dimensions section's visibility button.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
        is_3d : bool
            Whether this is a 3D plot.
        """
        if not self.sections["Dimensions"].visibility_button.isChecked():
            ax.set_xlabel("")
            ax.set_ylabel("")
            if is_3d:
                ax.set_zlabel("")
            ax.set_xticks([])
            ax.set_yticks([])
            if is_3d:
                ax.set_zticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            return

        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params["Dimensions"]["nelxyz"]

        if is_3d:
            # Define the 8 vertices of the box
            verts: list[tuple[int, int, int]] = [
                (0, 0, 0),
                (nelx, 0, 0),
                (nelx, nely, 0),
                (0, nely, 0),
                (0, 0, nelz),
                (nelx, 0, nelz),
                (nelx, nely, nelz),
                (0, nely, nelz),
            ]
            # Define the 12 edges by connecting the vertices
            edges: list[tuple[int, int]] = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 0),
                (4, 5),
                (5, 6),
                (6, 7),
                (7, 4),
                (0, 4),
                (1, 5),
                (2, 6),
                (3, 7),
            ]
            for edge in edges:
                points = [verts[edge[0]], verts[edge[1]]]
                x, y, z = zip(*points)
                ax.plot(x, y, z, color="gray", linestyle=":", linewidth=1.5)
        else:
            rect: Rectangle = Rectangle(
                (0, 0),
                nelx,
                nely,
                fill=False,
                edgecolor="gray",
                linestyle=":",
                linewidth=1.5,
            )
            ax.add_patch(rect)

        ax.set_xlabel("X", color="black")
        ax.set_ylabel("Y", color="black")
        ax.yaxis.label.set_rotation(0)  # Display Y label vertically
        if is_3d:
            ax.set_zlabel("Z", color="black")
        ax.tick_params(axis="x", colors="black")
        ax.tick_params(axis="y", colors="black")
        if is_3d:
            ax.tick_params(axis="z", colors="black")
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor("black")

    def _plot_forces(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Plot force vectors on the axes.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
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
            self._plot_deformed_forces(ax, is_3d, pd, length)
        else:
            self._plot_initial_forces(ax, is_3d, pf, length)

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

    def _plot_initial_forces(
        self, ax: plt.Axes, is_3d: bool, pf: dict, length: float
    ) -> None:
        """
        Plot force vectors at their initial positions.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
        is_3d : bool
            Whether this is a 3D plot.
        pf : dict
            Forces parameter dictionary.
        length : float
            Arrow length scaling factor.
        """
        for prefix, color in [("fi", "r"), ("fo", "b")]:
            dirs: np.ndarray = np.array(pf[f"{prefix}dir"])
            active: np.ndarray = dirs != "-"
            if not np.any(active):
                continue

            x: np.ndarray = np.array(pf[f"{prefix}x"])[active]
            y: np.ndarray = np.array(pf[f"{prefix}y"])[active]
            z: np.ndarray | None = (
                np.array(pf.get(f"{prefix}z", []))[active] if is_3d else None
            )

            dx: np.ndarray
            dy: np.ndarray
            dz: np.ndarray | None
            dx, dy, dz = self._arrow_vectors(dirs[active], length, is_3d)

            if is_3d:
                ax.quiver(
                    x,
                    y,
                    z,
                    dx,
                    dy,
                    dz,
                    color=color,
                    length=length,
                    normalize=True,
                    arrow_length_ratio=0.3,
                )
            else:
                ax.quiver(x, y, dx, dy, color=color, units="xy", scale=1, width=0.5)

    def _plot_deformed_forces(
        self, ax: plt.Axes, is_3d: bool, pd: dict, length: float
    ) -> None:
        """
        Plot force vectors at their displaced positions.

        Parameters
        ----------
        ax : plt.Axes
            Matplotlib axes object.
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
            ("fix", "fiy", "fiz", "fidir", "r"),
            ("fox", "foy", "foz", "fodir", "b"),
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
            ux: np.ndarray = self.u[dof * idx, 0] * disp_factor
            uy: np.ndarray = self.u[dof * idx + 1, 0] * disp_factor
            uz: np.ndarray | None = (
                self.u[dof * idx + 2, 0] * disp_factor if is_3d else None
            )

            fx: np.ndarray = fx + ux  # using += will give an error
            fy: np.ndarray = fy + uy if is_3d else fy - uy
            if is_3d:
                fz: np.ndarray = fz + uz

            dx: np.ndarray
            dy: np.ndarray
            dz: np.ndarray | None
            dx, dy, dz = self._arrow_vectors(dirs, length, is_3d)

            if is_3d:
                ax.quiver(
                    fx,
                    fy,
                    fz,
                    dx,
                    dy,
                    dz,
                    color=color,
                    length=length,
                    normalize=True,
                    arrow_length_ratio=0.3,
                )
            else:
                ax.quiver(fx, fy, dx, dy, color=color, units="xy", scale=1, width=0.5)

    def _plot_supports(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Plot support markers on the provided axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes (2D or 3D) where supports will be drawn.
        is_3d : bool
            Whether the axes represent a 3D plot. Controls marker placement
            and plotting API used.
        """
        if not self.sections["Supports"].visibility_button.isChecked():
            return
        if not self.last_params or "Supports" not in self.last_params:
            return
        # No need to consider the case is_displaying_deformation since the supports don't move
        ps: dict = self.last_params["Supports"]
        for i, d in enumerate(ps["sdim"]):
            if d == "-":
                continue
            pos: list = [ps["sx"][i], ps["sy"][i], ps["sz"][i]]
            size: int = 80 + 200 * ps["sr"][i] ** 2
            if is_3d:
                ax.scatter(
                    pos[0],
                    pos[1],
                    pos[2],
                    s=size,
                    marker="^",
                    c="black",
                    depthshade=False,
                )
            else:
                ax.scatter(pos[0], pos[1], s=size, marker="^", c="black")

    def _plot_regions(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Draw region outlines (square/cube or circle/sphere) on the axes.

        Regions are drawn with a green, dashed outline. In 2D a `matplotlib`
        patch (Rectangle or Circle) is added, while in 3D the function plots
        the corresponding edges or a wireframe sphere.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes for drawing.
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
                    # Define the 8 vertices of the cube
                    verts = np.array(
                        [
                            [rx - r, ry - r, rz - r],
                            [rx + r, ry - r, rz - r],
                            [rx + r, ry + r, rz - r],
                            [rx - r, ry + r, rz - r],
                            [rx - r, ry - r, rz + r],
                            [rx + r, ry - r, rz + r],
                            [rx + r, ry + r, rz + r],
                            [rx - r, ry + r, rz + r],
                        ]
                    )
                    # Define the 12 edges connecting the vertices
                    edges = [
                        (0, 1),
                        (1, 2),
                        (2, 3),
                        (3, 0),
                        (4, 5),
                        (5, 6),
                        (6, 7),
                        (7, 4),
                        (0, 4),
                        (1, 5),
                        (2, 6),
                        (3, 7),
                    ]
                    for edge in edges:
                        points = verts[list(edge)]
                        # Note: Matplotlib's 3D axes are ordered (X, Y, Z)
                        ax.plot(
                            points[:, 0],
                            points[:, 1],
                            points[:, 2],
                            color="green",
                            linestyle=":",
                        )

                elif shape == "◯":  # Circle/Sphere
                    # Create the surface grid for the sphere
                    u = np.linspace(0, 2 * np.pi, 20)
                    v = np.linspace(0, np.pi, 20)
                    # Parametric equations for a sphere
                    x = rx + r * np.outer(np.cos(u), np.sin(v))
                    y = ry + r * np.outer(np.sin(u), np.sin(v))
                    z = rz + r * np.outer(np.ones(np.size(u)), np.cos(v))
                    ax.plot_wireframe(x, y, z, color="green", linestyle=":")

            else:
                if shape == "□":  # Square/Cube
                    rect: plt.Rectangle = plt.Rectangle(
                        (rx - r, ry - r),
                        2 * r,
                        2 * r,
                        fill=False,
                        edgecolor="green",
                        linestyle=":",
                    )
                    ax.add_patch(rect)
                elif shape == "◯":  # Circle/Sphere
                    circ: plt.Circle = plt.Circle(
                        (rx, ry), r, fill=False, edgecolor="green", linestyle=":"
                    )
                    ax.add_patch(circ)

    def _plot_displacement_preview(self, ax: plt.Axes, is_3d: bool) -> None:
        """
        Overlay displacement preview vectors (quivers) on the plot.

        The function samples a subset of elements/nodes and draws quiver
        arrows representing the displacement vector at those locations. Only
        elements considered to contain material are shown. Scaling and arrow
        density depend on the current UI controls.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Target axes to draw displacement arrows.
        is_3d : bool
            Whether to draw 3D quivers or 2D arrows.
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
        step: int = (
            max(nelx, nely, nelz) / 10
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

            ux: np.ndarray = self.u[3 * node_valid, 0] * factor
            uy: np.ndarray = -self.u[3 * node_valid + 1, 0] * factor
            uz: np.ndarray = self.u[3 * node_valid + 2, 0] * factor

            ax.quiver(
                x_valid,
                y_valid,
                z_valid,
                ux,
                uy,
                uz,
                color="red",
                length=disp_factor / 4.0,
                normalize=True,
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

            ux = self.u[2 * node_valid, 0] * factor
            uy = -self.u[2 * node_valid + 1, 0] * factor

            ax.quiver(
                x_valid,
                y_valid,
                ux,
                uy,
                color="red",
                scale=40,
                scale_units="xy",
                angles="xy",
            )
