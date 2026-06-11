# app/ui/parameter_manager.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Manage parameters (json file to UI, comparison, ...).

import copy
import os
import numpy as np
from PySide6.QtWidgets import QMessageBox
from matplotlib.colors import to_hex


class ParameterManagerMixin:
    """Mixin for MainWindow to handle parameter gathering, validation, and equivalency checks."""

    def _gather_parameters(self) -> dict:
        """
        Collect all parameters from UI controls into a nested dictionary.

        Returns
        -------
        dict
            Complete parameters dictionary with Dimensions, Regions, Forces,
            Supports, Materials, Optimizer, and Displacement sections.
        """
        params = {}

        # --- Dimensions ---
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = (
            self.dim_widget.nx.value(),
            self.dim_widget.ny.value(),
            self.dim_widget.nz.value(),
        )
        params["Dimensions"] = {
            "nelxyz": [nelx, nely, nelz],
            "volfrac": self.dim_widget.volfrac.value(),
        }

        # --- Regions (optional) ---
        Regions = {
            "rshape": [],
            "rstate": [],
            "rradius": [],
            "rx": [],
            "ry": [],
            "rz": [],
        }
        for rw in self.regions_widget.inputs:
            Regions["rshape"].append(rw["rshape"].currentText())
            Regions["rstate"].append(rw["rstate"].currentText())
            Regions["rradius"].append(rw["rradius"].value())
            Regions["rx"].append(rw["rx"].value())
            Regions["ry"].append(rw["ry"].value())
            Regions["rz"].append(rw["rz"].value())
        if Regions["rshape"]:  # only add if there is at least one region
            params["Regions"] = Regions

        # --- Forces ---
        Forces = {
            "fix": [],
            "fiy": [],
            "fiz": [],
            "fidir": [],
            "finorm": [],
            "fox": [],
            "foy": [],
            "foz": [],
            "fodir": [],
            "fonorm": [],
        }
        for fw in self.forces_widget.inputs:
            if "fix" in fw:  # Input force
                Forces["fix"].append(fw["fix"].value())
                Forces["fiy"].append(fw["fiy"].value())
                Forces["fiz"].append(fw["fiz"].value())
                Forces["fidir"].append(fw["fidir"].currentText())
                Forces["finorm"].append(fw["finorm"].value())
            elif "fox" in fw:  # Output force
                Forces["fox"].append(fw["fox"].value())
                Forces["foy"].append(fw["foy"].value())
                Forces["foz"].append(fw["foz"].value())
                Forces["fodir"].append(fw["fodir"].currentText())
                Forces["fonorm"].append(fw["fonorm"].value())
        params["Forces"] = Forces

        # --- Supports (optional) ---
        Supports = {"sx": [], "sy": [], "sz": [], "sdim": [], "sr": []}
        for sw in self.supports_widget.inputs:
            Supports["sx"].append(sw["sx"].value())
            Supports["sy"].append(sw["sy"].value())
            Supports["sz"].append(sw["sz"].value())
            Supports["sdim"].append(sw["sdim"].currentText())
            Supports["sr"].append(sw["sr"].value())
        if Supports["sx"]:
            params["Supports"] = Supports

        # --- Materials ---
        Materials = {"E": [], "nu": [], "percent": [], "color": []}
        for mat in self.materials_widget.inputs:
            Materials["E"].append(mat["E"].value())
            Materials["nu"].append(mat["nu"].value())
            Materials["percent"].append(mat["percent"].value())
            Materials["color"].append(to_hex(mat["color"].get_color()))
        Materials["init_type"] = self.materials_widget.mat_init_type.currentIndex()
        params["Materials"] = Materials

        # --- Optimizer ---
        Optimizer = {
            "filter_type": (
                "Sensitivity"
                if self.optimizer_widget.opt_ft.currentIndex() == 0
                else (
                    "Density"
                    if self.optimizer_widget.opt_ft.currentIndex() == 1
                    else "None"
                )
            ),
            "filter_radius_min": self.optimizer_widget.opt_fr.value(),
            "penal": self.optimizer_widget.opt_p.value(),
            "eta": self.optimizer_widget.opt_eta.value(),
            "max_change": self.optimizer_widget.opt_max_change.value(),
            "n_it": self.optimizer_widget.opt_n_it.value(),
            "solver": self.optimizer_widget.opt_solver.currentText(),
            "save_frames": self.optimizer_widget.save_frames_cb.isChecked(),
        }
        params["Optimizer"] = Optimizer

        # --- Displacement (optional) ---
        Displacement = {
            "disp_factor": self.displacement_widget.mov_disp.value(),
            "disp_iterations": self.displacement_widget.mov_iter.value(),
            "save_frames": self.displacement_widget.save_frames_cb.isChecked(),
        }
        params["Displacement"] = Displacement

        return params

    def _get_time_estimation_indicators(self, params: dict) -> tuple[str, str]:
        """
        Calculate optimization time estimation indicators.

        Parameters
        ----------
        params : dict
            Current parameters dictionary.

        Returns
        -------
        tuple[str, str]
            (color, tooltip_text) - Color hex code and tooltip description
            indicating estimated runtime.
        """
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.last_params.get("Dimensions", {}).get("nelxyz", [1, 1, 1])
        is_3d: bool = nelz > 0
        dim = 3 if is_3d else 2

        # Get sizeMatrix (get number of fixed dofs for it)
        supports: dict = self.last_params.get("Supports", {})
        fixed_dofs: int = 0
        if supports:
            sx: list = supports.get("sx", [])
            sy: list = supports.get("sy", [])
            sz: list = supports.get("sz", [])
            sr: list = supports.get("sr", [0] * len(sx))
            sdim: list = supports.get("sdim", [])
            active_sup: list = [i for i, val in enumerate(sdim) if val != "-"]

            for i in active_sup:
                fixed_dofs += 1

                if i < len(sr) and sr[i] > 0:
                    radius = sr[i]
                    x_range = range(
                        max(0, int(sx[i] - radius)),
                        min(nelx + 1, int(sx[i] + radius + 1)),
                    )
                    y_range = range(
                        max(0, int(sy[i] - radius)),
                        min(nely + 1, int(sy[i] + radius + 1)),
                    )
                    z_range = (
                        range(
                            max(0, int(sz[i] - radius)),
                            min(nelz + 1, int(sz[i] + radius + 1)),
                        )
                        if is_3d
                        else range(1)
                    )

                    for z in z_range:
                        for x in x_range:
                            for y in y_range:
                                dist_sq = (
                                    (x - sx[i]) ** 2
                                    + (y - sy[i]) ** 2
                                    + ((z - sz[i]) ** 2 if is_3d else 0)
                                )
                                if dist_sq <= radius**2:
                                    fixed_dofs += 1
                coef = 0
                if "X" in sdim[i]:
                    coef += 1
                if "Y" in sdim[i]:
                    coef += 1
                if is_3d and "Z" in sdim[i]:
                    coef += 1
                fixed_dofs *= coef

        sizeMatrix: int = (nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1) * dim
        if sizeMatrix < 0:
            sizeMatrix = 0

        # Get nbIteration
        forces: dict = self.last_params.get("Forces", {})
        nbIteration: int = len([d for d in forces.get("fidir", []) if d != "-"]) + len(
            [d for d in forces.get("fodir", []) if d != "-"]
        )

        # Get nbSolves
        nbSolves: int = self.last_params.get("Optimizer", {}).get("n_it", 50)

        def time_estimated(nbIter: int, nbSolves: int, sMatrix: int) -> float:
            # 1.1 to add 10% of the time to consider filtering, ...
            # NbIteration is the number of force (both input and output)
            # NbIteration+1 to consider preparation time (which is roughly as long as an iteration)
            # SizeMatrix is the number of degree of freedom: (nelx+1)(nely+1)(nelz+1)*2 - fixed degree of freedom (see supports)
            # SizeMatrix^1.5 is the solver time for sparse matrix
            return 1.1 * (nbIter + 1) * nbSolves * (sMatrix**1.5)

        est_time = time_estimated(nbIteration, nbSolves, sizeMatrix)

        tooltip_text = "Start the optimization process\n\nEstimation:"
        if est_time < 10000000:
            color = "#00FF0D"
            tooltip_text += " Very fast (a few seconds)"
        elif est_time < 100000000:
            color = "#91FF00"
            tooltip_text += " Fast (less than a minute)"
        elif est_time < 1000000000:
            color = "#FBC02D"
            tooltip_text += " Medium (a few minutes)"
        elif est_time < 10000000000:
            color = "#FF7300"
            tooltip_text += " Slow (less than an hour)"
        else:
            color = "#FF0000"
            tooltip_text += " Very slow (more than an hour)"
        return color, tooltip_text

    def on_parameter_changed(self) -> None:
        """React when a parameter is changed."""
        # First, check if a valid result from a previous run exists.
        if self.xPhys is not None:
            self.xPhys = None
            self.u = None
            self.is_displaying_deformation = False
            self.last_displayed_frame_data = None

            # Disable buttons that require a valid result
            self.footer.binarize_button.setEnabled(False)
            self.footer.save_button.setEnabled(False)
            self.analysis_widget.run_analysis_button.setEnabled(False)
            self.displacement_widget.run_disp_button.setEnabled(False)
            self.displacement_widget.button_stack.setCurrentWidget(
                self.displacement_widget.run_disp_button
            )
            self.sections["Displacement"].visibility_button.setEnabled(False)

            # Reset the analysis
            self.analysis_widget.checkerboard_result.setText("-")
            self.analysis_widget.watertight_result.setText("-")
            self.analysis_widget.threshold_result.setText("-")
            self.analysis_widget.efficiency_result.setText("-")
            self.analysis_widget.checkerboard_result.setStyleSheet("")
            self.analysis_widget.watertight_result.setStyleSheet("")
            self.analysis_widget.threshold_result.setStyleSheet("")
            self.analysis_widget.efficiency_result.setStyleSheet("")

            # Inform the user what happened
            self.status_bar.showMessage(
                "Parameters changed. Please run 'Create' for a new result.", 3000
            )

        # Replot
        self.last_params = self._gather_parameters()
        self.replot()

        # Play the animation and update tooltip
        color, tooltip_text = self._get_time_estimation_indicators(self.last_params)
        self.footer.start_create_button_effect(color_hex=color)
        self.footer.create_button.setToolTip(tooltip_text)

        # Check if the current state matches the selected preset
        current_preset_name = self.preset.presets_combo.currentText()
        if current_preset_name in self.presets:
            if not self._are_parameters_equivalent(
                self.presets[current_preset_name], self.last_params
            ):
                # The parameters have changed, so deselect the preset
                self.preset.presets_combo.blockSignals(True)
                self.preset.presets_combo.setCurrentIndex(
                    0
                )  # Set to "Select a preset..."
                self.preset.presets_combo.blockSignals(False)
                self.preset.delete_preset_button.setEnabled(False)

    def _are_parameters_equivalent(self, params1: dict, params2: dict) -> bool:
        """Compares two parameter dictionaries, ignoring irrelevant data."""
        # Create deep copies to avoid modifying the original dictionaries
        p1 = copy.deepcopy(params1)
        p2 = copy.deepcopy(params2)

        self._normalize_params(p1)
        self._normalize_params(p2)

        return p1 == p2

    def _normalize_params(self, p: dict) -> None:
        """
        Normalize parameters for comparison by removing irrelevant keys.

        Parameters
        ----------
        p : dict
            Parameters dictionary to normalize (modified in place).
        """
        pd: dict = p["Dimensions"]
        if "nelxyz" in pd:
            is_2d: bool = len(pd["nelxyz"]) < 3 or pd["nelxyz"][2] == 0.0
            if is_2d:
                pd["nelxyz"] = pd["nelxyz"][:2]

        self._normalize_regions(p, is_2d)
        self._normalize_supports(p, is_2d)
        self._normalize_forces(p, is_2d)
        self._normalize_materials(p)

        # Remove irrelevant keys
        p.pop("Displacement", None)
        p.pop("Optimizer", None)

    def _normalize_regions(self, p: dict, is_2d: bool) -> None:
        """
        Normalize region parameters for comparison.

        Parameters
        ----------
        p : dict
            Parameters dictionary containing Regions.
        is_2d : bool
            Whether this is a 2D problem.
        """
        if "Regions" in p:
            pr: dict = p["Regions"]
            if "rshape" in pr:
                zipped = zip(
                    pr.get("rshape", []),
                    pr.get("rstate", []),
                    pr.get("rradius", []),
                    pr.get("rx", []),
                    pr.get("ry", []),
                    pr.get("rz", []) if not is_2d else [0] * len(pr.get("rx", [])),
                )
                active = [r for r in zipped if r[0] != "-"]
                if active:
                    (
                        pr["rshape"],
                        pr["rstate"],
                        pr["rradius"],
                        pr["rx"],
                        pr["ry"],
                        pr["rz"],
                    ) = map(list, zip(*active))
                else:
                    for key in ["rshape", "rstate", "rradius", "rx", "ry", "rz"]:
                        pr.pop(key, None)  # pop them, not just empty them

                if is_2d and "rz" in pr:
                    pr.pop("rz")

    def _normalize_supports(self, p: dict, is_2d: bool) -> None:
        """
        Normalize support parameters for comparison.

        Parameters
        ----------
        p : dict
            Parameters dictionary containing Supports.
        is_2d : bool
            Whether this is a 2D problem.
        """
        if "Supports" in p:
            ps: dict = p["Supports"]
            if "sdim" in ps:
                zipped = zip(
                    ps.get("sx", []),
                    ps.get("sy", []),
                    ps.get("sz", []) if not is_2d else [0] * len(ps.get("sx", [])),
                    ps.get("sdim", []),
                    ps.get("sr", []),
                )
                active = [s for s in zipped if s[3] != "-"]
                if active:
                    (ps["sx"], ps["sy"], ps["sz"], ps["sdim"], ps["sr"]) = map(
                        list, zip(*active)
                    )
                else:
                    p.pop("Supports", None)
                if is_2d and "sz" in ps:
                    ps.pop("sz")

    def _normalize_forces(self, p: dict, is_2d: bool) -> None:
        """
        Normalize force parameters for comparison.

        Parameters
        ----------
        p : dict
            Parameters dictionary containing Forces.
        is_2d : bool
            Whether this is a 2D problem.
        """
        pf: dict = p["Forces"]
        for prefix in ["fi", "fo"]:
            dir_key: str = f"{prefix}dir"
            if dir_key in pf:
                keys = [
                    f"{prefix}x",
                    f"{prefix}y",
                    f"{prefix}z",
                    f"{prefix}dir",
                    f"{prefix}norm",
                ]
                vals: list = [pf.get(k, []) for k in keys]
                zipped: tuple = zip(*vals)
                active: list = []
                # Keep if direction is not "-"
                active: list = [item for item in zipped if item[3] != "-"]

                if active:
                    unzipped: list = list(zip(*active))
                    for i, k in enumerate(keys):
                        pf[k] = list(unzipped[i])
                else:
                    for k in keys:
                        pf.pop(k, None)

                if is_2d and f"{prefix}z" in pf:
                    pf.pop(f"{prefix}z")

    def _normalize_materials(self, p: dict) -> None:
        """
        Normalize material parameters for comparison.

        Parameters
        ----------
        p : dict
            Parameters dictionary containing Materials.
        """
        if "Materials" in p:
            pm: dict = p["Materials"]
            if len(pm["E"]) == 1:
                # If only one material, ignore the percentage
                pm["percent"] = [100]
                if "color" not in pm or pm["color"] == [""]:
                    pm["color"] = ["#000000"]  # Default black color
                if "init_type" not in pm:
                    pm["init_type"] = 0

    def _validate_parameters(self, params: dict) -> str | None:
        """
        Validate the full parameters dictionary before running optimization.

        Performs a sequence of targeted checks (domain, regions, forces,
        supports, materials and optimizer settings). Returns an error string
        describing the first detected issue, or `None` when the parameters are
        valid.

        Parameters
        ----------
        params : dict
            Complete parameters dictionary as produced by `_gather_parameters`.

        Returns
        -------
        str or None
            Error message describing the invalid parameter, or `None` when
            validation passes.
        """
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        if nelx <= 0 or nely <= 0 or nelz < 0:
            return "Nx, Ny, Nz must be positive."

        # Validate initialization from current result if chosen
        if params.get("Materials", {}).get("init_type") == 3:
            last_x = getattr(self, "last_successful_xPhys", None)
            if last_x is None:
                return "No current result available to initialize from."
            nel: int = nelx * nely * (nelz if nelz > 0 else 1)
            last_x_nel: int = last_x.shape[1] if last_x.ndim == 2 else last_x.size
            if last_x_nel != nel:
                return f"Current result grid size ({last_x_nel}) does not match the active grid dimensions ({nel})."

        err = (
            self._check_domain(params)
            or self._check_regions(params)
            or self._check_forces(params)
            or self._check_supports(params)
            or self._check_materials(params)
            or self._check_optimizer(params)
        )
        return err

    def _check_domain(self, params: dict) -> str | None:
        """
        Validate domain-related configuration values.
        """
        pd: dict = params.get("Dimensions", {})
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = pd.get("nelxyz", [0, 0, 0])
        if nelx < 0 or nely < 0 or nelz < 0:
            return "Dimensions must not be negative."
        volfrac: float = pd.get("volfrac", 0.0)
        if volfrac <= 0.0 or volfrac > 1.0:
            return "Volume fraction must be between 0 and 1."

    def _check_optimizer(self, params: dict) -> str | None:
        """
        Validate optimizer-related parameters.
        """
        po: dict = params.get("Optimizer", {})
        if po.get("filter_radius_min", 0.0) < 0.0:
            return "Filter radius must be non-negative."
        eta: float = po.get("eta", 0.0)
        if eta <= 0.0:
            return "Optimizer eta must be positive."
        if eta > 1.0:
            return "Optimizer eta must be at most 1.0."
        penal: float = po.get("penal", 0.0)
        if penal < 1.0:
            return "Penalization must be at least 1.0."
        if penal > 10.0:
            return "Penalization must be at most 10.0."
        if po.get("max_change", 0.0) <= 0.0:
            return "Optimizer max_change must be positive."
        if po.get("n_it", 0) <= 0:
            return "Optimizer iteration count must be positive."

    def _check_duplicates(
        self, indices: list, keyfunc: callable, msg: callable
    ) -> str | None:
        """
        Check for duplicate entries and return error message if found.

        Parameters
        ----------
        indices : list
            List of indices to check.
        keyfunc : callable
            Function to extract comparison key from index.
        msg : callable
            Function to generate error message (takes two indices).

        Returns
        -------
        str or None
            Error message if duplicate found, None otherwise.
        """
        seen = {}
        for i in indices:
            k = keyfunc(i)
            if k in seen:
                return msg(seen[k], i)
            seen[k] = i

    def _check_forces(self, params: dict) -> str | None:
        """
        Validate that at least one input force is active.

        Parameters
        ----------
        params : dict
            Parameters dictionary.

        Returns
        -------
        str or None
            Error message if validation fails, None otherwise.
        """
        pf: dict = params["Forces"]
        ps: dict = params.get("Supports", {})

        if not any(d != "-" for d in pf["fidir"]):
            return "At least one input force must be active"

        if not any(d != "-" for d in pf["fodir"]) and not any(
            d != "-" for d in ps.get("sdim", [])
        ):
            return "At least one output force (for compliant mechanisms) or support (for rigid mechanisms) must be active"

        err: str | None = self._check_duplicates(
            [i for i, d in enumerate(pf["fidir"]) if d != "-"],
            lambda i: (pf["fix"][i], pf["fiy"][i], pf["fiz"][i], pf["fidir"][i]),
            lambda a, b: f"Input forces {a + 1} and {b + 1} are identical.",
        )
        if err:
            return err

        return self._check_duplicates(
            [i for i, d in enumerate(pf["fodir"]) if d != "-"],
            lambda i: (pf["fox"][i], pf["foy"][i], pf["foz"][i], pf["fodir"][i]),
            lambda a, b: f"Output forces {a + 1} and {b + 1} are identical.",
        )

    def _check_regions(self, params: dict) -> str | None:
        """
        Check for duplicate regions.

        Parameters
        ----------
        params : dict
            Parameters dictionary.

        Returns
        -------
        str or None
            Error message if duplicate found, None otherwise.
        """
        pr: dict = params.get("Region")
        if not pr:
            return

        idx = [i for i, s in enumerate(pr["rshape"]) if s != "-"]

        return self._check_duplicates(
            idx,
            lambda i: (
                pr["rshape"][i],
                pr["rradius"][i],
                pr["rx"][i],
                pr["ry"][i],
                pr["rz"][i],
            ),
            lambda a, b: f"Regions {a + 1} and {b + 1} are identical.",
        )

    def _check_supports(self, params: dict) -> str | None:
        """
        Check for duplicate supports.

        Parameters
        ----------
        params : dict
            Parameters dictionary.

        Returns
        -------
        str or None
            Error message if duplicate found, None otherwise.
        """
        ps: dict = params.get("Supports")
        if not ps:
            return

        idx: list = [i for i, s in enumerate(ps["sdim"]) if s != "-"]

        return self._check_duplicates(
            idx,
            lambda i: (ps["sx"][i], ps["sy"][i], ps["sz"][i], ps["sdim"][i]),
            lambda a, b: f"Supports {a + 1} and {b + 1} are identical.",
        )

    def _check_materials(self, params: dict) -> str | None:
        """
        Check for duplicate materials and validate percentages sum to 100.

        Parameters
        ----------
        params : dict
            Parameters dictionary.

        Returns
        -------
        str or None
            Error message if validation fails, None otherwise.
        """
        pm: dict = params["Materials"]

        err = self._check_duplicates(
            range(len(pm["E"])),
            lambda i: (pm["E"][i], pm["nu"][i], pm.get("percent", [100])[i]),
            lambda a, b: f"Materials {a + 1} and {b + 1} are identical.",
        )
        if err:
            return err

        if len(pm["E"]) > 1 and sum(pm["percent"]) != 100:
            return "Material percentages don't sum up to 100%."

    def _update_position_ranges(self) -> None:
        """
        Update maximum values for position spin boxes based on current dimensions.

        Ensures position inputs don't exceed the current mesh size.
        """
        nelx = self.dim_widget.nx.value()
        nely = self.dim_widget.ny.value()
        nelz = self.dim_widget.nz.value()

        # Update ranges for all regions
        for rw in self.regions_widget.inputs:
            rw["rx"].setMaximum(nelx)
            rw["ry"].setMaximum(nely)
            rw["rz"].setMaximum(nelz)
            rw["rradius"].setMaximum(
                min(nelx, nely, nelz) if nelz > 0 else min(nelx, nely)
            )

        # Update input forces
        for fw in self.forces_widget.inputs:
            if "fix" in fw:
                fw["fix"].setMaximum(nelx)
                fw["fiy"].setMaximum(nely)
                fw["fiz"].setMaximum(nelz)
            elif "fox" in fw:
                fw["fox"].setMaximum(nelx)
                fw["foy"].setMaximum(nely)
                fw["foz"].setMaximum(nelz)

        # Update supports
        for sw in self.supports_widget.inputs:
            sw["sx"].setMaximum(nelx)
            sw["sy"].setMaximum(nely)
            sw["sz"].setMaximum(nelz)

    def _scale_parameters(self) -> None:
        """
        Scale all dimensional and positional parameters by the scale factor.

        Shows error if scaling would move values out of range, and warning
        if rounding would lose proportions.
        """
        scale: float = self.dim_widget.scale.value()
        if scale == 1.0:
            self.status_bar.showMessage("Scale is 1.0, nothing to do.", 3000)
            return

        is_3d: bool = self.dim_widget.nz.value() > 0
        axes: list = ["x", "y", "z"] if is_3d else ["x", "y"]

        # Track dimensions separately so they can be scaled BEFORE positions
        dims_to_scale: list = [self.dim_widget.nx, self.dim_widget.ny]
        if is_3d:
            dims_to_scale.append(self.dim_widget.nz)

        pos_to_validate: list = []
        pos_to_scale: list = []

        def register(widget, active: bool, is_radius: bool = False) -> None:
            if active:
                pos_to_validate.append(widget)
            pos_to_scale.append((widget, is_radius))

        # --- Gather parameters ---
        # Gather Regions
        for rw in self.regions_widget.inputs:
            active = rw["rshape"].currentText() != "-"
            for ax in axes:
                register(rw["r" + ax], active)
            # Original logic validated rradius regardless of active state
            register(rw["rradius"], True, True)

        # Gather Forces
        for fw in self.forces_widget.inputs:
            if "fidir" in fw:
                active = fw["fidir"].currentText() != "-"
                for ax in axes:
                    register(fw["fi" + ax], active)
            elif "fodir" in fw:
                active = fw["fodir"].currentText() != "-"
                for ax in axes:
                    register(fw["fo" + ax], active)

        # Gather Supports
        for sw in self.supports_widget.inputs:
            active: bool = sw["sdim"].currentText() != "-"
            for ax in axes:
                register(sw["s" + ax], active)

        # --- Validation ---
        proceed_impossible: bool = False
        warn_needed: bool = False

        for w in dims_to_scale + pos_to_validate:
            val: float = w.value() * scale
            if (val < 1.0 or val > 1000.0) and w.value() > 0.0:
                proceed_impossible = True
            elif abs(val - round(val)) > 1e-6:
                warn_needed = True

        if proceed_impossible:
            QMessageBox.critical(
                self, "Scaling Error", "Scaling would lead position(s) out of range."
            )
            return

        if warn_needed:
            reply = QMessageBox.question(
                self,
                "Scaling Warning",
                "Scaling would lose initial proportions due to rounding(s). Proceed?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if reply == QMessageBox.StandardButton.No:
                return

        # --- Perform Scaling ---
        # Temporarily block signals to prevent multiple replots
        self._block_all_parameter_signals(True)

        for w in dims_to_scale:
            w.setValue(round(w.value() * scale))

        if scale > 1.0:
            self._update_position_ranges()  # Update max ranges before scaling positions otherwise they might get clamped

        for w, is_radius in pos_to_scale:
            new_val = round(w.value() * scale)
            if is_radius:
                new_val = max(1, new_val)
            w.setValue(new_val)

        if scale < 1.0:
            self._update_position_ranges()  # Update max ranges after scaling positions otherwise values might be clamped before scaling

        self._block_all_parameter_signals(False)

        # Manually trigger a single, final update
        self.on_parameter_changed()
        self.status_bar.showMessage(
            f"All parameters scaled by a factor of {scale}.", 3000
        )

    def _block_all_parameter_signals(self, block: bool) -> None:
        """
        Block or unblock signals for all parameter widgets.

        Parameters
        ----------
        block : bool
            True to block signals, False to unblock.
        """
        all_widgets = [
            self.dim_widget.nx,
            self.dim_widget.ny,
            self.dim_widget.nz,
            self.dim_widget.volfrac,
            self.materials_widget.mat_init_type,
            self.optimizer_widget.opt_ft,
            self.optimizer_widget.opt_fr,
            self.optimizer_widget.opt_p,
            self.optimizer_widget.opt_eta,
            self.optimizer_widget.opt_max_change,
            self.optimizer_widget.opt_n_it,
            self.optimizer_widget.opt_solver,
        ]
        for w in all_widgets:
            w.blockSignals(block)
        for group in (
            self.regions_widget.inputs
            + self.forces_widget.inputs
            + self.supports_widget.inputs
            + self.materials_widget.inputs
        ):
            for w in group.values():
                w.blockSignals(block)
