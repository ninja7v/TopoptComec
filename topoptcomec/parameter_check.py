# topoptcomec/parameter_check.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Check input parameters.

import numpy as np
import numpy.typing as npt

from topoptcomec.core import preset_format
from topoptcomec.presets_io import (
    check_directions,
    validate_preset_structure,
)

# Type aliases
FloatArray = npt.NDArray[np.float64]


class ParameterCheck:
    """
    Validate optimization parameter dictionaries for both the GUI and the CLI.

    Instances are stateless except for an optional reference to the last
    successful density field, which is required when the user requests
    initialization from the current result (``init_type == 3``).

    Parameters
    ----------
    last_successful_xPhys : FloatArray or None
        Density field produced by the last successful optimization run, or
        ``None`` when no such result is available.
    """

    def __init__(self, last_successful_xPhys: FloatArray | None = None) -> None:
        self.last_successful_xPhys = last_successful_xPhys

    def validate(self, params: dict) -> str | None:
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
        # Reject malformed presets before any direct indexing below; this
        # keeps shareable JSON files from raising KeyError/TypeError.
        structure_error = validate_preset_structure(params)
        if structure_error:
            return structure_error

        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        if nelx <= 0 or nely <= 0 or nelz < 0:
            return "Nx, Ny, Nz must be positive."

        # Validate initialization from current result if chosen
        if params.get("Materials", {}).get("init_type") == 3:
            last_x = self.last_successful_xPhys
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
        solver: str = po.get("solver", "Auto")
        if solver not in ("Direct", "Iterative", "Auto"):
            return f"Unknown solver '{solver}' (expected Direct, Iterative or Auto)."
        filter_type: str = po.get("filter_type", "Sensitivity")
        if filter_type not in ("Sensitivity", "Density", "None"):
            return (
                f"Unknown filter type '{filter_type}' "
                "(expected Sensitivity, Density or None)."
            )

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
        Validate the force entries.

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
        output_directions = pf.get("fodir", [])
        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        is_3d: bool = nelz > 0

        if not any(d != "-" for d in pf["fidir"]):
            return "At least one input force must be active"

        if not any(d != "-" for d in output_directions) and not any(
            d != "-" for d in ps.get("sdim", [])
        ):
            return "At least one output force (for compliant mechanisms) or support (for rigid mechanisms) must be active"

        for prefix, label, norm_key in (
            ("fi", "Input", "finorm"),
            ("fo", "Output", "fonorm"),
        ):
            dirs: list = pf.get(f"{prefix}dir", [])
            # z coordinates are only required for 3D problems.
            keys = [f"{prefix}x", f"{prefix}y", norm_key]
            if is_3d:
                keys.append(f"{prefix}z")
            if any(len(pf.get(k, [])) != len(dirs) for k in keys):
                return f"The {label.lower()} force arrays have inconsistent lengths."
            for i, direction in enumerate(dirs):
                if direction == "-":
                    continue
                x = pf[f"{prefix}x"][i]
                y = pf[f"{prefix}y"][i]
                z = pf[f"{prefix}z"][i] if is_3d else 0
                if not (0 <= x <= nelx and 0 <= y <= nely):
                    return (
                        f"{label} force {i + 1} at ({x}, {y}) is outside the "
                        f"design domain [0, {nelx}] x [0, {nely}]."
                    )
                if is_3d and not 0 <= z <= nelz:
                    return (
                        f"{label} force {i + 1} is outside the design domain "
                        f"along z (0..{nelz})."
                    )
                if not np.isfinite(pf[norm_key][i]) or pf[norm_key][i] <= 0:
                    return f"{label} force {i + 1} must have a positive magnitude."

        def z_of(key: str, i: int) -> int:
            return pf[key][i] if is_3d else 0

        err: str | None = self._check_duplicates(
            [i for i, d in enumerate(pf["fidir"]) if d != "-"],
            lambda i: (pf["fix"][i], pf["fiy"][i], z_of("fiz", i), pf["fidir"][i]),
            lambda a, b: f"Input forces {a + 1} and {b + 1} are identical.",
        )
        if err:
            return err

        return self._check_duplicates(
            [i for i, d in enumerate(output_directions) if d != "-"],
            lambda i: (pf["fox"][i], pf["foy"][i], z_of("foz", i), pf["fodir"][i]),
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
        pr: dict = params.get("Regions")
        if not pr:
            return

        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        for i, shape in enumerate(pr.get("rshape", [])):
            if shape == "-":
                continue
            x = pr["rx"][i]
            y = pr["ry"][i]
            if not (0 <= x <= nelx and 0 <= y <= nely):
                return (
                    f"Region {i + 1} at ({x}, {y}) is outside the design "
                    f"domain [0, {nelx}] x [0, {nely}]."
                )
            if nelz > 0 and not 0 <= pr["rz"][i] <= nelz:
                return (
                    f"Region {i + 1} is outside the design domain along z (0..{nelz})."
                )
            if pr["rradius"][i] <= 0:
                return f"Region {i + 1} must have a positive radius."
            states = pr.get("rstate", [])
            state = states[i] if i < len(states) else "Void"
            if state != "Void":
                material = preset_format.region_material_index(state)
                n_materials = len(params.get("Materials", {}).get("E", [1.0]))
                if material is None or not 0 <= material < n_materials:
                    return (
                        f"Region {i + 1} references {state!r}, but only "
                        f"{n_materials} material(s) are defined."
                    )

        idx = [i for i, s in enumerate(pr["rshape"]) if s != "-"]

        rz = pr.get("rz", [0] * len(pr["rshape"]))
        return self._check_duplicates(
            idx,
            lambda i: (
                pr["rshape"][i],
                pr["rradius"][i],
                pr["rx"][i],
                pr["ry"][i],
                rz[i] if nelz > 0 else 0,
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

        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        error = check_directions(list(ps.get("sdim", [])), "Support")
        if error:
            return error
        for i, dim in enumerate(ps.get("sdim", [])):
            if dim == "-":
                continue
            x = ps["sx"][i]
            y = ps["sy"][i]
            if not (0 <= x <= nelx and 0 <= y <= nely):
                return (
                    f"Support {i + 1} at ({x}, {y}) is outside the design "
                    f"domain [0, {nelx}] x [0, {nely}]."
                )
            if nelz > 0 and not 0 <= ps["sz"][i] <= nelz:
                return (
                    f"Support {i + 1} is outside the design domain along z (0..{nelz})."
                )
            if not any(axis in dim for axis in ("X", "Y", "Z")):
                return f"Support {i + 1} must fix at least one direction."

        idx: list = [i for i, s in enumerate(ps["sdim"]) if s != "-"]

        sz = ps.get("sz", [0] * len(ps["sdim"]))
        return self._check_duplicates(
            idx,
            lambda i: (
                ps["sx"][i],
                ps["sy"][i],
                sz[i] if nelz > 0 else 0,
                ps["sdim"][i],
            ),
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

        for i, (E, nu) in enumerate(zip(pm["E"], pm["nu"])):
            if not np.isfinite(E) or E <= 0:
                return f"Material {i + 1}: Young's modulus must be positive."
            if not np.isfinite(nu) or nu < 0.0 or nu >= 0.5:
                return f"Material {i + 1}: Poisson's ratio must satisfy 0 <= nu < 0.5."
        percent = pm.get("percent", [100] * len(pm["E"]))
        if len(percent) != len(pm["E"]):
            return "Each material must have a volume percentage."
        if any(p < 0 or p > 100 for p in percent):
            return "Material percentages must be between 0 and 100."

        err = self._check_duplicates(
            range(len(pm["E"])),
            lambda i: (pm["E"][i], pm["nu"][i], percent[i]),
            lambda a, b: f"Materials {a + 1} and {b + 1} are identical.",
        )
        if err:
            return err

        if len(pm["E"]) > 1 and sum(percent) != 100:
            return "Material percentages don't sum up to 100%."
