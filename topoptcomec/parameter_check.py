# topoptcomec/parameter_check.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Check input parameters.

import numpy as np
import numpy.typing as npt

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
        output_directions = pf.get("fodir", [])

        if not any(d != "-" for d in pf["fidir"]):
            return "At least one input force must be active"

        if not any(d != "-" for d in output_directions) and not any(
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
            [i for i, d in enumerate(output_directions) if d != "-"],
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
        pr: dict = params.get("Regions")
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
