# topoptcomec/time_estimation.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Estimate runtime for optimization, displacement and analysis.


class TimeEstimation:
    """
    Estimate runtime for the different stages of the workflow
    (optimization, displacement, analysis) from a parameters dictionary.

    The estimates are rough orders of magnitude used to color the GUI's
    "Create" button and to print indicative messages in the CLI.

    Parameters
    ----------
    params : dict
        Complete parameters dictionary as produced by ``_gather_parameters``.
    """

    # Thresholds on the estimated cost metric (1.1 * (nbIter+1) * nbSolves *
    # sizeMatrix**1.5) and the associated (color, label) pairs.
    _OPTIMIZATION_TIERS: list[tuple[float, str, str]] = [
        (10_000_000, "#00FF0D", "Very fast (a few seconds)"),
        (100_000_000, "#91FF00", "Fast (less than a minute)"),
        (1_000_000_000, "#FBC02D", "Medium (a few minutes)"),
        (10_000_000_000, "#FF7300", "Slow (less than an hour)"),
    ]
    _OPTIMIZATION_VERY_SLOW: tuple[str, str] = (
        "#FF0000",
        "Very slow (more than an hour)",
    )

    def __init__(self, params: dict) -> None:
        self.params: dict = params

    # ------------------------------------------------------------------ #
    # Optimization                                                       #
    # ------------------------------------------------------------------ #

    def _compute_size_matrix(self) -> int:
        """
        Compute the number of degrees of freedom of the FE problem.

        This is ``(nelx+1)(nely+1)(nelz+1)*dim`` minus the fixed dofs
        induced by supports (with their optional radius).
        """
        nelx: int
        nely: int
        nelz: int
        nelx, nely, nelz = self.params.get("Dimensions", {}).get("nelxyz", [1, 1, 1])
        is_3d: bool = nelz > 0
        dim: int = 3 if is_3d else 2

        supports: dict = self.params.get("Supports", {})
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

        size_matrix: int = (nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1) * dim
        if size_matrix < 0:
            size_matrix = 0
        return size_matrix

    @staticmethod
    def _estimated_solving_cost(
        nb_iter: int, nb_solves: int, size_matrix: int
    ) -> float:
        """
        Rough cost metric for a FE-based stage.

        ``nb_iter + 1`` accounts for preparation (roughly one iteration).
        ``size_matrix**1.5`` is the sparse solver cost.
        """
        return (nb_iter + 1) * nb_solves * (size_matrix**1.5)

    def optimization_cost(self) -> float:
        """
        Return the rough cost metric for the optimization stage.
        """
        size_matrix = self._compute_size_matrix()
        forces: dict = self.params.get("Forces", {})
        nb_iteration: int = len([d for d in forces.get("fidir", []) if d != "-"]) + len(
            [d for d in forces.get("fodir", []) if d != "-"]
        )
        nb_solves: int = self.params.get("Optimizer", {}).get("n_it", 50)

        # 1.1 adds 10% to account for filtering and bookkeeping.
        return 1.1 * self._estimated_solving_cost(nb_iteration, nb_solves, size_matrix)

    def optimization_indicators(self) -> tuple[str, str]:
        """
        Return ``(color, label)`` for the optimization stage.

        ``color`` is a hex color code used by the GUI's "Create" button;
        ``label`` is a short human-readable description used in tooltips and
        CLI messages.
        """
        est_time = self.optimization_cost()
        for threshold, color, label in self._OPTIMIZATION_TIERS:
            if est_time < threshold:
                return color, label
        color, label = self._OPTIMIZATION_VERY_SLOW
        return color, label

    def displacement_cost(self) -> float:
        """
        Return the rough cost metric for the displacement stage.
        """
        size_matrix = self._compute_size_matrix()
        forces: dict = self.params.get("Forces", {})
        nb_iteration: int = len([d for d in forces.get("fidir", []) if d != "-"])
        nb_solves: int = self.params.get("Optimizer", {}).get("n_it", 50)

        return self._estimated_solving_cost(nb_iteration, nb_solves, size_matrix)

    def analysis_cost(self) -> float:
        """
        Return the rough cost metric for the analysis stage.
        """
        return 0.0  # Instant
