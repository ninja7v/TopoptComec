# topoptcomec/time_estimation.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Estimate runtime for optimization, displacement and analysis.

from typing import ClassVar


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
    _OPTIMIZATION_TIERS: ClassVar[list[tuple[float, str, str]]] = [
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
        Compute the number of free degrees of freedom of the FE problem.

        This is ``(nelx+1)(nely+1)(nelz+1)*dim`` minus the DOFs removed by
        supports (with their optional radius). The result is never below one.
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
                coef = 0
                if "X" in sdim[i]:
                    coef += 1
                if "Y" in sdim[i]:
                    coef += 1
                if is_3d and "Z" in sdim[i]:
                    coef += 1
                if coef == 0:
                    continue
                nodes: set[tuple[int, int, int]] = set()
                radius = sr[i] if i < len(sr) else 0
                z0 = sz[i] if i < len(sz) else 0
                if radius > 0:
                    for z in range(
                        max(0, int(z0 - radius)),
                        min((nelz if is_3d else 0) + 1, int(z0 + radius + 1)),
                    ):
                        for x in range(
                            max(0, int(sx[i] - radius)),
                            min(nelx + 1, int(sx[i] + radius + 1)),
                        ):
                            for y in range(
                                max(0, int(sy[i] - radius)),
                                min(nely + 1, int(sy[i] + radius + 1)),
                            ):
                                if (
                                    (x - sx[i]) ** 2
                                    + (y - sy[i]) ** 2
                                    + ((z - z0) ** 2 if is_3d else 0)
                                ) <= radius**2:
                                    nodes.add((x, y, z))
                else:
                    nodes.add((int(sx[i]), int(sy[i]), int(z0)))
                fixed_dofs += len(nodes) * coef

        size_matrix: int = (nelx + 1) * (nely + 1) * (nelz + 1 if is_3d else 1) * dim
        return max(1, size_matrix - fixed_dofs)

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
        nb_iteration: int = int(any(d != "-" for d in forces.get("fidir", []))) + int(
            any(d != "-" for d in forces.get("fodir", []))
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
        nb_iteration: int = int(any(d != "-" for d in forces.get("fidir", [])))
        nb_solves: int = self.params.get("Optimizer", {}).get("n_it", 50)

        return self._estimated_solving_cost(nb_iteration, nb_solves, size_matrix)

    def analysis_cost(self) -> float:
        """
        Return the rough cost metric for the analysis stage.
        """
        return 0.0  # Instant
