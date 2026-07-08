# tests/references/regenerate_references.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Regenerate the .npz reference files used by the regression tests.
#
# Run from the repository root whenever an intentional numerical change is
# made to the core (document the change in the commit message):
#
#     python tests/references/regenerate_references.py

from __future__ import annotations
import copy
import json
from pathlib import Path

import numpy as np

from topoptcomec.core import displacements, optimizers

TESTS_DIR = Path(__file__).parent.parent
REFERENCES_DIR = Path(__file__).parent


def _load_presets() -> dict:
    with open(TESTS_DIR / "presets_test.json", "r") as f:
        return json.load(f)


def regenerate_optimizer_references() -> None:
    for preset_name, preset_params in _load_presets().items():
        if preset_params["Materials"]["init_type"] == 2:
            continue  # random init: not compared against references
        is_multi = len(preset_params.get("Materials", {}).get("E", [1.0])) > 1
        params = copy.deepcopy(preset_params)
        params.pop("Displacement", None)
        params["Materials"].pop("color", None)
        if not is_multi:
            params["Materials"].pop("percent", None)

        if is_multi:
            result, u_vec = optimizers.optimize_multimaterial(**params, verbose=False)
        else:
            result, u_vec = optimizers.optimize(**params, verbose=False)

        out = REFERENCES_DIR / f"test_optimizers_with_presets_{preset_name}.npz"
        np.savez_compressed(out, result=result, u_vec=u_vec)
        print(f"wrote {out.name}")


def regenerate_displacement_references() -> None:
    """Mirror the setup of tests/test_displacements.py exactly."""
    for preset_name, preset_params in _load_presets().items():
        if preset_params["Materials"]["init_type"] == 2:
            continue
        params = copy.deepcopy(preset_params)
        nelx, nely, nelz = params["Dimensions"]["nelxyz"]
        is_3d = nelz > 0
        keys_to_remove = ["filter_type", "filter_radius_min", "max_change", "n_it"]
        if not is_3d:
            keys_to_remove += ["rz", "fz", "sz"]
        for key in keys_to_remove:
            params.pop(key, None)

        nel = nelx * nely * (nelz if is_3d else 1)
        ndof = (
            (3 if is_3d else 2) * (nelx + 1) * (nely + 1) * ((nelz + 1) if is_3d else 1)
        )
        p = 1 / preset_params["Dimensions"]["volfrac"] - 1
        x = np.linspace(0, 1, nel)
        densities = x**p
        result = densities[::-1]
        n_forces = sum(1 for fdir in params["Forces"]["fidir"] if fdir != "-")
        u_vec = np.linspace(0.0, 1.0, ndof * n_forces, dtype=float).reshape(
            ndof, n_forces
        )

        linear_arrays = displacements.single_linear_displacement(
            u_vec, nelx, nely, nelz, 1.0
        )

        params["Displacement"]["disp_iterations"] = 2
        for frame in displacements.run_iterative_displacement(params, result):
            last_result_displaced = frame

        params["Displacement"]["disp_factor"] = 0.0
        for frame in displacements.run_iterative_displacement(params, result):
            zero_factor_result_displaced = frame

        out = REFERENCES_DIR / f"test_displacement_with_presets_{preset_name}.npz"
        np.savez_compressed(
            out,
            **{f"linear_{i}": arr for i, arr in enumerate(linear_arrays)},
            last_result_displaced=last_result_displaced,
            zero_factor_result_displaced=zero_factor_result_displaced,
        )
        print(f"wrote {out.name}")


if __name__ == "__main__":
    regenerate_optimizer_references()
    regenerate_displacement_references()
