# tests/test_exporters.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the exporters.

from pathlib import Path

import numpy as np

from app.ui import exporters


def test_exporters():
    """Unit Test: Runs the exporters."""
    filepath = Path(__file__).parent / "test_output_file"

    # Generate a 2D single material mock result
    (nelx_2d, nely_2d, nelz_2d) = 30, 20, 0
    nel_2d = nelx_2d * nely_2d
    p = (
        1 / 0.3 - 1
    )  # f(x) = (x/volfrac)^p -> integral(f(x)) from 0 to nel = volfrac * nel
    x = np.linspace(0, 1, nel_2d)
    densities_2d = x**p
    np.random.shuffle(densities_2d)
    result_2d = densities_2d

    # Generate a 3D single material mock result
    nelx_3d, nely_3d, nelz_3d = 30, 20, 10
    nel_3d = nelx_3d * nely_3d * nelz_3d
    x = np.linspace(0, 1, nel_3d)
    densities_3d = x**p
    np.random.shuffle(densities_3d)
    result_3d = densities_3d

    # Generate Multi-material mock results
    result_2d_multi = np.vstack([result_2d * 0.5, result_2d * 0.3])
    result_3d_multi = np.vstack([result_3d * 0.5, result_3d * 0.3])
    colors = ["#FF0000", "#00FF00"]

    png_path = str(Path(__file__).parent / "test_output_file.png")

    for res, dims, is_multi in [
        (result_2d, (nelx_2d, nely_2d, nelz_2d), False),
        (result_3d, (nelx_3d, nely_3d, nelz_3d), False),
        (result_2d_multi, (nelx_2d, nely_2d, nelz_2d), True),
        (result_3d_multi, (nelx_3d, nely_3d, nelz_3d), True),
    ]:
        args = (res, dims, png_path) if not is_multi else (res, dims, png_path, colors)
        success, error_msg = exporters.save_as_png(*args)
        assert success, f"PNG export failed: {error_msg}"

        args = (
            (res, dims, str(filepath) + ".vti")
            if not is_multi
            else (res, dims, str(filepath) + ".vti")
        )
        success, error_msg = exporters.save_as_vti(*args)
        assert success, f"VTI export failed: {error_msg}"

        args = (
            (res, dims, str(filepath) + ".stl")
            if not is_multi
            else (res, dims, str(filepath) + ".stl")
        )
        success, error_msg = exporters.save_as_stl(*args)
        assert success, f"STL export failed: {error_msg}"

        threemf_path = str(Path(__file__).parent / "test_output_file.3mf")
        args = (
            (res, dims, threemf_path)
            if not is_multi
            else (res, dims, threemf_path, colors)
        )
        success, error_msg = exporters.save_as_3mf(*args)
        assert success, f"3MF export failed: {error_msg}"
