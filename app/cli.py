# app/cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# CLI entry point of TopoptComec.

from __future__ import annotations
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt

from app.core import analyzers, optimizers
from app.ui import exporters

# Type aliases
FloatArray = npt.NDArray[np.float64]


def _load_or_run_optimization(
    preset_name: str,
    params: dict,
    optimizer_params: dict,
    base_dir: str,
    save_frames: bool,
    verbose: bool,
) -> tuple[FloatArray | None, FloatArray | None, str | None]:
    """
    Load a cached density field or run the optimizer to generate a new one.

    Parameters
    ----------
    preset_name : str
        Preset identifier used to name cache files and logging messages.
    params : dict
        Full preset configuration dictionary.
    optimizer_params : dict
        Parameters passed to the optimizer, stripped of UI-only keys.
    base_dir : str
        Base output directory to use for cached files.
    save_frames : bool
        Whether to save intermediate optimization frames.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    tuple[FloatArray | None, FloatArray | None, str | None]
        xPhys, displacement field u, and an error message if one occurred.
    """
    cache_file: Path = Path(base_dir) / preset_name / f"{preset_name}_density_field.npz"

    if cache_file.exists():
        if verbose:
            print(f"[{preset_name}] Loading cached density field...")
        try:
            data: np.lib.npyio.NpzFile = np.load(cache_file)
            return data["xPhys"], data["u"], None
        except Exception as e:
            if verbose:
                print(f"[{preset_name}] Failed to load cache: {e}")

    if save_frames:

        def progress_callback(
            iteration: int, objective: float, change: float, xPhys_frame: FloatArray
        ) -> bool:
            folder: Path = Path(base_dir) / preset_name / f"{preset_name}_creation"
            folder.mkdir(parents=True, exist_ok=True)
            filename: Path = folder / f"{preset_name}_creation_{iteration}.png"
            colors: list[str] | None = params.get("Materials", {}).get("color", None)
            exporters.save_as_png(
                xPhys_frame, params["Dimensions"]["nelxyz"], str(filename), colors
            )
            return False

        optimizer_params["progress_callback"] = progress_callback(
            preset_name, base_dir, params
        )

    is_multimaterial: bool = (
        len(optimizer_params.get("Materials", {}).get("E", [1.0])) > 1
    )
    try:
        if is_multimaterial:
            xPhys, u = optimizers.optimize_multimaterial(
                **optimizer_params, verbose=verbose
            )
        else:
            xPhys, u = optimizers.optimize(**optimizer_params, verbose=verbose)

        cache_file.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_file, xPhys=xPhys, u=u)
        if verbose:
            print(f"[{preset_name}] Cached density field.")
        return xPhys, u, None
    except Exception as e:
        import traceback

        traceback.print_exc()
        return None, None, f"Optimization failed: {e}"


def _run_analysis(
    preset_name: str,
    params: dict,
    xPhys: FloatArray,
    u: FloatArray,
    verbose: bool,
) -> str | None:
    """
    Run analysis indicators for terminal output.

    Parameters
    ----------
    preset_name : str
        Preset identifier used for logging.
    params : dict
        Full preset configuration dictionary.
    xPhys : FloatArray
        Density field produced by the optimizer.
    u : FloatArray
        Displacement field produced by the solver.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    str | None
        Error message if analysis fails, otherwise None.
    """
    if verbose:
        print(f"[{preset_name}] Running analysis...")

    try:
        analysis_results = analyzers.analyze(
            xPhys, u, params["Dimensions"], params["Forces"]
        )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Analysis failed: {e}"

    print(f"[{preset_name}] Analysis results:")
    labels: list[str] = [
        "Contains checkerboard",
        "Watertight",
        "Thresholded",
        "Efficient",
    ]
    print(
        "\n".join(
            f"{label}: {'yes' if value else 'no'}"
            for label, value in zip(labels, analysis_results)
        )
    )
    return None


def _run_displacement(
    preset_name: str,
    params: dict,
    xPhys: FloatArray,
    base_dir: str,
    save_frames: bool,
    verbose: bool,
) -> str | None:
    """
    Run the displacement simulation and optionally save displacement frames.

    Parameters
    ----------
    preset_name : str
        Preset identifier used to name output files.
    params : dict
        Full preset configuration dictionary.
    xPhys : FloatArray
        Density field produced by the optimizer.
    base_dir : str
        Base output directory to use for displacement frames.
    save_frames : bool
        Whether to write displacement frames to disk.
    verbose : bool
        Whether to print progress messages.

    Returns
    -------
    str | None
        Error message if the displacement run failed, otherwise None.
    """
    if verbose:
        print(f"[{preset_name}] Running displacement...")

    from app.core import displacements

    try:
        disp_iter: int = 0

        def disp_callback(iteration: int) -> bool:
            nonlocal disp_iter
            disp_iter = iteration
            return False

        for frame_data in displacements.run_iterative_displacement(
            params, xPhys, disp_callback
        ):
            if save_frames:
                folder: Path = (
                    Path(base_dir) / preset_name / f"{preset_name}_displacement"
                )
                folder.mkdir(parents=True, exist_ok=True)
                filename: Path = folder / f"{preset_name}_displacement_{disp_iter}.png"
                colors: list[str] | None = params.get("Materials", {}).get(
                    "color", None
                )
                exporters.save_as_png(
                    frame_data, params["Dimensions"]["nelxyz"], str(filename), colors
                )
    except Exception as e:
        import traceback

        traceback.print_exc()
        return f"Displacement failed: {e}"

    return None


def _export_results(
    preset_name: str,
    xPhys: FloatArray,
    nelxyz: list[int],
    format: str,
    verbose: bool,
) -> None:
    """
    Export the optimized density field to the requested output formats.

    Parameters
    ----------
    preset_name : str
        Preset identifier used to name exported files.
    xPhys : FloatArray
        Optimized density field.
    nelxyz : list[int]
        Element counts in the x, y, and z dimensions.
    format : str
        Output format or 'all' for every supported export format.
    verbose : bool
        Whether to print export progress messages.
    """
    results_dir: Path = Path("results")
    results_dir.mkdir(exist_ok=True)
    base_filename: Path = results_dir / preset_name

    formats: list[str] = ["png", "stl", "vti", "3mf"] if format == "all" else [format]
    for f in formats:
        filename: str = str(base_filename.with_suffix(f".{f}"))
        if verbose:
            print(f"[{preset_name}] Saving {f.upper()} to {filename}...")

        success, error_msg = _export(xPhys, nelxyz, filename, f)

        if not success:
            print(f"[{preset_name}] Error saving {f}: {error_msg}")
        else:
            print(f"[{preset_name}] Saved {filename}")


def _run_single_preset(
    preset_name: str,
    params: dict,
    format: str,
    threshold: bool,
    analysis: bool = False,
    run_disp: bool = False,
    save_frames: bool = False,
    preview: bool = False,
    verbose: bool = False,
) -> tuple[str, str | None]:
    """
    Execute a single preset from the CLI workflow.

    This function handles caching, optimization, optional displacement,
    thresholding, and exporting of the final result.

    Parameters
    ----------
    preset_name : str
        Name of the preset to process.
    params : dict
        Preset configuration dictionary.
    format : str
        Export format or 'all' to export all supported formats.
    threshold : bool
        Whether to apply a binary threshold to the final density field.
    verbose : bool, optional
        Whether to print diagnostic messages.
    run_disp : bool, optional
        Whether to run displacement analysis after optimization.
    preview : bool, optional
        Whether to render a terminal preview of the result.
    save_frames : bool, optional
        Whether to save intermediate frames during optimization and displacement.

    Returns
    -------
    tuple[str, str | None]
        The preset name and an error message if execution failed.
    """
    is_pytest = os.environ.get("PYTEST_CURRENT_TEST") is not None
    base_dir = "tests" if is_pytest else "results"

    optimizer_params: dict = params.copy()
    optimizer_params.pop("Displacement", None)

    is_multimaterial: bool = (
        len(optimizer_params.get("Materials", {}).get("E", [1.0])) > 1
    )
    if "Materials" in optimizer_params:
        optimizer_params["Materials"].pop("color", None)
        if not is_multimaterial:
            optimizer_params["Materials"].pop("percent", None)

    xPhys: FloatArray | None
    u: FloatArray | None
    error: str | None
    xPhys, u, error = _load_or_run_optimization(
        preset_name, params, optimizer_params, base_dir, save_frames, verbose
    )
    if error:
        return preset_name, error

    if analysis and u is not None:
        error = _run_analysis(preset_name, params, xPhys, u, verbose)
        if error:
            return preset_name, error

    if run_disp and u is not None:
        error = _run_displacement(
            preset_name, params, xPhys, base_dir, save_frames, verbose
        )
        if error:
            return preset_name, error

    if threshold:
        if verbose:
            print(f"[{preset_name}] Applying threshold (0.5)...")
        xPhys = np.where(xPhys > 0.5, 1.0, 0.0)

    if preview:
        from app.cli_preview import render_preview

        print(render_preview(xPhys, params["Dimensions"]["nelxyz"]))

    _export_results(preset_name, xPhys, params["Dimensions"]["nelxyz"], format, verbose)

    if not verbose:
        print(f"Preset '{preset_name}' completed.")
    return preset_name, None


def _export(
    xPhys: FloatArray, nelxyz: list[int], filename: str, format: str
) -> tuple[bool, str | None]:
    """
    Dispatch export to the correct exporter function.

    Parameters
    ----------
    xPhys : FloatArray
        Element densities.
    nelxyz : list[int]
        Number of elements in [x, y, z] directions.
    filename : str
        Output file path.
    format : str
        Export format ("png", "vti", "stl", or "3mf").

    Returns
    -------
    tuple[bool, str | None]
        True and None when export succeeds, otherwise False and an error message.
    """
    if format == "png":
        return exporters.save_as_png(xPhys, nelxyz, filename)
    elif format == "vti":
        return exporters.save_as_vti(xPhys, nelxyz, filename)
    elif format == "stl":
        return exporters.save_as_stl(xPhys, nelxyz, filename)
    elif format == "3mf":
        return exporters.save_as_3mf(xPhys, nelxyz, filename)
    return False, f"Unknown format: {format}"


def run_cli() -> None:
    """
    Parse command-line arguments and execute the requested preset workflows.

    The CLI supports single or parallel preset execution, optional displacement,
    and output export to several file formats.
    """
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="TopoptComec CLI - Topology Optimization for Compliant Mechanisms"
    )
    parser.add_argument(
        "-p",
        "--preset",
        type=str,
        required=True,
        help="Preset name(s) from presets.json (comma-separated for parallel runs)",
    )
    parser.add_argument(
        "-f",
        "--format",
        type=str,
        default="all",
        choices=["png", "stl", "vti", "3mf", "all"],
        help="Output format (png, stl, vti, 3mf). Default: all",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        action="store_true",
        help="Binarize the result (black and white)",
    )
    parser.add_argument(
        "-a",
        "--analysis",
        action="store_true",
        help="Run quick analysis on the optimized result and print it to the terminal",
    )
    parser.add_argument(
        "-d",
        "--displacement",
        action="store_true",
        help="Run displacement automatically after optimization",
    )
    parser.add_argument(
        "-i",
        "--intermediate",
        action="store_true",
        help="Save intermediate frames for both optimization and displacement",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Show a terminal preview of the result after optimization",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Suppress intermediate optimizer output (useful for parallel runs)",
    )

    args: argparse.Namespace = parser.parse_args()

    # Load presets
    presets_path: Path = Path("presets.json")
    if not presets_path.exists():
        print(f"Error: presets.json not found at {presets_path.absolute()}")
        sys.exit(1)

    try:
        with open(presets_path, "r") as f:
            presets: dict[str, dict] = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error reading presets.json: {e}")
        sys.exit(1)

    # Parse and validate preset names
    preset_names: list[str] = [name.strip() for name in args.preset.split(",")]
    for name in preset_names:
        if name not in presets:
            print(f"Error: Preset '{name}' not found in presets.json")
            print("Available presets:", ", ".join(presets.keys()))
            sys.exit(1)

    # Run
    if len(preset_names) == 1:
        _, error = _run_single_preset(
            preset_names[0],
            presets[preset_names[0]],
            format=args.format,
            threshold=args.threshold,
            analysis=args.analysis,
            run_disp=args.displacement,
            save_frames=args.intermediate,
            preview=args.preview,
            verbose=args.verbose,
        )
        if error:
            print(error)
            sys.exit(1)
    else:
        max_workers: int = min(len(preset_names), os.cpu_count() or 1)
        print(
            f"Running {len(preset_names)} presets in parallel ({max_workers} workers)"
        )
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures: dict[object, str] = {
                executor.submit(
                    _run_single_preset,
                    name,
                    presets[name],
                    format=args.format,
                    threshold=args.threshold,
                    analysis=args.analysis,
                    run_disp=args.displacement,
                    save_frames=args.intermediate,
                    preview=args.preview,
                    verbose=args.verbose,
                ): name
                for name in preset_names
            }
            errors: list[str] = []
            for future in futures:
                preset_name, error = future.result()
                if error:
                    errors.append(f"  {preset_name}: {error}")

        if errors:
            print("Errors occurred:\n" + "\n".join(errors))
            sys.exit(1)

    print("Done.")
