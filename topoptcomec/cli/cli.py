# topoptcomec/cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# CLI entry point of TopoptComec.

from __future__ import annotations

import argparse
import copy
import json
import multiprocessing
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import numpy.typing as npt

from topoptcomec import exporters, result_cache
from topoptcomec.core import analyzers, optimizers
from topoptcomec.core.displacements import combine_load_case_displacements
from topoptcomec.parameter_check import ParameterCheck
from topoptcomec.presets_io import (
    load_presets,
    validate_preset_name,
    validate_preset_structure,
)
from topoptcomec.time_estimation import TimeEstimation
from topoptcomec.version import __version__

# Type aliases
FloatArray = npt.NDArray[np.float64]

#: Convergence threshold on the design-variable change (mirrors the optimizer).
CONVERGENCE_TOLERANCE = 0.01

#: Native math-library thread-pool variables capped in parallel runs.
_BLAS_THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def _limit_blas_threads(worker_count: int) -> None:
    """Cap native BLAS/OpenMP threads per parallel worker.

    Each worker process would otherwise start one thread per core, so N
    workers oversubscribe the machine by a factor of N. The environment is
    set before the workers are spawned so every child imports NumPy with the
    reduced thread count.
    """
    threads: int = max(1, (os.cpu_count() or 1) // max(1, worker_count))
    for variable in _BLAS_THREAD_ENV:
        os.environ[variable] = str(threads)


# Backwards-compatible alias (the hash lives in :mod:`topoptcomec.result_cache`).
_params_hash = result_cache.params_hash


def _resolve_starting_field(
    preset_name: str,
    params: dict,
    base_dir: str,
    verbose: bool,
    restart_from: str | None,
) -> tuple[
    FloatArray | None,
    tuple[FloatArray, FloatArray] | None,
    str | None,
]:
    """
    Decide whether to reuse a cached result or restart from a density field.

    Parameters
    ----------
    preset_name : str
        Preset identifier used to name cache files.
    params : dict
        Full preset configuration dictionary.
    base_dir : str
        Base output directory holding the cache.
    verbose : bool
        Whether to print why a cache was ignored.
    restart_from : str | None
        Optional npz file used when the preset initializes from a result.

    Returns
    -------
    tuple
        ``(last_xPhys, cached_result, error)``. ``cached_result`` is
        ``(xPhys, u)`` when the cache can be reused as-is; ``last_xPhys`` is
        the restart field for ``init_type == 3``.
    """
    cache_file: Path = result_cache.cache_path(base_dir, preset_name)
    nelxyz: list[int] = params["Dimensions"]["nelxyz"]
    init_type: int = int(params.get("Materials", {}).get("init_type", 0))
    cached = result_cache.load_density_cache(cache_file)

    if init_type == 3:
        # Restart: use an explicit file when given, otherwise continue from
        # the previous result of this preset (whatever its hash).
        source = Path(restart_from) if restart_from else cache_file
        data = result_cache.load_density_cache(source)
        if data is None:
            if restart_from:
                return None, None, f"Restart file not found or invalid: {source}"
            return (
                None,
                None,
                (
                    f"No previous result for {preset_name!r} to restart from. "
                    "Run the preset once, or pass --restart-from."
                ),
            )
        if not result_cache.cache_matches_grid(data, nelxyz):
            return (
                None,
                None,
                f"The restart field in {source} does not match the grid {nelxyz}.",
            )
        field_error = result_cache.validate_density_field(data)
        if field_error:
            return None, None, f"Invalid restart file {source}: {field_error}"
        return np.asarray(data["xPhys"], dtype=np.float64), None, None

    if cached is None:
        return None, None, None

    cached_hash = str(cached["params_hash"]) if "params_hash" in cached else None
    if cached_hash != _params_hash(params):
        if verbose:
            print(
                f"[{preset_name}] Cached result was produced with different "
                "parameters; re-running optimization."
            )
    elif not result_cache.cache_matches_grid(cached, nelxyz):
        if verbose:
            print(
                f"[{preset_name}] Cached result does not match the grid; "
                "re-running optimization."
            )
    else:
        if verbose:
            print(f"[{preset_name}] Loading cached density field...")
        return (
            None,
            (
                np.asarray(cached["xPhys"], dtype=np.float64),
                combine_load_case_displacements(cached["u"]),
            ),
            None,
        )
    return None, None, None


def _load_or_run_optimization(
    preset_name: str,
    params: dict,
    optimizer_params: dict,
    base_dir: str,
    save_frames: bool,
    verbose: bool,
    restart_from: str | None = None,
) -> tuple[FloatArray | None, FloatArray | None, str | None]:
    """
    Load a cached density field or run the optimizer to generate a new one.

    The cache stores a hash of the preset parameters and the grid it was
    produced for; if either does not match, the cache is ignored and the
    optimization reruns (C8: no stale results).

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
    restart_from : str | None
        Optional npz file used as the starting density field when the preset
        initializes from the current result (``init_type == 3``).

    Returns
    -------
    tuple[FloatArray | None, FloatArray | None, str | None]
        xPhys, displacement field u, and an error message if one occurred.
    """
    # Validate the preset before _resolve_starting_field indexes its keys: a
    # malformed shared preset must produce a message, not a KeyError.
    error: str | None = validate_preset_structure(params)
    if error:
        return None, None, f"Invalid preset '{preset_name}': {error}"

    last_xPhys, cached_result, error = _resolve_starting_field(
        preset_name, params, base_dir, verbose, restart_from
    )
    if error:
        return None, None, error
    if cached_result is not None:
        return cached_result[0], cached_result[1], None

    error = ParameterCheck(last_xPhys).validate(params)
    if error:
        return None, None, error

    cache_file: Path = result_cache.cache_path(base_dir, preset_name)
    nelxyz: list[int] = params["Dimensions"]["nelxyz"]
    loss_history: list[tuple[int, float]] = []
    last_change: list[float] = [1.0]

    def progress_callback(
        iteration: int, objective: float, change: float, xPhys_frame: FloatArray
    ) -> bool:
        last_change[0] = change
        if iteration > 0:
            # The final frame reuses the last iteration index to report the
            # objective of the final design; replace, do not duplicate.
            if loss_history and loss_history[-1][0] == iteration:
                loss_history[-1] = (iteration, objective)
            else:
                loss_history.append((iteration, objective))
        if save_frames:
            folder: Path = Path(base_dir) / preset_name / f"{preset_name}_creation"
            folder.mkdir(parents=True, exist_ok=True)
            filename: Path = folder / f"{preset_name}_creation_{iteration}.png"
            colors: list[str] | None = params.get("Materials", {}).get("color", None)
            exporters.save_as_png(
                xPhys_frame, params["Dimensions"]["nelxyz"], str(filename), colors
            )
        return False

    optimizer_params["progress_callback"] = progress_callback
    optimizer_params["current_xPhys"] = last_xPhys

    is_multimaterial: bool = (
        len(optimizer_params.get("Materials", {}).get("E", [1.0])) > 1
    )
    try:
        xPhys, u = optimizers.optimize(
            **optimizer_params, multimaterial=is_multimaterial, verbose=verbose
        )

        mean_density: float = (
            np.mean(xPhys.sum(axis=0)) if xPhys.ndim == 2 else np.mean(xPhys)
        )
        if mean_density < 0.01:
            return (
                None,
                None,
                "No valid structure was found. Please adjust your parameters and try again.",
            )

        result_cache.save_density_cache(
            cache_file,
            xPhys,
            u,
            params,
            nelxyz,
            converged=last_change[0] <= CONVERGENCE_TOLERANCE,
        )
        if verbose:
            print(f"[{preset_name}] Cached density field.")

        exporters.save_loss(loss_history, base_dir, preset_name)

        return xPhys, u, None
    except Exception as e:  # noqa: BLE001
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
    except Exception as e:  # noqa: BLE001
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

    from topoptcomec.core import displacements

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
    except Exception as e:  # noqa: BLE001
        return f"Displacement failed: {e}"

    return None


def _export_results(
    preset_name: str,
    xPhys: FloatArray,
    nelxyz: list[int],
    format: str,
    verbose: bool,
    output_dir: str = "results",
    colors: list[str] | None = None,
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
    output_dir : str
        Directory where exported files are written.
    colors : list[str] | None
        Per-material colors used by the PNG and 3MF exporters.
    """
    results_dir: Path = Path(output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    base_filename: Path = results_dir / preset_name

    formats: list[str] = ["png", "stl", "vti", "3mf"] if format == "all" else [format]
    for f in formats:
        filename: str = str(base_filename.with_suffix(f".{f}"))
        if verbose:
            print(f"[{preset_name}] Saving {f.upper()} to {filename}...")

        success, error_msg = _export(xPhys, nelxyz, filename, f, colors)

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
    output_dir: str = "results",
    restart_from: str | None = None,
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
    output_dir : str, optional
        Directory for caches, frames and exported files.
    restart_from : str, optional
        Density-field npz used as the starting point (implies initialization
        from a result, ``init_type == 3``).

    Returns
    -------
    tuple[str, str | None]
        The preset name and an error message if execution failed.
    """
    base_dir = output_dir

    if restart_from is not None:
        # An explicit restart file implies initialization from a result.
        params = copy.deepcopy(params)
        params.setdefault("Materials", {})["init_type"] = 3

    optimizer_params: dict = {k: v for k, v in params.items() if k != "Displacement"}

    is_multimaterial: bool = (
        len(optimizer_params.get("Materials", {}).get("E", [1.0])) > 1
    )
    if "Materials" in optimizer_params:
        materials = dict(optimizer_params["Materials"])
        materials.pop("color", None)
        if not is_multimaterial:
            materials.pop("percent", None)
        optimizer_params["Materials"] = materials

    xPhys: FloatArray | None
    u: FloatArray | None
    error: str | None
    if verbose:
        _, label = TimeEstimation(params).optimization_indicators()
        print(f"[{preset_name}] Estimated optimization time: {label}")
    xPhys, u, error = _load_or_run_optimization(
        preset_name,
        params,
        optimizer_params,
        base_dir,
        save_frames,
        verbose,
        restart_from=restart_from,
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
        from topoptcomec.cli.cli_preview import render_preview

        preview_colors = params.get("Materials", {}).get("color", None)
        print(
            render_preview(xPhys, params["Dimensions"]["nelxyz"], colors=preview_colors)
        )

    _export_results(
        preset_name,
        xPhys,
        params["Dimensions"]["nelxyz"],
        format,
        verbose,
        output_dir=output_dir,
        colors=params.get("Materials", {}).get("color", None),
    )

    if not verbose:
        print(f"Preset '{preset_name}' completed.")
    return preset_name, None


def _export(
    xPhys: FloatArray,
    nelxyz: list[int],
    filename: str,
    format: str,
    colors: list[str] | None = None,
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
    colors : list[str] | None
        Per-material colors for the PNG and 3MF exporters.

    Returns
    -------
    tuple[bool, str | None]
        True and None when export succeeds, otherwise False and an error message.
    """
    if format == "png":
        return exporters.save_as_png(xPhys, nelxyz, filename, colors)
    elif format == "vti":
        return exporters.save_as_vti(xPhys, nelxyz, filename)
    elif format == "stl":
        return exporters.save_as_stl(xPhys, nelxyz, filename)
    elif format == "3mf":
        return exporters.save_as_3mf(xPhys, nelxyz, filename, colors)
    return False, f"Unknown format: {format}"


def _build_arg_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser."""
    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="TopoptComec CLI - Topology Optimization for Compliant Mechanisms",
        epilog=(
            "Run 'topoptcomec' without arguments to start the GUI, and "
            "'topoptcomec update' to check for a new release."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"TopoptComec {__version__}",
        help="Show the installed TopoptComec version and exit",
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
        help="Print intermediate optimizer output during optimization",
    )
    parser.add_argument(
        "--presets",
        type=str,
        default=None,
        help=(
            "Presets JSON file to use exclusively. By default the built-in "
            "presets, ~/.topoptcomec/presets.json and ./presets.json are merged."
        ),
    )
    parser.add_argument(
        "--restart-from",
        type=str,
        default=None,
        help=(
            "Density-field .npz to restart from (implies initialization from "
            "the current result). Defaults to the previous result of the same "
            "preset."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="results",
        help="Output directory for caches, frames and exports (default: ./results)",
    )

    return parser


def run_cli() -> None:
    """
    Parse command-line arguments and execute the requested preset workflows.

    The CLI supports single or parallel preset execution, optional displacement,
    and output export to several file formats.
    """
    args: argparse.Namespace = _build_arg_parser().parse_args()

    # Load presets (explicit file, or the merged built-in/user/local layers)
    try:
        presets: dict[str, dict] = load_presets(args.presets)
    except FileNotFoundError as e:
        print(
            f"Error: presets file not found at {Path(str(e)).absolute()}\n"
            "Use --presets to point to a presets JSON file."
        )
        sys.exit(1)
    except (json.JSONDecodeError, TypeError) as e:
        print(f"Error reading presets: {e}")
        sys.exit(1)

    # Parse and validate preset names. Names double as filesystem path
    # components, and presets are shareable files, so reject anything that
    # could escape the output directory.
    preset_names: list[str] = [name.strip() for name in args.preset.split(",")]
    if any(not name for name in preset_names):
        print("Error: Empty preset name in --preset.")
        sys.exit(1)
    duplicates = sorted({name for name in preset_names if preset_names.count(name) > 1})
    if duplicates:
        print(f"Error: Preset(s) listed more than once: {', '.join(duplicates)}")
        sys.exit(1)
    for name in preset_names:
        if name not in presets:
            print(f"Error: Preset '{name}' not found in the presets")
            print("Available presets:", ", ".join(presets.keys()))
            sys.exit(1)
        name_error = validate_preset_name(name)
        if name_error:
            print(f"Error: {name_error}")
            sys.exit(1)
        structure_error = validate_preset_structure(presets[name])
        if structure_error:
            print(f"Error: Preset '{name}' is invalid: {structure_error}")
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
            output_dir=args.output,
            restart_from=args.restart_from,
        )
        if error:
            print(error)
            sys.exit(1)
    else:
        if args.restart_from:
            print("Error: --restart-from only supports a single preset.")
            sys.exit(1)
        max_workers: int = min(len(preset_names), os.cpu_count() or 1)
        print(
            f"Running {len(preset_names)} presets in parallel ({max_workers} workers)"
        )
        # Spawn so each child imports NumPy after the thread caps below are in
        # place (fork would inherit already-initialised BLAS thread pools).
        _limit_blas_threads(max_workers)
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=multiprocessing.get_context("spawn"),
        ) as executor:
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
                    output_dir=args.output,
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
