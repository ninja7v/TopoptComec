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

from app.core import optimizers
from app.ui import exporters

# Type aliases
FloatArray = npt.NDArray[np.float64]


def _run_single_preset(
    preset_name: str,
    params: dict,
    format: str,
    threshold: bool,
    verbose: bool = False,
    run_disp: bool = False,
    save_frames: bool = False,
) -> tuple[str, str | None]:
    """Run optimization and export for a single preset. Returns (preset_name, error)."""
    if verbose:
        print(f"Running optimization for preset: {preset_name}")

    xPhys: FloatArray | None = None
    u: FloatArray | None = None
    cache_file: Path = (
        Path("results") / preset_name / f"{preset_name}_density_field.npz"
    )
    if cache_file.exists():
        if verbose:
            print(f"[{preset_name}] Loading cached density field...")
        try:
            data = np.load(cache_file)
            xPhys, u = data["xPhys"], data["u"]
        except Exception as e:
            if verbose:
                print(f"[{preset_name}] Failed to load cache: {e}")

    if xPhys is None:
        # Clean params for optimizer
        optimizer_params: dict = params.copy()
        if "Displacement" in optimizer_params:
            optimizer_params.pop("Displacement")

        is_multimaterial: bool = (
            len(optimizer_params.get("Materials", {}).get("E", [1.0])) > 1
        )
        if "Materials" in optimizer_params:
            optimizer_params["Materials"].pop("color", None)
            if not is_multimaterial:
                optimizer_params["Materials"].pop("percent", None)

        if save_frames:

            def progress_callback(
                iteration: int, objective: float, change: float, xPhys_frame: FloatArray
            ) -> bool:
                folder: Path = Path("results") / preset_name / f"{preset_name}_creation"
                folder.mkdir(parents=True, exist_ok=True)
                filename: Path = folder / f"{preset_name}_creation_{iteration}.png"
                colors: list[str] | None = params.get("Materials", {}).get(
                    "color", None
                )
                exporters.save_as_png(
                    xPhys_frame, params["Dimensions"]["nelxyz"], str(filename), colors
                )
                return False

            optimizer_params["progress_callback"] = progress_callback

        # Run optimization
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
        except Exception as e:
            import traceback

            traceback.print_exc()
            return preset_name, f"Optimization failed: {e}"

    if run_disp and u is not None:
        if verbose:
            print(f"[{preset_name}] Running displacement...")
        from app.core import displacements

        disp_params: dict = params.copy()
        try:
            disp_iter: int = 0

            def disp_callback(iteration: int) -> bool:
                nonlocal disp_iter
                disp_iter = iteration
                return False

            for frame_data in displacements.run_iterative_displacement(
                disp_params, xPhys, disp_callback
            ):
                if save_frames:
                    folder: Path = (
                        Path("results") / preset_name / f"{preset_name}_displacement"
                    )
                    folder.mkdir(parents=True, exist_ok=True)
                    filename: Path = (
                        folder / f"{preset_name}_displacement_{disp_iter}.png"
                    )
                    colors: list[str] | None = params.get("Materials", {}).get(
                        "color", None
                    )
                    exporters.save_as_png(
                        frame_data,
                        params["Dimensions"]["nelxyz"],
                        str(filename),
                        colors,
                    )
        except Exception as e:
            import traceback

            traceback.print_exc()
            return preset_name, f"Displacement failed: {e}"

    # Apply threshold if requested
    if threshold:
        if verbose:
            print(f"[{preset_name}] Applying threshold (0.5)...")
        xPhys = np.where(xPhys > 0.5, 1.0, 0.0)

    # Create results directory
    results_dir: Path = Path("results")
    results_dir.mkdir(exist_ok=True)
    base_filename: Path = results_dir / preset_name

    # Export
    nelxyz: list[int] = params["Dimensions"]["nelxyz"]
    formats: list[str] = ["png", "stl", "vti", "3mf"] if format == "all" else [format]
    for f in formats:
        filename: str = str(base_filename.with_suffix(f".{f}"))
        if verbose:
            print(f"[{preset_name}] Saving {f.upper()} to {filename}...")

        success: bool
        error_msg: str | None
        success, error_msg = _export(xPhys, nelxyz, filename, f)

        if not success:
            print(f"[{preset_name}] Error saving {f}: {error_msg}")
        else:
            print(f"[{preset_name}] Saved {filename}")

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
        (success, error_message) - True if successful, error string otherwise.
    """
    """Dispatch export to the correct exporter."""
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
    """Parses arguments and runs the optimization from the CLI."""
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
            presets: dict = json.load(f)
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
            args.format,
            args.threshold,
            args.verbose,
            args.displacement,
            args.intermediate,
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
            futures: dict = {
                executor.submit(
                    _run_single_preset,
                    name,
                    presets[name],
                    args.format,
                    args.threshold,
                    args.verbose,
                    args.displacement,
                    args.intermediate,
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
