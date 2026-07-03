# Architecture Overview

TopoptComec has two front doors built on the same core engine:

- GUI mode for interactive design, optimization, visualization, and export
- CLI mode for preset-based batch runs

## Main Building Blocks

### Entry Points

- `app/__main__.py`: chooses GUI when no arguments are passed, otherwise runs the CLI
- `main.py`: convenience wrapper, delegates to `app/__main__:main`
- `app/cli/cli.py`: loads presets, runs optimization, exports results
- `app/gui/main_window.py`: owns the main GUI workflow

### Shared Logic
- `app/parameter_check.py`: `ParameterCheck` class, validates parameter dictionaries (used by GUI and CLI)
- `app/time_estimation.py`: `TimeEstimation` class, estimates runtime for optimization (extensible to displacement and analysis)
- `app/exporters.py`: shared export logic used by GUI and CLI

### Core Engine

The numerical core lives in `app/core/`.
- `fem.py`: finite element model, stiffness assembly, boundary conditions, solves, sensitivities
- `optimizers.py`: SIMP optimization loops, including single-material and multi-material paths
- `initializers.py`: starting density fields
- `displacements.py`: iterative post-optimization displacement simulation
- `analyzers.py`: heuristic quality checks on finished results

### GUI Layer

The GUI layer lives in `app/gui/`.
- `widgets.py`: parameter-entry widgets
- `parameter_manager.py`: gathers, normalizes, and scales parameters
- `plotting.py`: 2D/3D visualization
- `workers.py`: background threads for optimization, analysis, and displacement

## Runtime Flow

### GUI Flow

1. The user edits parameters in the UI.
2. `parameter_manager.py` builds the nested parameter dictionary.
3. `parameter_check.py` validates the parameters.
4. `main_window.py` launches the optimizer in a worker thread.
5. The optimizer calls into the FEM core.
6. The result is plotted and can then be analyzed, displaced, or exported.

### CLI Flow

1. The CLI reads one or more presets from `presets.json`.
2. `parameter_check.py` validates the parameters.
3. It runs the optimizer directly.
4. It optionally thresholds the result.
5. It exports files into `results/`.

## Key Design Constraints

- 2D and 3D behavior share the same code paths; `nelz == 0` means 2D.
- Parameters are passed as nested dictionaries across GUI, CLI, and core code.
- GUI and CLI share the solver, exporters, parameter validation, and time estimation, so changes in shared logic affect both.
- Multi-material support exists in the core, but not every surrounding workflow is equally mature.
