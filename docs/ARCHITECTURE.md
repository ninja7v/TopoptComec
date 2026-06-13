# Architecture Overview

TopoptComec has two front doors built on the same core engine:

- GUI mode for interactive design, optimization, visualization, and export
- CLI mode for preset-based batch runs

## Main Building Blocks

### Entry Points

- `app/__main__.py`: chooses GUI when no arguments are passed, otherwise runs the CLI
- `main.py`: convenience wrapper, delegates to `app/__main__:main`
- `app/cli.py`: loads presets, runs optimization, exports results
- `app/ui/main_window.py`: owns the main GUI workflow

### Core Engine

The numerical core lives in `app/core/`.

- `fem.py`: finite element model, stiffness assembly, boundary conditions, solves, sensitivities
- `optimizers.py`: SIMP optimization loops, including single-material and multi-material paths
- `initializers.py`: starting density fields
- `displacements.py`: iterative post-optimization displacement simulation
- `analyzers.py`: heuristic quality checks on finished results

### UI Layer

The GUI layer lives in `app/ui/`.

- `widgets.py`: parameter-entry widgets
- `parameter_manager.py`: gathers, normalizes, validates, and scales parameters
- `plotting.py`: 2D/3D visualization
- `workers.py`: background threads for optimization, analysis, and displacement
- `exporters.py`: shared export logic used by GUI and CLI

## Runtime Flow

### GUI Flow

1. The user edits parameters in the UI.
2. `parameter_manager.py` builds the nested parameter dictionary.
3. `main_window.py` launches the optimizer in a worker thread.
4. The optimizer calls into the FEM core.
5. The result is plotted and can then be analyzed, displaced, or exported.

### CLI Flow

1. The CLI reads one or more presets from `presets.json`.
2. It runs the optimizer directly.
3. It optionally thresholds the result.
4. It exports files into `results/`.

## Key Design Constraints

- 2D and 3D behavior share the same code paths; `nelz == 0` means 2D.
- Parameters are passed as nested dictionaries across GUI, CLI, and core code.
- GUI and CLI share the solver and exporters, so changes in core behavior affect both.
- Multi-material support exists in the core, but not every surrounding workflow is equally mature.
