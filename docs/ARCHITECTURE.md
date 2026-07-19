# Architecture Overview

TopoptComec has two front doors built on the same core engine:

- GUI mode for interactive design, optimization, visualization, and export
- CLI mode for preset-based batch runs

## Main Building Blocks

### Entry Points

- `topoptcomec/__main__.py`: chooses GUI when no arguments are passed, otherwise runs the CLI
- `main.py`: convenience wrapper, delegates to `topoptcomec/__main__:main`
- `topoptcomec/cli/cli.py`: loads presets, runs optimization, exports results
- `topoptcomec/gui/main_window.py`: owns the main GUI workflow

### Shared Logic
- `topoptcomec/parameter_check.py`: `ParameterCheck` class, validates parameter dictionaries (used by GUI and CLI)
- `topoptcomec/time_estimation.py`: `TimeEstimation` class, estimates runtime for optimization (extensible to displacement and analysis)
- `topoptcomec/exporters.py`: shared export logic used by GUI and CLI
- `topoptcomec/presets_io.py`: locates the presets file (local, per-user, or packaged default)

### Core Engine

The numerical core lives in `topoptcomec/core/`. It is UI-free and can be
used as a library.

- `grid.py`: `StructuredGrid` — the single source of truth for element/node
  ordering (flat element index `ez*nelx*nely + ex*nely + ey`) and all
  spatial reshapes
- `model.py`: typed problem definition — `Load`, `Support`, `Region`
  dataclasses. This is the boundary API for embedding the core in other
  tools (CAD plugins, scripts, other physics front ends)
- `preset_format.py`: the *only* place that understands the legacy preset
  dictionaries and their UI glyphs (`←`, `↑`, `◯`, `-`); translates them
  into the typed model
- `fem.py`: finite element solver — stiffness assembly, boundary
  conditions, linear solves, objective and sensitivity computation. Takes a
  `StructuredGrid` and typed `Load`/`Support`/`Region` objects only
- `optimizers.py`: SIMP optimization loop (single- and multi-material share
  one implementation); accepts legacy preset dictionaries and translates
  them at the boundary
- `initializers.py`: starting density fields
- `displacements.py`: iterative post-optimization displacement simulation
- `analyzers.py`: heuristic quality checks on finished results
- `post_processing.py`: density/displacement field rescaling shared by
  optimization, displacement, and export

### GUI Layer

The GUI layer lives in `topoptcomec/gui/`.
- `widgets.py`: parameter-entry widgets
- `parameter_manager.py`: gathers, normalizes, and scales parameters
- `plotting.py`: interactive 2D/3D visualization using PyVista/PyVistaQt
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

1. The CLI resolves the presets file (`--presets`, else `./presets.json`,
   else `~/.topoptcomec/presets.json`, else the packaged defaults).
2. `parameter_check.py` validates the parameters.
3. It runs the optimizer directly, caching results per preset. The cache
   stores a hash of the parameters and is invalidated automatically when the
   preset changes.
4. It optionally thresholds the result.
5. It exports files into the output directory (`-o`, default `results/`).

## Key Design Constraints

- 2D and 3D behavior share the same code paths; `nelz == 0` means 2D.
- All element/node index arithmetic goes through `StructuredGrid`; no module
  may reimplement the flat ordering.
- The numerical core only consumes typed `Load`/`Support`/`Region` objects;
  preset dictionaries (and their UI glyphs) are translated once, in
  `preset_format.py`.
- GUI and CLI share the solver, exporters, parameter validation, and time
  estimation, so changes in shared logic affect both.
- Multi-material support exists in the core, but not every surrounding
  workflow is equally mature.
