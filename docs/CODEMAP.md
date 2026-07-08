# Codebase Map

Quick map of the repository.

## Root

- `main.py`: convenience wrapper, delegates to `topoptcomec/__main__:main`
- `topoptcomec/__main__.py`: application entry point (console script target)
- `requirements.txt`: dependencies
- `README.md`: user-facing overview
- `CONTRIBUTING.md`: contribution guide
- `LICENSE.txt`: license

## Source

### `topoptcomec/`
- `__main__.py`: application entry point (console script target)
- `exporters.py`: PNG, STL, VTI, 3MF export (shared by GUI and CLI)
- `parameter_check.py`: `ParameterCheck` class, parameter validation (shared)
- `time_estimation.py`: `TimeEstimation` class, runtime estimation (shared)
- `presets_io.py`: presets file resolution (local / per-user / packaged)
- `presets.json`: packaged default preset definitions

### `topoptcomec/core/`
- `grid.py`: `StructuredGrid`, canonical element/node ordering and reshapes
- `model.py`: typed problem definition (`Load`, `Support`, `Region`)
- `preset_format.py`: legacy preset dictionary -> typed model translation
- `fem.py`: FEM model and solves
- `optimizers.py`: optimization loops (single- and multi-material)
- `initializers.py`: density initialization
- `displacements.py`: displacement simulation
- `analyzers.py`: result checks
- `post_processing.py`: density/displacement field rescaling

### `topoptcomec/cli/`
- `cli.py`: CLI execution, preset loading, parallel runs, export dispatch
- `cli_preview.py`: terminal rendering of density fields

### `topoptcomec/gui/`
- `main_window.py`: main window and action flow
- `parameter_manager.py`: parameter gathering, normalization, and scaling
- `widgets.py`: GUI widgets
- `plotting.py`: plotting and visualization
- `workers.py`: background workers
- `themes.py`: stylesheets
- `icons.py`: icon handling
- `resource_path_finder.py`: path getter

## Tests

- `tests/test_cli.py`: CLI behavior (mocked and end-to-end)
- `tests/test_fem.py`: FEM behavior
- `tests/test_numerics.py`: stiffness matrices vs Gauss quadrature, finite-difference sensitivity checks, solver equivalence, filter behavior
- `tests/test_optimizers.py`: optimizer behavior
- `tests/test_initializers.py`: initializer behavior
- `tests/test_displacements.py`: displacement behavior
- `tests/test_analyzers.py`: analyzer behavior
- `tests/test_exporters.py`: exporter behavior
- `tests/test_parameter_validation.py`: GUI parameter validation
- `tests/test_main_window.py`: main window behavior
- `tests/test_widgets.py`: widget behavior
- `tests/test_workers.py`: worker behavior
- `tests/conftest.py`: shared fixtures
- `tests/presets_test.json`: preset fixture data
- `tests/references/`: result references (`regenerate_references.py` rebuilds them after intentional numerical changes)

## Documentation

- `docs/ARCHITECTURE.md`: high-level system design
- `docs/CODEMAP.md`: this file
- `docs/GLOSSARY.md`: terminology
- `docs/EXAMPLE.md`: usage examples
- `docs/SKILL.md`: agent-oriented repo guide

## Assets and Output

- `topoptcomec/icons/`: GUI SVG assets (packaged)
- `results/`: generated output files (default CLI output directory)
