# Codebase Map

Quick map of the repository.

## Root

- `main.py`: convenience wrapper, delegates to `app/__main__:main`
- `app/__main__.py`: application entry point (console script target)
- `presets.json`: built-in preset definitions
- `requirements.txt`: dependencies
- `README.md`: user-facing overview
- `CONTRIBUTING.md`: contribution guide
- `LICENSE.txt`: license

## Source

- `app/cli.py`: CLI execution, preset loading, parallel runs, export dispatch
- `app/cli_preview.py`: CLI execution, preset loading, parallel runs, export dispatch

### `app/core/`
- `fem.py`: FEM model and solves
- `optimizers.py`: optimization loops
- `initializers.py`: density initialization
- `displacements.py`: displacement simulation
- `analyzers.py`: result checks

### `app/ui/`
- `main_window.py`: main window and action flow
- `parameter_manager.py`: parameter gathering and validation
- `widgets.py`: GUI widgets
- `plotting.py`: plotting and visualization
- `workers.py`: background workers
- `exporters.py`: PNG, STL, VTI, 3MF export
- `themes.py`: stylesheets
- `icons.py`: icon handling
- `resource_path_finder.py`: path getter

## Tests

- `tests/test_cli.py`: CLI behavior
- `tests/test_fem.py`: FEM behavior
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

## Documentation

- `docs/ARCHITECTURE.md`: high-level system design
- `docs/CODEMAP.md`: this file
- `docs/GLOSSARY.md`: terminology
- `docs/EXAMPLE.md`: usage examples
- `docs/SKILL.md`: agent-oriented repo guide

## Assets and Output

- `icons/`: GUI SVG assets
- `results/`: generated output files
