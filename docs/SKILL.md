# TopoptComec Agent Skill

TopoptComec is a Python topology optimization app with a GUI (`topoptcomec`) and a CLI (`topoptcomec -p <preset>`).

## Edit Map

- `topoptcomec/core/fem.py`: FEM, boundary conditions, solves, sensitivities
- `topoptcomec/core/optimizers.py`: optimization loops
- `topoptcomec/core/displacements.py`: post-run displacement simulation
- `topoptcomec/core/analyzers.py`: result checks
- `topoptcomec/gui/main_window.py`: GUI flow
- `topoptcomec/gui/parameter_manager.py`: parameter gathering, normalization, and scaling
- `topoptcomec/gui/widgets.py`: GUI inputs
- `topoptcomec/exporters.py`: file export (shared by GUI and CLI)
- `topoptcomec/parameter_check.py`: parameter validation (shared)
- `topoptcomec/time_estimation.py`: runtime estimation (shared)
- `topoptcomec/cli/cli.py`: CLI runs
- `presets.json`: preset definitions

## Rules

- `nelz == 0` means 2D; otherwise 3D.
- Parameters are passed as nested dictionaries.
- Keep GUI, CLI, presets, and tests consistent when changing parameter schema.
- A valid model needs at least one input force and at least one output force or support.
- Parameter changes invalidate an existing result in the GUI.

## Workflow

1. Find the main file involved.
2. Make the smallest coherent change.
3. Run focused tests.
4. Update docs or presets if behavior changed.

Useful tests:

```bash
ruff format --check .
ruff check .
lizard -L 150 -C 30 -w topoptcomec tests
pytest
```
