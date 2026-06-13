# Usage Examples

## GUI

```bash
topoptcomec
```

Typical flow:

1. Load a preset.
2. Adjust parameters.
3. Click `Create`.
4. Inspect, analyze, or export the result.

If you change parameters after a run, create the result again.

## CLI

Run one preset:

```bash
topoptcomec -p ForceInverter_2Sup_2D
```

Export only PNG:

```bash
topoptcomec -p ForceInverter_2Sup_2D -f png
```

Threshold before export:

```bash
topoptcomec -p ForceInverter_2Sup_2D -f png -t
```

Run several presets in parallel:

```bash
topoptcomec -p ForceInverter_2Sup_2D,Gripper_2D -f png
```

## Notes

- Run commands from the repository root.
- CLI outputs are written to `results/`.
- Install dependencies with `pip install -r requirements.txt` if needed.
