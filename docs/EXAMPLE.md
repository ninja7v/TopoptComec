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

Use a custom presets file and output directory:

```bash
topoptcomec -p MyPreset --presets /path/to/presets.json -o /path/to/output
```

## Notes

- The CLI can run from any directory. Presets are resolved in this order:
  `--presets` path, `./presets.json`, `~/.topoptcomec/presets.json`, then
  the packaged example presets.
- CLI outputs are written to `results/` by default (override with `-o`).
- Results are cached per preset; the cache is invalidated automatically when
  the preset parameters change.
- Install dependencies with `pip install -r requirements.txt` if needed.
