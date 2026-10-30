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

Restart from a previous result (this implies "From current result", i.e.
`Materials` → init type 3):

```bash
topoptcomec -p Shelf_2D --restart-from results/Shelf_2D/Shelf_2D_density_field.npz
```

Without `--restart-from`, a restart preset continues from its own previous
result (`results/<preset>/<preset>_density_field.npz`).

## Notes

- The CLI can run from any directory. `--presets FILE` uses that file only;
  otherwise the built-in presets (`./topoptcomec/presets.json` in a source
  checkout, else the packaged ones), `~/.topoptcomec/presets.json` and
  `./presets.json` are merged, the later one winning on a name clash.
- CLI outputs are written to `results/` by default (override with `-o`).
  `results/` is generated output and is not tracked by git.
- Results are cached per preset. The cache records a hash of the parameters
  and the grid it was produced for, and is invalidated automatically when
  either changes.
- The GUI stores only the presets you save in `~/.topoptcomec/presets.json`,
  so editing presets never modifies the source checkout, and new built-in
  presets of a release always appear.
- Install dependencies with `pip install -r requirements.txt` if needed.
