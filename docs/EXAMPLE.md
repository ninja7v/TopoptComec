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

## Large displacements (non-linear analysis)

The default analysis is linear: it assumes small displacements, and the
result does not depend on the force magnitude. To design a mechanism for a
large stroke, select **Analysis: Nonlinear** in the Optimizer section (GUI)
or add two keys to the `Optimizer` section of a preset:

```json
"Optimizer": {
    "filter_type": "Sensitivity", "filter_radius_min": 1.3, "penal": 3.0,
    "eta": 0.3, "max_change": 0.05, "n_it": 60, "solver": "Auto",
    "analysis": "Nonlinear", "load_factor": 8.0
}
```

- `load_factor` multiplies the input forces without changing the port
  springs. Because the spring stiffness equals the force magnitude, the input
  port can move by at most `load_factor` elements: this is the knob that sets
  how large the displacements are. `1` applies the preset forces as written
  (displacements of about one element), which is not the linear limit: use
  the linear analysis for small displacements.
- The load is ramped up automatically during the first half of the
  iterations, so give the optimizer more iterations than for a linear run
  (60 instead of 30 is a good start).
- A non-linear run costs roughly 5 to 10 linear runs.
- If the log ends with "only reached N% of the load", the structure buckles
  before the full load: reduce `load_factor`.

## Notes

- The CLI can run from any directory. `--presets FILE` uses that file only;
  otherwise the built-in presets (`./topoptcomec/presets.json` in a source
  checkout, else the packaged ones), `~/.topoptcomec/presets.json` and
  `./presets.json` are merged, the later one winning on a name clash.
- CLI outputs are written to `results/` by default (override with `-o`).
  `results/` is generated output tracked in git, so regenerated results show
  up in diffs.
- Results are cached per preset. The cache records a hash of the parameters
  and the grid it was produced for, and is invalidated automatically when
  either changes.
- The GUI stores only the presets you save in `~/.topoptcomec/presets.json`,
  so editing presets never modifies the source checkout, and new built-in
  presets of a release always appear.
- Install dependencies with `pip install -r requirements.txt` if needed.
