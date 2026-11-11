<img width="30" height="30" alt="TopoptComec_logo" src="https://github.com/user-attachments/assets/b93640d7-cda4-4a47-9a7a-783906d2c19c" /> TopoptComec
============
**Interactive topology optimization for compliant mechanisms — unleash your creativity!**

<img width="900" alt="TopoptComec_demo" src="https://github.com/user-attachments/assets/cfb2207c-4adf-4bfe-94c9-83a1fbd19766" />

## What is TopoptComec?
TopoptComec helps you design compliant mechanisms — flexible structures that achieve motion through material deformation rather than joints.

Simply draw your domain, set forces and supports, choose your material and optimizer, then watch the algorithm sculpt the optimal shape — in 2D or 3D.

## Why using TopoptComec?
None of the existing competing solutions combine all of these features in a single tool:
- 🔨 **Rigid structure *and* compliant mechanism**  — Same solver, just change the loads.
- 🧊 **2D *and* 3D support** — Real engineering isn’t flat.
- 🌀 **Linear *and* non-linear analysis** — Optimize for small or large displacements.
- 🚀 **Fast, like really fast** — Designed for performance, not academic demos.
- 🧪 **Flexible** — Tons of parameters to tweak → infinite design possibilities.
- 🍰 **Easy to use** — Intuitive GUI *and* CLI, piece of cake to use.
- 🔓 **Open source** — Transparent, extensible. No black boxes.

## 🚀Quick Start
### 1. Clone the repo
Open the folder you want TopoptComec to be in a terminal and run:
```cmd
git clone https://github.com/ninja7V/topoptcomec.git
cd topoptcomec
```

### 2. Run
Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't already have it.
#### GUI:
```bash
uv run topoptcomec
```
#### CLI:
```bash
uv run topoptcomec -p ForceInverter_2Sup_2D --preview
```
#### Faster solver (optional, x86-64 only):
```bash
uv sync --extra fast
```
Installs [pypardiso](https://github.com/haasad/PyPardisoProject) (Intel MKL PARDISO), which is used automatically: several times faster in 2D, tens of times faster in 3D.
<img width="600" alt="TopoptComec_cli" src="https://github.com/user-attachments/assets/62b4be04-5085-4edc-ad68-3be169d11fd7" />

### 3. Create
Tweak the parameters or choose a preset and hit "Create"!

### Update
Check for a new release from the terminal:
```bash
uv run topoptcomec update --check
uv run topoptcomec update
```
or click the update button in the GUI header. Git checkouts update
automatically when they have no uncommitted changes. See
[docs/UPDATING.md](docs/UPDATING.md) for supported installation methods,
exit codes, and manual update steps.

> **Units:** TopoptComec is unit-agnostic — one grid element = one length unit, and `E`/forces are whatever consistent unit system you choose. See `docs/GLOSSARY.md` (Units and Conventions) for details, including the artificial-spring convention at load points.

> **Presets:** the CLI and GUI combine the built-in presets, your saved presets (`~/.topoptcomec/presets.json`) and a `presets.json` in the current directory; on a name clash the later one wins. The GUI saves only your own presets, to `~/.topoptcomec/presets.json`, so the checkout stays clean for the updater (CLI: `--presets FILE` uses that file only).

### Export
Once you’re happy with your mechanism, export it for visualization in ParaView or for refinement in your favorite CAD software.

## 📖Wiki
The interface should feel intuitive, but you’ll find detailed visual explanations in the [Wiki](https://github.com/ninja7v/TopoptComec/wiki).

<img width="700" alt="BeeSting_3D" src="https://github.com/user-attachments/assets/38f82626-73b6-4946-abf6-5451c18381c5" />

## ✍️Contribute
Ideas, bug reports, or pull requests — all are welcome. Let’s build something awesome together!

See the CONTRIBUTING.md file for details. 

Thank you for using TopoptComec 🙂

> Just optimize!

## Licensing
This project is licensed under the MIT License - see the LICENSE.txt file for details.

This project also uses the PySide6 library, which is licensed under the GNU Lesser General Public License v3.0 (LGPLv3). The source code for PySide6 can be obtained from its official repository: [https://github.com/pyside/pyside-setup](https://github.com/pyside/pyside-setup).
