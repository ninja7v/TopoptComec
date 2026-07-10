<img width="30" height="30" alt="TopoptComec_logo" src="https://github.com/user-attachments/assets/b93640d7-cda4-4a47-9a7a-783906d2c19c" /> TopoptComec
============
**Interactive topology optimization for compliant mechanisms — unleash your creativity!**
![TopoptComec_demo](https://github.com/user-attachments/assets/24a2853e-c53b-44d8-89d0-d7df75364ca9)

## What is TopoptComec?
TopoptComec helps you design compliant mechanisms — flexible structures that achieve motion through material deformation rather than joints.

Simply draw your domain, set forces and supports, choose your material and optimizer, then watch the algorithm sculpt the optimal shape — in 2D or 3D.

## Why using TopoptComec?
None of the existing competing solutions combine all of these features in a single tool:
- 🔨 **Rigid structure *and* compliant mechanism**  — Same solver, just change the loads.
- 🧊 **2D *and* 3D support** — Real engineering isn’t flat.
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
<img width="500" alt="image" src="https://github.com/user-attachments/assets/bf4e7643-2014-4521-bf0c-4d2cc78c7fe7" />

### 3. Create
Tweak the parameters or choose a preset and hit "Create"!

> **Units:** TopoptComec is unit-agnostic — one grid element = one length unit, and `E`/forces are whatever consistent unit system you choose. See `docs/GLOSSARY.md` (Units and Conventions) for details, including the artificial-spring convention at load points.

> **Presets:** the CLI and GUI look for `presets.json` in the current directory first, then `~/.topoptcomec/presets.json`, then fall back to the packaged examples (CLI: override with `--presets`).

### Export
Once you’re happy with your mechanism, export it for visualization in ParaView or for refinement in your favorite CAD software.

## 📖Wiki
The interface should feel intuitive, but you’ll find detailed visual explanations in the [Wiki](https://github.com/ninja7v/TopoptComec/wiki).

![TopoptComec_intro](https://github.com/user-attachments/assets/b36106a4-f642-4f50-9926-128de2fab463)

## ✍️Contribute
Ideas, bug reports, or pull requests — all are welcome. Let’s build something awesome together!

See the CONTRIBUTING.md file for details. 

Thank you for using TopoptComec 🙂

> Just optimize!

## Licensing
This project is licensed under the MIT License - see the LICENSE.txt file for details.

This project also uses the PySide6 library, which is licensed under the GNU Lesser General Public License v3.0 (LGPLv3). The source code for PySide6 can be obtained from its official repository: [https://github.com/pyside/pyside-setup](https://github.com/pyside/pyside-setup).
