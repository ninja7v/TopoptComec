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
```cmd
git clone https://github.com/ninja7V/topoptcomec.git
cd topoptcomec
```
### 2. Set up the environment (once)
Create and activate a virtual environment:
```bash
python -m venv .venv
```
- macOS / Linux:
```bash
source .venv/bin/activate
```
- Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
- Windows CMD:
```cmd
.\.venv\Scripts\activate.bat
```
Then install:
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

### 3. Run
> **Note:** Always activate the virtual environment first (see step 2) before running `topoptcomec`.

GUI:
```bash
topoptcomec
```
CLI:
```bash
topoptcomec -p ForceInverter_2Sup_2D --preview
```
### Create
Tweak the parameters or choose a preset and hit "Create"!

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
