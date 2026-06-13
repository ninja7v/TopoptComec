# Contributing to TopoptComec

Thank you for helping improve TopoptComec!

Everything happens on GitHub: https://github.com/ninja7v/topoptcomec

## Maintainers

- Developer: Luc Prevost — GitHub: @ninja7v
- Tester: Claire Prevost — GitHub: @clairette22

## How to Contribute

### Bug report

Create an [issues](https://github.com/ninja7v/topoptcomec/issues).

### Bug fixes & improvements

1. Create a branch for a single, focused change.
2. Implement and add tests where appropriate.
3. Install the dev dependencies: `python -m pip install -r requirements-dev.txt`
4. Run formatting `ruff format .` and checks `ruff format --check . && ruff check . && lizard -L 150 -C 30 -w app tests && pytest`.
5. Open a pull request describing what you changed and why.

Vibe coders are welcome, just make sure to read and understand every line!

### Questions / Suggestions

Use [GitHub Discussions](https://github.com/ninja7v/topoptcomec/discussions).

## Current Project Focus

1. Stability — fix bugs
2. Performance — faster display
3. Features — new functionality

---

Thanks again for contributing! ❤️