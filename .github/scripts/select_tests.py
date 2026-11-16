# .github/scripts/select_tests.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Decide which CI checks and test files a change needs.
#
# Reads changed paths (one per line) on stdin and prints GitHub Actions
# outputs (key=value lines) on stdout; the reasoning goes to stderr.
#
# Rules, applied to each changed path:
#   - documentation (DOCS_ONLY): nothing to run;
#   - topoptcomec/**/*.py: every test file that imports the module, directly
#     or through other topoptcomec modules (lazy imports included);
#   - tests/**/test_*.py: that test file;
#   - DATA_OWNERS: the tests of the module that owns the data file;
#   - anything else (dependencies, CI, conftest, references, presets...):
#     the full suite. Unknown paths therefore never skip tests.
#
# `--all` selects the full suite without reading stdin (scheduled and
# manual runs).

import ast
import fnmatch
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "topoptcomec"

# Paths that cannot affect any test or lint result.
DOCS_ONLY = (
    "*.md",
    "docs/*",
    "LICENSE.txt",
    ".github/FUNDING.yml",
    ".gitignore",
    ".vscode/*",
)

# Data files read at runtime, mapped to the module that reads them (its tests
# run). Other data (presets.json, results/ caches, test references) runs the
# full suite.
DATA_OWNERS = {
    "topoptcomec/icons/": "topoptcomec.gui.icons",
}

# Test files run on Windows and macOS (platform-sensitive numerics, file
# export, parameter validation).
PLATFORM_TESTS = (
    "tests/test_numerics.py",
    "tests/test_fem.py",
    "tests/test_optimizers.py",
    "tests/test_nonlinear.py",
    "tests/test_exporters.py",
    "tests/test_parameter_validation.py",
)

# Test files that run without VTK, which has no wheels for Python 3.15 yet.
PY315_TESTS = (
    "tests/test_numerics.py",
    "tests/test_fem.py",
    "tests/test_optimizers.py",
    "tests/test_nonlinear.py",
    "tests/test_analyzers.py",
    "tests/test_initializers.py",
    "tests/test_displacements.py",
    "tests/test_workers.py",
    "tests/test_widgets.py",
    "tests/test_main.py",
)


def module_name(path: str) -> str:
    """'topoptcomec/core/fem.py' -> 'topoptcomec.core.fem'."""
    parts = list(Path(path).with_suffix("").parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def imported_modules(path: Path, name: str) -> set[str]:
    """Return the topoptcomec modules a file imports, anywhere in the file.

    ``from pkg import x`` records both ``pkg`` and ``pkg.x``: x may be a
    submodule, and recording a name that is not a module is harmless.
    """
    is_package = path.name == "__init__.py"
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:  # relative import
                base = name.split(".")
                base = base[: len(base) - node.level + is_package]
                base = ".".join(base + ([node.module] if node.module else []))
            else:
                base = node.module or ""
            found.add(base)
            found.update(f"{base}.{alias.name}" for alias in node.names)
    deps = {m for m in found if m == PACKAGE or m.startswith(PACKAGE + ".")}
    # Importing a module runs its parent packages' __init__ first.
    for m in list(deps):
        parts = m.split(".")
        deps.update(".".join(parts[:i]) for i in range(1, len(parts)))
    return deps


def import_graph() -> dict[str, set[str]]:
    """Map each topoptcomec module and test file to what it imports."""
    graph = {}
    for path in sorted((ROOT / PACKAGE).rglob("*.py")):
        name = module_name(path.relative_to(ROOT).as_posix())
        graph[name] = imported_modules(path, name)
    for path in sorted((ROOT / "tests").rglob("test_*.py")):
        # Plain names only: they are written into the workflow's outputs.
        rel = path.relative_to(ROOT).as_posix()
        if not re.fullmatch(r"[\w/]+\.py", rel):
            continue
        graph[rel] = imported_modules(path, "tests")
    return graph


def closure(graph: dict[str, set[str]], start: str) -> set[str]:
    """Every module reachable from ``start`` through imports."""
    seen: set[str] = set()
    todo = list(graph.get(start, ()))
    while todo:
        module = todo.pop()
        if module not in seen:
            seen.add(module)
            todo.extend(graph.get(module, ()))
    return seen


def select(changed: list[str]) -> dict:
    """Classify changed paths and return the selection (see module docstring)."""
    graph = import_graph()
    tests = sorted(k for k in graph if k.startswith("tests/"))
    reach = {t: closure(graph, t) for t in tests}

    modules: set[str] = set()
    selected: set[str] = set()
    full: list[str] = []
    code_changed = False
    for path in changed:
        if any(fnmatch.fnmatch(path, pattern) for pattern in DOCS_ONLY):
            continue
        code_changed = True
        owner = next((o for p, o in DATA_OWNERS.items() if path.startswith(p)), None)
        if owner:
            modules.add(owner)
        elif path.startswith(PACKAGE + "/") and path.endswith(".py"):
            modules.add(module_name(path))
        elif path.startswith("tests/") and fnmatch.fnmatch(
            Path(path).name, "test_*.py"
        ):
            selected.add(path)  # may be deleted: dropped below if so
        else:
            full.append(path)

    selected |= {t for t in tests if reach[t] & modules}
    selected &= set(tests)
    return {
        "lint": code_changed,
        "full": bool(full),
        "full_reason": full,
        "tests": tests if full else sorted(selected),
        "cli_smoke": bool(full)
        or "tests/test_cli.py" in selected
        or f"{PACKAGE}.__main__" in modules,
    }


def outputs(result: dict) -> dict[str, str]:
    """Format the selection as GitHub Actions step outputs."""
    tests = result["tests"]
    platform = [t for t in tests if t in PLATFORM_TESTS]
    return {
        "lint": str(result["lint"]).lower(),
        "full": str(result["full"]).lower(),
        "tests": " ".join(tests),
        "py315_tests": " ".join(t for t in tests if t in PY315_TESTS),
        "platform_tests": " ".join(platform),
        "platforms": str(bool(platform) or result["cli_smoke"]).lower(),
        "cli_smoke": str(result["cli_smoke"]).lower(),
    }


def main() -> None:
    if "--all" in sys.argv[1:]:
        changed = ["<full run requested>"]
    else:
        changed = [line.strip() for line in sys.stdin if line.strip()]
    result = select(changed)
    out = outputs(result)

    print(f"Changed paths ({len(changed)}):", *changed, sep="\n  ", file=sys.stderr)
    if result["full"]:
        print(
            "Full suite, because of:",
            *result["full_reason"],
            sep="\n  ",
            file=sys.stderr,
        )
    for key, value in out.items():
        print(f"{key}: {value or '(none)'}", file=sys.stderr)
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
