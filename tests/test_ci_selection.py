# tests/test_ci_selection.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the CI test selection (.github/scripts/select_tests.py).

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / ".github" / "scripts" / "select_tests.py"
_spec = importlib.util.spec_from_file_location("select_tests", SCRIPT)
select_tests = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(select_tests)

CORE_TESTS = {"tests/test_fem.py", "tests/test_numerics.py", "tests/test_optimizers.py"}
GUI_TESTS = {"tests/test_main_window.py", "tests/test_widgets.py"}


def _run(*changed: str) -> dict:
    return select_tests.select(list(changed))


def test_every_test_file_is_reachable_from_source():
    """A test importing nothing from topoptcomec would never be selected."""
    graph = select_tests.import_graph()
    for test in (k for k in graph if k.startswith("tests/")):
        if test == "tests/test_ci_selection.py":
            continue
        assert select_tests.closure(graph, test), test


def test_documentation_runs_nothing():
    result = _run("README.md", "docs/ARCHITECTURE.md", "docs/images/a.png")
    assert not result["lint"] and not result["tests"] and not result["cli_smoke"]
    assert select_tests.outputs(result)["platforms"] == "false"


def test_gui_change_skips_core_and_cli_tests():
    tests = set(_run("topoptcomec/gui/widgets.py")["tests"])
    assert GUI_TESTS <= tests
    assert not tests & (CORE_TESTS | {"tests/test_cli.py"})


def test_core_change_runs_dependent_gui_and_cli_tests():
    result = _run("topoptcomec/core/fem.py")
    assert not result["full"] and result["cli_smoke"]
    assert CORE_TESTS | {"tests/test_main_window.py", "tests/test_cli.py"} <= set(
        result["tests"]
    )


def test_shared_module_change_runs_all_its_users():
    tests = set(_run("topoptcomec/presets_io.py")["tests"])
    assert {
        "tests/test_presets_io.py",
        "tests/test_cli.py",
        "tests/test_main_window.py",
    } <= tests


def test_relative_imports_are_followed():
    """gui/widgets.py imports gui/icons.py with `from .icons import icons`."""
    assert "tests/test_widgets.py" in _run("topoptcomec/gui/icons.py")["tests"]
    assert "tests/test_widgets.py" in _run("topoptcomec/icons/x.svg")["tests"]


def test_test_file_change_runs_that_file():
    result = _run("tests/test_widgets.py")
    assert result["tests"] == ["tests/test_widgets.py"] and not result["cli_smoke"]


def test_entry_point_change_runs_cli_smoke():
    assert _run("topoptcomec/__main__.py")["cli_smoke"]


@pytest.mark.parametrize(
    "path",
    [
        "pyproject.toml",
        "requirements.txt",
        ".github/workflows/tests.yml",
        "tests/conftest.py",
        "tests/references/x.npz",
        "topoptcomec/presets.json",
        "results/Bridge_2D/Bridge_2D_density_field.npz",
        "some/unknown/file",
    ],
)
def test_infrastructure_and_unknown_files_run_everything(path):
    result = _run("README.md", path)
    assert result["full"] and result["cli_smoke"]
    assert set(result["tests"]) >= CORE_TESTS | GUI_TESTS
