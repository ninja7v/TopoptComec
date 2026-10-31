# tests/test_cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the CLI.

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from topoptcomec.cli.cli import _params_hash, run_cli
from topoptcomec.cli.cli_preview import (
    _density_to_2d_rgb,
    _downscale_rgb,
    _render_lines_half_block,
    render_preview,
)


@pytest.fixture
def mock_presets_data():
    """Load presets from tests/presets_test.json."""
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_cli_help():
    """Test that -h/--help exits with code 0."""
    with patch.object(sys, "argv", ["main.py", "-h"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 0


# --- CLI run tests (real files, mocked optimizer/exporter where useful) ---

FIXTURE_PRESETS = Path(__file__).parent / "presets_test.json"


def _tiny_preset(n_it: int = 4) -> dict:
    """A small preset that optimizes in well under a second."""
    presets = json.loads(FIXTURE_PRESETS.read_text(encoding="utf-8"))
    preset = json.loads(json.dumps(presets["ForceInverter_2Sup_2D"]))
    preset["Optimizer"]["n_it"] = n_it
    return preset


def _prepare(tmp_path: Path, preset: dict, name: str = "Tiny"):
    """Write a one-preset JSON file and return (presets_file, output_dir)."""
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(json.dumps({name: preset}), encoding="utf-8")
    return presets_file, tmp_path / "out"


def _argv(presets_file: Path, out_dir: Path, *extra: str, name: str = "Tiny"):
    return [
        "main.py",
        "-p",
        name,
        "--presets",
        str(presets_file),
        "-o",
        str(out_dir),
        *extra,
    ]


def test_run_cli_valid_png(tmp_path):
    """A tiny preset runs end to end and writes the PNG and the cache."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png")):
        run_cli()

    assert (out_dir / "Tiny.png").is_file()
    cache = out_dir / "Tiny" / "Tiny_density_field.npz"
    assert cache.is_file()
    with np.load(cache) as data:
        assert "params_hash" in data
        assert "nelxyz" in data


def test_run_cli_all_formats(tmp_path):
    """The default format exports the four supported files."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with patch.object(sys, "argv", _argv(presets_file, out_dir)):
        run_cli()

    for suffix in ("png", "stl", "vti", "3mf"):
        assert (out_dir / f"Tiny.{suffix}").is_file(), suffix


def test_run_cli_threshold(tmp_path):
    """-t binarizes the field handed to the exporter."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch("topoptcomec.cli.cli.exporters") as mock_exporters,
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png", "-t")),
    ):
        mock_exporters.save_as_png.return_value = (True, None)
        run_cli()

    exported = mock_exporters.save_as_png.call_args.args[0]
    assert set(np.unique(exported)) <= {0.0, 1.0}


def test_run_cli_invalid_preset(tmp_path, capsys):
    """An unknown preset name exits with code 1 before optimizing."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch.object(sys, "argv", _argv(presets_file, out_dir, name="DoesNotExist")),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()
    assert cm.value.code == 1
    assert "not found" in capsys.readouterr().out


def test_run_cli_presets_file_not_found(tmp_path, capsys):
    """A missing presets file exits with code 1."""
    missing = tmp_path / "nope.json"
    with (
        patch.object(sys, "argv", _argv(missing, tmp_path / "out")),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()
    assert cm.value.code == 1
    assert "presets file not found" in capsys.readouterr().out


def test_run_cli_json_decode_error(tmp_path, capsys):
    """Malformed JSON exits with code 1."""
    bad = tmp_path / "presets.json"
    bad.write_text("{not json", encoding="utf-8")
    with (
        patch.object(sys, "argv", _argv(bad, tmp_path / "out")),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()
    assert cm.value.code == 1
    assert "Error reading" in capsys.readouterr().out


def test_run_cli_optimization_failure(tmp_path):
    """An optimizer exception exits with code 1."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch(
            "topoptcomec.cli.cli.optimizers.optimize",
            side_effect=RuntimeError("Solver diverged"),
        ),
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png")),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()
    assert cm.value.code == 1


def test_run_cli_optimization_empty_structure(tmp_path):
    """An empty structure is rejected and never cached or exported."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch(
            "topoptcomec.cli.cli.optimizers.optimize",
            return_value=(np.zeros(150), np.zeros(300)),
        ),
        patch("topoptcomec.cli.cli.exporters") as mock_exporters,
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png")),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()

    assert cm.value.code == 1
    mock_exporters.save_as_png.assert_not_called()
    assert not (out_dir / "Tiny" / "Tiny_density_field.npz").exists()


def test_run_cli_export_failure(tmp_path, capsys):
    """An exporter failure is reported but the run still completes."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch("topoptcomec.cli.cli.exporters") as mock_exporters,
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png")),
    ):
        mock_exporters.save_as_png.return_value = (False, "Disk full")
        run_cli()

    mock_exporters.save_as_png.assert_called_once()
    assert "Disk full" in capsys.readouterr().out


class MockFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
        return self._result


class MockExecutor:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def submit(self, fn, *args, **kwargs):
        return MockFuture(fn(*args, **kwargs))


def test_run_cli_multiple_presets(tmp_path):
    """Two comma-separated presets both run."""
    tiny = _tiny_preset()
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(json.dumps({"Tiny": tiny, "Tiny2": tiny}), encoding="utf-8")
    out_dir = tmp_path / "out"
    argv = [
        "main.py",
        "-p",
        "Tiny,Tiny2",
        "--presets",
        str(presets_file),
        "-o",
        str(out_dir),
        "-f",
        "png",
    ]
    with (
        patch("topoptcomec.cli.cli.ProcessPoolExecutor", MockExecutor),
        patch("topoptcomec.cli.cli.exporters") as mock_exporters,
        patch(
            "topoptcomec.cli.cli.optimizers.optimize",
            return_value=(np.ones(150) * 0.3, np.zeros(300)),
        ),
    ):
        mock_exporters.save_as_png.return_value = (True, None)
        with patch.object(sys, "argv", argv):
            run_cli()

    assert mock_exporters.save_as_png.call_count == 2


def test_run_cli_multiple_presets_one_invalid(tmp_path):
    """An invalid preset in a comma list exits before any optimization."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch.object(
            sys, "argv", _argv(presets_file, out_dir, name="Tiny,NonExistent")
        ),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()
    assert cm.value.code == 1


def test_run_cli_cache_hit(tmp_path, capsys):
    """A cache written for the same parameters is reused."""
    from topoptcomec import result_cache

    preset = _tiny_preset()
    presets_file, out_dir = _prepare(tmp_path, preset)
    nelxyz = preset["Dimensions"]["nelxyz"]
    nel = nelxyz[0] * nelxyz[1]
    ndof = 2 * (nelxyz[0] + 1) * (nelxyz[1] + 1)
    result_cache.save_density_cache(
        result_cache.cache_path(out_dir, "Tiny"),
        np.full(nel, 0.3),
        np.zeros(ndof),
        preset,
        nelxyz,
        converged=True,
    )

    with (
        patch("topoptcomec.cli.cli.optimizers.optimize") as mock_optimize,
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png", "-v")),
    ):
        run_cli()
        mock_optimize.assert_not_called()
    assert "Loading cached density field" in capsys.readouterr().out


def test_run_cli_saving_cache(tmp_path):
    """A fresh run stores a cache with the metadata needed to validate it."""
    preset = _tiny_preset()
    presets_file, out_dir = _prepare(tmp_path, preset)
    with (
        patch(
            "topoptcomec.cli.cli.optimizers.optimize",
            return_value=(np.ones(150) * 0.3, np.zeros(300)),
        ),
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png")),
    ):
        run_cli()

    cache = out_dir / "Tiny" / "Tiny_density_field.npz"
    assert cache.is_file()
    with np.load(cache) as data:
        assert str(data["params_hash"]) == _params_hash(preset)
        assert np.array_equal(
            data["nelxyz"], np.asarray(preset["Dimensions"]["nelxyz"])
        )


def test_run_cli_analysis_flag(tmp_path):
    """-a runs the analysis on the result."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch("topoptcomec.cli.cli.analyzers.analyze") as mock_analyze,
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png", "-a")),
    ):
        mock_analyze.return_value = (False, True, True, True)
        run_cli()
    mock_analyze.assert_called_once()


def test_run_cli_displacement_flag(tmp_path):
    """-d runs the displacement simulation."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    with (
        patch("topoptcomec.core.displacements.run_iterative_displacement") as mock_disp,
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png", "-d")),
    ):
        mock_disp.return_value = [np.zeros(150)]
        run_cli()
    mock_disp.assert_called_once()


def test_run_cli_restart_from(tmp_path):
    """--restart-from feeds a previous field into the optimizer."""
    from topoptcomec import result_cache

    preset = _tiny_preset()
    preset["Materials"]["init_type"] = 3
    presets_file, out_dir = _prepare(tmp_path, preset)
    restart = tmp_path / "restart.npz"
    result_cache.save_density_cache(
        restart,
        np.full(150, 0.3),
        np.zeros(2 * 16 * 11),
        preset,
        preset["Dimensions"]["nelxyz"],
        converged=True,
    )

    with (
        patch(
            "topoptcomec.cli.cli.optimizers.optimize",
            return_value=(np.full(150, 0.3), np.zeros(352)),
        ) as mock_optimize,
        patch.object(
            sys,
            "argv",
            _argv(presets_file, out_dir, "-f", "png", "--restart-from", str(restart)),
        ),
    ):
        run_cli()

    assert mock_optimize.call_args.kwargs["current_xPhys"] is not None


class TestDensityTo2DRgb:
    """Tests for _density_to_2d_rgb conversion."""

    def test_2d_shape(self):
        """2-D field produces (ny, nx, 3) output."""
        nx, ny = 6, 4
        xPhys = np.random.rand(nx * ny)
        result = _density_to_2d_rgb(xPhys, [nx, ny, 0], None)
        assert result.shape == (ny, nx, 3)

    def test_3d_shape(self):
        """3-D field produces (ny, nx, 3) XY projection."""
        nx, ny, nz = 5, 4, 3
        xPhys = np.random.rand(nz * nx * ny)
        result = _density_to_2d_rgb(xPhys, [nx, ny, nz], None)
        assert result.shape == (ny, nx, 3)

    def test_3d_max_projection(self):
        """3-D projection picks the max density along Z."""
        nx, ny, nz = 2, 2, 3
        xPhys = np.zeros(nz * nx * ny)
        idx = 1 * (nx * ny) + 0 * ny + 0
        xPhys[idx] = 0.9
        result = _density_to_2d_rgb(xPhys, [nx, ny, nz], None)
        # (x=0, y=0) → row (ny-1) after flipud; black material → rgb 0.1
        assert result[ny - 1, 0, 0] == pytest.approx(0.1, abs=1e-6)

    def test_2d_values_clipped(self):
        """Output values are clipped to [0, 1]."""
        xPhys = np.array([1.5, -0.5, 0.5, 0.8])
        result = _density_to_2d_rgb(xPhys, [2, 2, 0], None)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_void_is_white(self):
        """Zero-density elements appear white (1, 1, 1)."""
        xPhys = np.zeros(6)
        result = _density_to_2d_rgb(xPhys, [3, 2, 0], None)
        np.testing.assert_allclose(result, 1.0, atol=1e-10)

    def test_full_density_black_default(self):
        """Full density with no color defaults to black (0, 0, 0)."""
        xPhys = np.ones(6)
        result = _density_to_2d_rgb(xPhys, [3, 2, 0], None)
        np.testing.assert_allclose(result, 0.0, atol=1e-10)

    def test_custom_color(self):
        """Custom color is blended correctly."""
        xPhys = np.ones(6)
        result = _density_to_2d_rgb(xPhys, [3, 2, 0], ["#FF0000"])
        np.testing.assert_allclose(result[:, :, 0], 1.0, atol=1e-6)  # red
        np.testing.assert_allclose(result[:, :, 1], 0.0, atol=1e-6)  # green 0
        np.testing.assert_allclose(result[:, :, 2], 0.0, atol=1e-6)  # blue 0


class TestDownscaleRgb:
    """Tests for _downscale_rgb."""

    def test_no_downscale_when_smaller(self):
        """Image not wider than target is returned unchanged."""
        image = np.random.rand(10, 20, 3)
        result = _downscale_rgb(image, 30)
        np.testing.assert_array_equal(result, image)

    def test_downscale_width(self):
        """Output width matches the target."""
        image = np.random.rand(20, 100, 3)
        result = _downscale_rgb(image, 40)
        assert result.shape[1] == 40

    def test_downscale_preserves_range(self):
        """Downscaled values remain in [0, 1]."""
        image = np.random.rand(50, 200, 3)
        result = _downscale_rgb(image, 60)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_uniform_image_stays_uniform(self):
        """A uniform image remains uniform after downscale."""
        image = np.full((40, 120, 3), 0.7)
        result = _downscale_rgb(image, 30)
        np.testing.assert_allclose(result, 0.7, atol=1e-10)


class TestRenderLinesHalfBlock:
    """Tests for _render_lines_half_block."""

    def test_line_count_even_rows(self):
        """Even row count produces rows/2 output lines."""
        image = np.random.rand(6, 10, 3)
        lines = _render_lines_half_block(image)
        assert len(lines) == 3

    def test_line_count_odd_rows(self):
        """Odd row count pads to even and produces ceil(rows/2) lines."""
        image = np.random.rand(5, 10, 3)
        lines = _render_lines_half_block(image)
        assert len(lines) == 3

    def test_contains_reset(self):
        """Each line ends with the ANSI reset sequence."""
        image = np.random.rand(4, 8, 3)
        lines = _render_lines_half_block(image)
        for line in lines:
            assert line.endswith("\033[0m")


class TestRenderPreview:
    """Integration tests for render_preview."""

    def test_returns_string(self):
        """render_preview returns a non-empty string."""
        xPhys = np.random.rand(20)
        result = render_preview(xPhys, [4, 5, 0], max_width=40)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_no_line_exceeds_max_width(self):
        """No visible line exceeds the max_width limit."""
        import re

        xPhys = np.random.rand(200)
        width = 30
        result = render_preview(xPhys, [20, 10, 0], max_width=width)
        ansi_re = re.compile(r"\033\[[0-9;]*m")
        for line in result.split("\n"):
            visible = ansi_re.sub("", line)
            assert len(visible) <= width

    def test_3d_header_mentions_projection(self):
        """3-D preview header mentions XY projection."""
        xPhys = np.random.rand(5 * 4 * 3)
        result = render_preview(xPhys, [5, 4, 3], max_width=40)
        assert "XY projection" in result

    def test_2d_header_no_projection(self):
        """2-D preview header does not mention projection."""
        xPhys = np.random.rand(20)
        result = render_preview(xPhys, [4, 5, 0], max_width=40)
        assert "XY projection" not in result

    def test_produces_ansi_escapes(self):
        """Preview contains ANSI escape sequences."""
        xPhys = np.random.rand(12)
        result = render_preview(xPhys, [3, 4, 0], max_width=20)
        assert "\033[" in result


# --- End-to-end tests (no mocks) ---


def test_run_cli_end_to_end(tmp_path, capsys):
    """Full CLI run on a tiny preset: optimize, export, cache and reuse."""
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r", encoding="utf-8") as f:
        presets = json.load(f)
    small = {"Tiny": presets["ForceInverter_2Sup_2D"]}
    preset_file = tmp_path / "presets.json"
    preset_file.write_text(json.dumps(small))
    out_dir = tmp_path / "out"

    argv = [
        "main.py",
        "-p",
        "Tiny",
        "-f",
        "png",
        "--presets",
        str(preset_file),
        "-o",
        str(out_dir),
    ]
    with patch.object(sys, "argv", argv):
        run_cli()
    assert (out_dir / "Tiny.png").exists()
    cache = out_dir / "Tiny" / "Tiny_density_field.npz"
    assert cache.exists()
    data = np.load(cache)
    assert "params_hash" in data

    csv_file = out_dir / "Tiny" / "Tiny_loss_function.csv"
    assert csv_file.exists()
    csv_content = csv_file.read_text(encoding="utf-8")
    assert csv_content.startswith("iteration,objective")
    assert len(csv_content.splitlines()) > 1

    # Second run with identical parameters: cache hit, no re-optimization.
    capsys.readouterr()
    with patch.object(sys, "argv", argv + ["-v"]):
        run_cli()
    assert "Loading cached density field" in capsys.readouterr().out


def test_run_cli_cache_invalidated_on_param_change(tmp_path, capsys):
    """Editing a preset must invalidate the cached result."""
    presets_path = Path(__file__).parent / "presets_test.json"
    with open(presets_path, "r", encoding="utf-8") as f:
        presets = json.load(f)
    preset = json.loads(json.dumps(presets["ForceInverter_2Sup_2D"]))
    preset_file = tmp_path / "presets.json"
    preset_file.write_text(json.dumps({"Tiny": preset}))
    out_dir = tmp_path / "out"

    def argv():
        return [
            "main.py",
            "-p",
            "Tiny",
            "-f",
            "png",
            "-v",
            "--presets",
            str(preset_file),
            "-o",
            str(out_dir),
        ]

    with patch.object(sys, "argv", argv()):
        run_cli()

    # Change a physical parameter and rerun: cache must be ignored.
    preset["Dimensions"]["volfrac"] = 0.42
    preset_file.write_text(json.dumps({"Tiny": preset}))
    capsys.readouterr()
    with patch.object(sys, "argv", argv()):
        run_cli()
    out = capsys.readouterr().out
    assert "different parameters" in out
    assert "Loading cached density field" not in out


def test_run_cli_rejects_path_traversal_preset_name(tmp_path, capsys):
    """A preset name must never write outside the output directory."""
    preset = _tiny_preset()
    presets_file = tmp_path / "presets.json"
    presets_file.write_text(json.dumps({"../evil": preset}), encoding="utf-8")
    out_dir = tmp_path / "out"
    argv = _argv(presets_file, out_dir, "-f", "png", name="../evil")
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as cm:
        run_cli()
    assert cm.value.code == 1
    assert "path separators" in capsys.readouterr().out
    assert not (tmp_path / "evil.png").exists()


def test_run_cli_rejects_duplicate_preset_names(tmp_path, capsys):
    """Two workers must not race on the same cache and export paths."""
    presets_file, out_dir = _prepare(tmp_path, _tiny_preset())
    argv = _argv(presets_file, out_dir, "-f", "png", name="Tiny,Tiny")
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as cm:
        run_cli()
    assert cm.value.code == 1
    assert "more than once" in capsys.readouterr().out


def test_run_cli_reports_malformed_preset_without_traceback(tmp_path, capsys):
    """Malformed presets get a friendly message instead of a traceback."""
    preset = _tiny_preset()
    del preset["Dimensions"]["nelxyz"]
    presets_file, out_dir = _prepare(tmp_path, preset)
    with (
        patch.object(sys, "argv", _argv(presets_file, out_dir, "-f", "png")),
        pytest.raises(SystemExit) as cm,
    ):
        run_cli()
    out = capsys.readouterr().out
    assert cm.value.code == 1
    assert "is invalid" in out
    assert "Traceback" not in out


def test_run_cli_rejects_corrupt_restart_field(tmp_path, capsys):
    """NaN/inf/out-of-range restart fields are rejected before the solver."""
    preset = _tiny_preset()
    presets_file, out_dir = _prepare(tmp_path, preset)
    nelxyz = preset["Dimensions"]["nelxyz"]
    nel = nelxyz[0] * nelxyz[1]
    ndof = 2 * (nelxyz[0] + 1) * (nelxyz[1] + 1)
    restart = tmp_path / "restart.npz"
    np.savez_compressed(
        restart,
        xPhys=np.full(nel, np.nan),
        u=np.zeros(ndof),
    )
    argv = _argv(presets_file, out_dir, "-f", "png") + [
        "--restart-from",
        str(restart),
    ]
    with patch.object(sys, "argv", argv), pytest.raises(SystemExit) as cm:
        run_cli()
    assert cm.value.code == 1
    assert "NaN or infinite" in capsys.readouterr().out
