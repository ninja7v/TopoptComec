# tests/test_cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the CLI.

import sys
import json
from unittest.mock import patch
import pytest
import numpy as np
from pathlib import Path
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


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.Path.mkdir")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_valid_png(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_mkdir,
    mock_savez,
    mock_presets_data,
):
    """Test running CLI with a valid preset and png output."""
    # Setup mocks
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    # Mock result from optimize (valid non-zero density)
    # ForceInverter_2Sup_2D has nelxyz = [15, 10, 0] -> nel = 150
    mock_xPhys = np.ones(150)
    mock_optimize.return_value = (mock_xPhys, None)

    # Mock exporters success
    mock_exporters.save_as_png.return_value = (True, None)

    # Run (on a real preset, not a test one since the function looks for presets.json)
    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-p", preset_name, "-f", "png"]):
        run_cli()

    # Verify optimize called with correct parameters
    mock_optimize.assert_called_once()
    call_kwargs = mock_optimize.call_args.kwargs
    assert call_kwargs["Dimensions"]["nelxyz"] == [15, 10, 0]
    assert "disp_factor" not in call_kwargs  # Should be removed

    # Verify export called
    mock_exporters.save_as_png.assert_called_once()
    args, _ = mock_exporters.save_as_png.call_args
    # Check filename ends with .png and contains preset name
    assert str(args[2]).endswith(f"{preset_name}.png")


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.Path.mkdir")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_all_formats(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_mkdir,
    mock_savez,
    mock_presets_data,
):
    """Test running CLI with default format (all)."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data
    mock_optimize.return_value = (np.ones(150), None)
    mock_exporters.save_as_png.return_value = (True, None)
    mock_exporters.save_as_vti.return_value = (True, None)
    mock_exporters.save_as_stl.return_value = (True, None)
    mock_exporters.save_as_3mf.return_value = (True, None)

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-p", preset_name]):
        run_cli()

    mock_exporters.save_as_png.assert_called_once()
    mock_exporters.save_as_vti.assert_called_once()
    mock_exporters.save_as_stl.assert_called_once()
    mock_exporters.save_as_3mf.assert_called_once()


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.Path.mkdir")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_threshold(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_mkdir,
    mock_savez,
    mock_presets_data,
):
    """Test running CLI with threshold option."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    # Return gray values: 0.2 and 0.8
    mock_xPhys = np.array([0.2, 0.8])
    mock_optimize.return_value = (mock_xPhys, None)

    mock_exporters.save_as_png.return_value = (True, None)

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-p", preset_name, "-f", "png", "-t"]):
        run_cli()

    # Verify the exporter received binary values: 0.0 and 1.0
    args, _ = mock_exporters.save_as_png.call_args
    exported_xPhys = args[0]
    expected = np.array([0.0, 1.0])
    np.testing.assert_array_equal(exported_xPhys, expected)


@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_invalid_preset(
    mock_exists, mock_json_load, mock_open, mock_presets_data
):
    """Test behavior when preset name is invalid."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    with patch.object(sys, "argv", ["main.py", "-p", "NonExistentPreset"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1


@patch.object(Path, "exists")
def test_run_cli_presets_file_not_found(mock_exists):
    """Test behavior when presets.json is missing."""
    mock_exists.return_value = False

    with patch.object(sys, "argv", ["main.py", "-p", "TestPreset"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1


@patch("builtins.open")
@patch.object(Path, "exists")
def test_run_cli_json_decode_error(mock_exists, mock_open):
    """Test behavior when presets.json contains invalid JSON."""
    mock_exists.return_value = True
    mock_open.return_value.__enter__ = lambda s: s
    mock_open.return_value.__exit__ = lambda s, *a: None

    with patch("json.load", side_effect=json.JSONDecodeError("err", "doc", 0)):
        with patch.object(sys, "argv", ["main.py", "-p", "TestPreset"]):
            with pytest.raises(SystemExit) as cm:
                run_cli()
            assert cm.value.code == 1


@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_optimization_failure(
    mock_exists, mock_json_load, mock_open, mock_optimize, mock_presets_data
):
    """Test behavior when the optimizer raises an exception."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data
    mock_optimize.side_effect = RuntimeError("Solver diverged")

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-p", preset_name, "-f", "png"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_optimization_empty_structure(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_savez,
    mock_presets_data,
):
    """Test behavior when the optimizer produces an empty (invalid) structure."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data
    mock_optimize.return_value = (np.zeros(150), None)
    mock_exporters.save_as_png.return_value = (True, None)

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-p", preset_name, "-f", "png"]):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1

    # Exporter must not be called for an invalid result
    mock_exporters.save_as_png.assert_not_called()
    # Invalid result must not be cached
    mock_savez.assert_not_called()


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.Path.mkdir")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_export_failure(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_mkdir,
    mock_savez,
    mock_presets_data,
):
    """Test behavior when an exporter returns failure."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data
    mock_optimize.return_value = (np.ones(150), None)
    mock_exporters.save_as_png.return_value = (False, "Disk full")

    preset_name = "ForceInverter_2Sup_2D"
    with patch.object(sys, "argv", ["main.py", "-p", preset_name, "-f", "png"]):
        # Should not raise, just print error message
        run_cli()

    mock_exporters.save_as_png.assert_called_once()


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


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.Path.mkdir")
@patch("topoptcomec.cli.cli.ProcessPoolExecutor", MockExecutor)
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_multiple_presets(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_mkdir,
    mock_savez,
    mock_presets_data,
):
    """Test running CLI with multiple comma-separated presets."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data
    mock_optimize.return_value = (np.ones(150), None)
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys,
        "argv",
        ["main.py", "-p", "ForceInverter_2Sup_2D,Gripper_2D", "-f", "png"],
    ):
        run_cli()

    assert mock_optimize.call_count == 2
    assert mock_exporters.save_as_png.call_count == 2


@patch("builtins.open")
@patch("json.load")
@patch.object(Path, "exists")
def test_run_cli_multiple_presets_one_invalid(
    mock_exists, mock_json_load, mock_open, mock_presets_data
):
    """Test that an invalid preset in a comma list exits before any optimization."""
    mock_exists.return_value = True
    mock_json_load.return_value = mock_presets_data

    with patch.object(
        sys,
        "argv",
        ["main.py", "-p", "ForceInverter_2Sup_2D,NonExistent"],
    ):
        with pytest.raises(SystemExit) as cm:
            run_cli()
        assert cm.value.code == 1


@patch("topoptcomec.cli.cli.np.load")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("topoptcomec.cli.cli.Path.exists")
def test_run_cli_cache_hit(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_np_load,
    mock_presets_data,
):
    """Test that optimize is not called when cache is found."""
    # First call: preset.json exists. Second call: cache exists
    mock_exists.side_effect = [True, True]
    mock_json_load.return_value = mock_presets_data

    mock_np_load.return_value = {
        "xPhys": np.zeros(150),
        "u": np.zeros(300),
        "params_hash": _params_hash(mock_presets_data["ForceInverter_2Sup_2D"]),
    }
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys, "argv", ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png"]
    ):
        run_cli()

    mock_optimize.assert_not_called()
    mock_np_load.assert_called_once()
    mock_exporters.save_as_png.assert_called_once()


@patch("topoptcomec.cli.cli.np.savez_compressed")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("topoptcomec.cli.cli.Path.exists")
def test_run_cli_saving_cache(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_savez,
    mock_presets_data,
):
    """Test that cache is saved after a successful optimization."""
    # First call: preset.json exists. Second call: cache does not exist
    mock_exists.side_effect = [True, False]
    mock_json_load.return_value = mock_presets_data

    mock_optimize.return_value = (np.ones(150), np.zeros(300))
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys, "argv", ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png"]
    ):
        run_cli()

    mock_optimize.assert_called_once()
    mock_savez.assert_called_once()
    mock_exporters.save_as_png.assert_called_once()


@patch("topoptcomec.cli.cli.analyzers.analyze")
@patch("topoptcomec.cli.cli.np.load")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("topoptcomec.cli.cli.Path.exists")
def test_run_cli_analysis_flag(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_np_load,
    mock_analyze,
    mock_presets_data,
):
    """Test that analysis runs when -a is passed."""
    mock_exists.side_effect = [True, True]
    mock_json_load.return_value = mock_presets_data
    mock_np_load.return_value = {
        "xPhys": np.zeros(150),
        "u": np.zeros(300),
        "params_hash": _params_hash(mock_presets_data["ForceInverter_2Sup_2D"]),
    }
    mock_analyze.return_value = (False, True, False, True)
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys,
        "argv",
        ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png", "-a"],
    ):
        run_cli()

    mock_analyze.assert_called_once()


@patch("topoptcomec.core.displacements.run_iterative_displacement")
@patch("topoptcomec.cli.cli.np.load")
@patch("topoptcomec.cli.cli.optimizers.optimize")
@patch("topoptcomec.cli.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("topoptcomec.cli.cli.Path.exists")
def test_run_cli_displacement_flag(
    mock_exists,
    mock_json_load,
    mock_open,
    mock_exporters,
    mock_optimize,
    mock_np_load,
    mock_disp,
    mock_presets_data,
):
    """Test that displacement runs when -d is passed."""
    mock_exists.side_effect = [True, True]
    mock_json_load.return_value = mock_presets_data
    mock_np_load.return_value = {
        "xPhys": np.zeros(150),
        "u": np.zeros(300),
        "params_hash": _params_hash(mock_presets_data["ForceInverter_2Sup_2D"]),
    }
    mock_disp.return_value = [np.zeros((3, 150))]
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys, "argv", ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png", "-d"]
    ):
        run_cli()

    mock_disp.assert_called_once()


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
