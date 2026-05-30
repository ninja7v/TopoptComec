# tests/test_cli.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the CLI.

import sys
import json
from unittest.mock import patch
import pytest
import numpy as np
from pathlib import Path
from app.cli import run_cli


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


@patch("app.cli.np.savez_compressed")
@patch("app.cli.Path.mkdir")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
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

    # Mock result from optimize
    # ForceInverter_2Sup_2D has nelxyz = [15, 10, 0] -> nel = 150
    mock_xPhys = np.zeros(150)
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


@patch("app.cli.np.savez_compressed")
@patch("app.cli.Path.mkdir")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
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
    mock_optimize.return_value = (np.zeros(150), None)
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


@patch("app.cli.np.savez_compressed")
@patch("app.cli.Path.mkdir")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
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


@patch("app.cli.optimizers.optimize")
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


@patch("app.cli.np.savez_compressed")
@patch("app.cli.Path.mkdir")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
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
    mock_optimize.return_value = (np.zeros(150), None)
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


@patch("app.cli.np.savez_compressed")
@patch("app.cli.Path.mkdir")
@patch("app.cli.ProcessPoolExecutor", MockExecutor)
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
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
    mock_optimize.return_value = (np.zeros(150), None)
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


@patch("app.cli.np.load")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("app.cli.Path.exists")
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

    mock_np_load.return_value = {"xPhys": np.zeros(150), "u": np.zeros(300)}
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys, "argv", ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png"]
    ):
        run_cli()

    mock_optimize.assert_not_called()
    mock_np_load.assert_called_once()
    mock_exporters.save_as_png.assert_called_once()


@patch("app.cli.np.savez_compressed")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("app.cli.Path.exists")
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

    mock_optimize.return_value = (np.zeros(150), np.zeros(300))
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys, "argv", ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png"]
    ):
        run_cli()

    mock_optimize.assert_called_once()
    mock_savez.assert_called_once()
    mock_exporters.save_as_png.assert_called_once()


@patch("app.cli.analyzers.analyze")
@patch("app.cli.np.load")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("app.cli.Path.exists")
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
    mock_np_load.return_value = {"xPhys": np.zeros(150), "u": np.zeros(300)}
    mock_analyze.return_value = (False, True, False, True)
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys,
        "argv",
        ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png", "-a"],
    ):
        run_cli()

    mock_analyze.assert_called_once()


@patch("app.core.displacements.run_iterative_displacement")
@patch("app.cli.np.load")
@patch("app.cli.optimizers.optimize")
@patch("app.cli.exporters")
@patch("builtins.open")
@patch("json.load")
@patch("app.cli.Path.exists")
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
    mock_np_load.return_value = {"xPhys": np.zeros(150), "u": np.zeros(300)}
    mock_disp.return_value = [np.zeros((3, 150))]
    mock_exporters.save_as_png.return_value = (True, None)

    with patch.object(
        sys, "argv", ["main.py", "-p", "ForceInverter_2Sup_2D", "-f", "png", "-d"]
    ):
        run_cli()

    mock_disp.assert_called_once()
