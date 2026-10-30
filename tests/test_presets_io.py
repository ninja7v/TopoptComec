# tests/test_presets_io.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for presets file resolution and layering.

import json

import pytest

from topoptcomec.presets_io import (
    load_preset_layers,
    load_presets,
    packaged_presets_path,
    resolve_presets_file,
    save_user_presets,
    user_presets_path,
)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Isolated working directory and home directory."""
    home = tmp_path / "home"
    home.mkdir()
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.chdir(cwd)
    return cwd


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def test_explicit_path_is_the_only_layer(workdir):
    """An explicit --presets file replaces every other layer."""
    explicit = workdir / "custom.json"
    _write(explicit, {"Only": {"a": 1}})
    _write(user_presets_path(), {"User": {}})
    assert resolve_presets_file(str(explicit)) == explicit
    assert load_presets(str(explicit)) == {"Only": {"a": 1}}


def test_missing_explicit_file_raises(workdir):
    """A wrong --presets path is reported, not silently replaced."""
    with pytest.raises(FileNotFoundError):
        load_presets(str(workdir / "missing.json"))


def test_packaged_presets_are_the_default(workdir):
    """Without any other file, the packaged presets are used."""
    packaged = json.loads(packaged_presets_path().read_text(encoding="utf-8"))
    assert load_presets() == packaged


def test_layers_merge_in_documented_precedence(workdir):
    """./presets.json > user presets > built-in presets."""
    builtin_names = set(load_presets())
    some_builtin = min(builtin_names)
    _write(user_presets_path(), {"Mine": {"u": 1}, some_builtin: {"user": True}})
    _write(workdir / "presets.json", {"Project": {"p": 1}, "Mine": {"local": True}})

    layers = load_preset_layers()
    merged = layers.merged
    assert builtin_names <= set(merged)
    assert merged[some_builtin] == {"user": True}
    assert merged["Mine"] == {"local": True}
    assert merged["Project"] == {"p": 1}
    # The user layer only contains what the user saved.
    assert set(layers.user) == {"Mine", some_builtin}


def test_new_builtin_presets_stay_visible_for_existing_users(workdir):
    """A user file from an older release must not hide new built-in presets."""
    _write(user_presets_path(), {"Mine": {"u": 1}})
    merged = load_presets()
    assert "Mine" in merged
    assert "BeeSting_3D" in merged  # shipped with v4


def test_source_checkout_presets_file_is_the_builtin_layer(workdir):
    """Running from the repository root uses topoptcomec/presets.json."""
    _write(workdir / "topoptcomec" / "presets.json", {"Checkout": {}})
    assert load_presets() == {"Checkout": {}}


def test_saving_writes_only_user_owned_presets(workdir):
    """Saving never touches the checkout and stores only user presets."""
    checkout = workdir / "topoptcomec" / "presets.json"
    _write(checkout, {"Builtin": {"b": 1}})
    layers = load_preset_layers()
    presets = dict(layers.merged)
    presets["Mine"] = {"m": 1}
    presets["Builtin"] = {"b": 2}  # user override of a built-in preset

    path = save_user_presets(presets, layers.builtin)

    assert path == user_presets_path()
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "Mine": {"m": 1},
        "Builtin": {"b": 2},
    }
    assert json.loads(checkout.read_text(encoding="utf-8")) == {"Builtin": {"b": 1}}


def test_unchanged_builtin_presets_are_not_copied(workdir):
    """Built-in presets equal to their default are not stored in the user file."""
    layers = load_preset_layers()
    path = save_user_presets(dict(layers.merged), layers.builtin)
    assert json.loads(path.read_text(encoding="utf-8")) == {}


def test_invalid_json_is_reported(workdir):
    """A corrupt presets file raises instead of being ignored."""
    (workdir / "presets.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        load_presets()
