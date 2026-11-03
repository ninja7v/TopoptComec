# tests/test_result_cache.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the result cache validation helpers.

import numpy as np
import pytest

from topoptcomec import result_cache
from topoptcomec.preset_normalization import normalized_params

NELXYZ = [4, 3, 0]
NEL = 12
NDOF = 2 * 5 * 4


def _preset() -> dict:
    return {
        "Dimensions": {"nelxyz": [4, 3, 0]},
        "Forces": {
            "fix": [1],
            "fiy": [0],
            "fidir": ["Y:\u2191"],
            "finorm": [0.1],
            "fox": [2],
            "foy": [1],
            "fodir": ["Y:\u2193"],
            "fonorm": [0.1],
        },
        "Materials": {"E": [1.0], "nu": [0.3], "color": ["#ABCDEF"]},
        "Displacement": {"disp_iterations": 2},
    }


def _cache(x_phys, u=None) -> dict:
    return {"xPhys": np.asarray(x_phys), "u": np.zeros(NDOF) if u is None else u}


def test_has_structure():
    """Empty, near-void and non-finite fields are not usable results."""
    assert result_cache.has_structure(np.full(NEL, 0.3))
    assert result_cache.has_structure(np.full((2, NEL), 0.2))
    assert not result_cache.has_structure(None)
    assert not result_cache.has_structure(np.array([]))
    assert not result_cache.has_structure(np.zeros(NEL))
    assert not result_cache.has_structure(np.full(NEL, np.nan))


def test_cache_error_accepts_a_valid_cache():
    assert result_cache.cache_error(_cache(np.full(NEL, 0.3)), NELXYZ) is None
    assert result_cache.cache_error(_cache(np.full((2, NEL), 0.3)), NELXYZ) is None


@pytest.mark.parametrize(
    "data, message",
    [
        (_cache(np.full(NEL + 1, 0.3)), "does not match the grid"),
        (_cache(np.full(NEL, 0.3), np.zeros(NDOF + 2)), "does not match the grid"),
        (_cache(np.full(NEL, 0.3), np.array(0.0)), "does not match the grid"),
        (_cache(np.full(NEL, np.nan)), "NaN or infinite"),
        (_cache(np.full(NEL, 0.3), np.full(NDOF, np.inf)), "NaN or infinite"),
        (_cache(np.full(NEL, -1.0)), "between 0 and 1"),
        (_cache(np.full(NEL, 1e300)), "between 0 and 1"),
    ],
)
def test_cache_error_rejects_invalid_caches(data, message):
    """Wrong shapes, non-finite and out-of-range fields are rejected."""
    assert message in result_cache.cache_error(data, NELXYZ)


def test_normalization_is_idempotent():
    """Normalizing an already-normalized 2D preset changes nothing."""
    once = normalized_params(_preset())
    twice = normalized_params(once)
    assert once == twice


def test_params_hash_normalizes_like_the_gui():
    """Raw and normalized dictionaries hash identically."""
    preset = _preset()
    normalized = normalized_params(preset)
    assert result_cache.params_hash(preset) == result_cache.params_hash(normalized)


def test_params_hash_ignores_display_only_changes():
    """Color and displacement settings do not invalidate a cached result."""
    preset = _preset()
    base = result_cache.params_hash(preset)
    recolored = _preset()
    recolored["Materials"]["color"] = ["#000000"]
    changed_display = _preset()
    changed_display["Displacement"]["disp_iterations"] = 99
    assert result_cache.params_hash(recolored) == base
    assert result_cache.params_hash(changed_display) == base


def test_params_hash_tracks_optimization_changes():
    """A change that affects the result changes the hash."""
    preset = _preset()
    changed = _preset()
    changed["Forces"]["finorm"] = [0.2]
    assert result_cache.params_hash(changed) != result_cache.params_hash(preset)
