# tests/test_result_cache.py
# MIT License - Copyright (c) 2025-2026 Luc Prevost
# Tests for the result cache validation helpers.

import numpy as np
import pytest

from topoptcomec import result_cache

NELXYZ = [4, 3, 0]
NEL = 12
NDOF = 2 * 5 * 4


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
