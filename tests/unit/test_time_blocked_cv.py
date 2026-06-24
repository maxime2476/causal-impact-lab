"""Leakage tests for the purged time-blocked cross-validation splitter.

These are correctness tests, not coverage: a failure means the DML cross-fitting
could leak serial correlation across folds.
"""

from __future__ import annotations

import numpy as np
import pytest

from cil.estimators.time_blocked_cv import PurgedTimeBlockedCV


def _panel_time_codes(n_units: int = 8, n_periods: int = 24) -> np.ndarray:
    # Rows ordered (unit, period); period code repeats across units.
    return np.array([t for _ in range(n_units) for t in range(n_periods)])


def test_no_row_in_both_train_and_test() -> None:
    codes = _panel_time_codes()
    cv = PurgedTimeBlockedCV(codes, n_splits=4, embargo=2)
    for train_idx, test_idx in cv.split(np.zeros((codes.size, 1))):
        assert set(train_idx).isdisjoint(set(test_idx))


def test_embargo_purges_neighbouring_periods() -> None:
    codes = _panel_time_codes()
    embargo = 3
    cv = PurgedTimeBlockedCV(codes, n_splits=4, embargo=embargo)
    for train_idx, test_idx in cv.split(np.zeros((codes.size, 1))):
        train_periods = set(codes[train_idx].tolist())
        test_periods = set(codes[test_idx].tolist())
        # Every train period is strictly more than `embargo` away from every
        # test period (no adjacent-within-embargo leakage).
        min_gap = min(abs(tr - te) for tr in train_periods for te in test_periods)
        assert min_gap > embargo


def test_blocks_are_contiguous_and_cover_all_periods() -> None:
    codes = _panel_time_codes()
    cv = PurgedTimeBlockedCV(codes, n_splits=4, embargo=0)
    covered: set[int] = set()
    for _, test_idx in cv.split(np.zeros((codes.size, 1))):
        test_periods = sorted(set(codes[test_idx].tolist()))
        # Each test block is a contiguous run of periods.
        assert test_periods == list(range(test_periods[0], test_periods[-1] + 1))
        covered |= set(test_periods)
    assert covered == set(range(24))  # every period tested exactly once


def test_get_n_splits() -> None:
    cv = PurgedTimeBlockedCV(_panel_time_codes(), n_splits=5, embargo=1)
    assert cv.get_n_splits() == 5


def test_length_mismatch_raises() -> None:
    cv = PurgedTimeBlockedCV(_panel_time_codes(), n_splits=3, embargo=1)
    with pytest.raises(ValueError, match="length"):
        next(cv.split(np.zeros((5, 1))))


def test_invalid_config_raises() -> None:
    codes = _panel_time_codes()
    with pytest.raises(ValueError, match="n_splits"):
        PurgedTimeBlockedCV(codes, n_splits=1)
    with pytest.raises(ValueError, match="embargo"):
        PurgedTimeBlockedCV(codes, n_splits=3, embargo=-1)
    with pytest.raises(ValueError, match="distinct periods"):
        PurgedTimeBlockedCV(np.array([0, 0, 1]), n_splits=5)
