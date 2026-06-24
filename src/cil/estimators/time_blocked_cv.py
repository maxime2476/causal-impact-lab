"""Purged, embargoed, time-blocked cross-validation for DML cross-fitting.

Double/debiased ML cross-fitting must respect temporal dependence: a random
K-fold that places adjacent months in train and test leaks serial correlation
(and overlapping local-projection windows) across the split, biasing the
nuisance estimates. This splitter partitions the sample into contiguous time
blocks and, for each test block, *purges* train periods that fall within an
embargo distance of the test block. A random K-fold here is a correctness bug;
this is its replacement.

References
----------
Lopez de Prado (2018), *Advances in Financial Machine Learning* (purged CV);
Chernozhukov et al. (2018), cross-fitting.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import numpy.typing as npt

IntArray = npt.NDArray[np.int_]


class PurgedTimeBlockedCV:
    """Contiguous time-block CV with an embargo around each test block.

    Parameters
    ----------
    time_codes
        Integer period code for each observation, aligned to the rows of the
        data passed to :meth:`split` (e.g. a month index). Rows sharing a code
        always stay together.
    n_splits
        Number of contiguous time blocks.
    embargo
        Number of periods on each side of a test block removed from the training
        set, measured in distinct-period steps.
    """

    def __init__(
        self, time_codes: npt.ArrayLike, n_splits: int = 5, embargo: int = 1
    ) -> None:
        self._time_codes = np.asarray(time_codes)
        if n_splits < 2:
            msg = "n_splits must be at least 2."
            raise ValueError(msg)
        if embargo < 0:
            msg = "embargo must be non-negative."
            raise ValueError(msg)
        self.n_splits = n_splits
        self.embargo = embargo
        self._periods = np.unique(self._time_codes)
        if self._periods.size < n_splits:
            msg = "Fewer distinct periods than n_splits."
            raise ValueError(msg)

    def get_n_splits(
        self,
        x: object = None,
        y: object = None,
        groups: object = None,
    ) -> int:
        """Return the number of splits (sklearn-compatible signature)."""
        return self.n_splits

    def split(
        self,
        x: npt.ArrayLike | None = None,
        y: object = None,
        groups: object = None,
    ) -> Iterator[tuple[IntArray, IntArray]]:
        """Yield ``(train_idx, test_idx)`` for each contiguous time block.

        Parameters
        ----------
        x
            Ignored except for a length check against ``time_codes``.
        y, groups
            Ignored; present for sklearn compatibility.

        Yields
        ------
        train_idx, test_idx : numpy.ndarray
            Row indices for the training and test folds, with embargo periods
            purged from the training fold.
        """
        if x is not None and len(np.asarray(x)) != self._time_codes.size:
            msg = "x length does not match time_codes."
            raise ValueError(msg)
        period_blocks = np.array_split(self._periods, self.n_splits)
        period_rank = {p: i for i, p in enumerate(self._periods.tolist())}
        all_idx = np.arange(self._time_codes.size)
        for block in period_blocks:
            test_periods = set(block.tolist())
            test_ranks = [period_rank[p] for p in test_periods]
            lo, hi = min(test_ranks) - self.embargo, max(test_ranks) + self.embargo
            embargoed = {
                self._periods[r] for r in range(self._periods.size) if lo <= r <= hi
            }
            in_test = np.isin(self._time_codes, list(test_periods))
            in_excluded = np.isin(self._time_codes, list(embargoed))
            test_idx = all_idx[in_test]
            train_idx = all_idx[~in_excluded]
            if test_idx.size and train_idx.size:
                yield train_idx, test_idx
