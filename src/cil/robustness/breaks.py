"""Structural-break detection (Bai-Perron via ``ruptures``).

Detects multiple structural breaks in a univariate series (e.g. national
employment growth, or the aggregate response) using the ``ruptures`` library.
Reported as a diagnostic on the stability of the relationship over the sample.

References
----------
Bai & Perron (2003), *Computation and analysis of multiple structural change
models*, Journal of Applied Econometrics 18(1).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import ruptures as rpt


def detect_breaks(
    values: npt.ArrayLike, *, penalty_scale: float = 1.0, min_size: int = 12
) -> list[int]:
    """Return break indices via PELT with a BIC-style penalty.

    Parameters
    ----------
    values
        Univariate series.
    penalty_scale
        Multiplier on the ``scale * log(n) * variance`` penalty; larger means
        fewer breaks.
    min_size
        Minimum segment length (in observations) between breaks.

    Returns
    -------
    list of int
        Break indices (exclusive of the series endpoint).
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    n = arr.shape[0]
    variance = float(np.var(arr)) or 1.0
    penalty = penalty_scale * np.log(n) * variance
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(arr)
    breaks = algo.predict(pen=penalty)
    return [b for b in breaks if b < n]


def bai_perron(
    frame: pl.DataFrame,
    value_col: str,
    *,
    date_col: str = "date",
    penalty_scale: float = 1.0,
    min_size: int = 12,
) -> pl.DataFrame:
    """Detect structural breaks and return their dates.

    Parameters
    ----------
    frame
        Frame sorted by ``date_col`` with the series in ``value_col``.
    value_col
        Column holding the series.
    date_col
        Column holding the dates.
    penalty_scale, min_size
        Passed to :func:`detect_breaks`.

    Returns
    -------
    polars.DataFrame
        Column ``break_date`` with one row per detected break.
    """
    ordered = frame.sort(date_col).drop_nulls(value_col)
    indices = detect_breaks(
        ordered[value_col].to_numpy(), penalty_scale=penalty_scale, min_size=min_size
    )
    dates = ordered[date_col].to_list()
    return pl.DataFrame({"break_date": [dates[i] for i in indices if i < len(dates)]})
