"""Cross-correlation / triangulation of the monetary shock series.

Aligns the in-house Romer-Romer, proxy-SVAR, and benchmark BRW series on common
months and reports pairwise contemporaneous correlations. Agreement (or
documented disagreement) across the three is part of the results.
"""

from __future__ import annotations

import itertools

import polars as pl
from pydantic import BaseModel


class PairwiseCorrelation(BaseModel):
    """Contemporaneous correlation between two shock series.

    Parameters
    ----------
    series_a, series_b
        The two series names.
    correlation
        Pearson correlation on their common months.
    n_obs
        Number of common months.
    """

    series_a: str
    series_b: str
    correlation: float
    n_obs: int


def align(series: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Inner-join named shock series on ``date`` into one wide frame.

    Parameters
    ----------
    series
        Mapping of series name to a frame with ``date`` and one value column
        named exactly as the key.

    Returns
    -------
    polars.DataFrame
        Wide frame with ``date`` and one column per series, common months only.
    """
    frames = [df.select("date", name) for name, df in series.items()]
    out = frames[0]
    for frame in frames[1:]:
        out = out.join(frame, on="date", how="inner")
    return out.drop_nulls().sort("date")


def cross_correlations(series: dict[str, pl.DataFrame]) -> list[PairwiseCorrelation]:
    """Compute pairwise contemporaneous correlations among the series.

    Parameters
    ----------
    series
        Mapping of series name to a frame with ``date`` and a value column named
        as the key.

    Returns
    -------
    list of PairwiseCorrelation
        One entry per unordered pair.
    """
    wide = align(series)
    results: list[PairwiseCorrelation] = []
    for name_a, name_b in itertools.combinations(series, 2):
        corr = wide.select(pl.corr(name_a, name_b)).item()
        results.append(
            PairwiseCorrelation(
                series_a=name_a,
                series_b=name_b,
                correlation=float(corr) if corr is not None else float("nan"),
                n_obs=wide.height,
            )
        )
    return results
