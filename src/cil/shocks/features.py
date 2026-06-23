"""Real-time macro features for shock identification.

Builds the Fed's real-time information set from first-release ALFRED vintages
(per the project's PIT policy): year-on-year inflation and industrial-production
growth, the unemployment level, and its monthly change. These features proxy the
Greenbook forecasts in the Romer-Romer orthogonalization and serve as predictors
in the predictability test.
"""

from __future__ import annotations

import polars as pl

from cil.data import alfred


def realtime_macro_features(
    macro_pit: pl.DataFrame,
    *,
    cpi_id: str = "CPIAUCSL",
    ip_id: str = "INDPRO",
    unemployment_id: str = "UNRATE",
) -> pl.DataFrame:
    """Construct real-time macro features keyed by reference month.

    Parameters
    ----------
    macro_pit
        Point-in-time macro frame
        (:data:`cil.data.schemas.MACRO_PIT_SCHEMA`).
    cpi_id, ip_id, unemployment_id
        Series identifiers for the price level, industrial production, and the
        unemployment rate.

    Returns
    -------
    polars.DataFrame
        Columns ``date``, ``inflation`` (YoY %), ``ip_growth`` (YoY %),
        ``unemployment`` (level), ``d_unemployment`` (monthly change).
    """
    first = alfred.first_release_from_pit(macro_pit)
    wide = (
        first.pivot(values="value", index="reference_date", on="series_id")
        .rename({"reference_date": "date"})
        .sort("date")
    )
    return wide.select(
        "date",
        inflation=100.0 * (pl.col(cpi_id).log() - pl.col(cpi_id).log().shift(12)),
        ip_growth=100.0 * (pl.col(ip_id).log() - pl.col(ip_id).log().shift(12)),
        unemployment=pl.col(unemployment_id),
        d_unemployment=pl.col(unemployment_id) - pl.col(unemployment_id).shift(1),
    ).drop_nulls()


def add_lags(
    frame: pl.DataFrame, columns: list[str], n_lags: int, *, on: str = "date"
) -> pl.DataFrame:
    """Append ``n_lags`` monthly lags of *columns* (sorted by *on*).

    Parameters
    ----------
    frame
        Source frame.
    columns
        Columns to lag.
    n_lags
        Number of lags to append.
    on
        Ordering key (a monthly date column).

    Returns
    -------
    polars.DataFrame
        The frame with added ``{col}_l{lag}`` columns.
    """
    out = frame.sort(on)
    lagged = [
        pl.col(col).shift(lag).alias(f"{col}_l{lag}")
        for col in columns
        for lag in range(1, n_lags + 1)
    ]
    return out.with_columns(lagged)
