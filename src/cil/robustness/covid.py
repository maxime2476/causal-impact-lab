"""COVID handling and state-dependent local projections.

The baseline excludes the acute COVID window (Mar-Dec 2020); robustness keeps it.
A state-dependent aggregate LP (Auerbach-Gorodnichenko) lets the impulse response
differ between expansion and recession states, so the 2020 collapse and other
downturns can be separated from the typical response.

References
----------
Auerbach & Gorodnichenko (2012), state-dependent fiscal multipliers,
AEJ:Economic Policy 4(2).
"""

from __future__ import annotations

import datetime as dt

import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel

COVID_START = dt.date(2020, 3, 1)
COVID_END = dt.date(2020, 12, 1)


def exclude_covid(
    frame: pl.DataFrame,
    *,
    start: dt.date = COVID_START,
    end: dt.date = COVID_END,
    date_col: str = "date",
) -> pl.DataFrame:
    """Drop the acute COVID window from a date-indexed frame."""
    return frame.filter((pl.col(date_col) < start) | (pl.col(date_col) > end))


class StateDependentResult(BaseModel):
    """State-dependent aggregate response at one horizon.

    Parameters
    ----------
    horizon
        Response horizon.
    theta_expansion
        Response in the expansion state.
    theta_recession
        Response in the recession state.
    """

    horizon: int
    theta_expansion: float
    theta_recession: float


def state_dependent_lp(
    outcome: pl.DataFrame,
    shock: pl.DataFrame,
    recession: pl.DataFrame,
    *,
    shock_col: str,
    horizons: tuple[int, ...],
    scale: float = 100.0,
) -> pl.DataFrame:
    """Estimate expansion- and recession-state aggregate IRFs.

    Parameters
    ----------
    outcome
        National outcome (``date``, ``value``); the value is scaled and logged
        differences are taken internally.
    shock
        Shock frame (``date``, shock column).
    recession
        Frame (``date``, ``recession``) with a 0/1 recession indicator.
    shock_col
        Name of the shock column.
    horizons
        Response horizons.
    scale
        Multiplier on the outcome (``100`` -> percent).

    Returns
    -------
    polars.DataFrame
        One row per horizon with ``theta_expansion`` and ``theta_recession``.
    """
    base = (
        outcome.select("date", y=pl.col("value") * scale)
        .join(shock.select("date", shock_col), on="date", how="inner")
        .join(recession.select("date", "recession"), on="date", how="inner")
        .sort("date")
        .with_columns(
            s_recession=pl.col(shock_col) * pl.col("recession"),
            s_expansion=pl.col(shock_col) * (1 - pl.col("recession")),
        )
    )
    rows: list[dict[str, float]] = []
    for h in horizons:
        frame = (
            base.with_columns(target=(pl.col("y").shift(-h) - pl.col("y").shift(1)))
            .select(["target", "s_expansion", "s_recession"])
            .drop_nulls()
        )
        pdf = frame.to_pandas()
        x = sm.add_constant(pdf[["s_expansion", "s_recession"]])
        fit = sm.OLS(pdf["target"], x).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        rows.append(
            {
                "horizon": float(h),
                "theta_expansion": float(fit.params["s_expansion"]),
                "theta_recession": float(fit.params["s_recession"]),
            }
        )
    return pl.DataFrame(rows).sort("horizon")


def unemployment_recession_indicator(
    unemployment: pl.DataFrame, *, window: int = 3
) -> pl.DataFrame:
    """Build a recession indicator: unemployment rising over *window* months.

    Parameters
    ----------
    unemployment
        Frame (``date``, ``value``) of the unemployment rate.
    window
        Look-back window for the change.

    Returns
    -------
    polars.DataFrame
        Columns ``date`` and ``recession`` (1 if unemployment rose).
    """
    return (
        unemployment.select("date", value="value")
        .sort("date")
        .with_columns(
            recession=((pl.col("value") - pl.col("value").shift(window)) > 0).cast(
                pl.Int64
            )
        )
        .select("date", "recession")
        .drop_nulls()
    )
