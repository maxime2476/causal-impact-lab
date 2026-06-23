"""Aggregate time-series local projection (Jorda 2005).

Estimates the impulse response of national log employment to an identified
monetary shock:

    Y_{t+h} - Y_{t-1} = alpha_h + theta_h s_t + sum_l psi_{h,l} W_{t-l} + u_{t+h}

with Newey-West (HAC) standard errors. ``{theta_h}`` is the aggregate IRF. This
is the assumption-dependent *complement* to the headline relative effect: it
requires the shock to be exogenous to the aggregate state, an assumption stated
and stress-tested (see ``docs/results.md``), not the cleanly-identified relative
design.

References
----------
Jorda (2005), AER 95(1).
"""

from __future__ import annotations

import polars as pl
import statsmodels.api as sm
from pydantic import BaseModel
from scipy.stats import norm


class TimeSeriesLPConfig(BaseModel):
    """Configuration for the aggregate time-series LP.

    Parameters
    ----------
    horizons
        Non-negative response horizons.
    n_lags
        Number of lags of the outcome change and the shock used as controls.
    confidence_level
        Two-sided confidence level for intervals.
    scale
        Multiplier applied to log employment (``100`` gives percent responses).
    """

    horizons: tuple[int, ...]
    n_lags: int = 6
    confidence_level: float = 0.95
    scale: float = 100.0


def _prepare(
    outcome: pl.DataFrame,
    shock: pl.DataFrame,
    shock_col: str,
    n_lags: int,
    scale: float,
) -> pl.DataFrame:
    """Join outcome and shock, scale, and build lagged controls."""
    base = (
        outcome.select("date", y=pl.col("value") * scale)
        .join(shock.select("date", shock_col), on="date", how="inner")
        .sort("date")
    )
    lag_exprs = []
    for lag in range(1, n_lags + 1):
        lag_exprs.append(
            (pl.col("y").shift(lag) - pl.col("y").shift(lag + 1)).alias(f"dy_l{lag}")
        )
        lag_exprs.append(pl.col(shock_col).shift(lag).alias(f"s_l{lag}"))
    return base.with_columns(lag_exprs)


def time_series_lp(
    outcome: pl.DataFrame,
    shock: pl.DataFrame,
    config: TimeSeriesLPConfig,
    *,
    shock_col: str,
) -> pl.DataFrame:
    """Estimate the aggregate IRF of *outcome* to the shock across horizons.

    Parameters
    ----------
    outcome
        National outcome frame with ``date`` and ``value`` (e.g. log employment).
    shock
        Shock frame with ``date`` and the shock column.
    config
        Estimation configuration.
    shock_col
        Name of the shock column.

    Returns
    -------
    polars.DataFrame
        One row per horizon: ``theta`` (IRF), ``se``, ``ci_low``, ``ci_high``,
        ``p_value``, ``n_obs``.
    """
    controls = [f"dy_l{lag}" for lag in range(1, config.n_lags + 1)]
    controls += [f"s_l{lag}" for lag in range(1, config.n_lags + 1)]
    prepared = _prepare(outcome, shock, shock_col, config.n_lags, config.scale)
    z = float(norm.ppf(0.5 + config.confidence_level / 2.0))
    rows: list[dict[str, float]] = []
    for h in config.horizons:
        frame = (
            prepared.with_columns(target=(pl.col("y").shift(-h) - pl.col("y").shift(1)))
            .select(["target", shock_col, *controls])
            .drop_nulls()
        )
        pdf = frame.to_pandas()
        x = sm.add_constant(pdf[[shock_col, *controls]])
        fit = sm.OLS(pdf["target"], x).fit(cov_type="HAC", cov_kwds={"maxlags": h + 1})
        theta = float(fit.params[shock_col])
        se = float(fit.bse[shock_col])
        rows.append(
            {
                "horizon": float(h),
                "theta": theta,
                "se": se,
                "ci_low": theta - z * se,
                "ci_high": theta + z * se,
                "p_value": float(fit.pvalues[shock_col]),
                "n_obs": float(fit.nobs),
            }
        )
    return pl.DataFrame(rows).sort("horizon")
