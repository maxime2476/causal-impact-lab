"""Proxy-SVAR aggregate IRF by local-projection IV (LP-IV).

Estimates the impulse response of national log employment to a unit increase in
the policy rate, instrumenting the policy-rate change with a borrowed
high-frequency-identified shock (BRW). Per horizon this is a 2SLS local
projection. Because the first stage is weak (see ADR-0004), the point estimates
are reported alongside the first-stage F and an Anderson-Rubin weak-instrument-
robust confidence interval, which remains valid under weak identification.

References
----------
Stock & Watson (2018), Economic Journal 128; Jorda, Schularick & Taylor (2015),
LP-IV; Montiel Olea & Pflueger (2013), JBES; Anderson & Rubin (1949).
"""

from __future__ import annotations

import numpy as np
import polars as pl
import statsmodels.api as sm
from linearmodels.iv import IV2SLS
from pydantic import BaseModel
from scipy.stats import chi2, norm


class LPIVConfig(BaseModel):
    """Configuration for the LP-IV aggregate IRF.

    Parameters
    ----------
    horizons
        Non-negative response horizons.
    n_lags
        Lags of the outcome change and policy change used as controls.
    confidence_level
        Two-sided confidence level.
    scale
        Multiplier on log employment (``100`` -> percent).
    ar_grid
        Number of grid points for the Anderson-Rubin interval search.
    ar_span
        Half-width (in IRF units) of the AR grid around zero.
    """

    horizons: tuple[int, ...]
    n_lags: int = 6
    confidence_level: float = 0.95
    scale: float = 100.0
    ar_grid: int = 401
    ar_span: float = 20.0


def _prepare(
    outcome: pl.DataFrame,
    policy_rate: pl.DataFrame,
    instrument: pl.DataFrame,
    instrument_col: str,
    n_lags: int,
    scale: float,
) -> pl.DataFrame:
    """Join outcome, policy-rate change, and instrument with lagged controls."""
    pol = policy_rate.select(
        "date", d_policy=pl.col("policy_rate") - pl.col("policy_rate").shift(1)
    )
    base = (
        outcome.select("date", y=pl.col("value") * scale)
        .join(pol, on="date", how="inner")
        .join(instrument.select("date", instrument_col), on="date", how="inner")
        .sort("date")
    )
    lag_exprs = []
    for lag in range(1, n_lags + 1):
        lag_exprs.append(
            (pl.col("y").shift(lag) - pl.col("y").shift(lag + 1)).alias(f"dy_l{lag}")
        )
        lag_exprs.append(pl.col("d_policy").shift(lag).alias(f"dp_l{lag}"))
    return base.with_columns(lag_exprs)


def _ar_interval(
    target: np.ndarray,
    endog: np.ndarray,
    instrument: np.ndarray,
    exog: np.ndarray,
    *,
    confidence_level: float,
    grid: int,
    span: float,
) -> tuple[float, float]:
    """Anderson-Rubin weak-IV-robust interval for the IRF coefficient.

    For each candidate ``theta0`` the residual ``target - theta0*endog`` is
    regressed on the instrument and exogenous controls; ``theta0`` is in the
    interval if the instrument is jointly insignificant (AR statistic below the
    chi-squared critical value).
    """
    crit = float(chi2.ppf(confidence_level, df=1))
    grid_vals = np.linspace(-span, span, grid)
    accepted: list[float] = []
    x_full = np.column_stack([instrument, exog])
    for theta0 in grid_vals:
        resid = target - theta0 * endog
        fit = sm.OLS(resid, x_full).fit(cov_type="HAC", cov_kwds={"maxlags": 6})
        # AR stat: squared robust t on the instrument (df=1).
        ar_stat = float(fit.tvalues[0]) ** 2
        if ar_stat <= crit:
            accepted.append(float(theta0))
    if not accepted:
        return float("nan"), float("nan")
    return min(accepted), max(accepted)


def lp_iv(
    outcome: pl.DataFrame,
    policy_rate: pl.DataFrame,
    instrument: pl.DataFrame,
    config: LPIVConfig,
    *,
    instrument_col: str,
) -> pl.DataFrame:
    """Estimate the LP-IV aggregate IRF with first-stage F and AR intervals.

    Parameters
    ----------
    outcome
        National outcome frame (``date``, ``value``).
    policy_rate
        Policy-rate frame (``date``, ``policy_rate``).
    instrument
        Instrument frame (``date`` and the instrument column).
    config
        Estimation configuration.
    instrument_col
        Name of the instrument column.

    Returns
    -------
    polars.DataFrame
        Per horizon: ``theta`` (rate IRF), ``se``, ``ci_low``, ``ci_high``,
        ``first_stage_f``, ``ar_low``, ``ar_high``, ``n_obs``.
    """
    controls = [f"dy_l{lag}" for lag in range(1, config.n_lags + 1)]
    controls += [f"dp_l{lag}" for lag in range(1, config.n_lags + 1)]
    prepared = _prepare(
        outcome, policy_rate, instrument, instrument_col, config.n_lags, config.scale
    )
    z_crit = float(norm.ppf(0.5 + config.confidence_level / 2.0))
    rows: list[dict[str, float]] = []
    for h in config.horizons:
        frame = (
            prepared.with_columns(target=(pl.col("y").shift(-h) - pl.col("y").shift(1)))
            .select(["target", "d_policy", instrument_col, *controls])
            .drop_nulls()
        )
        pdf = frame.to_pandas()
        exog = sm.add_constant(pdf[controls])
        result = IV2SLS(
            pdf["target"], exog, pdf[["d_policy"]], pdf[[instrument_col]]
        ).fit(cov_type="kernel", kernel="bartlett")
        theta = float(result.params["d_policy"])
        se = float(result.std_errors["d_policy"])
        fs_f = float(
            result.first_stage.diagnostics.loc["d_policy", "f.stat"]  # type: ignore[attr-defined]
        )
        ar_low, ar_high = _ar_interval(
            pdf["target"].to_numpy(),
            pdf["d_policy"].to_numpy(),
            pdf[instrument_col].to_numpy(),
            exog.to_numpy(),
            confidence_level=config.confidence_level,
            grid=config.ar_grid,
            span=config.ar_span,
        )
        rows.append(
            {
                "horizon": float(h),
                "theta": theta,
                "se": se,
                "ci_low": theta - z_crit * se,
                "ci_high": theta + z_crit * se,
                "first_stage_f": fs_f,
                "ar_low": ar_low,
                "ar_high": ar_high,
                "n_obs": float(result.nobs),
            }
        )
    return pl.DataFrame(rows).sort("horizon")
