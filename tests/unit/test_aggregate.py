"""Aggregate IRF estimators: time-series LP and LP-IV (positive controls)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.estimators.proxy_svar import LPIVConfig, lp_iv
from cil.estimators.ts_lp import TimeSeriesLPConfig, time_series_lp


def _months(n: int) -> list[dt.date]:
    return [dt.date(1995 + i // 12, i % 12 + 1, 1) for i in range(n)]


def test_time_series_lp_recovers_impact_response() -> None:
    rng = np.random.default_rng(0)
    n = 240
    months = _months(n)
    s = rng.normal(size=n)
    theta = 0.5
    # Level whose one-month change at t is theta*s_t + noise (scale=1 for clarity).
    dy = theta * s + rng.normal(0, 0.1, n)
    y = np.cumsum(dy)
    outcome = pl.DataFrame({"date": months, "value": y})
    shock = pl.DataFrame({"date": months, "shock": s})
    res = time_series_lp(
        outcome,
        shock,
        TimeSeriesLPConfig(horizons=(0, 1), n_lags=3, scale=1.0),
        shock_col="shock",
    )
    theta0 = res.filter(pl.col("horizon") == 0)["theta"][0]
    assert abs(theta0 - theta) < 0.05


def test_lp_iv_recovers_theta_with_strong_instrument() -> None:
    rng = np.random.default_rng(1)
    n = 300
    months = _months(n)
    z = rng.normal(size=n)
    d_policy = 1.0 * z + rng.normal(0, 0.3, n)  # strong first stage
    theta = -0.8
    dy = theta * d_policy + rng.normal(0, 0.2, n)
    y = np.cumsum(dy)
    outcome = pl.DataFrame({"date": months, "value": y})
    policy = pl.DataFrame({"date": months, "policy_rate": np.cumsum(d_policy)})
    instrument = pl.DataFrame({"date": months, "brw_monthly": z})
    res = lp_iv(
        outcome,
        policy,
        instrument,
        # Fine grid: a strong instrument yields a very narrow AR region, so the
        # grid step must be small enough to land inside it.
        LPIVConfig(horizons=(0,), n_lags=3, scale=1.0, ar_grid=4001, ar_span=2.0),
        instrument_col="brw_monthly",
    )
    row = res.to_dicts()[0]
    assert abs(row["theta"] - theta) < 0.1
    assert row["first_stage_f"] > 10  # strong instrument
    # The AR interval is finite and, by construction, brackets the 2SLS point
    # estimate (it need not bracket the true value in a single strong-ID sample,
    # where the interval is very narrow).
    assert row["ar_low"] <= row["theta"] <= row["ar_high"]
