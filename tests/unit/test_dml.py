"""DML heterogeneity positive control (synthetic, no network).

Recovers an injected average effect with both LinearDML and CausalForestDML under
purged time-blocked cross-fitting, and confirms the placebo (time-permuted
treatment) collapses toward zero. The no-leakage property of the splitter itself
is tested in ``test_time_blocked_cv.py``.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.estimators.dml import build_sample, estimate_heterogeneity


def _panel() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, float]:
    rng = np.random.default_rng(0)
    states = [f"{s:02d}" for s in range(1, 12)]
    sectors = [f"10{k}" for k in range(11, 22)]
    months = [dt.date(2015 + i // 12, i % 12 + 1, 1) for i in range(30)]
    expo = dict(zip(sectors, np.linspace(-1.5, 1.5, 11), strict=True))
    s_t = {m: rng.normal() for m in months}
    beta = -0.8
    rows = []
    for st in states:
        for k in sectors:
            level = 0.0
            for m in months:
                level += beta * expo[k] * s_t[m] + rng.normal(0, 0.2)
                rows.append((f"{st}_{k}", st, k, m, level))
    panel = pl.DataFrame(
        rows,
        schema=["unit_id", "state_fips", "supersector_code", "date", "log_employment"],
        orient="row",
    )
    exposure = pl.DataFrame(
        {"supersector_code": list(expo), "exposure": list(expo.values())}
    )
    shock = pl.DataFrame({"date": months, "shock": [s_t[m] for m in months]})
    return panel, exposure, shock, beta


def test_build_sample_shapes_align() -> None:
    panel, exposure, shock, _ = _panel()
    y, t, x, w, codes = build_sample(
        panel, exposure, shock, shock_col="shock", horizon=0, n_lags=2
    )
    assert y.shape[0] == t.shape[0] == x.shape[0] == w.shape[0] == codes.shape[0]
    assert x.shape[1] == 1
    assert w.shape[1] == 2


def test_dml_recovers_ate_and_placebo_collapses() -> None:
    panel, exposure, shock, beta = _panel()
    res = estimate_heterogeneity(
        panel,
        exposure,
        shock,
        shock_col="shock",
        horizon=0,
        n_lags=2,
        n_splits=3,
        embargo=2,
    )
    assert abs(res.linear_ate - beta) < 0.1
    assert abs(res.forest_ate - beta) < 0.15
    assert abs(res.placebo_ate) < 0.15  # placebo collapses toward zero
    assert res.linear_ate_ci_low <= res.linear_ate <= res.linear_ate_ci_high
