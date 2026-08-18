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

from cil.estimators.dml import (
    build_driver_sample,
    build_sample,
    estimate_cate_drivers,
    estimate_heterogeneity,
)


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


def _panel_with_employment() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(1)
    states = [f"{s:02d}" for s in range(1, 12)]
    sectors = [f"10{k}" for k in range(11, 22)]
    months = [dt.date(2015 + i // 12, i % 12 + 1, 1) for i in range(30)]
    expo = dict(zip(sectors, np.linspace(-1.5, 1.5, 11), strict=True))
    s_t = {m: rng.normal() for m in months}
    # Marginal effect of the treatment T = exposure * shock is (a + b*exposure),
    # so the CATE genuinely varies with exposure (and nothing else).
    a, b = -0.8, -0.5
    rows = []
    for st in states:
        for k in sectors:
            level = 0.0
            base = 10.0 + rng.normal(0, 1.0)  # varies cell size / share
            for m in months:
                level += (a * expo[k] + b * expo[k] ** 2) * s_t[m] + rng.normal(0, 0.1)
                rows.append((f"{st}_{k}", st, k, m, float(np.exp(base)), level))
    panel = pl.DataFrame(
        rows,
        schema=[
            "unit_id",
            "state_fips",
            "supersector_code",
            "date",
            "employment",
            "log_employment",
        ],
        orient="row",
    )
    exposure = pl.DataFrame(
        {"supersector_code": list(expo), "exposure": list(expo.values())}
    )
    shock = pl.DataFrame({"date": months, "shock": [s_t[m] for m in months]})
    return panel, exposure, shock


def test_driver_sample_has_multiple_features() -> None:
    panel, exposure, shock = _panel_with_employment()
    y, _t, x, _w, codes, features = build_driver_sample(
        panel, exposure, shock, shock_col="shock", horizon=0, n_lags=2
    )
    assert features == ["exposure", "base_share", "log_base_emp"]
    assert x.shape[1] == 3
    assert y.shape[0] == x.shape[0] == codes.shape[0]


def test_cate_drivers_flag_exposure_as_top() -> None:
    panel, exposure, shock = _panel_with_employment()
    res = estimate_cate_drivers(
        panel,
        exposure,
        shock,
        shock_col="shock",
        horizon=0,
        n_lags=2,
        n_splits=3,
        embargo=2,
    )
    assert res.features == ["exposure", "base_share", "log_base_emp"]
    assert abs(sum(res.importances) - 1.0) < 1e-6
    # The DGP's heterogeneity is driven only by exposure.
    top = max(range(len(res.importances)), key=lambda i: res.importances[i])
    assert res.features[top] == "exposure"
