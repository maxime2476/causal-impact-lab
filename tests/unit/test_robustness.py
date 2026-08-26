"""Robustness suite: spec curve, placebo, breaks, COVID, QCEW bound.

Synthetic data (no network). Placebo and spec-curve tests use a DGP with a known
relative effect; the break test uses a series with a known mean shift.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.robustness import breaks, covid, placebo, qcew_revision
from cil.robustness import specification_curve as sc


def _panel(
    beta: float = -0.8, n_states: int = 12
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    rng = np.random.default_rng(0)
    states = [f"{s:02d}" for s in range(1, n_states + 1)]
    sectors = [f"10{k}" for k in range(11, 22)]
    months = [dt.date(2015 + i // 12, i % 12 + 1, 1) for i in range(48)]
    expo = dict(zip(sectors, np.linspace(-1.5, 1.5, 11), strict=True))
    s_t = {m: rng.normal() for m in months}
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
    return panel, exposure, shock


def test_specification_curve_grid_and_bh() -> None:
    panel, exposure, shock = _panel()
    curve = sc.run_specification_curve(
        panel,
        {"estimated": exposure},
        {"brw": (shock, "shock")},
        horizons=(0,),
        control_lags=(3, 6),
    )
    assert curve.height == 2  # 1 shock x 1 exposure x 2 lags x 1 sample x 1 horizon
    assert "p_value_bh" in curve.columns
    summary = sc.summarize_curve(curve)
    assert summary["share_negative"][0] == 1.0  # injected effect is negative


def test_placebo_collapses() -> None:
    panel, exposure, shock = _panel()
    res = placebo.permutation_test(
        panel,
        exposure,
        shock,
        shock_col="shock",
        horizon=0,
        mode="shock",
        n_permutations=25,
        seed=0,
    )
    assert abs(res.placebo_mean) < abs(res.actual_beta)  # placebo near zero
    assert 0.0 < res.placebo_p_value <= 1.0


def test_circular_shift_ri_detects_effect_and_preserves_autocov() -> None:
    panel, exposure, shock = _panel(beta=-0.8)
    # Circular shift preserves the shock's autocovariance (same multiset of values).
    vals = shock.sort("date")["shock"].to_numpy()
    rolled = np.roll(vals, 7)
    assert np.isclose(np.var(vals), np.var(rolled))
    assert np.isclose(sorted(vals), sorted(rolled)).all()

    frame, joint_p = placebo.circular_shift_ri(
        panel, exposure, shock, shock_col="shock", horizons=(0, 12), n_draws=50, seed=0
    )
    assert set(frame["horizon"].to_list()) == {0, 12}
    assert (frame["ri_p_value"] > 0).all() and (frame["ri_p_value"] <= 1).all()
    assert 0.0 < joint_p <= 1.0
    # The strong injected effect at h=0 is detected (small RI p-value).
    assert float(frame.filter(pl.col("horizon") == 0)["ri_p_value"][0]) < 0.1


def test_breaks_detects_mean_shift() -> None:
    rng = np.random.default_rng(0)
    series = np.concatenate([rng.normal(0, 0.3, 60), rng.normal(3.0, 0.3, 60)])
    idx = breaks.detect_breaks(series, min_size=12)
    assert any(abs(b - 60) <= 10 for b in idx)  # break near the true shift


def test_bai_perron_full_selects_one_break_with_ci() -> None:
    rng = np.random.default_rng(1)
    series = np.concatenate([rng.normal(0, 0.3, 60), rng.normal(3.0, 0.3, 60)])
    months = [dt.date(2000 + i // 12, i % 12 + 1, 1) for i in range(120)]
    frame = pl.DataFrame({"date": months, "growth": series})
    breaks_df, selection = breaks.bai_perron_full(
        frame, "growth", max_breaks=4, min_size=12, n_boot=100, seed=0
    )
    # BIC selects exactly one break, near the true shift, with a bracketing CI.
    assert int(selection.filter(pl.col("selected"))["n_breaks"][0]) == 1
    assert breaks_df.height == 1
    bd = breaks_df["break_date"][0]
    assert dt.date(2004, 7, 1) <= bd <= dt.date(2005, 7, 1)
    assert breaks_df["ci_low_date"][0] <= bd <= breaks_df["ci_high_date"][0]
    assert abs(float(breaks_df["delta"][0]) - 3.0) < 0.5


def test_covid_exclude_and_state_dependent() -> None:
    months = [dt.date(2019 + i // 12, i % 12 + 1, 1) for i in range(24)]
    frame = pl.DataFrame({"date": months, "x": range(24)})
    excluded = covid.exclude_covid(frame)
    assert all(
        not (covid.COVID_START <= d <= covid.COVID_END) for d in excluded["date"]
    )

    rng = np.random.default_rng(0)
    outcome = pl.DataFrame({"date": months, "value": np.cumsum(rng.normal(0, 0.1, 24))})
    shock = pl.DataFrame({"date": months, "shock": rng.normal(size=24)})
    rec = pl.DataFrame({"date": months, "recession": [0] * 12 + [1] * 12})
    sd = covid.state_dependent_lp(
        outcome, shock, rec, shock_col="shock", horizons=(0, 1)
    )
    assert {"theta_expansion", "theta_recession"}.issubset(sd.columns)


def test_qcew_revision_bound_contains_actual() -> None:
    panel, exposure, shock = _panel()
    bound = qcew_revision.revision_bound(
        panel,
        exposure,
        shock,
        shock_col="shock",
        horizon=0,
        revision_sd=0.003,
        n_draws=8,
        seed=0,
    )
    assert bound.beta_min <= bound.actual_beta <= bound.beta_max
    assert bound.beta_max >= bound.beta_min


def test_correlated_revision_bound_contains_actual() -> None:
    panel, exposure, shock = _panel()
    bound = qcew_revision.correlated_revision_bound(
        panel,
        exposure,
        shock,
        shock_col="shock",
        horizon=0,
        sigma_bench=0.01,
        sigma_idio=0.002,
        n_draws=8,
        seed=0,
    )
    assert bound.beta_min <= bound.actual_beta <= bound.beta_max
    assert bound.beta_sd >= 0.0
    assert bound.sigma_bench == 0.01


def test_growth_discrepancy_sd_positive() -> None:
    rng = np.random.default_rng(0)
    months = [dt.date(2015 + i // 12, i % 12 + 1, 1) for i in range(30)]
    rows_q, rows_c = [], []
    for k in ("100", "200"):
        base = 100.0
        for m in months:
            base *= 1.0 + rng.normal(0.0, 0.01)
            rows_q.append(("01", k, m, base))
            # CES differs from QCEW by a multiplicative growth wedge.
            rows_c.append(("01", k, m, base * (1.0 + rng.normal(0.0, 0.02))))
    schema = ["state_fips", "supersector_code", "date", "employment"]
    qcew = pl.DataFrame(rows_q, schema=schema, orient="row")
    ces = pl.DataFrame(rows_c, schema=schema, orient="row")
    sd = qcew_revision.growth_discrepancy_sd(qcew, ces)
    assert sd > 0.0 and np.isfinite(sd)
