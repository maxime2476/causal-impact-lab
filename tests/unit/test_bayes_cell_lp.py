"""Cell-level nested Bayesian LP positive control (synthetic, no network).

Stage 1 recovers known per-cell slopes; stage 2 recovers the grand mean and a
predominantly between-supersector variance decomposition under a nested DGP. Few
draws keep it fast; thresholds are correspondingly lenient.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.estimators.bayes_cell_lp import (
    fit_cell_hierarchy,
    per_cell_slopes,
    summarize_cell,
)


def test_per_cell_slopes_recovers_known_slope() -> None:
    rng = np.random.default_rng(0)
    months = [dt.date(2016 + i // 12, i % 12 + 1, 1) for i in range(48)]
    s_t = {m: rng.normal() for m in months}
    beta_cell = {"01_100": -0.8, "02_100": 0.4, "03_100": 0.0}
    rows = []
    for cell, b in beta_cell.items():
        level = 0.0
        for m in months:
            level += b * s_t[m] + rng.normal(0, 0.02)
            rows.append((cell, "100", m, level))
    panel = pl.DataFrame(
        rows,
        schema=["unit_id", "supersector_code", "date", "log_employment"],
        orient="row",
    )
    shock = pl.DataFrame({"date": months, "shock": [s_t[m] for m in months]})
    got = per_cell_slopes(panel, shock, shock_col="shock", horizon=0).sort("unit_id")
    # Horizon-0 growth is y_t - y_{t-1} = b * s_t (+ tiny noise): slope ~ b.
    for uid, b in beta_cell.items():
        row = got.filter(pl.col("unit_id") == uid)
        assert abs(float(row["beta_hat"][0]) - b) < 0.05
        assert float(row["se"][0]) > 0.0


def test_cell_hierarchy_recovers_grand_mean_and_between_share() -> None:
    rng = np.random.default_rng(1)
    n_sectors, cells_per = 5, 8
    mu0_true, tau_b_true, tau_w_true = -0.5, 0.6, 0.1
    sector_mean = mu0_true + tau_b_true * rng.standard_normal(n_sectors)
    sector_idx = np.repeat(np.arange(n_sectors), cells_per)
    beta = sector_mean[sector_idx] + tau_w_true * rng.standard_normal(
        n_sectors * cells_per
    )
    se = np.full(beta.shape, 0.1)
    beta_hat = beta + se * rng.standard_normal(beta.shape[0])
    idata = fit_cell_hierarchy(
        beta_hat, se, sector_idx, n_sectors, draws=300, tune=300, chains=2, seed=0
    )
    summ = summarize_cell(idata, 12, beta.shape[0], n_sectors)
    assert abs(summ.mu0_mean - mu0_true) < 0.4
    # Between-supersector variance dominates within (tau_b >> tau_w by construction).
    assert summ.between_share > 0.5
    assert summ.max_rhat < 1.2
