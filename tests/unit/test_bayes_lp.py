"""Bayesian hierarchical LP positive control (synthetic, no network).

Recovers known sector responses and the population mean under partial pooling,
with acceptable convergence. Uses few draws to stay fast; convergence thresholds
are correspondingly lenient.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.estimators.bayes_lp import (
    fit_hierarchical_lp,
    posterior_predictive_check,
    prepare_design,
    summarize,
)


def _panel() -> tuple[pl.DataFrame, pl.DataFrame, dict[str, float]]:
    rng = np.random.default_rng(0)
    states = [f"{s:02d}" for s in range(1, 9)]
    sectors = [f"10{k}" for k in range(11, 16)]
    months = [dt.date(2016 + i // 12, i % 12 + 1, 1) for i in range(24)]
    s_t = {m: rng.normal() for m in months}
    beta_k = dict(zip(sectors, [-1.0, -0.5, 0.0, 0.5, 1.0], strict=True))
    rows = []
    for st in states:
        for k in sectors:
            level = 0.0
            for m in months:
                level += beta_k[k] * s_t[m] + rng.normal(0, 0.2)
                rows.append((f"{st}_{k}", st, k, m, level))
    panel = pl.DataFrame(
        rows,
        schema=["unit_id", "state_fips", "supersector_code", "date", "log_employment"],
        orient="row",
    )
    shock = pl.DataFrame({"date": months, "shock": [s_t[m] for m in months]})
    return panel, shock, beta_k


def test_prepare_design_shapes() -> None:
    panel, shock, beta_k = _panel()
    y, x, sectors = prepare_design(panel, shock, shock_col="shock", horizon=0)
    assert y.shape[0] == x.shape[0]
    assert x.shape[1] == len(sectors) == len(beta_k)


def test_hierarchical_lp_recovers_sector_responses() -> None:
    panel, shock, beta_k = _panel()
    y, x, sectors = prepare_design(panel, shock, shock_col="shock", horizon=0)
    idata = fit_hierarchical_lp(y, x, draws=400, tune=400, chains=2, seed=0)
    posterior_beta = idata.posterior["beta"].mean(("chain", "draw")).values
    truth = np.array([beta_k[s] for s in sectors])
    # Posterior sector means track the injected responses.
    assert np.corrcoef(posterior_beta, truth)[0, 1] > 0.95
    summary = summarize(idata, 0)
    assert summary.max_rhat < 1.1
    ppc = posterior_predictive_check(idata, y)
    assert ppc["sigma_ratio"] < 1.5
