"""Estimators: panel LP, LP-DiD, Goodman-Bacon, and exposure (positive controls).

Synthetic DGPs with known effects (no network, no real data): the panel LP
recovers an injected relative semi-elasticity, LP-DiD recovers an injected ATT
(its TWFE equivalence on a clean two-group design), and the exposure shifter is
standardized with the expected sign.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.estimators.goodman_bacon import bacon_diagnostic
from cil.estimators.lp_did import LPDiDConfig, lp_did
from cil.estimators.panel_lp import PanelLPConfig, run_panel_lp
from cil.exposure import shift_share as ss


def _months(n: int) -> list[dt.date]:
    return [dt.date(2014 + i // 12, i % 12 + 1, 1) for i in range(n)]


def test_panel_lp_recovers_injected_beta() -> None:
    rng = np.random.default_rng(0)
    states = [f"{s:02d}" for s in range(1, 31)]
    sectors = [f"10{k}" for k in range(11, 22)]
    months = _months(60)
    expo = dict(zip(sectors, np.linspace(-1.5, 1.5, 11), strict=True))
    s_t = {m: rng.normal() for m in months}
    tau = {m: rng.normal() * 0.5 for m in months}
    beta_true = -0.8
    rows = []
    for st in states:
        for k in sectors:
            level = 0.0
            for m in months:
                level += beta_true * expo[k] * s_t[m] + tau[m] + rng.normal(0, 0.2)
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
    res = run_panel_lp(
        panel, exposure, shock, PanelLPConfig(horizons=(0, 1)), shock_col="shock"
    )
    beta0 = res.filter(pl.col("horizon") == 0)["beta"][0]
    assert abs(beta0 - beta_true) < 0.1


def test_panel_lp_omits_reference_lead() -> None:
    rng = np.random.default_rng(1)
    months = _months(36)
    rows = []
    for st in range(20):
        for k in [f"10{j}" for j in range(11, 14)]:
            lvl = 0.0
            for m in months:
                lvl += rng.normal(0, 0.1)
                rows.append((f"{st}_{k}", str(st), k, m, lvl))
    panel = pl.DataFrame(
        rows,
        schema=["unit_id", "state_fips", "supersector_code", "date", "log_employment"],
        orient="row",
    )
    exposure = pl.DataFrame(
        {
            "supersector_code": [f"10{j}" for j in range(11, 14)],
            "exposure": [-1.0, 0.0, 1.0],
        }
    )
    shock = pl.DataFrame({"date": months, "shock": rng.normal(size=len(months))})
    res = run_panel_lp(
        panel,
        exposure,
        shock,
        PanelLPConfig(horizons=(-2, -1, 0, 1)),
        shock_col="shock",
    )
    assert -1 not in res["horizon"].to_list()  # reference horizon omitted
    assert -2 in res["horizon"].to_list()


def _staggered_panel(theta: float = 0.5, seed: int = 0) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    months = _months(48)
    adopt = months[24]
    rows = []
    for u in range(40):
        treated_group = u < 20
        a = rng.normal()
        for i, m in enumerate(months):
            post = treated_group and m >= adopt
            y = a + 0.01 * i + (theta if post else 0.0) + rng.normal(0, 0.05)
            rows.append((f"u{u}", m, y, int(post)))
    return pl.DataFrame(
        rows, schema=["unit_id", "date", "log_employment", "treated"], orient="row"
    )


def test_lp_did_recovers_att_twfe_equivalence() -> None:
    df = _staggered_panel(theta=0.5)
    res = lp_did(df, LPDiDConfig(horizons=(0, 1, 2)))
    for att in res["att"].to_list():
        assert abs(att - 0.5) < 0.1


def test_bacon_diagnostic_small_gap_on_clean_design() -> None:
    df = _staggered_panel(theta=0.5)
    bacon = bacon_diagnostic(df)
    assert abs(bacon.twfe_estimate - bacon.clean_estimate) < 0.15
    assert 0.0 <= bacon.forbidden_share <= 1.0


def test_cell_exposure_standardized() -> None:
    sens = pl.DataFrame(
        {"supersector_code": ["a", "b", "c"], "sensitivity": [1.0, 2.0, 3.0]}
    )
    expo = ss.cell_exposure(sens)
    vals = expo["exposure"].to_numpy()
    assert abs(float(vals.mean())) < 1e-9
    assert abs(float(vals.std(ddof=1)) - 1.0) < 1e-9


def test_state_exposure_weights_shares() -> None:
    shares = pl.DataFrame(
        {
            "state_fips": ["01", "01", "02", "02"],
            "supersector_code": ["a", "b", "a", "b"],
            "share": [0.9, 0.1, 0.1, 0.9],
        }
    )
    sens = pl.DataFrame({"supersector_code": ["a", "b"], "sensitivity": [1.0, 0.0]})
    e_s = ss.state_exposure(shares, sens)
    # State 01 (heavy in sensitive sector a) is more exposed than state 02.
    e01 = e_s.filter(pl.col("state_fips") == "01")["exposure"][0]
    e02 = e_s.filter(pl.col("state_fips") == "02")["exposure"][0]
    assert e01 > e02


def test_duration_proxy_for_codes_maps_supersector_and_3digit() -> None:
    out = ss.duration_proxy_for_codes(["1013", "331", "236", "999"])
    mapping = dict(
        zip(
            out["supersector_code"].to_list(),
            out["sensitivity"].to_list(),
            strict=True,
        )
    )
    # 331 (manufacturing) -> 1013; 236 (construction) -> 1012.
    assert mapping["1013"] == ss.DURATION_PROXY["1013"]
    assert mapping["331"] == ss.DURATION_PROXY["1013"]
    assert mapping["236"] == ss.DURATION_PROXY["1012"]
    assert "999" not in mapping  # unmapped code omitted
