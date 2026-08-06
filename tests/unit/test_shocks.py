"""Shock identification: orthogonalization, first stage, info-effect, predictability.

All tests use synthetic data (no network). They are positive controls: OLS
residuals are orthogonal to the regressors, a strong instrument yields a strong
first stage and a noisy one is flagged weak, the info-effect classifier counts
co-movement correctly, and the predictability test separates signal from noise.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.shocks import compare, features, info_effect, predictability, proxy_svar
from cil.shocks import rr_orthogonalization as rro


def _months(n: int, start: dt.date = dt.date(2000, 1, 1)) -> list[dt.date]:
    return [
        dt.date(
            start.year + (start.month - 1 + i) // 12, (start.month - 1 + i) % 12 + 1, 1
        )
        for i in range(n)
    ]


def _synthetic_pit(n: int = 72, seed: int = 3) -> pl.DataFrame:
    """A single-vintage PIT frame for CPI, INDPRO, UNRATE."""
    rng = np.random.default_rng(seed)
    refs = _months(n)
    frames = []
    specs = {
        "CPIAUCSL": 100.0 * np.exp(np.cumsum(rng.normal(0.002, 0.002, n))),
        "INDPRO": 100.0 * np.exp(np.cumsum(rng.normal(0.001, 0.004, n))),
        "UNRATE": 5.0 + np.cumsum(rng.normal(0.0, 0.05, n)),
    }
    for sid, vals in specs.items():
        frames.append(
            pl.DataFrame(
                {
                    "series_id": [sid] * n,
                    "reference_date": refs,
                    "vintage_date": refs,
                    "value": vals,
                }
            )
        )
    return pl.concat(frames)


def test_rr_residual_is_orthogonal_to_regressors() -> None:
    pit = _synthetic_pit()
    rng = np.random.default_rng(0)
    refs = _months(72)
    policy = pl.DataFrame(
        {"date": refs, "policy_rate": np.cumsum(rng.normal(0, 0.1, 72))}
    )
    shock, diag = rro.romer_romer_shock(pit, policy, n_lags=2)
    assert shock.height > 0
    assert diag.n_obs == shock.height
    # OLS residuals are orthogonal to the regressors (here, inflation feature).
    merged = shock.join(features.realtime_macro_features(pit), on="date", how="inner")
    corr = merged.select(pl.corr("rr_shock", "inflation")).item()
    assert abs(float(corr)) < 1e-6


def test_proxy_svar_strong_vs_weak_instrument() -> None:
    rng = np.random.default_rng(1)
    n = 240
    dates = _months(n)
    z = rng.normal(size=n)
    # Policy rate driven partly by the instrument; block has activity/prices.
    policy = np.cumsum(0.3 * z + rng.normal(0, 0.2, n))
    block = pl.DataFrame(
        {
            "date": dates,
            "policy_rate": policy,
            "log_ip": np.cumsum(rng.normal(0, 0.3, n)),
            "log_cpi": np.cumsum(rng.normal(0, 0.2, n)),
            "unemployment": 5 + np.cumsum(rng.normal(0, 0.05, n)),
        }
    )
    strong = pl.DataFrame({"date": dates, "z": z})
    _, strong_diag = proxy_svar.first_stage(block, strong, instrument_col="z", n_lags=6)
    assert not strong_diag.weak
    assert strong_diag.robust_f > 10

    noise = pl.DataFrame({"date": dates, "z": rng.normal(size=n)})
    _, weak_diag = proxy_svar.first_stage(block, noise, instrument_col="z", n_lags=6)
    assert weak_diag.weak


def test_info_effect_counts_comovement() -> None:
    dates = _months(4)
    shock = pl.DataFrame({"date": dates, "s": [1.0, 1.0, -1.0, -1.0]})
    equity = pl.DataFrame({"date": dates, "equity_return": [2.0, -2.0, 3.0, -3.0]})
    classified, summary = info_effect.classify(shock, equity, shock_col="s")
    # Months 1 and 4 co-move (information); 2 and 3 do not.
    assert summary.n_months == 4
    assert summary.n_information == 2
    assert summary.contamination_share == 0.5
    assert classified.filter(pl.col("is_information"))["date"].to_list() == [
        dates[0],
        dates[3],
    ]


def test_info_effect_high_frequency_uses_same_window() -> None:
    dates = _months(4)
    fomc = pl.DataFrame(
        {
            "date": dates,
            "mps": [1.0, 1.0, -1.0, -1.0],
            "mps_orth": [0.0, 0.0, 0.0, 0.0],
            "sp500": [2.0, -2.0, 3.0, -3.0],
        }
    )
    classified, summary = info_effect.classify_high_frequency(fomc)
    # Events 1 and 4 co-move (information); 2 and 3 are conventional policy.
    assert summary.n_months == 4
    assert summary.n_information == 2
    assert summary.contamination_share == 0.5
    # The monetary component zeroes out the information events.
    assert classified.sort("date")["monetary_component"].to_list() == [
        0.0,
        1.0,
        -1.0,
        0.0,
    ]


def test_predictability_separates_signal_from_noise() -> None:
    rng = np.random.default_rng(2)
    n = 200
    dates = _months(n)
    x = rng.normal(size=n)
    predictors = pl.DataFrame({"date": dates, "inflation": x})
    predictable = pl.DataFrame(
        {"date": dates, "s": np.roll(x, 1) * 0.9 + rng.normal(0, 0.1, n)}
    )
    noise = pl.DataFrame({"date": dates, "s": rng.normal(size=n)})
    sig = predictability.predictability_test(
        predictable, predictors, shock_col="s", predictor_cols=["inflation"], n_lags=2
    )
    nul = predictability.predictability_test(
        noise, predictors, shock_col="s", predictor_cols=["inflation"], n_lags=2
    )
    assert sig.predictable
    assert sig.r_squared > nul.r_squared
    assert not nul.predictable


def test_cross_correlations() -> None:
    dates = _months(50)
    a = np.linspace(-1, 1, 50)
    series = {
        "x": pl.DataFrame({"date": dates, "x": a}),
        "y": pl.DataFrame({"date": dates, "y": a}),  # identical -> corr 1
        "z": pl.DataFrame({"date": dates, "z": a[::-1]}),  # reversed -> corr -1
    }
    results = {
        (c.series_a, c.series_b): c.correlation
        for c in compare.cross_correlations(series)
    }
    assert abs(results[("x", "y")] - 1.0) < 1e-9
    assert abs(results[("x", "z")] + 1.0) < 1e-9
