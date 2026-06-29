"""The committed app assets satisfy the contract the Streamlit app depends on.

These run in CI (the CSV artifacts are committed, unlike the raw data store), so
they guard the app's data contract without needing the analysis stack.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ASSETS = Path(__file__).resolve().parents[2] / "app" / "assets"

_REQUIRED: dict[str, set[str]] = {
    "headline_irf.csv": {"horizon", "beta", "ci_low", "ci_high", "p_value_bh"},
    "aggregate_ts_irf.csv": {"horizon", "theta", "ci_low", "ci_high"},
    "aggregate_lpiv_irf.csv": {
        "horizon",
        "theta",
        "first_stage_f",
        "ar_low",
        "ar_high",
    },
    "spec_curve.csv": {"shock", "exposure", "lags", "sample", "horizon", "beta"},
    "state_exposure.csv": {"state", "exposure"},
    "exposure_sigma.csv": {"supersector_code", "sensitivity"},
    "bayes_vs_freq.csv": {"horizon", "bayes_mu", "freq_beta"},
    "dml_results.csv": {"horizon", "linear_ate"},
}


@pytest.mark.parametrize(("filename", "columns"), list(_REQUIRED.items()))
def test_asset_present_with_columns(filename: str, columns: set[str]) -> None:
    path = ASSETS / filename
    assert path.exists(), f"missing committed app asset: {filename}"
    df = pd.read_csv(path)
    assert columns.issubset(df.columns), (
        f"{filename} missing {columns - set(df.columns)}"
    )
    assert len(df) > 0


def test_state_exposure_uses_postal_codes() -> None:
    df = pd.read_csv(ASSETS / "state_exposure.csv")
    assert df["state"].str.len().eq(2).all()
    assert df["state"].is_unique
