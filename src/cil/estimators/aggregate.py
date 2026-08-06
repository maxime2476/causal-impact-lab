"""Build the aggregate impulse responses on real data and store them.

Runs the aggregate time-series local projection (employment response to the
shock) and the LP-IV proxy-SVAR (response to a policy-rate increase, instrumented
by the shock), then writes both IRFs to the store. The aggregate effect is the
assumption-dependent complement; results and caveats are in ``docs/results.md``.

Run as a module::

    uv run python -m cil.estimators.aggregate
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.proxy_svar import LPIVConfig, lp_iv
from cil.estimators.ts_lp import TimeSeriesLPConfig, time_series_lp

_SHOCK_COL = "shock"

#: Quarterly response horizons for the narrative-shock LP (0..16 quarters).
_RR_HORIZONS = tuple(range(0, 17))
#: Headline Romer-Romer vintage (the longest, extended through 2012Q4).
_RR_SHOCK_COL = "rr12"


def build_narrative_shock_irf(settings: Settings | None = None) -> dict[str, float]:
    """Build and store the quarterly Romer-Romer narrative-shock aggregate IRF.

    Aggregates national employment to quarterly (mean level, logged) and runs the
    time-series LP against the updated Romer-Romer narrative shock (``rr12``), an
    identification that does not rely on a 30-minute market window. This is a
    second aggregate complement, reported alongside the HF-instrument LP-IV.

    Returns
    -------
    dict of str to float
        Narrative-shock employment response at h=4/8 quarters and the sample size.
    """
    settings = settings or get_settings()
    with Store(settings.paths.store_path) as store:
        macro_current = store.read_table("macro_current")
        rr = store.read_table("rr_shocks")

        employment_q = (
            macro_current.filter(
                pl.col("series_id") == settings.data.series.national_employment
            )
            .select(
                date=pl.col("reference_date").dt.truncate("1q"),
                value=pl.col("value"),
            )
            .group_by("date")
            .agg(pl.col("value").mean())
            .select("date", value=pl.col("value").log())
            .sort("date")
        )

        irf = time_series_lp(
            employment_q,
            rr,
            TimeSeriesLPConfig(
                horizons=_RR_HORIZONS,
                n_lags=4,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_RR_SHOCK_COL,
        )
        store.write_table("rr_lp_irf", irf)

        def _at(frame: pl.DataFrame, col: str, h: int) -> float:
            row = frame.filter(pl.col("horizon") == h)
            return float(row[col].to_numpy()[0]) if row.height else float("nan")

        return {
            "rr_lp_theta_h4": _at(irf, "theta", 4),
            "rr_lp_theta_h8": _at(irf, "theta", 8),
            "rr_lp_n_obs_h8": _at(irf, "n_obs", 8),
        }


def build_aggregate_irf(settings: Settings | None = None) -> dict[str, float]:
    """Build and store the aggregate IRFs. Return a small summary.

    Returns
    -------
    dict of str to float
        TS-LP employment response at h=12/24, the LP-IV response at h=12, and
        the minimum LP-IV first-stage F (weak-instrument flag context).
    """
    settings = settings or get_settings()
    horizons = tuple(range(0, settings.horizons.max_horizon + 1))
    with Store(settings.paths.store_path) as store:
        macro_current = store.read_table("macro_current")
        policy = store.read_table("policy_rate")
        brw = store.read_table("brw_shocks")

        employment = macro_current.filter(
            pl.col("series_id") == settings.data.series.national_employment
        ).select(date="reference_date", value=pl.col("value").log())
        shock = brw.rename({"brw_monthly": _SHOCK_COL})

        ts = time_series_lp(
            employment,
            shock,
            TimeSeriesLPConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
        )
        store.write_table("ts_lp_irf", ts)

        # Headline LP-IV instrument: the Bauer-Swanson HF surprise (a far stronger
        # first stage than BRW); BRW retained as a robustness variant.
        iv_config = LPIVConfig(
            horizons=horizons,
            confidence_level=settings.inference.confidence_level,
        )
        mps = store.read_table("mps")
        iv = lp_iv(employment, policy, mps, iv_config, instrument_col="mps")
        store.write_table("lpiv_irf", iv)
        iv_brw = lp_iv(employment, policy, brw, iv_config, instrument_col="brw_monthly")
        store.write_table("lpiv_irf_brw", iv_brw)
        # Predictability-robust variant: the orthogonalized surprise MPS_ORTH.
        iv_orth = lp_iv(employment, policy, mps, iv_config, instrument_col="mps_orth")
        store.write_table("lpiv_irf_orth", iv_orth)

        def _at(frame: pl.DataFrame, col: str, h: int) -> float:
            row = frame.filter(pl.col("horizon") == h)
            return float(row[col].to_numpy()[0]) if row.height else float("nan")

        return {
            "ts_lp_theta_h12": _at(ts, "theta", 12),
            "ts_lp_theta_h24": _at(ts, "theta", 24),
            "lpiv_theta_h12": _at(iv, "theta", 12),
            "lpiv_orth_theta_h12": _at(iv_orth, "theta", 12),
            "lpiv_mps_min_first_stage_f": float(iv["first_stage_f"].to_numpy().min()),
            "lpiv_orth_min_first_stage_f": float(
                iv_orth["first_stage_f"].to_numpy().min()
            ),
            "lpiv_brw_min_first_stage_f": float(
                iv_brw["first_stage_f"].to_numpy().min()
            ),
        }


def main() -> None:
    """Build the aggregate IRFs and print the summary (module entry point)."""
    summary = {**build_aggregate_irf(), **build_narrative_shock_irf()}
    for key, value in summary.items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
