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

        iv = lp_iv(
            employment,
            policy,
            brw,
            LPIVConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            instrument_col="brw_monthly",
        )
        store.write_table("lpiv_irf", iv)

        def _at(frame: pl.DataFrame, col: str, h: int) -> float:
            row = frame.filter(pl.col("horizon") == h)
            return float(row[col].to_numpy()[0]) if row.height else float("nan")

        return {
            "ts_lp_theta_h12": _at(ts, "theta", 12),
            "ts_lp_theta_h24": _at(ts, "theta", 24),
            "lpiv_theta_h12": _at(iv, "theta", 12),
            "lpiv_min_first_stage_f": float(iv["first_stage_f"].to_numpy().min()),
        }


def main() -> None:
    """Build the aggregate IRFs and print the summary (module entry point)."""
    for key, value in build_aggregate_irf().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
