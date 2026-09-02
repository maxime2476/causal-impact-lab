"""Headline relative effect on **wages** (a second outcome).

Applies the interacted panel local projection to log QCEW average weekly wage
instead of employment — the same clean cross-sectional identification, a second
honest estimand. Wages are quarterly, so the shock is aggregated to quarterly and
horizons are in quarters.

Run as a module::

    uv run python -m cil.estimators.wages
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.panel_lp import PanelLPConfig, leads_summary, run_panel_lp

_SHOCK_COL = "shock"
#: Quarterly response horizons and pre-trend leads.
_HORIZONS = tuple(range(-4, 9))
_MIN_QUARTERS_FRACTION = 0.9


def build_wage_estimates(settings: Settings | None = None) -> dict[str, float]:
    """Build and store the wage relative-effect IRF. Return a small summary."""
    settings = settings or get_settings()
    with Store(settings.paths.store_path) as store:
        wages = store.read_table("qcew_wages")
        brw = store.read_table("brw_shocks")
        exposure = store.read_table("exposure_cell")

        n_quarters = wages["date"].n_unique()
        panel = (
            wages.filter(pl.col("avg_weekly_wage") > 0)
            .with_columns(
                unit_id=pl.col("state_fips") + "_" + pl.col("supersector_code"),
                log_wage=pl.col("avg_weekly_wage").log(),
            )
            .select("unit_id", "state_fips", "supersector_code", "date", "log_wage")
        )
        # Coverage filter: keep cells present in >= 90% of quarters.
        keep = (
            panel.group_by("unit_id")
            .agg(n=pl.len())
            .filter(pl.col("n") >= _MIN_QUARTERS_FRACTION * n_quarters)
            .select("unit_id")
        )
        panel = panel.join(keep, on="unit_id", how="inner")

        # Aggregate the monthly BRW shock to quarterly (quarter-start dates).
        qshock = (
            brw.with_columns(date=pl.col("date").dt.truncate("1q"))
            .group_by("date")
            .agg(shock=pl.col("brw_monthly").mean())
            .sort("date")
        )

        result = run_panel_lp(
            panel,
            exposure,
            qshock,
            PanelLPConfig(
                horizons=_HORIZONS,
                n_control_lags=4,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
            outcome_col="log_wage",
        )
        store.write_table("wage_panel_lp_results", result)
        leads = leads_summary(result, settings.inference.fdr_alpha)

        def _beta(h: int) -> float:
            row = result.filter(pl.col("horizon") == h)
            return float(row["beta"][0]) if row.height else float("nan")

        response = result.filter(pl.col("horizon") >= 0)
        pbh = response["p_value_bh"].to_numpy()
        return {
            "n_cells": float(panel["unit_id"].n_unique()),
            "wage_beta_h0": _beta(0),
            "wage_beta_h4": _beta(4),
            "wage_beta_h8": _beta(8),
            "wage_leads_any_significant": leads["any_significant"],
            "wage_any_bh_significant": float(
                bool((pbh <= settings.inference.fdr_alpha).any())
            ),
        }


def main() -> None:
    """Build the wage estimates and print the summary (module entry point)."""
    for key, value in build_wage_estimates().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
