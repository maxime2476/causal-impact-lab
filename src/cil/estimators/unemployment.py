"""Relative effect on the **state unemployment rate** (a third outcome).

The headline interacted panel LP, at the state level: the change in a state's
unemployment rate is regressed on its Bartik exposure interacted with the national
shock, with state and time fixed effects. The expected sign is the *opposite* of
employment — a tightening should raise unemployment more in exposed states.

Run as a module::

    uv run python -m cil.estimators.unemployment
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.panel_lp import PanelLPConfig, leads_summary, run_panel_lp
from cil.exposure import shift_share as ss

_SHOCK_COL = "shock"


def build_unemployment_estimates(settings: Settings | None = None) -> dict[str, float]:
    """Build and store the state-unemployment relative-effect IRF. Return a summary."""
    settings = settings or get_settings()
    horizons = tuple(
        range(settings.horizons.min_horizon, settings.horizons.max_horizon + 1)
    )
    with Store(settings.paths.store_path) as store:
        unemp = store.read_table("state_unemployment")
        cells = store.read_table("qcew_cells")
        policy = store.read_table("policy_rate")
        brw = store.read_table("brw_shocks")

        sigma = ss.estimate_sigma_semielasticity(
            ss.national_sector_log_employment(cells), policy
        )
        exposure = ss.state_exposure(ss.base_period_shares(cells), sigma).rename(
            {"state_fips": "supersector_code"}
        )
        panel = (
            unemp.filter(
                (pl.col("date") >= settings.data.sample.start)
                & (pl.col("date") <= settings.data.sample.end)
            )
            .with_columns(
                unit_id=pl.col("state_fips"),
                supersector_code=pl.col("state_fips"),
            )
            .select("unit_id", "state_fips", "supersector_code", "date", "unemployment")
            .sort(["unit_id", "date"])
        )
        shock = brw.rename({"brw_monthly": _SHOCK_COL})

        result = run_panel_lp(
            panel,
            exposure,
            shock,
            PanelLPConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
            outcome_col="unemployment",
        )
        store.write_table("unemployment_panel_lp_results", result)
        leads = leads_summary(result, settings.inference.fdr_alpha)

        def _beta(h: int) -> float:
            row = result.filter(pl.col("horizon") == h)
            return float(row["beta"][0]) if row.height else float("nan")

        response = result.filter(pl.col("horizon") >= 0)
        pbh = response["p_value_bh"].to_numpy()
        return {
            "n_states": float(panel["unit_id"].n_unique()),
            "unemp_beta_h0": _beta(0),
            "unemp_beta_h12": _beta(12),
            "unemp_beta_h24": _beta(24),
            "unemp_share_positive": float((response["beta"].to_numpy() > 0).mean()),
            "unemp_leads_any_significant": leads["any_significant"],
            "unemp_any_bh_significant": float(
                bool((pbh <= settings.inference.fdr_alpha).any())
            ),
        }


def main() -> None:
    """Build the unemployment estimates and print the summary (entry point)."""
    for key, value in build_unemployment_estimates().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
