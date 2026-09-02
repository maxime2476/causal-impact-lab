"""Build the robustness suite on real data and store the results.

Runs the specification curve, placebo/permutation tests, Bai-Perron breaks, the
COVID state-dependent aggregate LP, and the QCEW revision bound, then writes
everything to the store. See ``docs/results.md`` for the honest write-up.

Run as a module::

    uv run python -m cil.robustness.build
"""

from __future__ import annotations

import numpy as np
import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.asymmetry import run_panel_lp_asymmetry
from cil.estimators.panel_lp import PanelLPConfig
from cil.exposure import shift_share as ss
from cil.robustness import breaks, covid, placebo, qcew_revision
from cil.robustness import specification_curve as sc
from cil.robustness.ces_reconciliation import aggregate_qcew_to_supersector
from cil.robustness.placebo import PlaceboMode

_PRIMARY = 12
_PLACEBO_MODES: tuple[PlaceboMode, ...] = ("shock", "exposure")


def build_robustness(settings: Settings | None = None) -> dict[str, float]:
    """Run and store the robustness suite. Return a small summary."""
    settings = settings or get_settings()
    with Store(settings.paths.store_path) as store:
        cells = store.read_table("qcew_cells")
        policy = store.read_table("policy_rate")
        brw = store.read_table("brw_shocks")
        panel = store.read_table("panel_cell")
        macro = store.read_table("macro_current")
        shocks_tbl = store.read_table("shocks")

        sigma = ss.estimate_sigma_semielasticity(
            ss.national_sector_log_employment(cells), policy
        )
        codes = cells["supersector_code"].unique().to_list()
        exposures = {
            "estimated": ss.cell_exposure(
                sigma.select("supersector_code", "sensitivity")
            ),
            "duration": ss.cell_exposure(ss.duration_proxy_for_codes(codes)),
        }
        brw_shock = brw.rename({"brw_monthly": "shock"})
        rr_shock = shocks_tbl.select("date", "rr_shock").drop_nulls()
        shock_map = {
            "brw": (brw_shock, "shock"),
            "rr": (rr_shock, "rr_shock"),
        }

        # Specification curve (+ COVID-excluded sample).
        curve = sc.run_specification_curve(
            panel,
            exposures,
            shock_map,
            horizons=(12, 24),
            control_lags=(3, 6, 12),
            samples={"full": lambda f: f, "ex_covid": sc.covid_window_filter()},
        )
        store.write_table("spec_curve", curve)
        store.write_table("spec_curve_summary", sc.summarize_curve(curve))

        # Placebo / permutation at the primary horizon (headline shock + exposure).
        placebo_rows = [
            placebo.permutation_test(
                panel,
                exposures["estimated"],
                brw_shock,
                shock_col="shock",
                horizon=_PRIMARY,
                mode=mode,
                n_permutations=200,
                seed=settings.inference.seed,
            ).model_dump()
            for mode in _PLACEBO_MODES
        ]
        store.write_table("placebo_results", pl.DataFrame(placebo_rows))

        # Randomization inference: circular-shift the shock (preserves its serial
        # dependence) and re-estimate; per-horizon + joint max|beta| p-values.
        ri_frame, ri_joint_p = placebo.circular_shift_ri(
            panel,
            exposures["estimated"],
            brw_shock,
            shock_col="shock",
            horizons=(0, 12, 24),
            n_draws=200,
            seed=settings.inference.seed,
        )
        store.write_table("randomization_inference", ri_frame)

        # Shock asymmetry: does the relative effect differ by the shock's sign
        # (tightening vs easing) or size (large vs small)?
        asym_config = PanelLPConfig(
            horizons=(0, 6, 12, 24),
            confidence_level=settings.inference.confidence_level,
        )
        asym_sign = run_panel_lp_asymmetry(
            panel,
            exposures["estimated"],
            brw_shock,
            asym_config,
            shock_col="shock",
            split="sign",
        )
        asym_size = run_panel_lp_asymmetry(
            panel,
            exposures["estimated"],
            brw_shock,
            asym_config,
            shock_col="shock",
            split="size",
        )
        store.write_table("asymmetry_sign", asym_sign)
        store.write_table("asymmetry_size", asym_size)

        # Bai-Perron breaks on national employment growth, over the analysis
        # window (breaks within the study period are the relevant diagnostic).
        national = (
            macro.filter(
                (pl.col("series_id") == settings.data.series.national_employment)
                & (pl.col("reference_date") >= settings.data.sample.start)
                & (pl.col("reference_date") <= settings.data.sample.end)
            )
            .select(date="reference_date", level=pl.col("value"))
            .sort("date")
            .with_columns(growth=pl.col("level").log().diff() * 100.0)
            .drop_nulls()
        )
        breaks_df, break_selection = breaks.bai_perron_full(
            national, "growth", seed=settings.inference.seed
        )
        store.write_table("structural_breaks", breaks_df)
        store.write_table("break_selection", break_selection)

        # COVID state-dependent aggregate LP.
        employment = macro.filter(
            pl.col("series_id") == settings.data.series.national_employment
        ).select(date="reference_date", value=pl.col("value").log())
        unemployment = macro.filter(
            pl.col("series_id") == settings.data.series.unemployment
        ).select(date="reference_date", value="value")
        recession = covid.unemployment_recession_indicator(unemployment)
        store.write_table(
            "state_dependent_irf",
            covid.state_dependent_lp(
                employment,
                brw,
                recession,
                shock_col="brw_monthly",
                horizons=(0, 6, 12, 24),
            ),
        )

        # QCEW revision bound at the primary horizon. iid (reference) + the
        # honest benchmark-step model, calibrated to the QCEW-vs-CES growth
        # discrepancy (real vintages are unavailable; see ADR-0022).
        bound = qcew_revision.revision_bound(
            panel,
            exposures["estimated"],
            brw_shock,
            shock_col="shock",
            horizon=_PRIMARY,
            n_draws=50,
            seed=settings.inference.seed,
        )
        store.write_table("qcew_revision_bound", pl.DataFrame([bound.model_dump()]))

        sigma_g = qcew_revision.growth_discrepancy_sd(
            aggregate_qcew_to_supersector(cells),
            store.read_table("ces_supersector"),
        )
        corr_bound = qcew_revision.correlated_revision_bound(
            panel,
            exposures["estimated"],
            brw_shock,
            shock_col="shock",
            horizon=_PRIMARY,
            sigma_bench=sigma_g / np.sqrt(2.0),
            n_draws=40,
            seed=settings.inference.seed,
        )
        store.write_table(
            "qcew_revision_bound_correlated", pl.DataFrame([corr_bound.model_dump()])
        )

        summ = sc.summarize_curve(curve)
        return {
            "spec_share_negative_h12": float(
                summ.filter(pl.col("horizon") == 12)["share_negative"][0]
            ),
            "spec_share_sig_h12": float(
                summ.filter(pl.col("horizon") == 12)["share_sig_bh"][0]
            ),
            "placebo_p_shock": placebo_rows[0]["placebo_p_value"],
            "placebo_p_exposure": placebo_rows[1]["placebo_p_value"],
            "n_breaks": float(store.read_table("structural_breaks").height),
            "qcew_bound_width": bound.beta_max - bound.beta_min,
            "qcew_bound_corr_width": corr_bound.beta_max - corr_bound.beta_min,
            "qcew_bound_corr_sigma_bench": corr_bound.sigma_bench,
            "ri_joint_p_value": ri_joint_p,
            "ri_p_value_h12": float(
                ri_frame.filter(pl.col("horizon") == 12)["ri_p_value"][0]
            ),
            "asym_sign_p_diff_h12": float(
                asym_sign.filter(pl.col("horizon") == 12)["p_diff"][0]
            ),
            "asym_size_p_diff_h12": float(
                asym_size.filter(pl.col("horizon") == 12)["p_diff"][0]
            ),
        }


def main() -> None:
    """Build the robustness suite and print a summary (entry point)."""
    for key, value in build_robustness().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
