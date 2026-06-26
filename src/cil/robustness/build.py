"""Build the robustness suite on real data and store the results.

Runs the specification curve, placebo/permutation tests, Bai-Perron breaks, the
COVID state-dependent aggregate LP, and the QCEW revision bound, then writes
everything to the store. See ``docs/results.md`` for the honest write-up.

Run as a module::

    uv run python -m cil.robustness.build
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.exposure import shift_share as ss
from cil.robustness import breaks, covid, placebo, qcew_revision
from cil.robustness import specification_curve as sc
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
        exposures = {
            "estimated": ss.cell_exposure(
                sigma.select("supersector_code", "sensitivity")
            ),
            "duration": ss.cell_exposure(ss.duration_proxy_sigma()),
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

        # Bai-Perron breaks on national employment growth.
        national = (
            macro.filter(
                pl.col("series_id") == settings.data.series.national_employment
            )
            .select(date="reference_date", level=pl.col("value"))
            .sort("date")
            .with_columns(growth=pl.col("level").log().diff() * 100.0)
            .drop_nulls()
        )
        store.write_table("structural_breaks", breaks.bai_perron(national, "growth"))

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

        # QCEW revision bound at the primary horizon.
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
        }


def main() -> None:
    """Build the robustness suite and print a summary (entry point)."""
    for key, value in build_robustness().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
