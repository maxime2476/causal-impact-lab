"""Build the headline estimates on real data and store them.

Constructs the shift-share exposure, runs the interacted panel local projection
(headline), the binary LP-DiD robustness on large-tightening episodes, and the
Goodman-Bacon diagnostic, then writes the results to the store. Results are
reported honestly, including nulls (see ``docs/results.md``).

Run as a module::

    uv run python -m cil.estimators.build
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.panel_lp import (
    PanelLPConfig,
    conley_cutoff_sensitivity,
    leads_summary,
    run_panel_lp,
    run_panel_lp_conley,
    run_panel_lp_exposure_robust,
)
from cil.exposure import shift_share as ss

_SHOCK_COL = "shock"


def build_estimates(settings: Settings | None = None) -> dict[str, float]:
    """Build and store the headline estimates. Return a small summary.

    Returns
    -------
    dict of str to float
        Key headline numbers (beta at h=0/12/24, lead significance flag,
        LP-DiD ATT at h=12, and the Bacon TWFE-vs-clean gap).
    """
    settings = settings or get_settings()
    horizons = tuple(
        range(settings.horizons.min_horizon, settings.horizons.max_horizon + 1)
    )
    with Store(settings.paths.store_path) as store:
        cells = store.read_table("qcew_cells")
        policy = store.read_table("policy_rate")
        brw = store.read_table("brw_shocks")
        panel = store.read_table("panel_cell")

        national = ss.national_sector_log_employment(cells)
        sigma = ss.estimate_sigma_semielasticity(national, policy)
        exposure = ss.cell_exposure(sigma.select("supersector_code", "sensitivity"))
        store.write_table("exposure_sigma", sigma)
        store.write_table("exposure_cell", exposure)

        shock = brw.rename({"brw_monthly": _SHOCK_COL})
        lp = run_panel_lp(
            panel,
            exposure,
            shock,
            PanelLPConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
        )
        store.write_table("panel_lp_results", lp)
        leads = leads_summary(lp, settings.inference.fdr_alpha)

        # Exposure-robust (BHJ) inference: cluster on the supersector exposure
        # dimension. Same point estimates; typically wider bands.
        lp_er = run_panel_lp_exposure_robust(
            panel,
            exposure,
            shock,
            PanelLPConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
        )
        store.write_table("panel_lp_exposure_robust", lp_er)

        # Conley spatial + serial HAC inference: robust to geographic correlation
        # between nearby states (500 km Bartlett kernel) plus serial correlation.
        lp_conley = run_panel_lp_conley(
            panel,
            exposure,
            shock,
            PanelLPConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
        )
        store.write_table("panel_lp_conley", lp_conley)

        # Cutoff sensitivity at h=12: the Conley SE converges to Driscoll-Kraay as
        # the spatial kernel widens, exposing the short-cutoff SE as a
        # distance-decay artifact (ADR-0021).
        conley_sens = conley_cutoff_sensitivity(
            panel,
            exposure,
            shock,
            PanelLPConfig(
                horizons=horizons,
                confidence_level=settings.inference.confidence_level,
            ),
            shock_col=_SHOCK_COL,
            horizon=12,
            cutoffs_km=(200.0, 500.0, 1000.0, 3000.0, 100000.0),
        )
        store.write_table("conley_cutoff_sensitivity", conley_sens)

        def _col(frame: pl.DataFrame, col: str, h: int) -> float:
            row = frame.filter(pl.col("horizon") == h)
            return float(row[col][0]) if row.height else float("nan")

        response_er = lp_er.filter(pl.col("horizon") >= 0)
        er_pbh = response_er["p_value_bh"].to_numpy()
        er_any_sig = float(bool((er_pbh <= settings.inference.fdr_alpha).any()))

        response_c = lp_conley.filter(pl.col("horizon") >= 0)
        c_pbh = response_c["p_value_bh"].to_numpy()
        conley_any_sig = float(bool((c_pbh <= settings.inference.fdr_alpha).any()))

        return {
            "beta_h0": _col(lp, "beta", 0),
            "beta_h12": _col(lp, "beta", 12),
            "beta_h24": _col(lp, "beta", 24),
            "leads_any_significant": leads["any_significant"],
            "se_h12_dk": _col(lp, "se", 12),
            "se_h12_exposure_robust": _col(lp_er, "se", 12),
            "se_h12_conley": _col(lp_conley, "se", 12),
            "se_h12_conley_wide": float(
                conley_sens.filter(pl.col("cutoff_km") >= 1e5)["se"].to_numpy()[0]
            ),
            "exposure_robust_any_significant": er_any_sig,
            "conley_any_significant": conley_any_sig,
        }


def main() -> None:
    """Build the estimates and print the summary (module entry point)."""
    for key, value in build_estimates().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
