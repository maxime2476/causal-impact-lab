"""Export analysis-ready result tables to committed CSV artifacts.

The Streamlit app reads these small, computed result tables (not raw data) so it
can run self-contained on a lightweight Hugging Face Space without the heavy
analysis stack or the FRED key. Regenerate after re-running the estimators::

    uv run python -m cil.report.export
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cil.config import Settings, get_settings
from cil.data.ces_sae import STATE_FIPS_ABBR
from cil.data.store import Store
from cil.exposure import shift_share as ss

#: Store tables copied verbatim into the app assets, with output file names.
_PASS_THROUGH = {
    "panel_lp_results": "headline_irf.csv",
    "ts_lp_irf": "aggregate_ts_irf.csv",
    "lpiv_irf": "aggregate_lpiv_irf.csv",
    "spec_curve": "spec_curve.csv",
    "spec_curve_summary": "spec_curve_summary.csv",
    "bayes_vs_freq": "bayes_vs_freq.csv",
    "dml_results": "dml_results.csv",
    "exposure_sigma": "exposure_sigma.csv",
    "state_dependent_irf": "state_dependent_irf.csv",
}


def _state_exposure(store: Store, settings: Settings) -> pl.DataFrame:
    """Compute state-level shift-share exposure with postal abbreviations."""
    cells = store.read_table("qcew_cells")
    policy = store.read_table("policy_rate")
    sigma = ss.estimate_sigma_semielasticity(
        ss.national_sector_log_employment(cells), policy
    )
    shares = ss.base_period_shares(cells)
    exposure = ss.state_exposure(
        shares, sigma.select("supersector_code", "sensitivity")
    )
    abbr = pl.DataFrame(
        {
            "state_fips": list(STATE_FIPS_ABBR.keys()),
            "state": list(STATE_FIPS_ABBR.values()),
        }
    )
    return exposure.join(abbr, on="state_fips", how="inner").sort("state_fips")


def export_app_assets(
    settings: Settings | None = None, out_dir: Path | None = None
) -> list[str]:
    """Write the result tables the app needs as CSVs. Return the file names.

    Parameters
    ----------
    settings
        Project settings (for the store path).
    out_dir
        Output directory; defaults to ``app/assets`` at the project root.

    Returns
    -------
    list of str
        The CSV file names written.
    """
    settings = settings or get_settings()
    out_dir = out_dir or (Path(__file__).resolve().parents[3] / "app" / "assets")
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []
    with Store(settings.paths.store_path) as store:
        for table, filename in _PASS_THROUGH.items():
            if store.table_exists(table):
                store.read_table(table).write_csv(out_dir / filename)
                written.append(filename)
        _state_exposure(store, settings).write_csv(out_dir / "state_exposure.csv")
        written.append("state_exposure.csv")
    return written


def main() -> None:
    """Export the app assets and print the files written (entry point)."""
    for name in export_app_assets():
        print(f"wrote {name}")


if __name__ == "__main__":
    main()
