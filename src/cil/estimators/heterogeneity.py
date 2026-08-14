"""Build DML heterogeneity estimates on real data and store them.

Runs the EconML LinearDML and CausalForestDML CATE estimation (with purged
time-blocked cross-fitting and a placebo refutation) at the primary horizons,
and writes the results to the store. See ``docs/results.md`` for the honest
write-up.

Run as a module::

    uv run python -m cil.estimators.heterogeneity
"""

from __future__ import annotations

import polars as pl

from cil.config import Settings, get_settings
from cil.data.store import Store
from cil.estimators.dml import estimate_cate_drivers, estimate_heterogeneity
from cil.exposure import shift_share as ss

_SHOCK_COL = "shock"
_HORIZONS = (0, 12, 24)
_DRIVER_HORIZON = 12


def build_heterogeneity(settings: Settings | None = None) -> dict[str, float]:
    """Estimate and store DML heterogeneity at the primary horizons.

    Returns
    -------
    dict of str to float
        LinearDML ATE and placebo ATE at each horizon.
    """
    settings = settings or get_settings()
    with Store(settings.paths.store_path) as store:
        cells = store.read_table("qcew_cells")
        policy = store.read_table("policy_rate")
        brw = store.read_table("brw_shocks")
        panel = store.read_table("panel_cell")

        national = ss.national_sector_log_employment(cells)
        sigma = ss.estimate_sigma_semielasticity(national, policy)
        exposure = ss.cell_exposure(sigma.select("supersector_code", "sensitivity"))
        shock = brw.rename({"brw_monthly": _SHOCK_COL})

        results = [
            estimate_heterogeneity(
                panel,
                exposure,
                shock,
                shock_col=_SHOCK_COL,
                horizon=h,
                seed=settings.inference.seed,
            )
            for h in _HORIZONS
        ]
        store.write_table(
            "dml_results", pl.DataFrame([r.model_dump() for r in results])
        )

        # CausalForest heterogeneity drivers (multi-feature X) at the primary
        # horizon: which predetermined cell characteristics drive the CATE.
        drivers = estimate_cate_drivers(
            panel,
            exposure,
            shock,
            shock_col=_SHOCK_COL,
            horizon=_DRIVER_HORIZON,
            seed=settings.inference.seed,
        )
        store.write_table(
            "cate_drivers",
            pl.DataFrame(
                {
                    "horizon": _DRIVER_HORIZON,
                    "feature": drivers.features,
                    "blp_coef": drivers.blp_coef,
                    "importance": drivers.importances,
                }
            ),
        )

        summary: dict[str, float] = {}
        for r in results:
            summary[f"linear_ate_h{r.horizon}"] = r.linear_ate
            summary[f"placebo_ate_h{r.horizon}"] = r.placebo_ate
        top_idx = max(
            range(len(drivers.importances)), key=lambda i: drivers.importances[i]
        )
        summary["cate_top_driver_importance"] = drivers.importances[top_idx]
        summary["cate_top_exposure_tercile"] = drivers.cate_top_exposure_tercile
        summary["cate_bottom_exposure_tercile"] = drivers.cate_bottom_exposure_tercile
        return summary


def main() -> None:
    """Build the heterogeneity estimates and print a summary (entry point)."""
    for key, value in build_heterogeneity().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
