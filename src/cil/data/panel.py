"""Assembly of the analysis-ready cell panel.

Turns the QCEW state-by-supersector employment cells into the balanced-enough
panel used by the headline estimators: cells with insufficient coverage (too
many months of zero/withheld employment) are dropped and reported, log
employment is computed, and the result is validated against the panel contract.
"""

from __future__ import annotations

import polars as pl

from cil.data.schemas import PANEL_CELL_SCHEMA, validate


def build_cell_panel(
    cells: pl.DataFrame, coverage_min_fraction: float
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build the analysis-ready cell panel and a dropped-cell report.

    Parameters
    ----------
    cells
        QCEW cell frame with columns ``state_fips``, ``supersector_code``,
        ``date``, ``employment``, ``suppressed``.
    coverage_min_fraction
        Minimum fraction of observed months with strictly positive employment
        required to retain a (state, supersector) cell.

    Returns
    -------
    panel : polars.DataFrame
        Frame validated against
        :data:`cil.data.schemas.PANEL_CELL_SCHEMA`.
    dropped : polars.DataFrame
        One row per dropped (state, supersector) with its coverage fraction.
    """
    coverage = (
        cells.group_by(["state_fips", "supersector_code"])
        .agg(
            n_months=pl.len(),
            n_positive=(pl.col("employment") > 0).sum(),
        )
        .with_columns(coverage=pl.col("n_positive") / pl.col("n_months"))
    )
    kept_keys = coverage.filter(pl.col("coverage") >= coverage_min_fraction).select(
        ["state_fips", "supersector_code"]
    )
    dropped = coverage.filter(pl.col("coverage") < coverage_min_fraction).sort(
        ["state_fips", "supersector_code"]
    )
    panel = (
        cells.join(kept_keys, on=["state_fips", "supersector_code"], how="inner")
        .filter(pl.col("employment") > 0)
        .with_columns(
            unit_id=pl.col("state_fips") + "_" + pl.col("supersector_code"),
            log_employment=pl.col("employment").log(),
        )
        .select(
            "unit_id",
            "state_fips",
            "supersector_code",
            "date",
            "employment",
            "log_employment",
        )
        .sort(["unit_id", "date"])
    )
    return validate(PANEL_CELL_SCHEMA, panel), dropped
