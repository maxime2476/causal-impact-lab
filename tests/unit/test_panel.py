"""Cell-panel assembly: coverage-based dropping and log employment."""

from __future__ import annotations

import datetime as dt
import math

import polars as pl

from cil.data import panel


def _cells() -> pl.DataFrame:
    months = [dt.date(2020, m, 1) for m in range(1, 5)]
    rows = []
    # Well-covered cell: all positive.
    for m in months:
        rows.append(("01", "1013", m, 100.0))
    # Poorly-covered cell: only one positive month (coverage 0.25).
    for i, m in enumerate(months):
        rows.append(("01", "1022", m, 100.0 if i == 0 else 0.0))
    return pl.DataFrame(
        rows,
        schema=["state_fips", "supersector_code", "date", "employment"],
        orient="row",
    )


def test_build_cell_panel_drops_low_coverage() -> None:
    kept, dropped = panel.build_cell_panel(_cells(), coverage_min_fraction=0.95)
    assert kept["unit_id"].unique().to_list() == ["01_1013"]
    assert dropped.height == 1
    assert dropped["supersector_code"].to_list() == ["1022"]


def test_build_cell_panel_computes_log_and_unit_id() -> None:
    kept, _ = panel.build_cell_panel(_cells(), coverage_min_fraction=0.95)
    assert kept["unit_id"].unique().to_list() == ["01_1013"]
    assert kept["log_employment"][0] == math.log(100.0)
    assert kept.height == 4


def test_build_cell_panel_keeps_all_when_threshold_zero() -> None:
    kept, dropped = panel.build_cell_panel(_cells(), coverage_min_fraction=0.0)
    # The poorly-covered cell's zero months are still dropped row-wise.
    assert set(kept["unit_id"].unique().to_list()) == {"01_1013", "01_1022"}
    assert dropped.height == 0
