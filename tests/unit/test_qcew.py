"""QCEW parsing: ownership summation, suppression flagging, and footprint."""

from __future__ import annotations

import datetime as dt

import polars as pl

from cil.data import qcew

_HEADER = (
    "area_fips,own_code,industry_code,agglvl_code,disclosure_code,"
    "month1_emplvl,month2_emplvl,month3_emplvl"
)


def _csv() -> bytes:
    rows = [
        # State 01, Manufacturing: private disclosed, local withheld, federal small.
        "01000,5,1013,53,,100,110,120",
        "01000,3,1013,53,N,0,0,0",
        "01000,1,1013,53,,10,10,10",
        # State 02, Manufacturing: all disclosed.
        "02000,5,1013,53,,50,50,50",
        # Rows that must be excluded:
        "US000,5,1013,53,,9999,9999,9999",  # national
        "01001,5,1013,53,,7,7,7",  # county
        "72000,5,1013,53,,5,5,5",  # territory (PR)
        "01000,5,1013,55,,3,3,3",  # wrong agglvl
    ]
    return ("\n".join([_HEADER, *rows]) + "\n").encode()


def test_parse_industry_sums_ownership_and_flags_suppression() -> None:
    cells = qcew.parse_industry(_csv(), 2020, 1)
    assert cells.height == 6  # 2 states x 3 months
    al_jan = cells.filter(
        (pl.col("state_fips") == "01") & (pl.col("date") == dt.date(2020, 1, 1))
    )
    assert al_jan["employment"].to_list() == [110.0]  # 100 + 0 + 10
    assert al_jan["suppressed"].to_list() == [True]
    ak_jan = cells.filter(
        (pl.col("state_fips") == "02") & (pl.col("date") == dt.date(2020, 1, 1))
    )
    assert ak_jan["employment"].to_list() == [50.0]
    assert ak_jan["suppressed"].to_list() == [False]


def test_parse_industry_maps_quarter_to_months() -> None:
    cells = qcew.parse_industry(_csv(), 2020, 2)  # Q2 -> Apr, May, Jun
    months = sorted(cells["date"].unique().to_list())
    assert months == [dt.date(2020, 4, 1), dt.date(2020, 5, 1), dt.date(2020, 6, 1)]


def test_suppression_footprint() -> None:
    fp = qcew.suppression_footprint(qcew.parse_industry(_csv(), 2020, 1))
    al = fp.filter(pl.col("state_fips") == "01")
    assert al["suppressed_fraction"].to_list() == [1.0]
    ak = fp.filter(pl.col("state_fips") == "02")
    assert ak["suppressed_fraction"].to_list() == [0.0]
