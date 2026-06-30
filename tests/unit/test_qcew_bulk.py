"""QCEW bulk (pre-2014) parser: synthetic zip, no network."""

from __future__ import annotations

import datetime as dt
import io
import zipfile

import polars as pl

from cil.data import qcew_bulk

_HEADER = (
    "area_fips,own_code,industry_code,agglvl_code,disclosure_code,year,qtr,"
    "month1_emplvl,month2_emplvl,month3_emplvl"
)


def _state_csv(area: str, rows: list[str]) -> bytes:
    return ("\n".join([_HEADER, *rows]) + "\n").encode()


def _zip_bytes() -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        # California statewide, NAICS 3-digit (agglvl 55): private + a withheld local.
        z.writestr(
            "2010.q1-q4.by_area/2010.q1-q4 06000 California -- Statewide.csv",
            _state_csv(
                "06000",
                [
                    "06000,5,443,55,,2010,1,100,110,120",
                    "06000,3,443,55,N,2010,1,0,0,0",
                    "06000,5,443,55,,2010,2,130,140,150",
                    "06000,5,999,55,,2010,1,5,5,5",  # Unclassified -> dropped
                ],
            ),
        )
        # A county file (must be ignored) and a wrong agglvl row.
        z.writestr(
            "2010.q1-q4.by_area/2010.q1-q4 06001 Alameda County.csv",
            _state_csv("06001", ["06001,5,443,55,,2010,1,9,9,9"]),
        )
    return buf.getvalue()


def test_parse_year_extracts_state_3digit_cells() -> None:
    cells = qcew_bulk.parse_year(_zip_bytes(), 2010, aggregation_level=55)
    # Only California statewide; county file ignored.
    assert cells["state_fips"].unique().to_list() == ["06"]
    jan = cells.filter(pl.col("date") == dt.date(2010, 1, 1))
    assert jan["employment"].to_list() == [100.0]  # 100 private + 0 withheld local
    assert jan["suppressed"].to_list() == [True]
    # Q1 maps to Jan/Feb/Mar and Q2 to Apr/May/Jun (the fixture has both).
    assert sorted(cells["date"].unique().to_list()) == [
        dt.date(2010, m, 1) for m in range(1, 7)
    ]
    apr = cells.filter(pl.col("date") == dt.date(2010, 4, 1))
    assert apr["employment"].to_list() == [130.0]  # Q2 month1, fully disclosed
    assert apr["suppressed"].to_list() == [False]
    # "Unclassified" (999) is excluded.
    assert cells["supersector_code"].unique().to_list() == ["443"]


def test_parse_year_empty_when_aggregation_absent() -> None:
    cells = qcew_bulk.parse_year(_zip_bytes(), 2010, aggregation_level=99)
    assert cells.height == 0
    assert set(cells.columns) == {
        "state_fips",
        "supersector_code",
        "date",
        "employment",
        "suppressed",
    }
