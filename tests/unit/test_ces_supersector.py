"""CES state x supersector ingestion and QCEW reconciliation (synthetic)."""

from __future__ import annotations

import datetime as dt

import polars as pl

from cil.data import ces_sae
from cil.robustness import ces_reconciliation as cr

_CATALOG = (
    b"series_id\tstate_code\tarea_code\tsupersector_code\tindustry_code\t"
    b"data_type_code\tseasonal\n"
    # CA manufacturing, statewide, All-Employees, SA, supersector aggregate -> keep
    b"SMS06000003000000001\t06\t00000\t30\t30000000\t01\tS\n"
    # wrong data_type (hours) -> drop
    b"SMS06000003000000002\t06\t00000\t30\t30000000\t02\tS\n"
    # not seasonally adjusted -> drop
    b"SMU06000003000000001\t06\t00000\t30\t30000000\t01\tU\n"
    # an MSA (area != 00000) -> drop
    b"SMS06310803000000001\t06\t31080\t30\t30000000\t01\tS\n"
)


def test_target_supersector_series_filters_and_maps() -> None:
    out = ces_sae.target_supersector_series(_CATALOG)
    assert out.height == 1
    row = out.to_dicts()[0]
    assert row["series_id"] == "SMS06000003000000001"
    assert row["state_fips"] == "06"
    assert row["supersector_code"] == "1013"  # CES 30 -> QCEW 1013


def test_parse_sm_data_to_monthly_persons() -> None:
    targets = ces_sae.target_supersector_series(_CATALOG)
    data = (
        b"series_id\tyear\tperiod\tvalue\tfootnote_codes\n"
        b"SMS06000003000000001\t2010\tM01\t100.0\t\n"
        b"SMS06000003000000001\t2010\tM02\t101.0\t\n"
        b"SMS06000003000000001\t2010\tM13\t100.5\t\n"  # annual avg -> excluded
        b"SMS99999999999999999\t2010\tM01\t9.0\t\n"  # not a target -> excluded
    )
    cells = ces_sae.parse_sm_data(data, targets)
    assert cells.height == 2  # M01, M02 only
    jan = cells.filter(pl.col("date") == dt.date(2010, 1, 1))
    assert jan["employment"].to_list() == [100000.0]  # thousands -> persons
    assert jan["supersector_code"].to_list() == ["1013"]


def test_aggregate_qcew_to_supersector() -> None:
    cells = pl.DataFrame(
        {
            "state_fips": ["06", "06", "06"],
            "supersector_code": ["331", "332", "236"],  # 33x -> 1013, 23x -> 1012
            "date": [dt.date(2010, 1, 1)] * 3,
            "employment": [100.0, 50.0, 70.0],
        }
    )
    agg = cr.aggregate_qcew_to_supersector(cells)
    by = dict(
        zip(agg["supersector_code"].to_list(), agg["employment"].to_list(), strict=True)
    )
    assert by["1013"] == 150.0  # 331 + 332
    assert by["1012"] == 70.0


def test_reconcile_perfect_agreement() -> None:
    months = [dt.date(2010 + i // 12, (i % 12) + 1, 1) for i in range(24)]
    emp = [100.0 + i for i in range(24)]
    ces = pl.DataFrame(
        {
            "state_fips": ["06"] * 24,
            "supersector_code": ["1013"] * 24,
            "date": months,
            "employment": emp,
        }
    )
    qcew = ces.with_columns(employment=pl.col("employment") * 1.0)  # identical
    recon = cr.reconcile(ces, qcew)
    assert recon.height == 1
    assert abs(recon["corr_level"][0] - 1.0) < 1e-9
