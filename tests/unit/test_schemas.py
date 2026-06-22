"""Schema contracts accept conforming frames and reject violations."""

from __future__ import annotations

import datetime as dt

import pandera.errors
import polars as pl
import pytest

from cil.data import schemas


def test_panel_schema_accepts_valid() -> None:
    df = pl.DataFrame(
        {
            "unit_id": ["01_1013"],
            "state_fips": ["01"],
            "supersector_code": ["1013"],
            "date": [dt.date(2020, 1, 1)],
            "employment": [100.0],
            "log_employment": [4.605170185988092],
        }
    )
    out = schemas.validate(schemas.PANEL_CELL_SCHEMA, df)
    assert out.height == 1


def test_panel_schema_rejects_nonpositive_employment() -> None:
    df = pl.DataFrame(
        {
            "unit_id": ["01_1013"],
            "state_fips": ["01"],
            "supersector_code": ["1013"],
            "date": [dt.date(2020, 1, 1)],
            "employment": [0.0],
            "log_employment": [0.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        schemas.validate(schemas.PANEL_CELL_SCHEMA, df)


def test_macro_pit_schema_rejects_duplicate_keys() -> None:
    df = pl.DataFrame(
        {
            "series_id": ["A", "A"],
            "reference_date": [dt.date(2020, 1, 1), dt.date(2020, 1, 1)],
            "vintage_date": [dt.date(2020, 2, 1), dt.date(2020, 2, 1)],
            "value": [1.0, 2.0],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        schemas.validate(schemas.MACRO_PIT_SCHEMA, df)


def test_schema_rejects_unexpected_column() -> None:
    df = pl.DataFrame(
        {
            "date": [dt.date(2020, 1, 1)],
            "brw_monthly": [0.1],
            "extra": [1],
        }
    )
    with pytest.raises(pandera.errors.SchemaError):
        schemas.validate(schemas.BRW_SHOCK_SCHEMA, df)
