"""Updated Romer-Romer narrative-shock .dta parsing (synthetic, no network)."""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from cil.data import rr


def _dta() -> bytes:
    pdf = pd.DataFrame(
        {
            "Date": pd.to_datetime(["1994-01-01", "1994-04-01", "1994-07-01"]),
            "MPORGQ": [0.10, -0.20, 0.30],
            "MP08Q": [0.11, -0.21, 0.31],
            "MP12Q": [0.12, -0.22, 0.32],
            "EXTRA": [9.0, 9.0, 9.0],  # extra column, ignored
        }
    )
    buf = io.BytesIO()
    pdf.to_stata(buf, write_index=False)
    return buf.getvalue()


def test_parse_renames_and_selects() -> None:
    df = rr.parse(_dta())
    assert df.columns == ["date", "rr_org", "rr08", "rr12"]
    assert df.height == 3


def test_parse_values() -> None:
    df = rr.parse(_dta())
    first = df.filter(df["date"] == dt.date(1994, 1, 1))
    assert first["rr_org"].to_list() == [0.10]
    assert first["rr12"].to_list() == [0.12]
