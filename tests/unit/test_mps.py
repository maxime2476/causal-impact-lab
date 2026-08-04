"""Bauer-Swanson MPS workbook parsing (synthetic xlsx, no network)."""

from __future__ import annotations

import datetime as dt
import io

import pandas as pd

from cil.data import mps


def _workbook() -> bytes:
    monthly = pd.DataFrame(
        {
            "Year": [1994, 1994],
            "Month": [1, 2],
            "MPS": [0.01, -0.02],
            "MPS_ORTH": [0.005, -0.015],
            "NFP_SURP": [0.1, 0.2],  # extra column, ignored
        }
    )
    fomc = pd.DataFrame(
        {
            "Date": [dt.date(1994, 2, 4), dt.date(1994, 3, 22)],
            "SP500": [0.3, -0.4],
            "MPS": [0.012, -0.024],
            "MPS_ORTH": [0.006, -0.018],
            "TNOTE10": [0.01, -0.02],  # extra column, ignored
        }
    )
    buf = io.BytesIO()
    with pd.ExcelWriter(buf) as writer:
        monthly.to_excel(writer, sheet_name="Monthly (update 2023)", index=False)
        fomc.to_excel(writer, sheet_name="FOMC (update 2023)", index=False)
        monthly.to_excel(writer, sheet_name="Monthly (original)", index=False)
    return buf.getvalue()


def test_parse_monthly() -> None:
    df = mps.parse_monthly(_workbook())
    assert df.columns == ["date", "mps", "mps_orth"]
    assert df.height == 2
    jan = df.filter(df["date"] == dt.date(1994, 1, 1))
    assert jan["mps"].to_list() == [0.01]
    assert jan["mps_orth"].to_list() == [0.005]


def test_parse_fomc_has_equity() -> None:
    df = mps.parse_fomc(_workbook())
    assert set(df.columns) == {"date", "mps", "mps_orth", "sp500"}
    assert df.height == 2
    first = df.filter(df["date"] == dt.date(1994, 2, 4))
    assert first["sp500"].to_list() == [0.3]
    assert first["mps"].to_list() == [0.012]


def test_pick_sheet_prefers_update() -> None:
    assert (
        mps._pick_sheet(
            ["Monthly (original)", "Monthly (update 2023)"], "Monthly (update"
        )
        == "Monthly (update 2023)"
    )
