"""BRW shock-series parsing."""

from __future__ import annotations

import datetime as dt

from cil.data import brw


def _csv() -> bytes:
    lines = [
        "month,BRW_monthly (updated)",
        "1994m1,0.5",
        "1994m2,-0.3",
        "1994m12,0.0",
        ",",  # trailing blank row to be dropped
    ]
    return ("\n".join(lines) + "\n").encode()


def test_parse_brw_months_and_values() -> None:
    out = brw.parse(_csv()).sort("date")
    assert out.height == 3
    assert out["date"].to_list() == [
        dt.date(1994, 1, 1),
        dt.date(1994, 2, 1),
        dt.date(1994, 12, 1),
    ]
    assert out["brw_monthly"].to_list() == [0.5, -0.3, 0.0]
