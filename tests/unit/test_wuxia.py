"""Wu-Xia / EFFR policy-rate splice logic."""

from __future__ import annotations

import datetime as dt

import polars as pl

from cil.config import ZlbWindow
from cil.data import wuxia

_WINDOWS = (ZlbWindow(start=dt.date(2008, 12, 1), end=dt.date(2015, 12, 1)),)


def _frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "date": [dt.date(2007, 6, 1), dt.date(2014, 6, 1), dt.date(2010, 1, 1)],
            "effr": [5.0, 0.1, 0.2],
            # No shadow value for the 2010-01 ZLB month -> must fall back to EFFR.
            "shadow_rate": [None, -2.8, None],
        }
    )


def test_splice_uses_shadow_inside_window() -> None:
    out = wuxia.splice(_frame(), _WINDOWS, use_shadow_rate=True)
    by_date = {r["date"]: r for r in out.to_dicts()}
    assert by_date[dt.date(2014, 6, 1)]["policy_rate"] == -2.8
    assert by_date[dt.date(2014, 6, 1)]["is_zlb_splice"] is True


def test_splice_uses_effr_outside_window() -> None:
    out = wuxia.splice(_frame(), _WINDOWS, use_shadow_rate=True)
    by_date = {r["date"]: r for r in out.to_dicts()}
    assert by_date[dt.date(2007, 6, 1)]["policy_rate"] == 5.0
    assert by_date[dt.date(2007, 6, 1)]["is_zlb_splice"] is False


def test_splice_falls_back_when_shadow_missing() -> None:
    out = wuxia.splice(_frame(), _WINDOWS, use_shadow_rate=True)
    by_date = {r["date"]: r for r in out.to_dicts()}
    assert by_date[dt.date(2010, 1, 1)]["policy_rate"] == 0.2
    assert by_date[dt.date(2010, 1, 1)]["is_zlb_splice"] is False


def test_splice_gate_disables_shadow() -> None:
    out = wuxia.splice(_frame(), _WINDOWS, use_shadow_rate=False)
    assert out["is_zlb_splice"].sum() == 0
    assert out.filter(pl.col("date") == dt.date(2014, 6, 1))[
        "policy_rate"
    ].to_list() == [0.1]
