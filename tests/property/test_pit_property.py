"""Property: the as-of construction never leaks future vintages.

For any vintage history and any as-of date ``V``, :func:`cil.data.alfred.as_of`
must return, for each reference period that has at least one vintage on or before
``V``, the value from the latest such vintage -- and must omit periods first
published after ``V``.
"""

from __future__ import annotations

import datetime as dt

import polars as pl
from hypothesis import given
from hypothesis import strategies as st

from cil.data import alfred

_REFERENCES = [dt.date(2020, m, 1) for m in range(1, 7)]
_VINTAGES = [dt.date(2020, m, 15) for m in range(1, 10)]


@st.composite
def _histories(draw: st.DrawFn) -> tuple[pl.DataFrame, dt.date]:
    """Draw a valid PIT frame (unique series/ref/vintage) and an as-of date."""
    rows: list[dict[str, object]] = []
    for ref in _REFERENCES:
        chosen = draw(st.lists(st.sampled_from(_VINTAGES), unique=True, max_size=4))
        for i, vintage in enumerate(chosen):
            rows.append(
                {
                    "series_id": "S",
                    "reference_date": ref,
                    "vintage_date": vintage,
                    "value": float(i),
                }
            )
    frame = pl.DataFrame(
        rows,
        schema={
            "series_id": pl.Utf8,
            "reference_date": pl.Date,
            "vintage_date": pl.Date,
            "value": pl.Float64,
        },
    )
    as_of_date = draw(st.sampled_from(_VINTAGES))
    return frame, as_of_date


@given(_histories())
def test_as_of_matches_latest_vintage_no_leak(
    case: tuple[pl.DataFrame, dt.date],
) -> None:
    frame, as_of_date = case
    result = alfred.as_of(frame, as_of_date)
    seen = dict(
        zip(result["reference_date"].to_list(), result["value"].to_list(), strict=True)
    )
    for ref in _REFERENCES:
        visible = frame.filter(
            (pl.col("reference_date") == ref) & (pl.col("vintage_date") <= as_of_date)
        )
        if visible.height == 0:
            assert ref not in seen  # not yet published -> absent
            continue
        latest = visible.sort("vintage_date").tail(1)
        assert seen[ref] == latest["value"].item()
        # No vintage after the as-of date contributed.
        assert latest["vintage_date"].item() <= as_of_date
