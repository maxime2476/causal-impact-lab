"""ALFRED point-in-time parsing and the no-look-ahead invariant.

Uses a small synthetic ALFRED payload (no network); the construction mirrors the
real API shape: each observation carries a ``realtime_start`` vintage date.
"""

from __future__ import annotations

import datetime as dt
import json

import polars as pl

from cil.data import alfred


def _payload() -> bytes:
    """A two-period series with one revision to the first period."""
    observations = [
        # Reference 2020-01: first release 100 on 2020-02-01, revised to 105.
        {"date": "2020-01-01", "realtime_start": "2020-02-01", "value": "100"},
        {"date": "2020-01-01", "realtime_start": "2020-03-01", "value": "105"},
        # Reference 2020-02: first released 2020-03-01.
        {"date": "2020-02-01", "realtime_start": "2020-03-01", "value": "200"},
        # A missing value encoded as ".".
        {"date": "2020-03-01", "realtime_start": "2020-04-01", "value": "."},
    ]
    return json.dumps({"observations": observations}).encode()


def test_parse_pit_shape_and_missing() -> None:
    pit = alfred.parse_pit(_payload(), "TEST")
    assert pit.height == 4
    assert set(pit.columns) == {"series_id", "reference_date", "vintage_date", "value"}
    missing = pit.filter(pl.col("reference_date") == dt.date(2020, 3, 1))
    assert missing["value"].to_list() == [None]


def test_latest_from_pit_takes_newest_vintage() -> None:
    latest = alfred.latest_from_pit(alfred.parse_pit(_payload(), "TEST"))
    jan = latest.filter(pl.col("reference_date") == dt.date(2020, 1, 1))
    assert jan["value"].to_list() == [105.0]  # revised value, not first release


def test_as_of_uses_no_future_vintages() -> None:
    pit = alfred.parse_pit(_payload(), "TEST")
    # As known on 2020-02-15: only the first release of 2020-01 is available.
    early = alfred.as_of(pit, dt.date(2020, 2, 15))
    assert early.sort("reference_date").to_dicts() == [
        {"series_id": "TEST", "reference_date": dt.date(2020, 1, 1), "value": 100.0}
    ]
    # As known on 2020-03-15: 2020-01 revised, 2020-02 now visible.
    later = alfred.as_of(pit, dt.date(2020, 3, 15))
    values = dict(
        zip(later["reference_date"].to_list(), later["value"].to_list(), strict=True)
    )
    assert values[dt.date(2020, 1, 1)] == 105.0
    assert values[dt.date(2020, 2, 1)] == 200.0


def test_pit_invariant_monotone_information() -> None:
    """Later as-of dates reveal a (weak) superset of reference periods."""
    pit = alfred.parse_pit(_payload(), "TEST")
    refs_early = set(
        alfred.as_of(pit, dt.date(2020, 2, 15))["reference_date"].to_list()
    )
    refs_late = set(alfred.as_of(pit, dt.date(2020, 3, 15))["reference_date"].to_list())
    assert refs_early <= refs_late
