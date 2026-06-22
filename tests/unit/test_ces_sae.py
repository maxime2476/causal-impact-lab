"""CES-SAE state cross-check parsing (thousands -> persons, FIPS mapping)."""

from __future__ import annotations

import datetime as dt
import json

from cil.data import ces_sae


def _payload() -> bytes:
    observations = [
        {"date": "2020-01-01", "realtime_start": "2020-02-01", "value": "1000.0"},
        {"date": "2020-02-01", "realtime_start": "2020-03-01", "value": "1010.5"},
    ]
    return json.dumps({"observations": observations}).encode()


def test_parse_state_converts_thousands_and_tags_metadata() -> None:
    out = ces_sae.parse_state(_payload(), "06").sort("date")
    assert out["state_fips"].unique().to_list() == ["06"]
    assert out["supersector_code"].unique().to_list() == ["00"]
    assert out["seasonal"].unique().to_list() == ["S"]
    assert out["employment"].to_list() == [1_000_000.0, 1_010_500.0]
    assert out["date"].to_list() == [dt.date(2020, 1, 1), dt.date(2020, 2, 1)]


def test_fred_series_id() -> None:
    assert ces_sae.fred_series_id("CA") == "CANA"
    assert ces_sae.STATE_FIPS_ABBR["06"] == "CA"
