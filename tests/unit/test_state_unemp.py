"""State unemployment-rate parsing (synthetic ALFRED payload, no network)."""

from __future__ import annotations

import datetime as dt
import json

from cil.data import state_unemp


def test_series_id() -> None:
    assert state_unemp.series_id("CA") == "CAUR"
    assert state_unemp.series_id("TX") == "TXUR"


def test_parse_state_columns_and_values() -> None:
    payload = {
        "observations": [
            {"date": "2015-01-01", "realtime_start": "2015-02-01", "value": "6.5"},
            {"date": "2015-02-01", "realtime_start": "2015-03-01", "value": "6.3"},
            {"date": "2015-03-01", "realtime_start": "2015-04-01", "value": "."},
        ]
    }
    # state_fips 06 -> CA (present in STATE_FIPS_ABBR).
    df = state_unemp.parse_state(json.dumps(payload).encode(), "06")
    assert df.columns == ["state_fips", "date", "unemployment"]
    assert df["state_fips"].unique().to_list() == ["06"]
    jan = df.filter(df["date"] == dt.date(2015, 1, 1))
    assert jan["unemployment"].to_list() == [6.5]
