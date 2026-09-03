"""State unemployment-rate ingestion (BLS LAUS via FRED).

Each state's seasonally-adjusted unemployment rate is the FRED series ``{ABBR}UR``
(e.g. ``CAUR``). Used as a second, state-level outcome for the relative design
(state Bartik exposure x shock).
"""

from __future__ import annotations

import httpx
import polars as pl

from cil.data import alfred
from cil.data.ces_sae import STATE_FIPS_ABBR


def series_id(state_abbr: str) -> str:
    """FRED series id for a state's unemployment rate (e.g. ``CAUR``)."""
    return f"{state_abbr}UR"


def fetch_raw_state(
    client: httpx.Client, fred_base: str, api_key: str, state_abbr: str
) -> tuple[bytes, str, dict[str, str]]:
    """Fetch one state's unemployment-rate series. Returns ``(content, url, ...)``."""
    return alfred.fetch_raw_series(client, fred_base, api_key, series_id(state_abbr))


def parse_state(content: bytes, state_fips: str) -> pl.DataFrame:
    """Parse a state's FRED payload into ``state_fips``, ``date``, ``unemployment``."""
    state_abbr = STATE_FIPS_ABBR[state_fips]
    latest = alfred.latest_from_pit(alfred.parse_pit(content, series_id(state_abbr)))
    return latest.select(
        state_fips=pl.lit(state_fips, dtype=pl.Utf8),
        date=pl.col("reference_date"),
        unemployment=pl.col("value").cast(pl.Float64),
    )
