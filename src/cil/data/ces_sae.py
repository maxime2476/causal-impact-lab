"""BLS CES State and Area Employment (SAE) ingestion -- state cross-check.

A secondary source used to cross-check QCEW state employment totals. We pull
total-nonfarm, seasonally-adjusted All-Employees by state from FRED's mirror of
the CES-SAE series (``{ABBR}NA``), reusing the FRED access already configured.

Supersector-level CES/QCEW reconciliation is deferred to the cross-check
robustness work; here the cell is total nonfarm (``supersector_code == "00"``).
CES values are published in thousands and converted to persons to match QCEW.
"""

from __future__ import annotations

import httpx
import polars as pl

from cil.data import alfred
from cil.data.schemas import CES_SAE_SCHEMA, validate

#: US state and DC FIPS codes mapped to their two-letter postal abbreviations.
STATE_FIPS_ABBR: dict[str, str] = {
    "01": "AL",
    "02": "AK",
    "04": "AZ",
    "05": "AR",
    "06": "CA",
    "08": "CO",
    "09": "CT",
    "10": "DE",
    "11": "DC",
    "12": "FL",
    "13": "GA",
    "15": "HI",
    "16": "ID",
    "17": "IL",
    "18": "IN",
    "19": "IA",
    "20": "KS",
    "21": "KY",
    "22": "LA",
    "23": "ME",
    "24": "MD",
    "25": "MA",
    "26": "MI",
    "27": "MN",
    "28": "MS",
    "29": "MO",
    "30": "MT",
    "31": "NE",
    "32": "NV",
    "33": "NH",
    "34": "NJ",
    "35": "NM",
    "36": "NY",
    "37": "NC",
    "38": "ND",
    "39": "OH",
    "40": "OK",
    "41": "OR",
    "42": "PA",
    "44": "RI",
    "45": "SC",
    "46": "SD",
    "47": "TN",
    "48": "TX",
    "49": "UT",
    "50": "VT",
    "51": "VA",
    "53": "WA",
    "54": "WV",
    "55": "WI",
    "56": "WY",
}

_THOUSANDS = 1000.0


def fred_series_id(state_abbr: str) -> str:
    """Return the FRED series id for total-nonfarm SA employment in a state."""
    return f"{state_abbr}NA"


def fetch_raw_state(
    client: httpx.Client,
    fred_base: str,
    api_key: str,
    state_abbr: str,
) -> tuple[bytes, str, dict[str, str]]:
    """Fetch one state's total-nonfarm series. Returns ``(content, url, params)``."""
    return alfred.fetch_raw_series(
        client, fred_base, api_key, fred_series_id(state_abbr)
    )


def parse_state(content: bytes, state_fips: str) -> pl.DataFrame:
    """Parse a state's FRED payload into CES-SAE cross-check rows.

    Parameters
    ----------
    content
        Raw JSON bytes from :func:`fetch_raw_state`.
    state_fips
        Two-digit FIPS code of the state.

    Returns
    -------
    polars.DataFrame
        Total-nonfarm rows (``supersector_code == "00"``, ``seasonal == "S"``),
        employment in persons, conforming to row-level
        :data:`cil.data.schemas.CES_SAE_SCHEMA`.
    """
    state_abbr = STATE_FIPS_ABBR[state_fips]
    pit = alfred.parse_pit(content, fred_series_id(state_abbr))
    latest = alfred.latest_from_pit(pit)
    rows = latest.select(
        state_fips=pl.lit(state_fips, dtype=pl.Utf8),
        supersector_code=pl.lit("00", dtype=pl.Utf8),
        date=pl.col("reference_date"),
        employment=pl.col("value") * _THOUSANDS,
        seasonal=pl.lit("S", dtype=pl.Utf8),
    )
    return validate(CES_SAE_SCHEMA, rows)
