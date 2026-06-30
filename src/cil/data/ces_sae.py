"""BLS CES State and Area Employment (SAE) ingestion -- state cross-check.

Two products:

* **State total nonfarm** from FRED's mirror (``{ABBR}NA``), the original
  aggregate cross-check (``supersector_code == "00"``).
* **State x supersector** seasonally-adjusted All-Employees from the BLS ``sm``
  flat files (per-state data files), mapped to QCEW supersectors, for the
  supersector-level reconciliation against QCEW (Tier 0.3).

CES values are published in thousands and converted to persons to match QCEW.
"""

from __future__ import annotations

import io

import httpx
import polars as pl

from cil.data import alfred, http
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


# --- State x supersector (BLS sm flat files) ------------------------------

#: CES supersector code -> QCEW supersector code (for the reconciliation).
CES_TO_QCEW_SUPERSECTOR: dict[str, str] = {
    "10": "1011",  # Mining and Logging <-> Natural Resources and Mining
    "20": "1012",  # Construction
    "30": "1013",  # Manufacturing
    "40": "1021",  # Trade, Transportation, and Utilities
    "50": "1022",  # Information
    "55": "1023",  # Financial Activities
    "60": "1024",  # Professional and Business Services
    "65": "1025",  # Education and Health Services
    "70": "1026",  # Leisure and Hospitality
    "80": "1027",  # Other Services
    "90": "1028",  # Government <-> Public Administration
}

_SM_SERIES = "/sm/sm.series"
_SM_DIR = "/sm/"
_KEYCOLS = (
    "series_id",
    "state_code",
    "area_code",
    "supersector_code",
    "industry_code",
    "data_type_code",
    "seasonal",
)


def fetch_sm_series_catalog(client: httpx.Client, bls_flat_base: str) -> bytes:
    """Download the BLS ``sm.series`` catalog. Returns the raw bytes."""
    return http.fetch(client, bls_flat_base + _SM_SERIES).content


def target_supersector_series(catalog: bytes) -> pl.DataFrame:
    """Select state x supersector SA All-Employees series from the catalog.

    Parameters
    ----------
    catalog
        Raw ``sm.series`` bytes.

    Returns
    -------
    polars.DataFrame
        Columns ``series_id``, ``state_fips``, ``supersector_code`` (QCEW), for
        statewide, All-Employees, seasonally-adjusted, supersector-level series.
    """
    df = pl.read_csv(
        io.BytesIO(catalog),
        separator="\t",
        infer_schema_length=50000,
        schema_overrides={c: pl.Utf8 for c in _KEYCOLS},
    )
    df = df.rename({c: c.strip() for c in df.columns})
    df = df.with_columns(pl.col(c).str.strip_chars().alias(c) for c in _KEYCOLS)
    valid_states = set(STATE_FIPS_ABBR)
    return (
        df.filter(
            (pl.col("area_code") == "00000")
            & (pl.col("data_type_code") == "01")
            & (pl.col("seasonal") == "S")
            # Supersector aggregate row: industry_code == supersector_code+"000000".
            & (pl.col("industry_code") == pl.col("supersector_code") + "000000")
            & pl.col("supersector_code").is_in(list(CES_TO_QCEW_SUPERSECTOR))
            & pl.col("state_code").is_in(list(valid_states))
        )
        .select(
            "series_id",
            state_fips=pl.col("state_code"),
            supersector_code=pl.col("supersector_code").replace(
                CES_TO_QCEW_SUPERSECTOR
            ),
        )
        .unique()
    )


def list_state_data_files(client: httpx.Client, bls_flat_base: str) -> list[str]:
    """Return the per-state full-history ``sm.data.N.State`` file names.

    Parsed from the actual directory hrefs (the per-supersector ``.Current``
    topic files hold only recent years; ``AllData`` is the 540 MB everything
    file). The per-state files are small (~1 MB each) and carry full history.
    """
    import re

    listing = http.fetch(client, bls_flat_base + _SM_DIR).text
    hrefs = re.findall(r'HREF="([^"]+)"', listing, re.IGNORECASE)
    names = {h.rsplit("/", 1)[-1] for h in hrefs if "/sm.data." in h.lower()}
    return sorted(n for n in names if not n.endswith(".Current") and "AllData" not in n)


def parse_sm_data(content: bytes, targets: pl.DataFrame) -> pl.DataFrame:
    """Parse one ``sm.data.*`` file into monthly cells for the target series.

    Parameters
    ----------
    content
        Raw tab-separated ``sm.data.*`` bytes.
    targets
        Output of :func:`target_supersector_series`.

    Returns
    -------
    polars.DataFrame
        Columns ``state_fips``, ``supersector_code`` (QCEW), ``date``,
        ``employment`` (persons), for rows matching the target series.
    """
    # BLS flat-file headers are whitespace-padded, so read everything as strings
    # (header names stripped below) and cast explicitly afterwards.
    raw = pl.read_csv(io.BytesIO(content), separator="\t", infer_schema=False)
    raw = raw.rename({c: c.strip() for c in raw.columns}).with_columns(
        pl.col("series_id").str.strip_chars(),
        pl.col("period").str.strip_chars(),
        value=pl.col("value").str.strip_chars().cast(pl.Float64, strict=False),
        year=pl.col("year").str.strip_chars().cast(pl.Int32),
    )
    matched = raw.join(targets, on="series_id", how="inner").filter(
        pl.col("period").str.starts_with("M")
        & (pl.col("period") != "M13")
        & pl.col("value").is_not_null()
    )
    if matched.height == 0:
        return pl.DataFrame(
            schema={
                "state_fips": pl.Utf8,
                "supersector_code": pl.Utf8,
                "date": pl.Date,
                "employment": pl.Float64,
            }
        )
    return matched.select(
        "state_fips",
        "supersector_code",
        date=pl.date(
            pl.col("year"), pl.col("period").str.slice(1, 2).cast(pl.Int32), 1
        ),
        employment=pl.col("value") * _THOUSANDS,
    ).sort(["state_fips", "supersector_code", "date"])
