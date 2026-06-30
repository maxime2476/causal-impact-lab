"""QCEW bulk flat-file ingestion (pre-2014 history extension).

The QCEW open-data API serves only 2014 onward; the annual ``by_area`` bulk zips
reach back to 1990 (NAICS reconstructed). Each zip holds one CSV per area with
all four quarters and all aggregation levels, so a single download per year
yields the state-level cells at any configured ``agglvl_code`` (e.g. 53
supersector, 55 NAICS 3-digit). State "Statewide" files are selected; ownership
components are summed and disclosure suppression is flagged, exactly as in the
API path (:mod:`cil.data.qcew`).

NAICS classification revised in 2002/2007/2012/2017; 3-digit codes are largely
stable across these, but the residual breaks are a documented caveat
(see ``docs/data.md``).
"""

from __future__ import annotations

import io
import re
import zipfile

import httpx
import polars as pl

from cil.data import http

# Statewide area files inside the zip, e.g. "... 06000 California -- Statewide.csv".
_STATE_FILE = re.compile(r"\b(\d{2}000)\b.*Statewide\.csv$", re.IGNORECASE)
_OWNERSHIP_CODES = (1, 2, 3, 5)
_MAX_STATE_FIPS = 56
# QCEW "Unclassified" buckets across aggregation levels (not real sectors).
_UNCLASSIFIED = frozenset({"1029", "99", "999", "9999", "99999", "999999"})


def fetch_raw_year(
    client: httpx.Client, bulk_template: str, year: int
) -> tuple[bytes, str, dict[str, str]]:
    """Download one year's QCEW ``by_area`` bulk zip.

    Parameters
    ----------
    client
        HTTP client.
    bulk_template
        URL template with a ``{year}`` field.
    year
        Calendar year.

    Returns
    -------
    content : bytes
        Raw zip payload.
    url : str
        The request URL.
    params : dict of str to str
        Selection keys, for provenance.
    """
    url = bulk_template.format(year=year)
    response = http.fetch(client, url)
    return response.content, url, {"year": str(year)}


def parse_year(content: bytes, year: int, *, aggregation_level: int) -> pl.DataFrame:
    """Parse a year's bulk zip into monthly state cells at *aggregation_level*.

    Parameters
    ----------
    content
        Raw zip bytes from :func:`fetch_raw_year`.
    year
        Calendar year of the zip.
    aggregation_level
        QCEW ``agglvl_code`` to extract (e.g. 53 supersector, 55 NAICS 3-digit).

    Returns
    -------
    polars.DataFrame
        Columns ``state_fips``, ``supersector_code`` (the industry code at the
        chosen level), ``date``, ``employment``, ``suppressed``.
    """
    archive = zipfile.ZipFile(io.BytesIO(content))
    frames: list[pl.DataFrame] = []
    for name in archive.namelist():
        if not _STATE_FILE.search(name):
            continue
        raw = pl.read_csv(
            io.BytesIO(archive.read(name)),
            infer_schema_length=10000,
            schema_overrides={"area_fips": pl.Utf8, "industry_code": pl.Utf8},
        )
        rows = (
            raw.filter(
                (pl.col("agglvl_code") == aggregation_level)
                & (pl.col("own_code").is_in(_OWNERSHIP_CODES))
            )
            .with_columns(
                state_fips=pl.col("area_fips").str.slice(0, 2),
                suppressed_component=pl.col("disclosure_code")
                .cast(pl.Utf8)
                .fill_null("")
                != "",
            )
            .filter(pl.col("state_fips").cast(pl.Int32) <= _MAX_STATE_FIPS)
        )
        if rows.height:
            frames.append(rows)
    if not frames:
        return pl.DataFrame(
            schema={
                "state_fips": pl.Utf8,
                "supersector_code": pl.Utf8,
                "date": pl.Date,
                "employment": pl.Float64,
                "suppressed": pl.Boolean,
            }
        )
    combined = pl.concat(frames)
    months = []
    for quarter in (1, 2, 3, 4):
        month_start = (quarter - 1) * 3 + 1
        q = combined.filter(pl.col("qtr") == quarter)
        for offset in range(3):
            col = f"month{offset + 1}_emplvl"
            months.append(
                q.select(
                    "state_fips",
                    supersector_code=pl.col("industry_code"),
                    date=pl.date(year, month_start + offset, 1),
                    emplvl=pl.col(col).cast(pl.Float64),
                    suppressed_component=pl.col("suppressed_component"),
                )
            )
    stacked = pl.concat(months).filter(~pl.col("supersector_code").is_in(_UNCLASSIFIED))
    return (
        stacked.group_by(["state_fips", "supersector_code", "date"])
        .agg(
            employment=pl.col("emplvl").sum(),
            suppressed=pl.col("suppressed_component").any(),
        )
        .sort(["state_fips", "supersector_code", "date"])
    )
