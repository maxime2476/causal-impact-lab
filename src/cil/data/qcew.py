"""QCEW state-by-supersector monthly employment ingestion.

The Quarterly Census of Employment and Wages publishes near-census employment
counts. We extract state-level NAICS *supersector* rows (``agglvl_code == 53``),
summing the ownership components (federal, state, local, private) into a total
employment figure per (state, supersector, month). Disclosure suppression is
recorded: if any ownership component is withheld, the summed total is a lower
bound and the cell is flagged.

The data is fetched via the *industry* slice endpoint (one supersector across
all areas per quarter), which is far fewer requests than per-area pulls. QCEW
NAICS files are not seasonally adjusted; that convention is carried downstream
explicitly (see ``docs/data.md``).
"""

from __future__ import annotations

import io

import httpx
import polars as pl

from cil.data import http

#: NAICS supersector codes published by QCEW at ``agglvl_code == 53``.
SUPERSECTOR_NAMES: dict[str, str] = {
    "1011": "Natural Resources and Mining",
    "1012": "Construction",
    "1013": "Manufacturing",
    "1021": "Trade, Transportation, and Utilities",
    "1022": "Information",
    "1023": "Financial Activities",
    "1024": "Professional and Business Services",
    "1025": "Education and Health Services",
    "1026": "Leisure and Hospitality",
    "1027": "Other Services",
    "1028": "Public Administration",
}

#: Ownership codes summed into the supersector total.
_OWNERSHIP_CODES = (1, 2, 3, 5)
_STATE_AREA_PATTERN = r"^\d{2}000$"
# Highest real US state FIPS (Wyoming = 56); excludes territories (PR=72, ...).
_MAX_STATE_FIPS = 56


def fetch_raw_industry(
    client: httpx.Client,
    qcew_area_template: str,
    supersector_code: str,
    year: int,
    quarter: int,
) -> tuple[bytes, str, dict[str, str]]:
    """Fetch one supersector's QCEW data for one quarter (all areas).

    Parameters
    ----------
    client
        HTTP client.
    qcew_area_template
        Area template URL; its host is reused to build the industry endpoint.
    supersector_code
        NAICS supersector code (a key of :data:`SUPERSECTOR_NAMES`).
    year
        Calendar year.
    quarter
        Quarter ``1``-``4``.

    Returns
    -------
    content : bytes
        Raw CSV payload.
    url : str
        The request URL.
    params : dict of str to str
        Selection keys, for provenance.
    """
    base = qcew_area_template.split("/data/api/")[0]
    url = f"{base}/data/api/{year}/{quarter}/industry/{supersector_code}.csv"
    response = http.fetch(client, url)
    params = {
        "supersector_code": supersector_code,
        "year": str(year),
        "quarter": str(quarter),
    }
    return response.content, url, params


def parse_industry(content: bytes, year: int, quarter: int) -> pl.DataFrame:
    """Parse a QCEW industry-quarter CSV into monthly state-supersector rows.

    Sums ownership components per (state, supersector, month) and flags any
    cell with a suppressed component.

    Parameters
    ----------
    content
        Raw CSV bytes from :func:`fetch_raw_industry`.
    year
        Calendar year of the file.
    quarter
        Quarter ``1``-``4`` of the file.

    Returns
    -------
    polars.DataFrame
        Columns ``state_fips``, ``supersector_code``, ``date``,
        ``employment``, ``suppressed``.
    """
    raw = pl.read_csv(
        io.BytesIO(content),
        infer_schema_length=10000,
        schema_overrides={"area_fips": pl.Utf8, "industry_code": pl.Utf8},
    )
    month_start = (quarter - 1) * 3 + 1
    long = (
        raw.filter(
            (pl.col("agglvl_code") == 53)
            & (pl.col("area_fips").str.contains(_STATE_AREA_PATTERN))
            & (pl.col("own_code").is_in(_OWNERSHIP_CODES))
        )
        .with_columns(
            state_fips=pl.col("area_fips").str.slice(0, 2),
            suppressed_component=pl.col("disclosure_code").cast(pl.Utf8).fill_null("")
            != "",
        )
        .filter(pl.col("state_fips").cast(pl.Int32) <= _MAX_STATE_FIPS)
    )
    months = []
    for offset in range(3):
        month = month_start + offset
        col = f"month{offset + 1}_emplvl"
        months.append(
            long.select(
                "state_fips",
                supersector_code=pl.col("industry_code"),
                date=pl.date(year, month, 1),
                emplvl=pl.col(col).cast(pl.Float64),
                suppressed_component=pl.col("suppressed_component"),
            )
        )
    stacked = pl.concat(months)
    return (
        stacked.group_by(["state_fips", "supersector_code", "date"])
        .agg(
            employment=pl.col("emplvl").sum(),
            suppressed=pl.col("suppressed_component").any(),
        )
        .sort(["state_fips", "supersector_code", "date"])
    )


def suppression_footprint(cells: pl.DataFrame) -> pl.DataFrame:
    """Summarise suppression per (state, supersector) over the panel.

    Parameters
    ----------
    cells
        A QCEW cell frame (as produced by :func:`parse_industry`, concatenated).

    Returns
    -------
    polars.DataFrame
        One row per (state, supersector) with the count of months, count of
        suppressed months, and the suppressed fraction.
    """
    return (
        cells.group_by(["state_fips", "supersector_code"])
        .agg(
            n_months=pl.len(),
            n_suppressed=pl.col("suppressed").sum(),
        )
        .with_columns(suppressed_fraction=pl.col("n_suppressed") / pl.col("n_months"))
        .sort(["state_fips", "supersector_code"])
    )
