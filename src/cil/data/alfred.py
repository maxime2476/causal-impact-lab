"""ALFRED (FRED real-time) ingestion for point-in-time macro series.

ALFRED returns the full revision history of a series: each observation row
carries a ``realtime_start`` marking when that value became the current value.
We treat ``realtime_start`` as the *vintage date*, yielding a long point-in-time
frame ``(series_id, reference_date, vintage_date, value)`` from which any as-of
slice (e.g. first release, or the value known at a given date) can be derived
without look-ahead.

References
----------
FRED/ALFRED API, ``/fred/series/observations`` with ``realtime_start`` /
``realtime_end`` spanning the full archive.
"""

from __future__ import annotations

import datetime as dt
import json

import httpx
import polars as pl

from cil.data import http
from cil.data.schemas import (
    MACRO_CURRENT_SCHEMA,
    MACRO_PIT_SCHEMA,
    validate,
)

# FRED's documented minimum and maximum real-time dates.
_REALTIME_MIN = "1776-07-04"
_REALTIME_MAX = "9999-12-31"
_MISSING = {".", "", "NA"}


def fetch_raw_series(
    client: httpx.Client,
    fred_base: str,
    api_key: str,
    series_id: str,
) -> tuple[bytes, str, dict[str, str]]:
    """Fetch the full real-time observation archive for one series.

    Parameters
    ----------
    client
        HTTP client.
    fred_base
        FRED API base URL.
    api_key
        FRED/ALFRED API key (sent as a query parameter, never recorded).
    series_id
        FRED series identifier.

    Returns
    -------
    content : bytes
        Raw JSON payload.
    record_url : str
        The request URL with the API key redacted, for provenance.
    record_params : dict of str to str
        Non-secret request parameters, for provenance.
    """
    url = f"{fred_base}/series/observations"
    record_params = {
        "series_id": series_id,
        "realtime_start": _REALTIME_MIN,
        "realtime_end": _REALTIME_MAX,
        "file_type": "json",
    }
    params: dict[str, str | int] = {**record_params, "api_key": api_key}
    response = http.fetch(client, url, params=params)
    return response.content, url, record_params


def parse_pit(content: bytes, series_id: str) -> pl.DataFrame:
    """Parse an ALFRED JSON payload into a point-in-time long frame.

    Parameters
    ----------
    content
        Raw JSON bytes from :func:`fetch_raw_series`.
    series_id
        Series identifier to stamp on each row.

    Returns
    -------
    polars.DataFrame
        Frame validated against
        :data:`cil.data.schemas.MACRO_PIT_SCHEMA`.
    """
    payload = json.loads(content)
    observations = payload["observations"]
    reference_dates: list[dt.date] = []
    vintage_dates: list[dt.date] = []
    values: list[float | None] = []
    for obs in observations:
        reference_dates.append(dt.date.fromisoformat(obs["date"]))
        vintage_dates.append(dt.date.fromisoformat(obs["realtime_start"]))
        raw = obs["value"]
        values.append(None if raw in _MISSING else float(raw))
    frame = pl.DataFrame(
        {
            "series_id": pl.Series([series_id] * len(values), dtype=pl.Utf8),
            "reference_date": pl.Series(reference_dates, dtype=pl.Date),
            "vintage_date": pl.Series(vintage_dates, dtype=pl.Date),
            "value": pl.Series(values, dtype=pl.Float64),
        }
    ).unique(subset=["series_id", "reference_date", "vintage_date"], keep="last")
    return validate(MACRO_PIT_SCHEMA, frame)


def latest_from_pit(pit: pl.DataFrame) -> pl.DataFrame:
    """Collapse a point-in-time frame to its latest vintage per observation.

    Parameters
    ----------
    pit
        A frame conforming to
        :data:`cil.data.schemas.MACRO_PIT_SCHEMA`.

    Returns
    -------
    polars.DataFrame
        The as-published (most recent vintage) series, validated against
        :data:`cil.data.schemas.MACRO_CURRENT_SCHEMA`.
    """
    latest = (
        pit.sort(["series_id", "reference_date", "vintage_date"])
        .group_by(["series_id", "reference_date"])
        .last()
        .select(["series_id", "reference_date", "value"])
        .sort(["series_id", "reference_date"])
    )
    return validate(MACRO_CURRENT_SCHEMA, latest)


def as_of(pit: pl.DataFrame, vintage: dt.date) -> pl.DataFrame:
    """Return the series as known on *vintage* (no look-ahead).

    For each reference period, selects the value from the latest vintage at or
    before *vintage*.

    Parameters
    ----------
    pit
        A point-in-time frame.
    vintage
        The as-of date.

    Returns
    -------
    polars.DataFrame
        Frame validated against
        :data:`cil.data.schemas.MACRO_CURRENT_SCHEMA`.
    """
    sliced = (
        pit.filter(pl.col("vintage_date") <= vintage)
        .sort(["series_id", "reference_date", "vintage_date"])
        .group_by(["series_id", "reference_date"])
        .last()
        .select(["series_id", "reference_date", "value"])
        .sort(["series_id", "reference_date"])
    )
    return validate(MACRO_CURRENT_SCHEMA, sliced)
