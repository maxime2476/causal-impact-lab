"""Bu-Rogers-Wu (2021) monetary policy shock series ingestion.

The published benchmark shock series, posted by the authors as a CSV. We read
the updated monthly column, indexed by month (``YYYYmM``), as the as-published
benchmark against which the in-house shocks are cross-correlated.

References
----------
Bu, Rogers, Wu (2021), *A unified measure of Fed monetary policy shocks*,
Journal of Monetary Economics 118.
"""

from __future__ import annotations

import datetime as dt
import io

import httpx
import polars as pl

from cil.data import http
from cil.data.schemas import BRW_SHOCK_SCHEMA, validate

_MONTHLY_COLUMN = "BRW_monthly (updated)"
_MONTH_COLUMN = "month"


def fetch_raw(client: httpx.Client, url: str) -> tuple[bytes, str, dict[str, str]]:
    """Download the BRW CSV. Returns ``(content, url, params)``."""
    response = http.fetch(client, url)
    return response.content, url, {}


def _parse_month(token: str) -> dt.date:
    """Convert a ``YYYYmM`` token (e.g. ``"1994m3"``) to a month-start date."""
    year_str, month_str = token.lower().split("m")
    return dt.date(int(year_str), int(month_str), 1)


def parse(content: bytes) -> pl.DataFrame:
    """Parse the BRW CSV into ``date`` and ``brw_monthly``.

    Parameters
    ----------
    content
        Raw CSV bytes from :func:`fetch_raw`.

    Returns
    -------
    polars.DataFrame
        Frame validated against
        :data:`cil.data.schemas.BRW_SHOCK_SCHEMA`.
    """
    raw = pl.read_csv(
        io.BytesIO(content),
        columns=[_MONTH_COLUMN, _MONTHLY_COLUMN],
        truncate_ragged_lines=True,
    )
    cleaned = raw.filter(
        pl.col(_MONTH_COLUMN).is_not_null()
        & (pl.col(_MONTH_COLUMN).str.strip_chars() != "")
    )
    dates = [_parse_month(m) for m in cleaned[_MONTH_COLUMN].to_list()]
    frame = pl.DataFrame(
        {
            "date": pl.Series(dates, dtype=pl.Date),
            "brw_monthly": cleaned[_MONTHLY_COLUMN].cast(pl.Float64),
        }
    )
    return validate(BRW_SHOCK_SCHEMA, frame)
