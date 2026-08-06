"""Bauer-Swanson monetary policy surprises (SF Fed) ingestion.

A maintained high-frequency FOMC surprise series (30-minute interest-rate changes
around announcements), published by the San Francisco Fed as an update of Bauer &
Swanson (2023). Provides the **raw** surprise (``MPS``), an **orthogonalized**
surprise cleaned of predictable variation (``MPS_ORTH``), and, per FOMC, the
same-window S&P 500 change (for the information-effect test). Used as the
external instrument for the proxy-SVAR (a far stronger first stage than the BRW
benchmark) and, orthogonalized, as a predictability-robust shock variant.

References
----------
Bauer & Swanson (2023), NBER Macroeconomics Annual 37; SF Fed data update.
"""

from __future__ import annotations

import io

import httpx
import polars as pl

from cil.data import http
from cil.data.schemas import MPS_FOMC_SCHEMA, MPS_MONTHLY_SCHEMA, validate

_MONTHLY_PREFIX = "Monthly (update"
_FOMC_PREFIX = "FOMC (update"


def fetch_raw(client: httpx.Client, url: str) -> tuple[bytes, str, dict[str, str]]:
    """Download the MPS workbook. Returns ``(content, url, params)``."""
    response = http.fetch(client, url)
    return response.content, url, {}


def _pick_sheet(sheet_names: list[str], prefix: str) -> str:
    matches = [s for s in sheet_names if s.startswith(prefix)]
    if not matches:
        msg = f"No MPS sheet starting with {prefix!r} in {sheet_names}."
        raise ValueError(msg)
    return sorted(matches)[-1]  # newest "update" sheet


def parse_monthly(content: bytes) -> pl.DataFrame:
    """Parse the monthly MPS sheet into ``date``, ``mps``, ``mps_orth``.

    Parameters
    ----------
    content
        Raw ``.xlsx`` bytes from :func:`fetch_raw`.

    Returns
    -------
    polars.DataFrame
        Frame validated against
        :data:`cil.data.schemas.MPS_MONTHLY_SCHEMA`.
    """
    import pandas as pd

    xls = pd.ExcelFile(io.BytesIO(content))
    sheet = _pick_sheet([str(s) for s in xls.sheet_names], _MONTHLY_PREFIX)
    pdf = pd.read_excel(xls, sheet_name=sheet)[
        ["Year", "Month", "MPS", "MPS_ORTH"]
    ].dropna(subset=["Year", "Month"])
    frame = pl.from_pandas(pdf).select(
        date=pl.date(pl.col("Year").cast(pl.Int32), pl.col("Month").cast(pl.Int32), 1),
        mps=pl.col("MPS").cast(pl.Float64),
        mps_orth=pl.col("MPS_ORTH").cast(pl.Float64),
    )
    return validate(MPS_MONTHLY_SCHEMA, frame)


def parse_fomc(content: bytes) -> pl.DataFrame:
    """Parse the per-FOMC MPS sheet into ``date``, ``mps``, ``mps_orth``, ``sp500``.

    Parameters
    ----------
    content
        Raw ``.xlsx`` bytes from :func:`fetch_raw`.

    Returns
    -------
    polars.DataFrame
        Per-FOMC frame (the ``sp500`` column is the same-window S&P 500 change),
        validated against :data:`cil.data.schemas.MPS_FOMC_SCHEMA`.
    """
    import pandas as pd

    xls = pd.ExcelFile(io.BytesIO(content))
    sheet = _pick_sheet([str(s) for s in xls.sheet_names], _FOMC_PREFIX)
    pdf = pd.read_excel(xls, sheet_name=sheet)[
        ["Date", "SP500", "MPS", "MPS_ORTH"]
    ].dropna(subset=["Date", "MPS"])
    frame = pl.from_pandas(pdf).select(
        date=pl.col("Date").cast(pl.Date),
        mps=pl.col("MPS").cast(pl.Float64),
        mps_orth=pl.col("MPS_ORTH").cast(pl.Float64),
        sp500=pl.col("SP500").cast(pl.Float64),
    )
    return validate(MPS_FOMC_SCHEMA, frame)
