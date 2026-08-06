"""Updated Romer-Romer (2004) narrative monetary policy shocks ingestion.

A **narrative / forecast-based** identification of monetary shocks: the change in
the intended federal funds rate around each FOMC meeting, orthogonalized to the
Fed's own Greenbook forecasts, so that the residual is the policy move *not*
explained by the Fed's real-time outlook. This is an alternative identification
to the high-frequency surprise (:mod:`cil.data.mps`) that does not rely on a
30-minute market window.

We use the maintained, directly-downloadable **Breitenlechner (2018) update** of
the original Romer & Romer (2004) series (quarterly), which reports the
original-method series (``MPORGQ``) and two extensions, through 2008 (``MP08Q``)
and through 2012 (``MP12Q``, the longest). A positive value is a contractionary
surprise (an unexpected tightening).

References
----------
Romer & Romer (2004), *A New Measure of Monetary Shocks*, AER 94(4).
Breitenlechner (2018), *An Update of Romer and Romer (2004) Narrative U.S.
Monetary Policy Shocks up to 2012Q4* (University of Innsbruck), data file
``UpdateRR04shocks.dta``.
"""

from __future__ import annotations

import io

import httpx
import polars as pl

from cil.data import http
from cil.data.schemas import RR_SHOCKS_SCHEMA, validate


def fetch_raw(client: httpx.Client, url: str) -> tuple[bytes, str, dict[str, str]]:
    """Download the updated RR ``.dta``. Returns ``(content, url, params)``."""
    response = http.fetch(client, url)
    return response.content, url, {}


def parse(content: bytes) -> pl.DataFrame:
    """Parse the updated RR ``.dta`` into ``date``, ``rr_org``, ``rr08``, ``rr12``.

    Parameters
    ----------
    content
        Raw Stata ``.dta`` bytes from :func:`fetch_raw`.

    Returns
    -------
    polars.DataFrame
        Quarterly frame (``date`` is the quarter-start), validated against
        :data:`cil.data.schemas.RR_SHOCKS_SCHEMA`.
    """
    import pandas as pd

    pdf = pd.read_stata(io.BytesIO(content))[["Date", "MPORGQ", "MP08Q", "MP12Q"]]
    frame = pl.from_pandas(pdf).select(
        date=pl.col("Date").cast(pl.Date),
        rr_org=pl.col("MPORGQ").cast(pl.Float64),
        rr08=pl.col("MP08Q").cast(pl.Float64),
        rr12=pl.col("MP12Q").cast(pl.Float64),
    )
    return validate(RR_SHOCKS_SCHEMA, frame)
