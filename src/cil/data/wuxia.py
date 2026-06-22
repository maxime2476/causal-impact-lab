"""Wu-Xia shadow federal funds rate ingestion and policy-rate splice.

The Atlanta Fed publishes the Wu-Xia (2016) shadow rate alongside the effective
federal funds rate (EFFR). We build a single monthly policy-rate series that
equals the EFFR away from the zero lower bound and the shadow rate within the
configured ZLB windows, gated behind ``use_shadow_rate``.

The Atlanta Fed suspended updates in March 2022; the shadow rate therefore ends
in early 2022, which still covers the ZLB windows after which the EFFR applies.

References
----------
Wu & Xia (2016), *Measuring the Macroeconomic Impact of Monetary Policy at the
Zero Lower Bound*, JMCB 48(2-3).
"""

from __future__ import annotations

import io
from typing import TYPE_CHECKING

import httpx
import pandas as pd
import polars as pl

from cil.data import http
from cil.data.schemas import POLICY_RATE_SCHEMA, validate

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cil.config import ZlbWindow


def fetch_raw(client: httpx.Client, url: str) -> tuple[bytes, str, dict[str, str]]:
    """Download the Wu-Xia workbook. Returns ``(content, url, params)``."""
    response = http.fetch(client, url)
    return response.content, url, {}


def parse(content: bytes) -> pl.DataFrame:
    """Parse the Wu-Xia workbook into ``date``, ``effr``, ``shadow_rate``.

    Parameters
    ----------
    content
        Raw ``.xlsx`` bytes from :func:`fetch_raw`.

    Returns
    -------
    polars.DataFrame
        Monthly frame with the EFFR and shadow rate (shadow may be null outside
        its estimated range).
    """
    pdf = pd.read_excel(io.BytesIO(content), sheet_name="Data")
    pdf.columns = ["date", "effr", "shadow_rate", *list(pdf.columns[3:])]
    pdf = pdf[["date", "effr", "shadow_rate"]]
    frame = pl.from_pandas(pdf).with_columns(
        pl.col("date").cast(pl.Date),
        pl.col("effr").cast(pl.Float64),
        pl.col("shadow_rate").cast(pl.Float64),
    )
    return frame.sort("date")


def splice(
    wuxia: pl.DataFrame,
    zlb_windows: Sequence[ZlbWindow],
    *,
    use_shadow_rate: bool,
) -> pl.DataFrame:
    """Splice EFFR and the shadow rate into one monthly policy rate.

    Within each ZLB window (and only if both ``use_shadow_rate`` is set and a
    shadow value exists) the policy rate equals the shadow rate; elsewhere it
    equals the EFFR.

    Parameters
    ----------
    wuxia
        Frame from :func:`parse`.
    zlb_windows
        Inclusive ZLB windows over which to splice.
    use_shadow_rate
        Master gate; when ``False`` the policy rate is the EFFR throughout.

    Returns
    -------
    polars.DataFrame
        Frame validated against
        :data:`cil.data.schemas.POLICY_RATE_SCHEMA`.
    """
    in_window = pl.lit(False)
    for window in zlb_windows:
        in_window = in_window | (
            (pl.col("date") >= pl.lit(window.start))
            & (pl.col("date") <= pl.lit(window.end))
        )
    use_shadow = (
        pl.lit(use_shadow_rate) & in_window & pl.col("shadow_rate").is_not_null()
    )
    spliced = wuxia.with_columns(
        is_zlb_splice=use_shadow,
        policy_rate=pl.when(use_shadow)
        .then(pl.col("shadow_rate"))
        .otherwise(pl.col("effr")),
    ).select(["date", "effr", "shadow_rate", "policy_rate", "is_zlb_splice"])
    return validate(POLICY_RATE_SCHEMA, spliced)
