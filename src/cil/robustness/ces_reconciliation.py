"""CES-SAE vs QCEW supersector reconciliation (Tier 0.3).

Cross-checks the QCEW state-by-supersector employment against the independent CES
state-and-area survey. QCEW 3-digit cells are aggregated up to supersectors (via
the NAICS 2-digit crosswalk) and compared to CES supersector employment by state:
the correlation of log levels and of year-on-year growth. High agreement
validates the QCEW panel; documented disagreement is itself a finding.

Run as a module::

    uv run python -m cil.robustness.ces_reconciliation
"""

from __future__ import annotations

import functools

import httpx
import numpy as np
import polars as pl

from cil.config import Settings, get_settings
from cil.data import ces_sae, http
from cil.data.pipeline import _cache_or_fetch
from cil.data.store import Store
from cil.exposure.shift_share import NAICS2_TO_SUPERSECTOR


def _fetch_sm_file(
    client: httpx.Client, bls_flat_base: str, filename: str
) -> tuple[bytes, str, dict[str, str]]:
    """Fetch one ``sm.data.*`` file. Returns ``(content, url, params)``."""
    url = bls_flat_base + "/sm/" + filename
    return http.fetch(client, url).content, url, {"file": filename}


def aggregate_qcew_to_supersector(qcew_cells: pl.DataFrame) -> pl.DataFrame:
    """Aggregate 3-digit QCEW cells up to QCEW supersectors.

    Parameters
    ----------
    qcew_cells
        Cell frame with ``state_fips``, ``supersector_code`` (the NAICS level in
        the panel, e.g. 3-digit), ``date``, ``employment``.

    Returns
    -------
    polars.DataFrame
        Columns ``state_fips``, ``supersector_code`` (QCEW supersector),
        ``date``, ``employment`` summed within supersector.
    """
    mapping = pl.DataFrame(
        {
            "naics2": list(NAICS2_TO_SUPERSECTOR.keys()),
            "qcew_supersector": list(NAICS2_TO_SUPERSECTOR.values()),
        }
    )
    return (
        qcew_cells.with_columns(naics2=pl.col("supersector_code").str.slice(0, 2))
        .join(mapping, on="naics2", how="inner")
        .group_by(["state_fips", "qcew_supersector", "date"])
        .agg(employment=pl.col("employment").sum())
        .rename({"qcew_supersector": "supersector_code"})
        .sort(["state_fips", "supersector_code", "date"])
    )


def reconcile(ces: pl.DataFrame, qcew: pl.DataFrame) -> pl.DataFrame:
    """Per (state, supersector) correlation of CES vs QCEW employment.

    Parameters
    ----------
    ces
        CES supersector panel (``state_fips``, ``supersector_code``, ``date``,
        ``employment``).
    qcew
        QCEW supersector panel (same columns), e.g. from
        :func:`aggregate_qcew_to_supersector`.

    Returns
    -------
    polars.DataFrame
        One row per (state, supersector): ``n_months``, ``corr_level``
        (log-employment), ``corr_growth`` (12-month log growth).
    """
    joined = (
        ces.select("state_fips", "supersector_code", "date", ces_emp="employment")
        .join(
            qcew.select(
                "state_fips", "supersector_code", "date", qcew_emp="employment"
            ),
            on=["state_fips", "supersector_code", "date"],
            how="inner",
        )
        .filter((pl.col("ces_emp") > 0) & (pl.col("qcew_emp") > 0))
        .sort(["state_fips", "supersector_code", "date"])
        .with_columns(
            ces_log=pl.col("ces_emp").log(),
            qcew_log=pl.col("qcew_emp").log(),
        )
        .with_columns(
            ces_g=(pl.col("ces_log") - pl.col("ces_log").shift(12)).over(
                ["state_fips", "supersector_code"]
            ),
            qcew_g=(pl.col("qcew_log") - pl.col("qcew_log").shift(12)).over(
                ["state_fips", "supersector_code"]
            ),
        )
    )
    return (
        joined.group_by(["state_fips", "supersector_code"])
        .agg(
            n_months=pl.len(),
            corr_level=pl.corr("ces_log", "qcew_log"),
            corr_growth=pl.corr("ces_g", "qcew_g"),
        )
        .sort(["state_fips", "supersector_code"])
    )


def build_ces_reconciliation(settings: Settings | None = None) -> dict[str, float]:
    """Ingest CES supersector data, reconcile against QCEW, and store both."""
    settings = settings or get_settings()
    client = http.build_client(
        settings.data.contact_email, settings.data.request_timeout_seconds
    )
    try:
        with Store(settings.paths.store_path) as store:
            catalog = ces_sae.fetch_sm_series_catalog(
                client, settings.data.urls.bls_flat_base
            )
            targets = ces_sae.target_supersector_series(catalog)
            frames: list[pl.DataFrame] = []
            for filename in ces_sae.list_state_data_files(
                client, settings.data.urls.bls_flat_base
            ):
                content = _cache_or_fetch(
                    store,
                    settings.paths.data_dir,
                    "ces_sm",
                    f"{filename}.tsv",
                    functools.partial(
                        _fetch_sm_file,
                        client,
                        settings.data.urls.bls_flat_base,
                        filename,
                    ),
                )
                frames.append(ces_sae.parse_sm_data(content, targets))
            ces = (
                pl.concat(frames)
                .filter(
                    (pl.col("date") >= settings.data.sample.start)
                    & (pl.col("date") <= settings.data.sample.end)
                )
                .sort(["state_fips", "supersector_code", "date"])
            )
            store.write_table("ces_supersector", ces)

            qcew = aggregate_qcew_to_supersector(store.read_table("qcew_cells"))
            recon = reconcile(ces, qcew)
            store.write_table("ces_qcew_reconciliation", recon)
            return {
                "ces_rows": float(ces.height),
                "n_pairs": float(recon.height),
                "median_corr_level": float(
                    np.nanmedian(recon["corr_level"].to_numpy())
                ),
                "median_corr_growth": float(
                    np.nanmedian(recon["corr_growth"].to_numpy())
                ),
            }
    finally:
        client.close()


def main() -> None:
    """Build the reconciliation and print a summary (entry point)."""
    for key, value in build_ces_reconciliation().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
