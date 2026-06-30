"""End-to-end ingestion pipeline.

Orchestrates the disjoint *backfill* path: for each source, fetch (or reuse a
cached raw payload), parse through its schema contract, write the analysis-ready
table to DuckDB, and record provenance. The analysis-ready cell panel is then
assembled from the QCEW cells.

Run as a module to build everything on real data::

    uv run python -m cil.data.pipeline

The refresh (incremental) path is intentionally not implemented here; it must
never read data dated after its reference date for the strict-PIT series.
"""

from __future__ import annotations

import datetime as dt
import functools
from collections.abc import Callable
from pathlib import Path

import httpx
import polars as pl

from cil.config import Settings, get_settings
from cil.data import alfred, brw, ces_sae, http, panel, qcew, qcew_bulk, wuxia
from cil.data.provenance import cache_raw, load_provenance
from cil.data.store import Store

FetchFn = Callable[[], tuple[bytes, str, dict[str, str]]]


def _cache_or_fetch(
    store: Store,
    data_dir: Path,
    source: str,
    filename: str,
    fetch_fn: FetchFn,
    *,
    vintage_date: dt.date | None = None,
) -> bytes:
    """Return cached raw bytes if present, else fetch, cache, and record them.

    Parameters
    ----------
    store
        Store used to persist provenance on a fresh fetch.
    data_dir
        Data root holding the raw cache.
    source
        Logical source name.
    filename
        Cache file name within the source directory.
    fetch_fn
        Zero-argument callable returning ``(content, url, params)``.
    vintage_date
        Vintage date for point-in-time sources.

    Returns
    -------
    bytes
        The raw payload.
    """
    cached = data_dir / "raw" / source / filename
    if cached.exists():
        prov = load_provenance(data_dir, source, filename)
        if prov is not None:
            store.record_provenance(prov)
        return cached.read_bytes()
    content, url, params = fetch_fn()
    prov = cache_raw(
        data_dir,
        source,
        filename,
        content,
        url,
        vintage_date=vintage_date,
        params=params,
    )
    store.record_provenance(prov)
    return content


def ingest_macro(settings: Settings, store: Store, client: httpx.Client) -> int:
    """Ingest ALFRED point-in-time macro series; return the PIT row count."""
    if settings.fred_api_key is None:
        msg = "CIL_FRED_API_KEY is required for ALFRED ingestion."
        raise ValueError(msg)
    api_key = settings.fred_api_key
    pit_frames: list[pl.DataFrame] = []
    for series_id in settings.data.series.all_ids:
        content = _cache_or_fetch(
            store,
            settings.paths.data_dir,
            "alfred",
            f"{series_id}.json",
            functools.partial(
                alfred.fetch_raw_series,
                client,
                settings.data.urls.fred_base,
                api_key,
                series_id,
            ),
        )
        pit_frames.append(alfred.parse_pit(content, series_id))
    pit = pl.concat(pit_frames)
    store.write_table("macro_pit", pit)
    store.write_table("macro_current", alfred.latest_from_pit(pit))
    return pit.height


def _quarters(start: dt.date, end: dt.date) -> list[tuple[int, int]]:
    """Return ``(year, quarter)`` pairs spanning *start*..*end* inclusive."""
    pairs: list[tuple[int, int]] = []
    for year in range(start.year, end.year + 1):
        for quarter in range(1, 5):
            pairs.append((year, quarter))
    return pairs


def ingest_qcew(settings: Settings, store: Store, client: httpx.Client) -> int:
    """Ingest QCEW cells (bulk flat files) and the suppression footprint.

    Uses the annual ``by_area`` bulk zips uniformly across the sample (one
    download per year, cached), at the configured aggregation level. This
    extends history before the API's 2014 floor and supports any NAICS level
    (e.g. supersector or 3-digit) from the same cached zips.
    """
    qcfg = settings.data.qcew
    start_year = max(qcfg.bulk_min_year, settings.data.sample.start.year)
    end_year = settings.data.sample.end.year
    frames: list[pl.DataFrame] = []
    for year in range(start_year, end_year + 1):
        content = _cache_or_fetch(
            store,
            settings.paths.data_dir,
            "qcew_bulk",
            f"{year}.zip",
            functools.partial(
                qcew_bulk.fetch_raw_year,
                client,
                settings.data.urls.qcew_bulk_template,
                year,
            ),
        )
        frames.append(
            qcew_bulk.parse_year(
                content, year, aggregation_level=qcfg.aggregation_level
            )
        )
    cells = (
        pl.concat(frames)
        .filter(
            (pl.col("date") >= settings.data.sample.start)
            & (pl.col("date") <= settings.data.sample.end)
        )
        .sort(["state_fips", "supersector_code", "date"])
    )
    store.write_table("qcew_cells", cells)
    store.write_table("qcew_suppression", qcew.suppression_footprint(cells))
    return cells.height


def ingest_wuxia(settings: Settings, store: Store, client: httpx.Client) -> int:
    """Ingest the Wu-Xia spliced policy rate; return the row count."""
    content = _cache_or_fetch(
        store,
        settings.paths.data_dir,
        "wuxia",
        "WuXiaShadowRate.xlsx",
        lambda: wuxia.fetch_raw(client, settings.data.urls.wuxia_xlsx),
    )
    spliced = wuxia.splice(
        wuxia.parse(content),
        settings.data.sample.zlb_windows,
        use_shadow_rate=settings.use_shadow_rate,
    )
    store.write_table("policy_rate", spliced)
    return spliced.height


def ingest_brw(settings: Settings, store: Store, client: httpx.Client) -> int:
    """Ingest the Bu-Rogers-Wu shock series; return the row count."""
    content = _cache_or_fetch(
        store,
        settings.paths.data_dir,
        "brw",
        "brw-shock-series.csv",
        lambda: brw.fetch_raw(client, settings.data.urls.brw_csv),
    )
    shocks = brw.parse(content)
    store.write_table("brw_shocks", shocks)
    return shocks.height


def ingest_ces(settings: Settings, store: Store, client: httpx.Client) -> int:
    """Ingest CES-SAE state total-nonfarm cross-check; return the row count."""
    if settings.fred_api_key is None:
        msg = "CIL_FRED_API_KEY is required for CES-SAE ingestion."
        raise ValueError(msg)
    api_key = settings.fred_api_key
    frames: list[pl.DataFrame] = []
    for state_fips, abbr in ces_sae.STATE_FIPS_ABBR.items():
        content = _cache_or_fetch(
            store,
            settings.paths.data_dir,
            "ces_sae",
            f"{abbr}NA.json",
            functools.partial(
                ces_sae.fetch_raw_state,
                client,
                settings.data.urls.fred_base,
                api_key,
                abbr,
            ),
        )
        frames.append(ces_sae.parse_state(content, state_fips))
    ces = pl.concat(frames).sort(["state_fips", "date"])
    store.write_table("ces_sae", ces)
    return ces.height


def build_panels(settings: Settings, store: Store) -> tuple[int, int]:
    """Assemble the analysis-ready cell panel from stored QCEW cells.

    Returns
    -------
    n_panel : int
        Rows in the analysis-ready panel.
    n_dropped : int
        Number of (state, supersector) cells dropped for low coverage.
    """
    cells = store.read_table("qcew_cells")
    panel_df, dropped = panel.build_cell_panel(
        cells, settings.data.qcew.coverage_min_fraction
    )
    store.write_table("panel_cell", panel_df)
    store.write_table("panel_dropped_cells", dropped)
    return panel_df.height, dropped.height


def run(settings: Settings | None = None) -> dict[str, int]:
    """Run the full backfill and panel assembly. Returns a counts summary."""
    settings = settings or get_settings()
    client = http.build_client(
        settings.data.contact_email, settings.data.request_timeout_seconds
    )
    summary: dict[str, int] = {}
    try:
        with Store(settings.paths.store_path) as store:
            summary["macro_pit_rows"] = ingest_macro(settings, store, client)
            summary["wuxia_rows"] = ingest_wuxia(settings, store, client)
            summary["brw_rows"] = ingest_brw(settings, store, client)
            summary["ces_rows"] = ingest_ces(settings, store, client)
            summary["qcew_cells"] = ingest_qcew(settings, store, client)
            n_panel, n_dropped = build_panels(settings, store)
            summary["panel_rows"] = n_panel
            summary["dropped_cells"] = n_dropped
    finally:
        client.close()
    return summary


def main() -> None:
    """Run the backfill and print a counts summary (module entry point)."""
    summary = run()
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
