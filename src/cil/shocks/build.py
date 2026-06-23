"""Build and triangulate the monetary shock series on real data.

Reads the analysis-ready store, constructs the three shock series (Romer-Romer
orthogonalized, proxy-SVAR external-instrument, and the BRW benchmark), runs the
information-effect and predictability tests, cross-correlates the series, and
writes the results back to the store.

Run as a module::

    uv run python -m cil.shocks.build
"""

from __future__ import annotations

import httpx
import polars as pl

from cil.config import Settings, get_settings
from cil.data import alfred, http
from cil.data.pipeline import _cache_or_fetch
from cil.data.store import Store
from cil.shocks import (
    compare,
    features,
    info_effect,
    predictability,
    proxy_svar,
    rr_orthogonalization,
)

# Predictability predictors: levels only (d_unemployment is collinear with the
# unemployment lags, so it is excluded to keep the design full rank).
_FEATURE_COLS = ["inflation", "ip_growth", "unemployment"]


def monthly_equity_returns(
    settings: Settings, store: Store, client: httpx.Client
) -> pl.DataFrame:
    """Fetch the equity index and return month-start log returns (percent).

    Parameters
    ----------
    settings
        Project settings (provides the equity series id and FRED access).
    store
        Store used for provenance on a fresh fetch.
    client
        An ``httpx.Client`` for fetching.

    Returns
    -------
    polars.DataFrame
        Columns ``date`` (first of month) and ``equity_return``.
    """
    series_id = settings.shocks.equity_series_id
    content = _cache_or_fetch(
        store,
        settings.paths.data_dir,
        "equity",
        f"{series_id}.json",
        lambda: alfred.fetch_raw_series(
            client,
            settings.data.urls.fred_base,
            settings.fred_api_key or "",
            series_id,
            pit=False,
        ),
    )
    daily = alfred.latest_from_pit(alfred.parse_pit(content, series_id))
    monthly = (
        daily.with_columns(month=pl.col("reference_date").dt.truncate("1mo"))
        .group_by("month")
        .agg(close=pl.col("value").sort_by("reference_date").last())
        .sort("month")
        .with_columns(
            equity_return=100.0
            * (pl.col("close").log() - pl.col("close").log().shift(1))
        )
        .select(date=pl.col("month"), equity_return=pl.col("equity_return"))
        .drop_nulls()
    )
    return monthly


def build_shocks(settings: Settings | None = None) -> dict[str, float]:
    """Construct, test, and store the shock series. Return a summary.

    Returns
    -------
    dict of str to float
        Headline diagnostics (correlations, first-stage F, contamination share,
        predictability p-values).
    """
    settings = settings or get_settings()
    client = http.build_client(
        settings.data.contact_email, settings.data.request_timeout_seconds
    )
    summary: dict[str, float] = {}
    try:
        with Store(settings.paths.store_path) as store:
            macro_pit = store.read_table("macro_pit")
            macro_current = store.read_table("macro_current")
            policy_rate = store.read_table("policy_rate")
            brw = store.read_table("brw_shocks")
            equity = monthly_equity_returns(settings, store, client)

            rr_shock, rr_diag = rr_orthogonalization.romer_romer_shock(
                macro_pit, policy_rate, n_lags=settings.shocks.rr_lags
            )
            block = proxy_svar.build_macro_block(policy_rate, macro_current)
            svar_shock, fs_diag = proxy_svar.first_stage(
                block,
                brw,
                instrument_col=settings.shocks.instrument,
                n_lags=settings.shocks.svar_lags,
                weak_f_threshold=settings.shocks.weak_iv_f_threshold,
            )

            shocks = (
                rr_shock.join(brw, on="date", how="full", coalesce=True)
                .join(svar_shock, on="date", how="full", coalesce=True)
                .join(equity, on="date", how="left")
                .sort("date")
            )
            store.write_table("shocks", shocks)

            _, info_summary = info_effect.classify(brw, equity, shock_col="brw_monthly")
            feats = features.realtime_macro_features(macro_pit)
            pred_brw = predictability.predictability_test(
                brw,
                feats,
                shock_col="brw_monthly",
                predictor_cols=_FEATURE_COLS,
                n_lags=settings.shocks.predictability_lags,
            )
            pred_rr = predictability.predictability_test(
                rr_shock,
                feats,
                shock_col="rr_shock",
                predictor_cols=_FEATURE_COLS,
                n_lags=settings.shocks.predictability_lags,
            )
            xcorr = compare.cross_correlations(
                {
                    "rr_shock": rr_shock,
                    "svar_shock": svar_shock,
                    "brw_monthly": brw,
                }
            )
            _store_diagnostics(
                store, rr_diag, fs_diag, info_summary, pred_brw, pred_rr, xcorr
            )

            summary = {
                "rr_r_squared": rr_diag.r_squared,
                "svar_first_stage_f": fs_diag.robust_f,
                "svar_weak": float(fs_diag.weak),
                "info_contamination_share": info_summary.contamination_share,
                "brw_predictability_p": pred_brw.f_pvalue,
                "rr_predictability_p": pred_rr.f_pvalue,
                **{f"corr_{c.series_a}_{c.series_b}": c.correlation for c in xcorr},
            }
    finally:
        client.close()
    return summary


def _store_diagnostics(
    store: Store,
    rr_diag: rr_orthogonalization.OrthogonalizationDiagnostics,
    fs_diag: proxy_svar.FirstStageDiagnostics,
    info_summary: info_effect.InfoEffectSummary,
    pred_brw: predictability.PredictabilityResult,
    pred_rr: predictability.PredictabilityResult,
    xcorr: list[compare.PairwiseCorrelation],
) -> None:
    """Persist shock diagnostics and cross-correlations to the store."""
    rows: list[dict[str, str | float]] = [
        {"metric": "rr_r_squared", "value": rr_diag.r_squared},
        {"metric": "rr_n_obs", "value": float(rr_diag.n_obs)},
        {"metric": "svar_first_stage_f", "value": fs_diag.robust_f},
        {"metric": "svar_partial_r_squared", "value": fs_diag.partial_r_squared},
        {"metric": "svar_weak", "value": float(fs_diag.weak)},
        {
            "metric": "info_contamination_share",
            "value": info_summary.contamination_share,
        },
        {"metric": "info_n_information", "value": float(info_summary.n_information)},
        {"metric": "brw_predictability_p", "value": pred_brw.f_pvalue},
        {"metric": "rr_predictability_p", "value": pred_rr.f_pvalue},
    ]
    store.write_table("shock_diagnostics", pl.DataFrame(rows))
    store.write_table(
        "shock_xcorr",
        pl.DataFrame([c.model_dump() for c in xcorr]),
    )


def main() -> None:
    """Build the shocks and print the summary (module entry point)."""
    for key, value in build_shocks().items():
        print(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main()
