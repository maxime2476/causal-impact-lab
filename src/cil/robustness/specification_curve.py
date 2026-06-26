"""Specification curve for the headline relative effect.

Re-estimates the interacted panel local projection across the pre-registered grid
of choices (shock series, exposure shifter, control-lag depth, and COVID-sample
handling) and reports the coefficient at the primary decision horizons for every
specification, with Benjamini-Hochberg FDR control across specifications. The
point is to show the full distribution of estimates, not a selected table.
"""

from __future__ import annotations

import datetime as dt
import itertools
from collections.abc import Callable

import polars as pl

from cil.estimators.panel_lp import PanelLPConfig, run_panel_lp
from cil.inference.bh_fdr import bh_adjust

#: A named exposure shifter and a named (shock frame, shock column).
ExposureMap = dict[str, pl.DataFrame]
ShockMap = dict[str, tuple[pl.DataFrame, str]]
SampleMap = dict[str, Callable[[pl.DataFrame], pl.DataFrame]]


def covid_window_filter(
    start: dt.date = dt.date(2020, 3, 1), end: dt.date = dt.date(2020, 12, 1)
) -> Callable[[pl.DataFrame], pl.DataFrame]:
    """Return a filter dropping the COVID window (by the frame's ``date``)."""

    def _filter(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.filter((pl.col("date") < start) | (pl.col("date") > end))

    return _filter


def run_specification_curve(
    panel: pl.DataFrame,
    exposures: ExposureMap,
    shocks: ShockMap,
    *,
    horizons: tuple[int, ...] = (12, 24),
    control_lags: tuple[int, ...] = (3, 6, 12),
    samples: SampleMap | None = None,
) -> pl.DataFrame:
    """Estimate the headline coefficient across the specification grid.

    Parameters
    ----------
    panel
        Analysis-ready cell panel.
    exposures
        Named exposure shifters (``supersector_code``, ``exposure``).
    shocks
        Named shocks, each a ``(frame, column)`` pair.
    horizons
        Decision horizons to record.
    control_lags
        Control-lag depths to sweep.
    samples
        Named sample transforms (e.g. full vs. COVID-excluded). Defaults to a
        single ``"full"`` identity sample.

    Returns
    -------
    polars.DataFrame
        One row per (specification, horizon): ``shock``, ``exposure``, ``lags``,
        ``sample``, ``horizon``, ``beta``, ``se``, ``p_value``, ``p_value_bh``
        (BH-adjusted within each horizon across specifications).
    """
    samples = samples or {"full": lambda frame: frame}
    rows: list[dict[str, object]] = []
    for (shock_name, (shock_df, col)), (expo_name, expo_df), lags, (
        samp_name,
        samp,
    ) in itertools.product(
        shocks.items(), exposures.items(), control_lags, samples.items()
    ):
        sub_panel = samp(panel)
        sub_shock = samp(shock_df)
        result = run_panel_lp(
            sub_panel,
            expo_df,
            sub_shock,
            PanelLPConfig(horizons=horizons, n_control_lags=lags),
            shock_col=col,
        )
        for h in horizons:
            row = result.filter(pl.col("horizon") == h)
            if row.height == 0:
                continue
            r = row.to_dicts()[0]
            rows.append(
                {
                    "shock": shock_name,
                    "exposure": expo_name,
                    "lags": lags,
                    "sample": samp_name,
                    "horizon": h,
                    "beta": r["beta"],
                    "se": r["se"],
                    "p_value": r["p_value"],
                }
            )
    curve = pl.DataFrame(rows)
    return _add_bh(curve)


def _add_bh(curve: pl.DataFrame) -> pl.DataFrame:
    """Append BH-FDR adjusted p-values within each horizon across specs."""
    parts = []
    for (_h,), group in curve.group_by(["horizon"], maintain_order=True):
        adjusted = bh_adjust(group["p_value"].to_numpy())
        parts.append(group.with_columns(p_value_bh=pl.Series(adjusted)))
    return pl.concat(parts).sort(["horizon", "shock", "exposure", "lags", "sample"])


def summarize_curve(curve: pl.DataFrame) -> pl.DataFrame:
    """Summarize the curve per horizon: share negative and share significant.

    Parameters
    ----------
    curve
        Output of :func:`run_specification_curve`.

    Returns
    -------
    polars.DataFrame
        Per horizon: ``n_specs``, ``share_negative``, ``share_sig_bh`` (BH < 0.10
        and negative), and the median beta.
    """
    return (
        curve.group_by("horizon")
        .agg(
            n_specs=pl.len(),
            share_negative=(pl.col("beta") < 0).mean(),
            share_sig_bh=((pl.col("p_value_bh") < 0.10) & (pl.col("beta") < 0)).mean(),
            median_beta=pl.col("beta").median(),
        )
        .sort("horizon")
    )
