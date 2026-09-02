"""Sign- and size-dependent asymmetry in the headline relative effect.

Splits the interacted-LP treatment ``E_i * s_t`` into two components and estimates
their coefficients jointly, so the relative response can differ by the shock's
**sign** (tightening ``s>0`` vs easing ``s<0``) or **size** (large vs small
``|s|``). A Wald test of ``beta_a = beta_b`` at each horizon quantifies the
asymmetry. This is the study's headline estimand, split — not the aggregate.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import polars as pl
from linearmodels.panel import PanelOLS
from scipy.stats import norm

from cil.estimators.panel_lp import PanelLPConfig
from cil.inference.bh_fdr import bh_adjust

Split = Literal["sign", "size"]
_LABELS: dict[Split, tuple[str, str]] = {
    "sign": ("tightening", "easing"),
    "size": ("large", "small"),
}


def _split_shock(shock: pl.DataFrame, shock_col: str, split: Split) -> pl.DataFrame:
    """Split the shock into two components (``s_a``, ``s_b``) per the rule."""
    s = pl.col(shock_col)
    if split == "sign":
        return shock.with_columns(
            s_a=pl.when(s > 0).then(s).otherwise(0.0),
            s_b=pl.when(s < 0).then(s).otherwise(0.0),
        )
    median_abs = float(np.nanmedian(shock[shock_col].abs().to_numpy()))
    return shock.with_columns(
        s_a=pl.when(s.abs() > median_abs).then(s).otherwise(0.0),
        s_b=pl.when(s.abs() <= median_abs).then(s).otherwise(0.0),
    )


def run_panel_lp_asymmetry(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    config: PanelLPConfig,
    *,
    shock_col: str,
    split: Split = "sign",
    outcome_col: str = "log_employment",
) -> pl.DataFrame:
    """Estimate the two-component (asymmetric) panel LP across response horizons.

    Returns
    -------
    polars.DataFrame
        One row per response horizon: ``beta_a``/``beta_b`` (with the split labels
        in ``label_a``/``label_b``), their SEs, the difference ``diff``, its SE,
        the Wald p-value ``p_diff``, and ``p_diff_bh`` (BH-FDR across horizons).
    """
    label_a, label_b = _LABELS[split]
    split_shock = _split_shock(shock, shock_col, split)
    base = (
        panel.join(exposure, on="supersector_code", how="inner")
        .join(split_shock.select("date", "s_a", "s_b"), on="date", how="inner")
        .sort(["unit_id", "date"])
        .with_columns(
            treat_a=pl.col("exposure") * pl.col("s_a"),
            treat_b=pl.col("exposure") * pl.col("s_b"),
        )
    )
    controls = [f"dy_l{lag}" for lag in range(1, config.n_control_lags + 1)]
    prepared = base.with_columns(
        [
            (pl.col(outcome_col).shift(lag) - pl.col(outcome_col).shift(lag + 1))
            .over("unit_id")
            .alias(f"dy_l{lag}")
            for lag in range(1, config.n_control_lags + 1)
        ]
    )
    z = float(norm.ppf(0.5 + config.confidence_level / 2.0))
    rows: list[dict[str, object]] = []
    for h in config.horizons:
        if h < 0:
            continue
        outcome = (pl.col(outcome_col).shift(-h) - pl.col(outcome_col).shift(1)).over(
            "unit_id"
        )
        frame = (
            prepared.with_columns(outcome=outcome)
            .select(["unit_id", "date", "outcome", "treat_a", "treat_b", *controls])
            .drop_nulls()
        )
        pdf = frame.to_pandas().set_index(["unit_id", "date"])
        exog = pdf[["treat_a", "treat_b", *controls]]
        fit = PanelOLS(
            pdf["outcome"], exog, entity_effects=True, time_effects=True
        ).fit(cov_type="kernel", kernel="bartlett")
        beta_a = float(fit.params["treat_a"])
        beta_b = float(fit.params["treat_b"])
        se_a = float(fit.std_errors["treat_a"])
        se_b = float(fit.std_errors["treat_b"])
        names = list(fit.params.index)
        idx_a, idx_b = names.index("treat_a"), names.index("treat_b")
        cov = np.asarray(fit.cov, dtype=float)
        var_diff = float(
            cov[idx_a, idx_a] + cov[idx_b, idx_b] - 2.0 * cov[idx_a, idx_b]
        )
        diff = beta_a - beta_b
        se_diff = math.sqrt(var_diff) if var_diff > 0 else float("nan")
        p_diff = (
            float(2.0 * (1.0 - norm.cdf(abs(diff / se_diff))))
            if se_diff > 0
            else float("nan")
        )
        rows.append(
            {
                "horizon": float(h),
                "label_a": label_a,
                "label_b": label_b,
                "beta_a": beta_a,
                "se_a": se_a,
                "beta_b": beta_b,
                "se_b": se_b,
                "diff": diff,
                "se_diff": se_diff,
                "ci_low": diff - z * se_diff,
                "ci_high": diff + z * se_diff,
                "p_diff": p_diff,
            }
        )
    result = pl.DataFrame(rows).sort("horizon")
    adjusted = bh_adjust(result["p_diff"].to_numpy())
    return result.with_columns(p_diff_bh=pl.Series(adjusted))
