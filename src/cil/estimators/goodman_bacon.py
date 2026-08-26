"""Goodman-Bacon diagnostics for staggered/non-absorbing treatment.

Two entry points:

- :func:`bacon_diagnostic` -- a light check: the static TWFE estimate vs the
  clean-control LP-DiD estimate and the already-treated ("forbidden") share.
- :func:`bacon_decompose` -- the **full** Goodman-Bacon decomposition: the TWFE
  coefficient written as the weighted sum of every 2x2 DiD between timing groups
  (treated-vs-never, earlier-vs-later, later-vs-already-treated), with the exact
  balanced-panel weights. It reports the weight on the "forbidden" later-vs-
  already-treated comparisons and verifies the decomposition reproduces TWFE.

The headline design is *not* a staggered DiD (the shock hits all cells at once,
differentiated by exposure), so :func:`build_staggered_treatment` constructs a
staggered treatment to decompose: a cell adopts (absorbing) once its
exposure-weighted cumulative tightening crosses a threshold, so high-exposure
cells adopt earlier. The decomposition then quantifies how much a naive TWFE on
such a design would lean on forbidden comparisons.

References
----------
Goodman-Bacon (2021), *Difference-in-differences with variation in treatment
timing*, Journal of Econometrics 225(2).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from linearmodels.panel import PanelOLS
from pydantic import BaseModel

from cil.estimators.lp_did import LPDiDConfig, lp_did

FloatArray = npt.NDArray[np.float64]


class BaconDiagnostic(BaseModel):
    """Comparison of TWFE and clean-control estimates.

    Parameters
    ----------
    twfe_estimate
        Static TWFE coefficient on the treatment level.
    clean_estimate
        Clean-control LP-DiD estimate at horizon 0.
    difference
        ``twfe_estimate - clean_estimate``; large magnitude flags negative
        weighting.
    forbidden_share
        Share of observations that are already-treated (used as controls by
        TWFE but excluded by the clean-control condition).
    """

    twfe_estimate: float
    clean_estimate: float
    difference: float
    forbidden_share: float


def twfe_estimate(
    data: pl.DataFrame,
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    outcome_col: str = "log_employment",
    treat_col: str = "treated",
) -> float:
    """Return the static TWFE coefficient on the treatment level."""
    pdf = data.select([unit_col, time_col, outcome_col, treat_col]).to_pandas()
    pdf = pdf.set_index([unit_col, time_col])
    result = PanelOLS(
        pdf[outcome_col], pdf[[treat_col]], entity_effects=True, time_effects=True
    ).fit(cov_type="clustered", cluster_entity=True)
    return float(result.params[treat_col])


def forbidden_share(
    data: pl.DataFrame, *, unit_col: str = "unit_id", treat_col: str = "treated"
) -> float:
    """Return the share of already-treated observations (forbidden controls)."""
    flagged = data.sort([unit_col, "date"]).with_columns(
        already=(pl.col(treat_col).shift(1) == 1).over(unit_col)
    )
    flags = flagged["already"].fill_null(value=False).to_numpy().astype(float)
    return float(flags.mean())


def bacon_diagnostic(
    data: pl.DataFrame,
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    outcome_col: str = "log_employment",
    treat_col: str = "treated",
) -> BaconDiagnostic:
    """Compute the TWFE-vs-clean diagnostic and the forbidden-comparison share.

    Parameters
    ----------
    data
        Long panel with unit, time, outcome, and a binary treatment column.
    unit_col, time_col, outcome_col, treat_col
        Column names.

    Returns
    -------
    BaconDiagnostic
        The comparison summary.
    """
    twfe = twfe_estimate(
        data,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treat_col=treat_col,
    )
    clean_series = lp_did(
        data,
        LPDiDConfig(horizons=(0,)),
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treat_col=treat_col,
    )["att"]
    clean = float(clean_series.to_numpy()[0])
    share = forbidden_share(data, unit_col=unit_col, treat_col=treat_col)
    return BaconDiagnostic(
        twfe_estimate=twfe,
        clean_estimate=clean,
        difference=twfe - clean,
        forbidden_share=share,
    )


def build_staggered_treatment(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    policy: pl.DataFrame,
    *,
    adopt_quantile: float = 0.6,
) -> pl.DataFrame:
    """Construct an absorbing, annually-staggered treatment for the decomposition.

    A cell adopts (absorbing, from January of the crossing year) once its
    exposure-weighted cumulative monetary tightening crosses a threshold — so
    high-exposure cells adopt earlier and negative/low-exposure cells never adopt
    (the never-treated group).

    Parameters
    ----------
    panel
        Cell panel (``unit_id``, ``supersector_code``, ``date``,
        ``log_employment``).
    exposure
        Cell exposure (``supersector_code``, ``exposure``).
    policy
        Policy-rate frame (``date``, ``policy_rate``).
    adopt_quantile
        Quantile of the per-cell maximum pressure used as the adoption threshold.

    Returns
    -------
    polars.DataFrame
        ``unit_id``, ``date``, ``log_employment``, ``treated`` (0/1, absorbing).
    """
    cum = (
        policy.sort("date")
        .with_columns(d=pl.col("policy_rate").diff())
        .with_columns(tighten=pl.when(pl.col("d") > 0).then(pl.col("d")).otherwise(0.0))
        .with_columns(cum=pl.col("tighten").cum_sum())
        .select("date", "cum")
    )
    base = (
        panel.join(exposure, on="supersector_code", how="inner")
        .join(cum, on="date", how="inner")
        .with_columns(pressure=pl.col("exposure") * pl.col("cum"))
    )
    tau_q = (
        base.group_by("unit_id")
        .agg(maxp=pl.col("pressure").max())["maxp"]
        .quantile(adopt_quantile)
    )
    tau = float(tau_q) if tau_q is not None else 0.0
    adopt = (
        base.filter(pl.col("pressure") >= tau)
        .group_by("unit_id")
        .agg(adopt_year=pl.col("date").min().dt.year())
    )
    return (
        panel.join(adopt, on="unit_id", how="left")
        .with_columns(
            treated=pl.when(
                pl.col("adopt_year").is_not_null()
                & (pl.col("date").dt.year() >= pl.col("adopt_year"))
            )
            .then(1)
            .otherwise(0)
        )
        .select("unit_id", "date", "log_employment", "treated")
    )


class BaconDecomposition(BaseModel):
    """Summary of the full Goodman-Bacon decomposition.

    Parameters
    ----------
    twfe
        The static TWFE coefficient (PanelOLS).
    twfe_implied
        The weighted sum of 2x2 estimates; should equal ``twfe`` (identity check).
    weight_treated_vs_untreated, weight_earlier_vs_later, weight_later_vs_earlier
        Total (normalised) Bacon weight in each comparison category.
    forbidden_weight
        Weight on the later-vs-already-treated ("forbidden") comparisons
        (== ``weight_later_vs_earlier``); large values warn of TWFE bias under
        heterogeneous dynamics.
    n_cohorts
        Number of adoption cohorts (excluding never-treated).
    """

    twfe: float
    twfe_implied: float
    weight_treated_vs_untreated: float
    weight_earlier_vs_later: float
    weight_later_vs_earlier: float
    forbidden_weight: float
    n_cohorts: int


def _did_2x2(
    y_treat: FloatArray, y_ctrl: FloatArray, adopt: int, lo: int, hi: int
) -> float:
    """2x2 DiD from group-period means over window ``[lo, hi)`` split at ``adopt``."""
    pre_t = y_treat[lo:adopt].mean()
    post_t = y_treat[adopt:hi].mean()
    pre_c = y_ctrl[lo:adopt].mean()
    post_c = y_ctrl[adopt:hi].mean()
    return float((post_t - pre_t) - (post_c - pre_c))


def bacon_decompose(
    data: pl.DataFrame,
    *,
    unit_col: str = "unit_id",
    time_col: str = "date",
    outcome_col: str = "log_employment",
    treat_col: str = "treated",
) -> tuple[pl.DataFrame, BaconDecomposition]:
    """Full Goodman-Bacon decomposition of the TWFE estimate.

    Returns
    -------
    components : polars.DataFrame
        One row per 2x2 comparison: ``category``, ``weight`` (normalised),
        ``estimate``, ``cohort_k``, ``cohort_l``.
    summary : BaconDecomposition
        Category weights and the TWFE identity check.
    """
    df = data.select(unit_col, time_col, outcome_col, treat_col)
    periods = df[time_col].unique().sort().to_list()
    n_t = len(periods)
    idx_frame = pl.DataFrame(
        {time_col: periods, "_tidx": list(range(n_t))}
    ).with_columns(pl.col("_tidx").cast(pl.Int64))
    df = df.join(idx_frame, on=time_col, how="inner")

    # Balanced panel only (the closed-form weights assume it). The TWFE identity
    # is checked on this same balanced subset.
    counts = df.group_by(unit_col).agg(n=pl.len())
    full = counts.filter(pl.col("n") == n_t).select(unit_col)
    df = df.join(full, on=unit_col, how="inner")
    balanced = data.join(full, on=unit_col, how="inner")

    adopt = (
        df.filter(pl.col(treat_col) == 1)
        .group_by(unit_col)
        .agg(a=pl.col("_tidx").min())
    )
    df = df.join(adopt, on=unit_col, how="left").with_columns(
        cohort=pl.col("a").fill_null(-1)
    )

    unit_cohort = df.group_by(unit_col).agg(cohort=pl.col("cohort").first())
    n_units = unit_cohort.height
    cohort_sizes = {
        int(r["cohort"]): int(r["n"])
        for r in unit_cohort.group_by("cohort").agg(n=pl.len()).to_dicts()
    }
    means = (
        df.group_by(["cohort", "_tidx"])
        .agg(y=pl.col(outcome_col).mean())
        .sort(["cohort", "_tidx"])
    )
    m: dict[int, FloatArray] = {}
    for cohort, sub in means.partition_by("cohort", as_dict=True).items():
        key = cohort[0] if isinstance(cohort, tuple) else cohort
        row = np.full(n_t, np.nan, dtype=np.float64)
        row[sub["_tidx"].to_numpy()] = sub["y"].to_numpy()
        m[int(key)] = row

    treated_cohorts = sorted(c for c in cohort_sizes if c >= 0)
    has_never = -1 in cohort_sizes
    n = {c: cohort_sizes[c] / n_units for c in cohort_sizes}
    dbar = {c: (n_t - c) / n_t for c in treated_cohorts}
    if has_never:
        dbar[-1] = 0.0

    rows: list[dict[str, object]] = []
    raw_weights: list[float] = []
    # Treated k vs never-treated U (needs a pre-period: 0 < k < T).
    if has_never:
        for k in treated_cohorts:
            if not 0 < k < n_t:
                continue
            f = n[k] / (n[k] + n[-1])
            v = f * (1 - f) * dbar[k] * (1 - dbar[k])
            w = (n[k] + n[-1]) ** 2 * v
            est = _did_2x2(m[k], m[-1], k, 0, n_t)
            raw_weights.append(w)
            rows.append(
                {
                    "category": "treated_vs_untreated",
                    "estimate": est,
                    "cohort_k": k,
                    "cohort_l": -1,
                }
            )
    # Timing-group pairs (k earlier than l).
    for i, k in enumerate(treated_cohorts):
        for lg in treated_cohorts[i + 1 :]:
            f = n[k] / (n[k] + n[lg])
            # Earlier k (treated) vs later l (not-yet-treated control), window [0, l).
            # Needs a pre-period for k within that window: 0 < k < l.
            if 0 < k < lg:
                v_e = (
                    f
                    * (1 - f)
                    * ((dbar[k] - dbar[lg]) / (1 - dbar[lg]))
                    * ((1 - dbar[k]) / (1 - dbar[lg]))
                )
                w_e = ((n[k] + n[lg]) * (1 - dbar[lg])) ** 2 * v_e
                est_e = _did_2x2(m[k], m[lg], k, 0, lg)
                raw_weights.append(w_e)
                rows.append(
                    {
                        "category": "earlier_vs_later",
                        "estimate": est_e,
                        "cohort_k": k,
                        "cohort_l": lg,
                    }
                )
            # Later l (treated) vs earlier k (already-treated control), window [k, T).
            v_l = f * (1 - f) * (dbar[lg] / dbar[k]) * ((dbar[k] - dbar[lg]) / dbar[k])
            w_l = ((n[k] + n[lg]) * dbar[k]) ** 2 * v_l
            est_l = _did_2x2(m[lg], m[k], lg, k, n_t)
            raw_weights.append(w_l)
            rows.append(
                {
                    "category": "later_vs_earlier",
                    "estimate": est_l,
                    "cohort_k": k,
                    "cohort_l": lg,
                }
            )

    total_w = sum(raw_weights)
    for r, rw in zip(rows, raw_weights, strict=True):
        r["weight"] = rw / total_w if total_w > 0 else 0.0
    components = pl.DataFrame(rows)

    def _cat_weight(cat: str) -> float:
        sub = components.filter(pl.col("category") == cat)
        return float(sub["weight"].sum()) if sub.height else 0.0

    twfe_implied = float((components["weight"] * components["estimate"]).sum())
    twfe = twfe_estimate(
        balanced,
        unit_col=unit_col,
        time_col=time_col,
        outcome_col=outcome_col,
        treat_col=treat_col,
    )
    summary = BaconDecomposition(
        twfe=twfe,
        twfe_implied=twfe_implied,
        weight_treated_vs_untreated=_cat_weight("treated_vs_untreated"),
        weight_earlier_vs_later=_cat_weight("earlier_vs_later"),
        weight_later_vs_earlier=_cat_weight("later_vs_earlier"),
        forbidden_weight=_cat_weight("later_vs_earlier"),
        n_cohorts=len(treated_cohorts),
    )
    return components, summary
