"""Placebo / permutation and randomization-inference tests.

Placebo permutations (:func:`permutation_test`): permuting the shock across time
or exposure across supersectors, which should yield a null. **iid** permutation of
the shock, however, destroys its serial correlation, so its placebo distribution
does not respect the design.

Randomization inference (:func:`circular_shift_ri`): the design-respecting test
for a serially-correlated time-series treatment. Circularly shifting the shock by
a random offset preserves its autocovariance exactly while breaking its alignment
with the outcomes, giving a valid sharp-null distribution. Reports per-horizon RI
p-values and a joint ``max|beta|`` p-value (family-wise across horizons).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from pydantic import BaseModel

from cil.estimators.panel_lp import PanelLPConfig, run_panel_lp

PlaceboMode = Literal["shock", "exposure"]


class PlaceboResult(BaseModel):
    """Outcome of a permutation test at one horizon.

    Parameters
    ----------
    mode
        Which assignment was permuted.
    horizon
        Response horizon tested.
    actual_beta
        The estimated coefficient on the real data.
    placebo_p_value
        Two-sided permutation p-value: the share of placebo draws whose absolute
        coefficient is at least the actual (with the usual ``+1`` smoothing).
    n_permutations
        Number of placebo draws.
    placebo_mean
        Mean placebo coefficient (should be near zero).
    """

    mode: PlaceboMode
    horizon: int
    actual_beta: float
    placebo_p_value: float
    n_permutations: int
    placebo_mean: float


def _beta(
    panel: pl.DataFrame, exposure: pl.DataFrame, shock: pl.DataFrame, col: str, h: int
) -> float:
    result = run_panel_lp(
        panel, exposure, shock, PanelLPConfig(horizons=(h,)), shock_col=col
    )
    return float(result["beta"][0])


def permutation_test(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizon: int,
    mode: PlaceboMode = "shock",
    n_permutations: int = 200,
    seed: int = 20260101,
) -> PlaceboResult:
    """Run a permutation test and return the placebo p-value.

    Parameters
    ----------
    panel, exposure, shock
        Headline inputs.
    shock_col
        Name of the shock column.
    horizon
        Response horizon to test.
    mode
        ``"shock"`` permutes the shock across dates; ``"exposure"`` permutes the
        exposure values across supersectors.
    n_permutations
        Number of placebo draws.
    seed
        Random seed.

    Returns
    -------
    PlaceboResult
        Actual coefficient and the permutation p-value.
    """
    rng = np.random.default_rng(seed)
    actual = _beta(panel, exposure, shock, shock_col, horizon)
    placebo_betas: list[float] = []
    for _ in range(n_permutations):
        if mode == "shock":
            permuted = shock.with_columns(
                pl.Series(shock_col, rng.permutation(shock[shock_col].to_numpy()))
            )
            placebo_betas.append(_beta(panel, exposure, permuted, shock_col, horizon))
        else:
            permuted = exposure.with_columns(
                pl.Series("exposure", rng.permutation(exposure["exposure"].to_numpy()))
            )
            placebo_betas.append(_beta(panel, permuted, shock, shock_col, horizon))
    draws = np.asarray(placebo_betas)
    at_least = int(np.sum(np.abs(draws) >= abs(actual)))
    p_value = (1 + at_least) / (n_permutations + 1)
    return PlaceboResult(
        mode=mode,
        horizon=horizon,
        actual_beta=actual,
        placebo_p_value=p_value,
        n_permutations=n_permutations,
        placebo_mean=float(draws.mean()),
    )


def circular_shift_ri(
    panel: pl.DataFrame,
    exposure: pl.DataFrame,
    shock: pl.DataFrame,
    *,
    shock_col: str,
    horizons: tuple[int, ...],
    n_draws: int = 200,
    seed: int = 20260101,
) -> tuple[pl.DataFrame, float]:
    """Circular-shift randomization inference for the headline coefficient.

    Each draw circularly shifts the (date-sorted) shock by a random offset,
    preserving its autocovariance, and re-estimates the panel LP at *horizons*.
    Returns per-horizon RI p-values and the joint ``max|beta|`` p-value.

    Parameters
    ----------
    panel, exposure, shock
        Headline inputs.
    shock_col
        Name of the shock column.
    horizons
        Response horizons to test jointly.
    n_draws
        Number of circular shifts.
    seed
        Random seed.

    Returns
    -------
    per_horizon : polars.DataFrame
        ``horizon``, ``actual_beta``, ``ri_p_value``, ``ri_mean``.
    joint_p_value : float
        RI p-value for the ``max|beta|`` statistic across *horizons*.
    """
    rng = np.random.default_rng(seed)
    config = PanelLPConfig(horizons=horizons)
    ordered = shock.sort("date")
    values = ordered[shock_col].to_numpy()
    n_t = values.shape[0]

    actual = run_panel_lp(panel, exposure, ordered, config, shock_col=shock_col)
    actual_beta = {
        int(h): float(actual.filter(pl.col("horizon") == h)["beta"][0])
        for h in horizons
    }
    draw_beta: dict[int, list[float]] = {int(h): [] for h in horizons}
    draw_max: list[float] = []
    for _ in range(n_draws):
        shift = int(rng.integers(1, n_t))
        shifted = ordered.with_columns(pl.Series(shock_col, np.roll(values, shift)))
        res = run_panel_lp(panel, exposure, shifted, config, shock_col=shock_col)
        betas = {
            int(h): float(res.filter(pl.col("horizon") == h)["beta"][0])
            for h in horizons
        }
        for h in horizons:
            draw_beta[int(h)].append(betas[int(h)])
        draw_max.append(max(abs(b) for b in betas.values()))

    rows = []
    for h in horizons:
        d = np.abs(np.asarray(draw_beta[int(h)]))
        a = abs(actual_beta[int(h)])
        rows.append(
            {
                "horizon": int(h),
                "actual_beta": actual_beta[int(h)],
                "ri_p_value": (1 + int(np.sum(d >= a))) / (n_draws + 1),
                "ri_mean": float(np.mean(draw_beta[int(h)])),
            }
        )
    actual_max = max(abs(v) for v in actual_beta.values())
    joint_p = (1 + int(np.sum(np.asarray(draw_max) >= actual_max))) / (n_draws + 1)
    return pl.DataFrame(rows), joint_p
