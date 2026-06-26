"""Placebo / permutation tests for the headline relative effect.

Two randomizations that should yield a null: permuting the shock across time
(breaking the timing) and permuting exposure across supersectors (breaking the
cross-sectional assignment). A non-null placebo is a red flag to investigate, not
to hide.
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
