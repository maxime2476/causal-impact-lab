"""Wild cluster bootstrap for a regression coefficient.

A cross-check on the Driscoll-Kraay inference for the headline coefficient,
clustering by state. Uses Rademacher weights at the cluster level on an
unrestricted residual bootstrap; the caller supplies the (within-transformed)
design so fixed effects are already partialled out.

References
----------
Cameron, Gelbach & Miller (2008), Review of Economics and Statistics 90(3).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def _ols_coef(x: FloatArray, y: FloatArray) -> FloatArray:
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    return np.asarray(coef, dtype=np.float64)


def wild_cluster_bootstrap(
    x: FloatArray,
    y: FloatArray,
    clusters: npt.NDArray[np.int_],
    target_index: int,
    *,
    n_boot: int = 999,
    seed: int = 20260101,
) -> FloatArray:
    """Return bootstrap draws of one OLS coefficient under a wild cluster scheme.

    Parameters
    ----------
    x
        Design matrix ``(N, k)`` (fixed effects already partialled out).
    y
        Outcome ``(N,)``.
    clusters
        Integer cluster id per row ``(N,)`` (e.g. state codes).
    target_index
        Column index of the coefficient of interest.
    n_boot
        Number of bootstrap replications.
    seed
        Random seed.

    Returns
    -------
    numpy.ndarray
        ``n_boot`` bootstrap draws of the target coefficient.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    beta = _ols_coef(x, y)
    fitted = x @ beta
    resid = y - fitted

    unique_clusters = np.unique(clusters)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        signs = rng.choice(np.array([-1.0, 1.0]), size=unique_clusters.size)
        sign_map = dict(zip(unique_clusters.tolist(), signs.tolist(), strict=True))
        weights = np.array([sign_map[c] for c in clusters], dtype=np.float64)
        y_star = fitted + resid * weights
        draws[b] = _ols_coef(x, y_star)[target_index]
    return draws


def bootstrap_ci(
    draws: FloatArray, confidence_level: float = 0.95
) -> tuple[float, float]:
    """Return the percentile bootstrap confidence interval from *draws*."""
    alpha = 1.0 - confidence_level
    lower = float(np.quantile(draws, alpha / 2.0))
    upper = float(np.quantile(draws, 1.0 - alpha / 2.0))
    return lower, upper
