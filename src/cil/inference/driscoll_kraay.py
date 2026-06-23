"""Driscoll-Kraay standard errors for panel regressions.

Driscoll-Kraay (1998) SEs are robust to cross-sectional dependence and serial
correlation. We implement the Hoechle (2007) estimator: form the regressor-times-
residual moments, sum them across entities within each period, and apply a
Newey-West HAC to the resulting time series. This is the primary inference for
the headline panel local projection.

References
----------
Driscoll & Kraay (1998), Review of Economics and Statistics 80(4);
Hoechle (2007), Stata Journal 7(3).
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]


def default_bandwidth(n_periods: int) -> int:
    """Return the default HAC bandwidth ``floor(4 (T/100)^(2/9))``."""
    return int(np.floor(4.0 * (n_periods / 100.0) ** (2.0 / 9.0)))


def driscoll_kraay_cov(
    x: FloatArray,
    residuals: FloatArray,
    time_codes: npt.NDArray[np.int_],
    *,
    bandwidth: int | None = None,
) -> FloatArray:
    """Compute the Driscoll-Kraay covariance of OLS coefficients.

    Parameters
    ----------
    x
        Regressor matrix ``(N, k)`` (include any constant/within-transform
        already applied).
    residuals
        OLS residuals ``(N,)``.
    time_codes
        Integer period index ``(N,)`` aligning each row to a time period.
    bandwidth
        Newey-West truncation lag; defaults to
        :func:`default_bandwidth` of the number of periods.

    Returns
    -------
    numpy.ndarray
        The ``(k, k)`` coefficient covariance matrix.
    """
    x = np.asarray(x, dtype=np.float64)
    residuals = np.asarray(residuals, dtype=np.float64)
    k = x.shape[1]
    moments = x * residuals[:, None]  # (N, k)

    periods = np.unique(time_codes)
    n_periods = periods.size
    # Sum moments across entities within each period -> h_t (T, k).
    h = np.zeros((n_periods, k), dtype=np.float64)
    for j, period in enumerate(periods):
        h[j] = moments[time_codes == period].sum(axis=0)

    band = default_bandwidth(n_periods) if bandwidth is None else bandwidth
    s = h.T @ h  # lag 0
    for lag in range(1, band + 1):
        weight = 1.0 - lag / (band + 1.0)
        gamma = h[lag:].T @ h[:-lag]
        s += weight * (gamma + gamma.T)

    xtx_inv = np.linalg.inv(x.T @ x)
    cov = xtx_inv @ s @ xtx_inv
    return (cov + cov.T) / 2.0  # enforce symmetry against fp drift


def coefficient_se(cov: FloatArray) -> FloatArray:
    """Return standard errors (sqrt of the diagonal) from a covariance matrix."""
    return np.sqrt(np.diag(np.asarray(cov, dtype=np.float64)))
