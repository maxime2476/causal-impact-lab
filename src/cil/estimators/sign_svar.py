"""Sign-restricted structural VAR — a third aggregate identification.

A contractionary monetary shock is identified by **sign restrictions** on the
impulse responses: the policy rate rises and the price level falls over the
restriction window, while output and employment are left **unrestricted** so the
data determine the employment response (Uhlig 2005-style agnostic identification).
This is a third assumption-dependent aggregate complement, alongside the
high-frequency-instrument LP-IV and the narrative shock.

References
----------
Uhlig (2005), *What are the effects of monetary policy on output?*, JME 52(2);
Rubio-Ramirez, Waggoner & Zha (2010) for the orthogonal-rotation draw.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
from statsmodels.tsa.api import VAR

FloatArray = npt.NDArray[np.float64]


def _haar_orthogonal(rng: np.random.Generator, n: int) -> FloatArray:
    """Draw an ``n x n`` orthogonal matrix from the Haar distribution."""
    a = rng.standard_normal((n, n))
    q, r = np.linalg.qr(a)
    return q * np.sign(np.diag(r))


def sign_restricted_svar(
    data: pl.DataFrame,
    var_order: list[str],
    *,
    rate: str,
    price: str,
    target: str,
    n_lags: int = 12,
    horizons: tuple[int, ...] = tuple(range(25)),
    restrict_horizons: tuple[int, ...] = (0, 1, 2, 3, 4, 5),
    n_draws: int = 2000,
    seed: int = 20260101,
) -> tuple[pl.DataFrame, float]:
    """Identify the contractionary monetary shock by sign restrictions.

    Parameters
    ----------
    data
        Frame with the (already-transformed) VAR variables as columns.
    var_order
        Column names in the VAR (order is irrelevant to sign restrictions).
    rate, price, target
        Column names of the policy rate (restricted up), the price level
        (restricted down), and the reported response (e.g. employment).
    n_lags
        VAR lag order.
    horizons
        Response horizons to report.
    restrict_horizons
        Horizons over which the sign restrictions are imposed.
    n_draws
        Random orthogonal rotations to draw.
    seed
        Random seed.

    Returns
    -------
    irf : polars.DataFrame
        ``horizon``, ``median``, ``lo_68``, ``hi_68`` of the *target* response to
        a one-SD contractionary monetary shock (in percent), over accepted draws.
    acceptance_rate : float
        Fraction of rotations satisfying the sign restrictions.
    """
    y = data.select(var_order).to_numpy().astype(np.float64)
    results = VAR(y).fit(n_lags)
    sigma = np.asarray(results.sigma_u, dtype=np.float64)
    chol = np.linalg.cholesky(sigma)
    max_h = max(max(horizons), max(restrict_horizons))
    psi = np.asarray(results.ma_rep(maxn=max_h), dtype=np.float64)  # (H+1, n, n)

    idx = {name: var_order.index(name) for name in (rate, price, target)}
    restrict_set = set(restrict_horizons)
    rng = np.random.default_rng(seed)
    n = len(var_order)

    accepted: list[list[float]] = []
    for _ in range(n_draws):
        q = _haar_orthogonal(rng, n)[:, 0]
        if (chol @ q)[idx[rate]] < 0:
            q = -q  # orient so the rate rises on impact
        responses = {}
        ok = True
        for h in range(max_h + 1):
            irf = psi[h] @ chol @ q
            if h in restrict_set and not (irf[idx[rate]] > 0 and irf[idx[price]] < 0):
                ok = False
                break
            responses[h] = float(irf[idx[target]])
        if ok:
            accepted.append([responses[h] * 100.0 for h in horizons])

    acceptance_rate = len(accepted) / n_draws
    if not accepted:
        empty = pl.DataFrame(
            {"horizon": list(horizons), "median": [], "lo_68": [], "hi_68": []}
        )
        return empty, acceptance_rate
    arr = np.asarray(accepted)  # (n_accepted, n_horizons)
    return (
        pl.DataFrame(
            {
                "horizon": [float(h) for h in horizons],
                "median": np.median(arr, axis=0),
                "lo_68": np.percentile(arr, 16, axis=0),
                "hi_68": np.percentile(arr, 84, axis=0),
            }
        ),
        acceptance_rate,
    )
