"""Structural-break detection (Bai-Perron).

Detects multiple structural breaks in a univariate series (e.g. national
employment growth). Two entry points:

- :func:`bai_perron` -- a fast PELT approximation (penalised), kept for cheap
  diagnostics.
- :func:`bai_perron_full` -- the full procedure: **exact** dynamic-programming
  global SSR minimisation for each number of breaks, break-number selection by
  BIC, and **bootstrap** confidence intervals for the break dates (a
  self-contained alternative to Bai's asymptotic-constant interval).

References
----------
Bai & Perron (2003), *Computation and analysis of multiple structural change
models*, Journal of Applied Econometrics 18(1); Bai (1997) for break-date CIs.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import polars as pl
import ruptures as rpt

FloatArray = npt.NDArray[np.float64]


def detect_breaks(
    values: npt.ArrayLike, *, penalty_scale: float = 1.0, min_size: int = 12
) -> list[int]:
    """Return break indices via PELT with a BIC-style penalty.

    Parameters
    ----------
    values
        Univariate series.
    penalty_scale
        Multiplier on the ``scale * log(n) * variance`` penalty; larger means
        fewer breaks.
    min_size
        Minimum segment length (in observations) between breaks.

    Returns
    -------
    list of int
        Break indices (exclusive of the series endpoint).
    """
    arr = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    n = arr.shape[0]
    variance = float(np.var(arr)) or 1.0
    penalty = penalty_scale * np.log(n) * variance
    algo = rpt.Pelt(model="l2", min_size=min_size).fit(arr)
    breaks = algo.predict(pen=penalty)
    return [b for b in breaks if b < n]


def bai_perron(
    frame: pl.DataFrame,
    value_col: str,
    *,
    date_col: str = "date",
    penalty_scale: float = 1.0,
    min_size: int = 12,
) -> pl.DataFrame:
    """Detect structural breaks and return their dates.

    Parameters
    ----------
    frame
        Frame sorted by ``date_col`` with the series in ``value_col``.
    value_col
        Column holding the series.
    date_col
        Column holding the dates.
    penalty_scale, min_size
        Passed to :func:`detect_breaks`.

    Returns
    -------
    polars.DataFrame
        Column ``break_date`` with one row per detected break.
    """
    ordered = frame.sort(date_col).drop_nulls(value_col)
    indices = detect_breaks(
        ordered[value_col].to_numpy(), penalty_scale=penalty_scale, min_size=min_size
    )
    dates = ordered[date_col].to_list()
    return pl.DataFrame({"break_date": [dates[i] for i in indices if i < len(dates)]})


def _segment_ssr(arr: FloatArray, bkps: list[int]) -> float:
    """Total within-segment sum of squared residuals for a segmentation."""
    ssr, start = 0.0, 0
    for end in bkps:
        seg = arr[start:end]
        if seg.size:
            ssr += float(((seg - seg.mean()) ** 2).sum())
        start = end
    return ssr


def _segment_means(arr: FloatArray, bkps: list[int]) -> list[float]:
    """Per-segment means for a segmentation (``bkps`` end indices incl. ``n``)."""
    means, start = [], 0
    for end in bkps:
        means.append(float(arr[start:end].mean()))
        start = end
    return means


def _exact_segmentation(
    arr: FloatArray, n_bkps: int, min_size: int
) -> list[int] | None:
    """Exact (dynamic-programming) global-SSR segmentation with ``n_bkps`` breaks."""
    if n_bkps == 0:
        return [arr.shape[0]]
    try:
        algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(arr.reshape(-1, 1))
        return list(algo.predict(n_bkps=n_bkps))
    except Exception:  # infeasible: too many breaks for the trimming
        return None


def _bootstrap_break_cis(
    arr: FloatArray,
    bkps: list[int],
    n_bkps: int,
    min_size: int,
    n_boot: int,
    seed: int,
) -> dict[int, tuple[float, float]]:
    """Residual-bootstrap 95% CIs (in indices) for each break date."""
    rng = np.random.default_rng(seed)
    n = arr.shape[0]
    means = _segment_means(arr, bkps)
    fitted = np.empty(n, dtype=np.float64)
    start = 0
    for k, end in enumerate(bkps):
        fitted[start:end] = means[k]
        start = end
    resid = arr - fitted
    draws: list[list[int]] = [[] for _ in range(n_bkps)]
    for _ in range(n_boot):
        x = fitted + rng.choice(resid, size=n, replace=True)
        boot = _exact_segmentation(x, n_bkps, min_size)
        if boot is None:
            continue
        idx = [b for b in boot if b < n]
        if len(idx) == n_bkps:
            for j in range(n_bkps):
                draws[j].append(idx[j])
    ci: dict[int, tuple[float, float]] = {}
    for j in range(n_bkps):
        if draws[j]:
            ci[j] = (
                float(np.percentile(draws[j], 2.5)),
                float(np.percentile(draws[j], 97.5)),
            )
    return ci


def bai_perron_full(
    frame: pl.DataFrame,
    value_col: str,
    *,
    date_col: str = "date",
    max_breaks: int = 5,
    min_size: int = 12,
    n_boot: int = 200,
    seed: int = 20260101,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Full Bai-Perron: exact DP segmentation, BIC selection, bootstrap CIs.

    Parameters
    ----------
    frame
        Frame sorted by ``date_col`` with the series in ``value_col``.
    value_col, date_col
        Column names.
    max_breaks
        Maximum number of breaks to consider.
    min_size
        Minimum segment length (trimming).
    n_boot
        Bootstrap replications for the break-date CIs.
    seed
        Random seed.

    Returns
    -------
    breaks : polars.DataFrame
        One row per selected break: ``break_date``, ``ci_low_date``,
        ``ci_high_date``, ``delta`` (mean shift across the break).
    selection : polars.DataFrame
        One row per candidate model: ``n_breaks``, ``ssr``, ``bic``, ``selected``.
    """
    ordered = frame.sort(date_col).drop_nulls(value_col)
    arr = ordered[value_col].to_numpy().astype(np.float64)
    dates = ordered[date_col].to_list()
    n = arr.shape[0]

    # Fit the DP cost matrix once and reuse it across the candidate break counts.
    base_algo = rpt.Dynp(model="l2", min_size=min_size, jump=1).fit(arr.reshape(-1, 1))
    models: dict[int, tuple[list[int], float, float]] = {}
    for m in range(max_breaks + 1):
        if m == 0:
            bkps: list[int] | None = [n]
        else:
            try:
                bkps = list(base_algo.predict(n_bkps=m))
            except Exception:  # infeasible: too many breaks for the trimming
                break
        if bkps is None:
            break
        ssr = _segment_ssr(arr, bkps)
        n_params = 2 * m + 1  # (m+1) segment means + m break locations
        bic = n * np.log(ssr / n) + n_params * np.log(n)
        models[m] = (bkps, ssr, float(bic))

    m_star = min(models, key=lambda k: models[k][2])
    bkps_star = models[m_star][0]
    break_idx = [b for b in bkps_star if b < n]
    means = _segment_means(arr, bkps_star)
    ci = (
        _bootstrap_break_cis(arr, bkps_star, m_star, min_size, n_boot, seed)
        if m_star > 0
        else {}
    )

    rows = []
    for j, bi in enumerate(break_idx):
        lo, hi = ci.get(j, (float(bi), float(bi)))
        lo_i = int(np.clip(round(lo), 0, n - 1))
        hi_i = int(np.clip(round(hi), 0, n - 1))
        rows.append(
            {
                "break_date": dates[bi],
                "ci_low_date": dates[lo_i],
                "ci_high_date": dates[hi_i],
                "delta": means[j + 1] - means[j],
            }
        )
    breaks_df = (
        pl.DataFrame(rows)
        if rows
        else pl.DataFrame(
            schema={
                "break_date": ordered[date_col].dtype,
                "ci_low_date": ordered[date_col].dtype,
                "ci_high_date": ordered[date_col].dtype,
                "delta": pl.Float64,
            }
        )
    )
    selection = pl.DataFrame(
        [
            {
                "n_breaks": m,
                "ssr": models[m][1],
                "bic": models[m][2],
                "selected": m == m_star,
            }
            for m in sorted(models)
        ]
    )
    return breaks_df, selection
