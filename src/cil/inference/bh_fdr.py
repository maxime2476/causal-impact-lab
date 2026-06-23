"""Benjamini-Hochberg false-discovery-rate control.

Used to adjust inference across horizons and subgroups; both raw and adjusted
results are reported throughout the project.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


def bh_adjust(p_values: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """Return Benjamini-Hochberg adjusted p-values (q-values).

    Parameters
    ----------
    p_values
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values, same order as the input, each in ``[0, 1]`` and
        monotone in the sorted raw p-values.
    """
    p = np.asarray(p_values, dtype=np.float64)
    n = p.size
    if n == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order]
    factors = n / np.arange(1, n + 1)
    adjusted_sorted = np.minimum.accumulate((ranked * factors)[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)
    out = np.empty(n, dtype=np.float64)
    out[order] = adjusted_sorted
    return out


def bh_reject(p_values: npt.ArrayLike, alpha: float) -> npt.NDArray[np.bool_]:
    """Return the BH reject/accept decisions at level *alpha*.

    Parameters
    ----------
    p_values
        Raw p-values.
    alpha
        Target false-discovery rate in ``(0, 1)``.

    Returns
    -------
    numpy.ndarray of bool
        ``True`` where the null is rejected.
    """
    return np.asarray(bh_adjust(p_values)) <= alpha
