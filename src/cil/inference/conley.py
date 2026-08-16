"""Conley (1999) spatial + serial HAC variance for the panel LP.

Standard errors robust to **spatial** correlation (cells in geographically nearby
states co-move) as well as serial correlation, complementing the Driscoll-Kraay
and two-way exposure-robust bands. The spatial kernel depends only on the
distance between the two cells' **states**, so the score contributions are
aggregated to a ``(state, time)`` grid and the sandwich "meat" is formed as
``sum_t g_t' W g_t`` (plus Bartlett-weighted serial lags), where ``W`` is the
51x51 state kernel -- avoiding an intractable sum over cell pairs.

References
----------
Conley (1999), *GMM estimation with cross sectional dependence*, J. Econometrics
92(1); the space-time HAC combines the spatial kernel with a Newey-West serial
kernel.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Approximate state geographic centroids (FIPS -> (lat, lon), degrees). Used only
#: to build the spatial distance kernel; coarse centroids are adequate for a
#: distance-band weight.
STATE_CENTROIDS: dict[str, tuple[float, float]] = {
    "01": (32.8, -86.8),
    "02": (63.6, -152.0),
    "04": (34.3, -111.7),
    "05": (34.9, -92.4),
    "06": (37.2, -119.4),
    "08": (39.0, -105.5),
    "09": (41.6, -72.7),
    "10": (39.0, -75.5),
    "11": (38.9, -77.0),
    "12": (28.6, -82.4),
    "13": (32.6, -83.4),
    "15": (20.3, -156.4),
    "16": (44.4, -114.6),
    "17": (40.0, -89.2),
    "18": (39.9, -86.3),
    "19": (42.0, -93.5),
    "20": (38.5, -98.4),
    "21": (37.5, -85.3),
    "22": (31.0, -92.0),
    "23": (45.4, -69.2),
    "24": (39.0, -76.8),
    "25": (42.3, -71.8),
    "26": (44.3, -85.4),
    "27": (46.3, -94.3),
    "28": (32.7, -89.7),
    "29": (38.4, -92.5),
    "30": (47.0, -109.6),
    "31": (41.5, -99.8),
    "32": (39.3, -116.9),
    "33": (43.7, -71.6),
    "34": (40.2, -74.7),
    "35": (34.4, -106.1),
    "36": (42.9, -75.5),
    "37": (35.5, -79.4),
    "38": (47.4, -100.5),
    "39": (40.3, -82.8),
    "40": (35.6, -97.5),
    "41": (44.0, -120.5),
    "42": (40.9, -77.8),
    "44": (41.7, -71.5),
    "45": (33.9, -80.9),
    "46": (44.4, -100.2),
    "47": (35.9, -86.4),
    "48": (31.5, -99.3),
    "49": (39.3, -111.7),
    "50": (44.1, -72.7),
    "51": (37.5, -78.9),
    "53": (47.4, -120.5),
    "54": (38.6, -80.6),
    "55": (44.6, -89.9),
    "56": (43.0, -107.6),
}

_EARTH_RADIUS_KM = 6371.0


def _haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Great-circle distance (km) between two ``(lat, lon)`` points."""
    lat1, lon1 = np.radians(a)
    lat2, lon2 = np.radians(b)
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return float(2.0 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(h)))


def spatial_kernel(fips: list[str], cutoff_km: float) -> FloatArray:
    """Bartlett spatial kernel over states: ``max(0, 1 - d/cutoff)``.

    Parameters
    ----------
    fips
        Ordered state FIPS codes indexing the kernel rows/columns.
    cutoff_km
        Distance beyond which the weight is zero.

    Returns
    -------
    numpy.ndarray
        Symmetric ``(n_states, n_states)`` kernel with unit diagonal. States
        absent from :data:`STATE_CENTROIDS` correlate only with themselves.
    """
    n = len(fips)
    w = np.eye(n, dtype=np.float64)
    for i in range(n):
        ci = STATE_CENTROIDS.get(fips[i])
        if ci is None:
            continue
        for j in range(i + 1, n):
            cj = STATE_CENTROIDS.get(fips[j])
            if cj is None:
                continue
            weight = max(0.0, 1.0 - _haversine_km(ci, cj) / cutoff_km)
            w[i, j] = w[j, i] = weight
    return w


def conley_meat(
    score: FloatArray,
    state_idx: npt.NDArray[np.int_],
    time_idx: npt.NDArray[np.int_],
    kernel: FloatArray,
    n_states: int,
    n_times: int,
    time_bandwidth: int,
) -> float:
    """Space-time HAC meat ``sum_{t,t'} K_time(|t-t'|) g_t' W g_{t'}``.

    Parameters
    ----------
    score
        Per-observation score ``x_tilde_it * u_it``.
    state_idx, time_idx
        Dense zero-based state and time codes per observation.
    kernel
        State spatial kernel ``W`` (``n_states x n_states``).
    n_states, n_times
        Grid dimensions.
    time_bandwidth
        Newey-West serial bandwidth (Bartlett); ``0`` gives the contemporaneous
        spatial-only meat.

    Returns
    -------
    float
        The scalar meat for a single-regressor sandwich.
    """
    g = np.zeros((n_states, n_times), dtype=np.float64)
    np.add.at(g, (state_idx, time_idx), score)
    wg = kernel @ g
    meat = float(np.sum(g * wg))
    for lag in range(1, time_bandwidth + 1):
        weight = 1.0 - lag / (time_bandwidth + 1.0)
        cross = float(np.sum(g[:, : n_times - lag] * wg[:, lag:]))
        meat += 2.0 * weight * cross
    return meat


def _demean_inplace(out: FloatArray, idx: npt.NDArray[np.int_], n_groups: int) -> None:
    """Subtract group means from every column of *out* in place."""
    counts = np.bincount(idx, minlength=n_groups).astype(np.float64)
    counts[counts == 0.0] = 1.0
    for col in range(out.shape[1]):
        sums = np.bincount(idx, weights=out[:, col], minlength=n_groups)
        out[:, col] -= (sums / counts)[idx]


def _two_way_within(
    mat: FloatArray,
    cell_idx: npt.NDArray[np.int_],
    time_idx: npt.NDArray[np.int_],
    *,
    iters: int = 4,
) -> FloatArray:
    """Absorb cell and time fixed effects by iterative within demeaning."""
    out = np.asarray(mat, dtype=np.float64).copy()
    n_cells = int(cell_idx.max()) + 1
    n_times = int(time_idx.max()) + 1
    for _ in range(iters):
        _demean_inplace(out, cell_idx, n_cells)
        _demean_inplace(out, time_idx, n_times)
    return out


def conley_regression_se(
    y: FloatArray,
    treatment: FloatArray,
    controls: FloatArray,
    cell_idx: npt.NDArray[np.int_],
    time_idx: npt.NDArray[np.int_],
    state_idx: npt.NDArray[np.int_],
    fips_order: list[str],
    *,
    cutoff_km: float,
    time_bandwidth: int,
) -> tuple[float, float]:
    """Two-way-FE slope of *y* on *treatment* with a Conley space-time HAC SE.

    Absorbs cell and time fixed effects by within demeaning, partials the
    controls out of the treatment and outcome (Frisch-Waugh-Lovell), then forms
    the single-regressor sandwich with the spatial + serial meat.

    Returns
    -------
    beta, se : float
        The point estimate (identical to the FE-OLS slope) and its Conley SE.
    """
    stack = (
        np.column_stack([y, treatment, controls])
        if controls.size
        else np.column_stack([y, treatment])
    )
    dem = _two_way_within(stack, cell_idx, time_idx)
    yd, td, cd = dem[:, 0], dem[:, 1], dem[:, 2:]
    if cd.shape[1] > 0:
        t_tilde = td - cd @ np.linalg.lstsq(cd, td, rcond=None)[0]
        y_res = yd - cd @ np.linalg.lstsq(cd, yd, rcond=None)[0]
    else:
        t_tilde, y_res = td, yd
    tt = float(t_tilde @ t_tilde)
    if tt <= 0.0:
        return float("nan"), float("nan")
    beta = float(t_tilde @ y_res) / tt
    resid = y_res - beta * t_tilde
    score = t_tilde * resid
    kernel = spatial_kernel(fips_order, cutoff_km)
    meat = conley_meat(
        score,
        state_idx,
        time_idx,
        kernel,
        len(fips_order),
        int(time_idx.max()) + 1,
        time_bandwidth,
    )
    var = meat / (tt**2)
    se = float(np.sqrt(var)) if var > 0.0 else float("nan")
    return beta, se
