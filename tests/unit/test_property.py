"""Property-based tests (Hypothesis) for pure functions with clear invariants."""

from __future__ import annotations

import numpy as np
import polars as pl
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from cil.exposure.shift_share import cell_exposure
from cil.inference.bh_fdr import bh_adjust
from cil.inference.conley import (
    STATE_CENTROIDS,
    _demean_inplace,
    _haversine_km,
    spatial_kernel,
)

_pvals = st.lists(
    st.floats(min_value=0.0, max_value=1.0, allow_nan=False), min_size=1, max_size=25
)
_MAX_GREAT_CIRCLE_KM = 20_100.0


@given(_pvals)
def test_bh_adjust_range_length_and_domination(ps: list[float]) -> None:
    p = np.asarray(ps)
    adj = bh_adjust(p)
    assert adj.shape == p.shape
    assert np.all(adj >= -1e-12) and np.all(adj <= 1.0 + 1e-12)
    # BH q-values dominate the raw p-values.
    assert np.all(adj + 1e-12 >= p)


@given(_pvals, st.data())
def test_bh_adjust_permutation_equivariant(
    ps: list[float], data: st.DataObject
) -> None:
    p = np.asarray(ps)
    perm = np.asarray(data.draw(st.permutations(range(p.size))), dtype=int)
    assert np.allclose(bh_adjust(p[perm]), bh_adjust(p)[perm])


_lat = st.floats(min_value=-90.0, max_value=90.0, allow_nan=False)
_lon = st.floats(min_value=-180.0, max_value=180.0, allow_nan=False)


@given(_lat, _lon, _lat, _lon)
def test_haversine_symmetric_nonneg_bounded(
    la1: float, lo1: float, la2: float, lo2: float
) -> None:
    a, b = (la1, lo1), (la2, lo2)
    d = _haversine_km(a, b)
    assert d >= 0.0
    assert abs(d - _haversine_km(b, a)) < 1e-6  # symmetric
    assert d <= _MAX_GREAT_CIRCLE_KM
    assert _haversine_km(a, a) < 1e-6  # identity


@given(
    st.lists(
        st.sampled_from(sorted(STATE_CENTROIDS)),
        min_size=1,
        max_size=8,
        unique=True,
    ),
    st.floats(min_value=50.0, max_value=20_000.0),
)
def test_spatial_kernel_symmetric_unit_diagonal_bounded(
    fips: list[str], cutoff: float
) -> None:
    w = spatial_kernel(fips, cutoff)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 1.0)
    assert np.all(w >= 0.0) and np.all(w <= 1.0)


@given(
    st.lists(
        st.floats(min_value=-1e4, max_value=1e4, allow_nan=False),
        min_size=2,
        max_size=25,
        unique=True,
    )
)
def test_cell_exposure_is_standardized(sens: list[float]) -> None:
    assume(float(np.std(sens, ddof=1)) > 1e-3)
    frame = pl.DataFrame(
        {
            "supersector_code": [f"s{i}" for i in range(len(sens))],
            "sensitivity": sens,
        }
    )
    exposure = cell_exposure(frame)["exposure"].to_numpy()
    assert abs(float(np.mean(exposure))) < 1e-6
    assert abs(float(np.std(exposure, ddof=1)) - 1.0) < 1e-6


@settings(max_examples=50)
@given(st.data())
def test_demean_inplace_zeroes_group_means(data: st.DataObject) -> None:
    n = data.draw(st.integers(min_value=2, max_value=40))
    vals = data.draw(
        st.lists(
            st.floats(min_value=-1e3, max_value=1e3, allow_nan=False),
            min_size=n,
            max_size=n,
        )
    )
    n_groups = data.draw(st.integers(min_value=1, max_value=n))
    idx = np.asarray(
        data.draw(
            st.lists(
                st.integers(min_value=0, max_value=n_groups - 1), min_size=n, max_size=n
            )
        ),
        dtype=int,
    )
    out = np.asarray(vals, dtype=np.float64).reshape(-1, 1)
    _demean_inplace(out, idx, n_groups)
    for group in range(n_groups):
        mask = idx == group
        if mask.any():
            assert abs(float(out[mask, 0].mean())) < 1e-6
