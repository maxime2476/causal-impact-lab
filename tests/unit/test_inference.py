"""Inference primitives: BH-FDR, Driscoll-Kraay, wild cluster bootstrap."""

from __future__ import annotations

import numpy as np

from cil.inference import bh_fdr
from cil.inference.driscoll_kraay import coefficient_se, driscoll_kraay_cov
from cil.inference.wild_bootstrap import bootstrap_ci, wild_cluster_bootstrap


def test_bh_adjust_known_values() -> None:
    # Classic BH example.
    p = [0.01, 0.02, 0.03, 0.04, 0.05]
    q = bh_fdr.bh_adjust(p)
    assert np.all(np.diff(np.sort(q)) >= -1e-12)  # monotone in sorted p
    assert np.all((q >= 0) & (q <= 1))
    assert abs(q[0] - 0.05) < 1e-12  # 0.01 * 5/1


def test_bh_adjust_all_null_is_identity_ish() -> None:
    q = bh_fdr.bh_adjust([1.0, 1.0, 1.0])
    assert np.allclose(q, 1.0)


def test_bh_reject() -> None:
    rejects = bh_fdr.bh_reject([0.001, 0.2, 0.9], alpha=0.10)
    assert rejects[0]
    assert not rejects[2]


def test_bh_empty() -> None:
    assert bh_fdr.bh_adjust([]).size == 0


def test_driscoll_kraay_positive_and_sane() -> None:
    rng = np.random.default_rng(0)
    n_e, n_t = 30, 40
    rows = []
    for e in range(n_e):
        for t in range(n_t):
            rows.append((e, t))
    time_codes = np.array([t for _, t in rows])
    x = np.column_stack([np.ones(len(rows)), rng.normal(size=len(rows))])
    beta = np.array([1.0, 2.0])
    y = x @ beta + rng.normal(size=len(rows))
    coef = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ coef
    cov = driscoll_kraay_cov(x, resid, time_codes)
    se = coefficient_se(cov)
    assert cov.shape == (2, 2)
    assert np.all(se > 0)
    # Covariance is symmetric positive semidefinite.
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov) > -1e-8)


def test_wild_cluster_bootstrap_brackets_truth() -> None:
    rng = np.random.default_rng(1)
    n_clusters, per = 40, 20
    clusters = np.repeat(np.arange(n_clusters), per)
    x = np.column_stack([np.ones(n_clusters * per), rng.normal(size=n_clusters * per)])
    y = x @ np.array([0.0, 1.5]) + rng.normal(size=n_clusters * per)
    draws = wild_cluster_bootstrap(x, y, clusters, target_index=1, n_boot=299, seed=0)
    low, high = bootstrap_ci(draws, 0.95)
    assert low < 1.5 < high
    assert draws.size == 299
