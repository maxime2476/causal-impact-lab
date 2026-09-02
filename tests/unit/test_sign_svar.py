"""Sign-restricted SVAR: orthogonal draws and recovery on a monetary DGP."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl

from cil.estimators.sign_svar import _haar_orthogonal, sign_restricted_svar


def test_haar_orthogonal_is_orthogonal() -> None:
    rng = np.random.default_rng(0)
    q = _haar_orthogonal(rng, 4)
    assert np.allclose(q @ q.T, np.eye(4), atol=1e-10)


def test_sign_svar_recovers_negative_employment_response() -> None:
    # DGP with a monetary shock e_t: rate up, price/employment/IP down.
    rng = np.random.default_rng(1)
    n = 400
    rate = np.zeros(n)
    price = np.zeros(n)
    emp = np.zeros(n)
    ip = np.zeros(n)
    for t in range(1, n):
        e = rng.normal()
        rate[t] = 0.5 * rate[t - 1] + e + 0.2 * rng.normal()
        price[t] = 0.5 * price[t - 1] - 0.8 * e + 0.2 * rng.normal()
        emp[t] = 0.5 * emp[t - 1] - 0.6 * e + 0.2 * rng.normal()
        ip[t] = 0.5 * ip[t - 1] - 0.4 * e + 0.2 * rng.normal()
    dates = [dt.date(1990 + i // 12, i % 12 + 1, 1) for i in range(n)]
    data = pl.DataFrame(
        {"date": dates, "rate": rate, "log_cpi": price, "log_emp": emp, "log_ip": ip}
    )
    irf, acceptance = sign_restricted_svar(
        data,
        ["rate", "log_cpi", "log_emp", "log_ip"],
        rate="rate",
        price="log_cpi",
        target="log_emp",
        n_lags=2,
        horizons=(0, 1, 2, 3, 4),
        restrict_horizons=(0, 1),
        n_draws=500,
        seed=0,
    )
    assert 0.0 < acceptance <= 1.0
    assert set(irf["horizon"].to_list()) == {0.0, 1.0, 2.0, 3.0, 4.0}
    assert bool(irf["median"].is_finite().all())
    # Employment falls to the contractionary shock in the DGP.
    assert float(irf.filter(pl.col("horizon") == 0)["median"][0]) < 0.0
    assert (irf["lo_68"] <= irf["hi_68"]).all()
