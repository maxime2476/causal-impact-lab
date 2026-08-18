"""Golden test for the LP-DiD estimator on a staggered design.

The Stata reference (github.com/danielegirardi/lpdid) cannot be run in this
environment, so we do **not** claim Stata parity here. Instead this is a
**known-DGP golden**: the committed fixture panel is built from deterministic
two-way fixed effects and a *known* dynamic treatment-effect path ``tau_h`` under
staggered adoption (three cohorts plus never-treated controls, no noise). With the
horizon-dependent clean-control condition, LP-DiD must recover ``tau_h`` at every
response horizon and zero at the pre-treatment lead -- exactly, up to numerical
precision. This is a stronger check than the non-staggered TWFE-equivalence unit
test, because contamination from not-yet-treated controls adopting *inside* the
outcome window would show up as a horizon-dependent bias.

Fixtures live in ``tests/golden/fixtures/lpdid/`` (``panel.parquet`` and the known
``expected.csv``); the generator is documented in ADR-0018. A true Stata
cross-validation can be dropped in later by overwriting ``expected.csv`` with
Stata ``lpdid`` output on the same panel.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures" / "lpdid"


@pytest.mark.golden
@pytest.mark.skipif(
    not (_FIXTURES / "panel.parquet").exists(),
    reason="LP-DiD golden fixtures missing; see module docstring.",
)
def test_lpdid_recovers_known_staggered_dgp() -> None:
    import polars as pl

    from cil.estimators.lp_did import LPDiDConfig, lp_did

    panel = pl.read_parquet(_FIXTURES / "panel.parquet")
    expected = pl.read_csv(_FIXTURES / "expected.csv").sort("horizon")
    horizons = tuple(int(h) for h in expected["horizon"].to_list())
    result = lp_did(panel, LPDiDConfig(horizons=horizons)).sort("horizon")
    for got, want in zip(
        result["att"].to_list(), expected["att"].to_list(), strict=True
    ):
        assert abs(got - want) < 1e-4
