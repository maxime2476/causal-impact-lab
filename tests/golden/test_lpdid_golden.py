"""Golden tests for the LP-DiD port against the Stata reference.

The Stata reference (github.com/danielegirardi/lpdid) cannot be run in this
environment, so the LP-DiD port is currently validated by analytic / two-way-FE
equivalence (see ``tests/unit/test_estimators.py``). This module is the drop-in
slot for true golden fixtures: place small panels and the Stata-produced
coefficients/standard errors under ``tests/golden/fixtures/lpdid/`` and the test
below will compare against them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_FIXTURES = Path(__file__).parent / "fixtures" / "lpdid"


@pytest.mark.golden
@pytest.mark.skipif(
    not _FIXTURES.exists(),
    reason="Stata LP-DiD golden fixtures not yet provided; see module docstring.",
)
def test_lpdid_matches_stata_reference() -> None:  # pragma: no cover - needs fixtures
    import polars as pl

    from cil.estimators.lp_did import LPDiDConfig, lp_did

    panel = pl.read_parquet(_FIXTURES / "panel.parquet")
    expected = pl.read_csv(_FIXTURES / "expected.csv")
    horizons = tuple(int(h) for h in expected["horizon"].to_list())
    result = lp_did(panel, LPDiDConfig(horizons=horizons)).sort("horizon")
    expected = expected.sort("horizon")
    for got, want in zip(
        result["att"].to_list(), expected["att"].to_list(), strict=True
    ):
        assert abs(got - want) < 1e-4
