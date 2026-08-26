"""Benchmark drift-detection logic and snapshot determinism (fast, no estimators)."""

from __future__ import annotations

from cil import benchmark


def test_compare_to_baseline_flags_drift_and_missing() -> None:
    baseline = {"a": 1.0, "b": 2.0}
    # Within tolerance (2% < 5% rel_tol) -> no drift.
    assert benchmark.compare_to_baseline({"a": 1.0, "b": 2.04}, baseline) == []
    # Beyond tolerance -> flagged.
    drifts = benchmark.compare_to_baseline({"a": 1.0, "b": 2.4}, baseline)
    assert len(drifts) == 1 and drifts[0].startswith("b:")
    # Missing baseline key -> flagged.
    assert benchmark.compare_to_baseline({"c": 1.0}, baseline)


def test_synthetic_snapshot_is_deterministic() -> None:
    p1, e1, s1 = benchmark.synthetic_snapshot()
    p2, _, _ = benchmark.synthetic_snapshot()
    assert p1.equals(p2)
    assert p1.height == benchmark._N_STATES * benchmark._N_SECTORS * benchmark._N_MONTHS
    assert set(p1.columns) >= {"unit_id", "state_fips", "supersector_code", "date"}
    assert e1.height == benchmark._N_SECTORS
    assert s1.height == benchmark._N_MONTHS
