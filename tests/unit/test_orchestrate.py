"""Orchestrator DAG ordering and stage selection (stubbed, no real stages)."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from cil import orchestrate
from cil.config import Settings, get_settings


def _stub_registry(calls: list[str]) -> dict[str, orchestrate.StageFn]:
    def make(name: str) -> orchestrate.StageFn:
        def _run(_settings: Settings) -> Mapping[str, float]:
            calls.append(name)
            return {"ok": 1.0}

        return _run

    return {name: make(name) for name in ("a", "b", "c")}


def test_available_stages_in_dependency_order() -> None:
    assert orchestrate.available_stages() == [
        "data",
        "ces_reconciliation",
        "shocks",
        "aggregate",
        "estimators",
        "heterogeneity",
        "bayesian",
        "robustness",
        "report",
    ]


def test_run_all_runs_in_canonical_order() -> None:
    calls: list[str] = []
    settings = get_settings()
    # Request out of order; must still run in registry (canonical) order.
    results = orchestrate.run_all(
        settings, stages=["c", "a"], stage_map=_stub_registry(calls)
    )
    assert calls == ["a", "c"]  # canonical order, not requested order
    assert set(results) == {"a", "c"}
    assert all("seconds" in r for r in results.values())


def test_run_all_full_registry() -> None:
    calls: list[str] = []
    orchestrate.run_all(get_settings(), stage_map=_stub_registry(calls))
    assert calls == ["a", "b", "c"]


def test_unknown_stage_raises() -> None:
    calls: list[str] = []
    with pytest.raises(ValueError, match="Unknown stage"):
        orchestrate.run_all(
            get_settings(), stages=["a", "nope"], stage_map=_stub_registry(calls)
        )
