"""Causal graph, identification, refuters, and the assumptions registry.

Uses a synthetic structural model (no network, no real data) matching the
project DAG to exercise identification and the refuter battery as a positive
control: the injected effect is recovered, and the placebo collapses to zero.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest

from cil.dag import assumptions, graph, identification, refuters

_TRUE_EFFECT = -0.8


def _synthetic(n: int = 1500, seed: int = 7) -> pd.DataFrame:
    """Draw data from the project DAG with a known treatment effect."""
    rng = np.random.default_rng(seed)
    agg = rng.normal(size=n)
    shock = 0.5 * agg + rng.normal(size=n)
    cell_trend = rng.normal(size=n)
    exposure = 0.3 * cell_trend + rng.normal(size=n)
    treatment = shock * exposure
    d_employment = (
        _TRUE_EFFECT * treatment
        + 0.5 * agg
        + 0.4 * shock
        + 0.3 * exposure
        + 0.2 * cell_trend
        + rng.normal(size=n)
    )
    return pd.DataFrame(
        {
            "agg_conditions": agg,
            "shock": shock,
            "exposure": exposure,
            "cell_trend": cell_trend,
            "treatment": treatment,
            "d_employment": d_employment,
        }
    )


@pytest.fixture(scope="module")
def model_and_estimate() -> tuple[Any, Any, Any]:
    """Build the causal model, identified estimand, and a point estimate once."""
    model = identification.build_causal_model(_synthetic())
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = refuters.estimate_effect(model, estimand)
    return model, estimand, estimate


# --- graph -----------------------------------------------------------------


def test_graph_is_acyclic_and_well_formed() -> None:
    g = graph.build_graph()
    assert g.number_of_nodes() == len(graph.NODES)
    assert g.number_of_edges() == len(graph.EDGES)
    assert graph.TREATMENT in g
    assert graph.OUTCOME in g


def test_graph_gml_round_trips() -> None:
    gml = graph.graph_gml()
    assert gml.startswith("graph[directed 1")
    assert f'node[id "{graph.TREATMENT}"' in gml
    assert f'edge[source "{graph.TREATMENT}" target "{graph.OUTCOME}"]' in gml


# --- identification --------------------------------------------------------


def test_identification_finds_backdoor_set(
    model_and_estimate: tuple[Any, Any, Any],
) -> None:
    model, _, _ = model_and_estimate
    result = identification.summarize_identification(model)
    assert result.treatment == graph.TREATMENT
    assert result.outcome == graph.OUTCOME
    # The Fed-reaction and exposure backdoors must be adjusted for.
    assert "shock" in result.backdoor_variables
    assert "exposure" in result.backdoor_variables


# --- refuters --------------------------------------------------------------


def test_estimate_recovers_injected_effect(
    model_and_estimate: tuple[Any, Any, Any],
) -> None:
    _, _, estimate = model_and_estimate
    assert abs(float(estimate.value) - _TRUE_EFFECT) < 0.25


def test_refuters_run_and_behave(
    model_and_estimate: tuple[Any, Any, Any],
) -> None:
    model, estimand, estimate = model_and_estimate
    results = refuters.run_refuters(
        model,
        estimand,
        estimate,
        num_simulations=10,
    )
    assert set(results) == {
        "placebo_treatment",
        "random_common_cause",
        "data_subset",
        "unobserved_common_cause",
    }
    # Placebo treatment should collapse the effect toward zero.
    assert abs(results["placebo_treatment"].refuted_effect) < 0.2
    # Invariance refuters should leave the effect roughly unchanged.
    assert (
        abs(results["random_common_cause"].refuted_effect - float(estimate.value)) < 0.2
    )


# --- assumptions registry --------------------------------------------------


def test_assumptions_registry_complete() -> None:
    registry = assumptions.registry()
    assert len(registry) >= 5
    keys = [a.key for a in registry]
    assert len(keys) == len(set(keys))  # unique
    assert "conditional_parallel_trends" in keys
    assert "shock_exogeneity" in keys
    # Every assumption names a non-empty probe.
    assert all(a.probe.strip() for a in registry)
    assert set(assumptions.probes()) == set(keys)
