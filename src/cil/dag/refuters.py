"""Refutation of an identified estimate via DoWhy refuters.

Wraps DoWhy's refuters into typed results: placebo treatment, random common
cause, data subset, and an unobserved-confounder sensitivity check. These probe
the identifying assumptions registered in :mod:`cil.dag.assumptions`. A robust
estimate should survive the placebo (effect collapses to ~0) and the invariance
refuters (effect roughly unchanged).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

if TYPE_CHECKING:
    from dowhy import CausalModel
    from dowhy.causal_estimator import CausalEstimate
    from dowhy.causal_identifier import IdentifiedEstimand


class Refutation(BaseModel):
    """Result of one refutation.

    Parameters
    ----------
    method
        DoWhy refuter method name.
    estimated_effect
        The original estimated effect.
    refuted_effect
        The effect under the refutation (the "new effect").
    p_value
        Refutation p-value if the refuter reports one, else ``None``.
    """

    method: str
    estimated_effect: float
    refuted_effect: float
    p_value: float | None = None


def estimate_effect(model: CausalModel, estimand: IdentifiedEstimand) -> CausalEstimate:
    """Estimate the identified effect by backdoor linear regression.

    Parameters
    ----------
    model
        The causal model.
    estimand
        The identified estimand from the model.

    Returns
    -------
    dowhy.causal_estimator.CausalEstimate
        The point estimate object.
    """
    return model.estimate_effect(estimand, method_name="backdoor.linear_regression")


def _p_value(refutation: Any) -> float | None:
    """Extract a refutation p-value when present (DoWhy returns dynamic types)."""
    result = getattr(refutation, "refutation_result", None)
    if isinstance(result, dict):
        raw = result.get("p_value")
        if isinstance(raw, (int, float)):
            return float(raw)
    return None


def run_refuters(
    model: CausalModel,
    estimand: IdentifiedEstimand,
    estimate: CausalEstimate,
    *,
    num_simulations: int = 50,
    seed: int = 20260101,
) -> dict[str, Refutation]:
    """Run the standard refuter battery and return typed results.

    Parameters
    ----------
    model
        The causal model.
    estimand
        The identified estimand.
    estimate
        The point estimate to refute.
    num_simulations
        Simulation count for the stochastic refuters.
    seed
        Random seed for reproducibility.

    Returns
    -------
    dict of str to Refutation
        Keyed by a short refuter name.
    """
    specs: dict[str, dict[str, Any]] = {
        "placebo_treatment": {
            "method_name": "placebo_treatment_refuter",
            "placebo_type": "permute",
            "num_simulations": num_simulations,
            "random_seed": seed,
        },
        "random_common_cause": {
            "method_name": "random_common_cause",
            "num_simulations": num_simulations,
            "random_seed": seed,
        },
        "data_subset": {
            "method_name": "data_subset_refuter",
            "subset_fraction": 0.8,
            "num_simulations": num_simulations,
            "random_seed": seed,
        },
        "unobserved_common_cause": {
            "method_name": "add_unobserved_common_cause",
            "confounders_effect_on_treatment": "linear",
            "confounders_effect_on_outcome": "linear",
            "effect_strength_on_treatment": 0.01,
            "effect_strength_on_outcome": 0.02,
        },
    }
    results: dict[str, Refutation] = {}
    for name, kwargs in specs.items():
        refutation = model.refute_estimate(estimand, estimate, **kwargs)
        results[name] = Refutation(
            method=str(kwargs["method_name"]),
            estimated_effect=float(estimate.value),
            refuted_effect=float(refutation.new_effect),
            p_value=_p_value(refutation),
        )
    return results
