"""Identification of the headline estimand via DoWhy.

Wraps the causal graph in a :class:`dowhy.CausalModel`, derives the identified
estimand (backdoor / instrumental-variable sets), and reports it in a typed
result. The estimand is the effect of the exposure-shock interaction on the
cumulative change in log employment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel

from cil.dag.graph import OUTCOME, TREATMENT, graph_gml

if TYPE_CHECKING:
    import pandas as pd
    from dowhy import CausalModel


class IdentificationResult(BaseModel):
    """Summary of an identified estimand.

    Parameters
    ----------
    treatment
        Treatment variable name.
    outcome
        Outcome variable name.
    backdoor_variables
        Variables forming a valid backdoor adjustment set (may be empty).
    instrumental_variables
        Instruments available for an IV strategy (may be empty).
    estimand_expression
        The identified estimand expression as rendered by DoWhy.
    """

    treatment: str
    outcome: str
    backdoor_variables: list[str]
    instrumental_variables: list[str]
    estimand_expression: str


def build_causal_model(
    data: pd.DataFrame,
    *,
    treatment: str = TREATMENT,
    outcome: str = OUTCOME,
) -> CausalModel:
    """Build a :class:`dowhy.CausalModel` on the project graph.

    Parameters
    ----------
    data
        Observations with columns for every graph node referenced.
    treatment
        Treatment node identifier.
    outcome
        Outcome node identifier.

    Returns
    -------
    dowhy.CausalModel
        The configured causal model.
    """
    from dowhy import CausalModel

    return CausalModel(
        data=data, treatment=treatment, outcome=outcome, graph=graph_gml()
    )


def summarize_identification(model: CausalModel) -> IdentificationResult:
    """Identify the estimand and return a typed summary.

    Parameters
    ----------
    model
        A causal model from :func:`build_causal_model`.

    Returns
    -------
    IdentificationResult
        The backdoor / IV sets and the estimand expression.
    """
    estimand = model.identify_effect(proceed_when_unidentifiable=True)
    backdoor = estimand.get_backdoor_variables()
    instruments = estimand.get_instrumental_variables()
    return IdentificationResult(
        treatment=str(model._treatment[0]),
        outcome=str(model._outcome[0]),
        backdoor_variables=[str(v) for v in backdoor],
        instrumental_variables=[str(v) for v in instruments],
        estimand_expression=str(estimand),
    )
