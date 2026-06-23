"""Assumptions registry: each identifying assumption and the probe for it.

Every causal claim names the assumption that, if violated, breaks it, and points
to the test or refuter that probes it. This registry is the single source of
truth linking assumptions to their checks; some probes run now (DoWhy refuters),
others in later phases (event-study leads, predictability and info-effect tests).
"""

from __future__ import annotations

from pydantic import BaseModel


class Assumption(BaseModel):
    """One identifying assumption and how it is probed.

    Parameters
    ----------
    key
        Stable short identifier.
    statement
        The assumption in plain terms.
    breaks_if
        The condition under which the claim fails.
    probe
        The test or refuter that probes the assumption.
    phase
        The roadmap phase where the probe is exercised.
    """

    key: str
    statement: str
    breaks_if: str
    probe: str
    phase: str


#: The registered identifying assumptions for the headline relative effect.
ASSUMPTIONS: tuple[Assumption, ...] = (
    Assumption(
        key="shock_exogeneity",
        statement=(
            "The identified monetary shock s_t is exogenous to the aggregate "
            "state of the economy after orthogonalisation and information-effect "
            "cleaning."
        ),
        breaks_if=(
            "Shocks are predictable from macro forecasts or co-move with equity "
            "prices (a central-bank information effect)."
        ),
        probe=(
            "Predictability regressions (Bauer-Swanson); information-effect test "
            "(Jarocinski-Karadi); placebo_treatment refuter."
        ),
        phase="P3",
    ),
    Assumption(
        key="conditional_parallel_trends",
        statement=(
            "Absent the shock, high- and low-exposure cells would have evolved "
            "in parallel, conditional on unit/time fixed effects and controls."
        ),
        breaks_if="Event-study leads (h < 0) are jointly significant.",
        probe="Event-study lead test; data_subset refuter.",
        phase="P4",
    ),
    Assumption(
        key="shift_share_exogeneity",
        statement=(
            "The national shock is plausibly exogenous and the base-period "
            "exposure shares are predetermined (Borusyak-Hull-Jaravel)."
        ),
        breaks_if=(
            "Exposure is correlated with omitted cell-level shocks that move "
            "with the cycle."
        ),
        probe="Exposure balance / pre-trend checks; random_common_cause refuter.",
        phase="P4",
    ),
    Assumption(
        key="no_anticipation",
        statement="Cells do not adjust employment before the shock is realised.",
        breaks_if="Pre-shock leads show systematic response by exposure.",
        probe="Event-study lead test.",
        phase="P4",
    ),
    Assumption(
        key="no_interference",
        statement=(
            "Treatment of one cell does not spill over into another's outcome "
            "(SUTVA across cells)."
        ),
        breaks_if="Cross-cell spillovers (e.g. supply chains) are large.",
        probe="Robustness with spatial/sector aggregation (documented).",
        phase="P8",
    ),
    Assumption(
        key="overlap",
        statement="Exposure has common support across cells (no degenerate cells).",
        breaks_if="A few cells dominate identification through extreme exposure.",
        probe="Exposure distribution and leave-one-out checks.",
        phase="P4",
    ),
)


def registry() -> tuple[Assumption, ...]:
    """Return the registered assumptions."""
    return ASSUMPTIONS


def probes() -> dict[str, str]:
    """Return a mapping of assumption key to its probe description."""
    return {a.key: a.probe for a in ASSUMPTIONS}
