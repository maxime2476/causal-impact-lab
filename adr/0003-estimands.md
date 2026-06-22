# ADR-0003: Estimand definitions and identification

- Status: Accepted
- Date: 2026-06-22

## Context

Phase 2 fixes the causal targets and the identification strategy before any
estimator is run on real outcomes. This record defines the estimands formally,
states the identifying assumptions, and records the priority order. The frozen
falsifiable claim is in `docs/analysis_plan.md`; the methods and graph are in
`docs/methods.md` and `cil.dag`.

## Decision

### Two estimands, in an explicit hierarchy

1. **Headline — relative effect (cleanly identified).** `β_h` in the interacted
   panel local projection with unit and time fixed effects, the effect of the
   exposure-shock interaction `E_i · s_t` on `y_{i,t+h} - y_{i,t-1}`. Time fixed
   effects absorb the aggregate component; `β_h` is therefore a relative,
   cross-sectional semi-elasticity and does not identify the level effect.
2. **Complement — aggregate dynamic effect (assumption-dependent).** `θ_h` in a
   time-series local projection of national log employment on the shock,
   complemented by a proxy-SVAR. Reported separately, with assumptions stated and
   fragility quantified.

### Sign convention

`s_t > 0` is contractionary; the headline-supporting sign is `β_h < 0`.

### Identification

- The headline relative effect is identified under **conditional parallel
  trends**, tested by the event-study leads.
- The shift-share exposure leans on the **exogenous-shocks** justification
  (Borusyak-Hull-Jaravel 2022), not on shares-exogeneity.
- The causal graph (`cil.dag.graph`) encodes the Fed reaction function as the
  aggregate backdoor and a predetermined cell trend as the cross-sectional
  backdoor. DoWhy yields a backdoor set `{shock, exposure}` (and correlates),
  which the two-way fixed-effects estimator closes.

### Assumptions registry and refuters

Every identifying assumption is registered in `cil.dag.assumptions` with the
test or refuter that probes it. DoWhy refuters (placebo, random common cause,
data subset, unobserved confounder) are wired in `cil.dag.refuters` and
validated as a positive control on a synthetic DGP.

## Consequences

- The relative effect is the headline because it is cleanly identified; the
  aggregate effect is explicitly secondary and caveated.
- The analysis plan is frozen and dated; results cannot redefine the claim.
- The DoWhy graph is the contract for identification; estimator choices (two-way
  FE, controls) are justified by the backdoor set it implies.
