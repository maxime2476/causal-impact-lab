# Methods: estimands and identification

This page states the estimands, the identification strategy, and the causal
graph behind them. The frozen falsifiable claim lives in
`docs/analysis_plan.md`; the formal estimand definitions are fixed in ADR-0003.

Notation: unit `i` = (state × supersector) cell; month `t`;
`y_{i,t} = log(employment_{i,t})`; identified national shock `s_t`; predetermined
exposure `E_i`.

## Estimand 1 — headline relative effect (cleanly identified)

Interacted panel local projection, for each horizon `h`:

```
y_{i,t+h} - y_{i,t-1} = β_h (E_i · s_t) + γ_i + δ_t + Σ_l φ'_{h,l} X_{i,t-l} + ε_{i,t+h}
```

`γ_i` are unit fixed effects, `δ_t` time fixed effects. The time effects absorb
the aggregate/common component, so `β_h` is the **relative** effect per unit
exposure per unit shock. By construction `{β_h}` does not identify the aggregate
level effect.

Identification rests on **conditional parallel trends**: absent the shock, high-
and low-exposure cells would have evolved in parallel, conditional on the fixed
effects and controls. This is testable via the event-study leads (`h < 0`).

## Estimand 2 — aggregate dynamic effect (assumption-dependent, complement)

Time-series local projection of national log employment on the shock:

```
Y_{t+h} - Y_{t-1} = α_h + θ_h s_t + Σ_l ψ'_{h,l} W_{t-l} + u_{t+h}
```

`{θ_h}` is the IRF, complemented by a proxy-SVAR / external-instrument VAR with
first-stage strength and a weak-instrument-robust check. Reported separately,
with assumptions stated and fragility quantified — never as "the" answer.

## Exposure (shift-share / Bartik)

```
E_s = Σ_k ω_{s,k} · σ_k
```

`ω_{s,k}` is the predetermined base-period employment share of supersector `k` in
state `s`; `σ_k` is the interest-rate sensitivity of supersector `k`.
Identification leans on the exogenous-shocks justification (Borusyak-Hull-Jaravel
2022): the national shock is plausibly exogenous and the shares are
predetermined.

## The causal graph

The graph (`cil.dag.graph`, encoded for DoWhy) makes the identification explicit:

- **Fed reaction function (the aggregate backdoor):** aggregate conditions
  (output gap, inflation, oil/commodity shocks, fiscal shocks, global demand)
  drive the policy decision and hence the shock, and also drive employment.
- **Cross-sectional backdoor:** a predetermined cell trend correlated with
  exposure also affects employment growth.
- **Effect of interest:** the exposure-shock interaction on employment.

DoWhy identifies a backdoor adjustment set of `{shock, exposure}` (plus their
correlates). The mapping to the estimator is deliberate:

| Backdoor | Closed by |
|---|---|
| aggregate conditions, shock (aggregate effect) | **time fixed effects** `δ_t` |
| exposure, predetermined cell trend | **unit fixed effects** `γ_i` + lagged controls |

## Assumptions registry

`cil.dag.assumptions` is the single source of truth linking each identifying
assumption to the test or refuter that probes it:

| Assumption | Breaks if | Probe | Phase |
|---|---|---|---|
| Shock exogeneity | Shocks predictable / information effect | Predictability + info-effect tests; placebo refuter | P3 |
| Conditional parallel trends | Leads jointly significant | Event-study leads; data-subset refuter | P4 |
| Shift-share exogeneity | Exposure correlated with omitted shocks | Balance/pre-trend; random-common-cause refuter | P4 |
| No anticipation | Pre-shock response by exposure | Event-study leads | P4 |
| No interference (SUTVA) | Large cross-cell spillovers | Aggregation robustness | P8 |
| Overlap | Extreme cells dominate | Exposure distribution / leave-one-out | P4 |

## Refuters

DoWhy refuters (`cil.dag.refuters`) probe the estimates: placebo treatment
(effect should collapse to ~0), random common cause and data subset (effect
should be ~unchanged), and an unobserved-confounder sensitivity check. They are
validated as a positive control on a synthetic DGP with a known effect.

## References

See the reference list in the repository contract; key anchors: Jordà (2005);
Dube-Girardi-Jordà-Taylor (2025); Goodman-Bacon (2021); Stock-Watson (2018);
Romer-Romer (2004); Bu-Rogers-Wu (2021); Jarociński-Karadi (2020);
Bauer-Swanson (2023); Wu-Xia (2016); Chernozhukov et al. (2018);
Borusyak-Hull-Jaravel (2022); Driscoll-Kraay (1998).
