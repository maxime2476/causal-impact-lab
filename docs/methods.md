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

## Monetary shocks (three series, triangulated)

Three monetary-shock series are built and compared (`cil.shocks`):

1. **In-house Romer-Romer orthogonalization** (`rr_orthogonalization`): the
   monthly change in the policy rate regressed on the Fed's real-time information
   set (first-release ALFRED vintages: real-time inflation, IP growth,
   unemployment and lags, plus the lagged rate). The residual is the shock. The
   real-time vintages stand in for the Greenbook forecasts used by Romer-Romer
   (2004) — a reproducible public-data proxy.
2. **Proxy-SVAR external instrument** (`proxy_svar`): a reduced-form VAR whose
   policy-equation residual is instrumented by a borrowed high-frequency series
   (the BRW shock); the first-stage strength is reported with a robust effective
   F and a weak-instrument flag. (Intraday tick data is not freely reproducible,
   so the instrument is borrowed; the SVAR is ours.)
3. **Published benchmark** — Bu-Rogers-Wu (`brw_shocks`, ingested in the data
   layer): spans pre/post-2008 and is documented as largely unpredictable.

Diagnostics reported (`info_effect`, `predictability`, `compare`):

- **Information-effect test** (Jarocinski-Karadi, monthly proxy): classify each
  surprise as policy vs. information by the sign co-movement of the shock with
  broad equity returns; report the contamination share. The monthly proxy
  overstates contamination relative to the high-frequency test.
- **Predictability test** (Bauer-Swanson): regress the shock on lagged real-time
  predictors; report R-squared and the joint F p-value. A clean shock is
  unpredictable.
- **Cross-correlation** of the three series.

Findings on the real sample (1994–2020 where available; see `shock_diagnostics`):
the in-house RR series correlates only weakly with BRW (~0.09), the BRW series
shows modest but significant predictability from the real-time information set
(p ≈ 0.02, R² ≈ 0.08 — consistent with Bauer-Swanson), and the proxy-SVAR first
stage is **weak** in the monthly effective-funds-rate configuration (robust
F ≈ 0.7), which is reported, not hidden. The headline-shock choice is recorded
in ADR-0004.

### External shock series (Tier 1)

Two purpose-built external series strengthen and cross-check identification:

- **High-frequency instrument** — the Bauer-Swanson MPS (`cil.data.mps`; SF Fed)
  replaces BRW as the headline LP-IV instrument. The first stage strengthens to
  robust F ≈ 13–15 at h = 0/12 (vs BRW's ~3–5), though the resulting aggregate
  IRF is wrong-signed at the medium run (ADR-0013).
- **Narrative shock** — the updated Romer-Romer series (`cil.data.rr`;
  Breitenlechner 2018) is a forecast-purged, non-market identification used in a
  quarterly aggregate LP (`rr_lp_irf`); correctly signed but imprecise (ADR-0014).

The two aggregate identifications disagree on sign, which is itself the message:
the aggregate response is assumption-dependent. See `docs/results.md`.

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
