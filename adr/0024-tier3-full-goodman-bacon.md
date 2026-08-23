# ADR-0024: Tier 3.3 — full Goodman-Bacon decomposition

- Status: Accepted
- Date: 2026-08-20

## Context

The Phase-8 Bacon step was a *light* diagnostic (TWFE vs clean-control LP-DiD, and
the already-treated share). It did not perform the actual Goodman-Bacon
decomposition, and it was never run on real data — the headline design is an
exposure x shock interaction, not a staggered DiD, so no staggered treatment
existed to decompose.

## Decision

- `goodman_bacon.bacon_decompose`: the **full** decomposition. The TWFE
  coefficient is written as the weighted sum of every 2x2 DiD between adoption
  cohorts, using the exact balanced-panel Goodman-Bacon weights, split into three
  categories — treated-vs-never, earlier-vs-later (clean), and
  later-vs-already-treated (**forbidden**). The panel is balanced first (the
  closed-form weights require it) and the TWFE identity is checked on that same
  subset.
- `goodman_bacon.build_staggered_treatment`: constructs a treatment to decompose —
  a cell adopts (absorbing, annual) once its exposure-weighted cumulative monetary
  tightening crosses a threshold, so high-exposure cells adopt earlier and
  low/negative-exposure cells never adopt.
- Stored as `bacon_decomposition` (per-2x2 rows) and `bacon_summary`.

## Result (honest)

On the constructed staggered treatment (real panel, 6 cohorts, balanced):

- The decomposition **exactly reproduces TWFE** (`-0.0217`; identity gap
  `4e-14`) — a strong correctness check.

| Comparison category | weight | contribution to TWFE |
|---|---|---|
| treated vs never-treated (clean) | 0.609 | +0.012 |
| earlier vs later (clean) | 0.012 | +0.000 |
| **later vs already-treated (forbidden)** | **0.378** | **-0.034** |

**~38% of the TWFE weight lands on forbidden comparisons**, and they pull the
estimate the opposite way from the clean comparisons (average forbidden 2x2
`~ -0.09` vs clean treated-vs-untreated `~ +0.02`). So a naive TWFE on such a
staggered design would be materially contaminated by negative weighting.

The estimate itself is not a monetary effect (the treatment is a constructed
adoption rule, not the shock); the deliverable is the **weight decomposition**.

## Consequences

- Quantifies exactly why the project avoids naive staggered TWFE: on a staggered
  operationalization of the exposure design, more than a third of the TWFE weight
  is forbidden. The headline LP/interaction design and the clean-control LP-DiD
  (ADR-0018) sidestep this entirely.
- The machine-precision TWFE identity validates the implementation.
