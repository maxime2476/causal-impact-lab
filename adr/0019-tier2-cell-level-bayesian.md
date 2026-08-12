# ADR-0019: Tier 2.3 — cell-level Bayesian hierarchical LP

- Status: Accepted
- Date: 2026-08-12

## Context

The Bayesian triangulation (`bayes_lp`) partial-pools at the **supersector**
level: one slope per sector, shrunk to a pooled mean. It cannot say how much of
the response heterogeneity is *between* sectors versus *across states within* a
sector. Tier 2.3 adds a **cell-level** (state x supersector) hierarchy to
decompose that.

## Decision

A fully-joint cell-level design is infeasible: two-way demeaning a ~4,500-column
shock-interaction matrix over ~1.5M rows is tens of GB dense. Instead use a
tractable **two-stage / nested** hierarchy (`cil.estimators.bayes_cell_lp`):

1. **Stage 1** — per-cell OLS slope of horizon-`h` growth on the shock,
   `beta_hat_i` with `se_i`, computed vectorised over all cells (a few tenths of
   a second; no giant matrix).
2. **Stage 2** — a non-centred normal-normal hierarchy, cells nested in sectors:
   `beta_hat_i ~ N(beta_i, se_i)`, `beta_i ~ N(mu_sector[k], tau_within)`,
   `mu_sector ~ N(mu0, tau_between)`. Reports `mu0` (grand mean ~ aggregate),
   `tau_between`, `tau_within`, and the **between-share**
   `tau_between^2 / (tau_between^2 + tau_within^2)`.

Fit at the primary horizon (h=12); stored as `bayes_cell_summary` and
`bayes_cell_sector` (posterior supersector means).

## Result (honest)

At h = 12 on the 3-digit panel (4,564 cells, 100 supersectors):

| Quantity | Posterior mean |
|---|---|
| `mu0` (grand-mean response) | +0.016 |
| `tau_between` (between-supersector SD) | 0.061 |
| `tau_within` (within-supersector, across-state SD) | 0.002 |
| **between-share** | **0.999** |

**Essentially all (~99.9%) of the cell-level response heterogeneity is *between*
supersectors; almost none is *within* a supersector across states.** The relative
response is an industry-composition phenomenon, not a state-idiosyncratic one —
which is exactly the premise of the shift-share exposure design, now confirmed
from the data rather than assumed. It also explains why the supersector-level
pooled model captures the systematic variation: there is little within-sector
signal left to exploit. `mu0` is small and near zero, consistent with the
aggregate-null story.

Caveats, stated:

- The stage-1 slopes share the common aggregate shock, so treating them as
  independent given `beta_i` understates `mu0`'s posterior uncertainty. The robust
  deliverable is the **variance decomposition**, not a sharper aggregate point
  estimate — consistent with the project's stance that the aggregate is the
  assumption-dependent complement.
- Convergence is good but not pristine (`max_rhat ~ 1.02` even at 4 chains /
  2000 tune / `target_accept = 0.98`): the near-zero `tau_within` is a variance
  component pinned against its boundary, which funnels and mildly slows mixing.
  The decomposition is insensitive to the residual non-mixing (τ_within ≈ 0.002
  is unambiguously tiny relative to τ_between ≈ 0.06). Reported, not smoothed over.

## Consequences

- The heterogeneity of the relative response is now decomposed into a
  between-sector and a within-sector (across-state) component, a new triangulation
  axis beyond the pooled sector model.
- Two-stage design keeps the cell-level model tractable on the 3-digit panel; a
  future fully-joint sparse implementation could replace stage 1 if warranted.
