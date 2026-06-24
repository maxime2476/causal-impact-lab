# ADR-0007: DML heterogeneity and time-respecting cross-fitting

- Status: Accepted
- Date: 2026-06-23

## Context

Phase 6 estimates the heterogeneous effect of the exposure-shock interaction
with double/debiased ML (EconML). The decisive correctness concern is that
cross-fitting must respect temporal dependence: a random K-fold places adjacent
months in train and test, leaking serial correlation (and overlapping
local-projection windows) into the nuisance estimates. The contract designates
random K-fold here a bug.

## Decision

- **Purged time-blocked cross-fitting** (`PurgedTimeBlockedCV`): folds are
  contiguous time blocks; for each test block, training periods within an
  `embargo` distance are purged on both sides. This is the only cross-fitting
  used by the DML estimators. Its no-leakage property is enforced by dedicated
  tests (disjoint folds; every train period strictly more than `embargo` from
  every test period).
- **Estimators:** EconML `LinearDML` (linear CATE in exposure -- headline
  average effect and heterogeneity slope) and `CausalForestDML` (nonparametric
  heterogeneity), both with random-forest nuisances and the time-blocked
  splitter.
- **Treatment:** the exposure-shock interaction `D = E_i * s_t` (consistent with
  the headline and ADR-0003); the effect modifier is exposure.
- **Refutation:** a placebo with the treatment permuted across time, expected to
  collapse the effect toward zero, is reported alongside the estimates; the
  DoWhy refuter battery (Phase 2) applies to the identified estimand.

## Consequences

- Heterogeneity estimates are free of the random-K-fold leakage that would
  otherwise bias DML on serially dependent panels.
- Positive controls confirm both estimators recover an injected effect and the
  placebo collapses; on real data the estimates inherit the same power
  limitations as the headline (short 2014-2020 window, sector-level exposure).
- `CausalForestDML` is heavier; it is run at the primary horizons only.
