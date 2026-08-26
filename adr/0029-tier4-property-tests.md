# ADR-0029: Tier 4.4 — property-based tests

- Status: Accepted
- Date: 2026-08-27

## Context

The suite was entirely example-based. Example tests check fixed inputs; they miss
edge cases (empty groups, extreme values, adversarial orderings) that a bug could
hit in production. Several pure functions in the project have clean mathematical
invariants that hold for *all* inputs and are natural to check with generated data.

## Decision

Add Hypothesis property tests (`tests/unit/test_property.py`) for invariants of the
pure building blocks:

- **BH-FDR** (`bh_adjust`): outputs in `[0, 1]`, length preserved, the q-values
  **dominate** the raw p-values, and the transform is **permutation-equivariant**
  (adjusting a permutation of the inputs permutes the outputs).
- **Great-circle distance** (`_haversine_km`): symmetric, non-negative, zero on
  identical points, and bounded by half the Earth's circumference.
- **Spatial kernel** (`spatial_kernel`): symmetric, unit diagonal, entries in
  `[0, 1]`.
- **Exposure standardisation** (`cell_exposure`): output has mean 0 and unit
  sample standard deviation for any non-degenerate input.
- **Fixed-effect demeaning** (`_demean_inplace`): every non-empty group's column
  mean is zero after the transform.

## Consequences

- These invariants are now checked across a wide swathe of generated inputs on
  every test run, catching regressions that fixed examples would miss (e.g. a
  standardisation using the wrong `ddof`, or a demean that leaves a residual mean).
- Hypothesis (already a dev dependency) shrinks any failure to a minimal
  counter-example, so a regression is reported with the smallest triggering input.
