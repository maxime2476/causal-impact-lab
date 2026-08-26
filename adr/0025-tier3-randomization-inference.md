# ADR-0025: Tier 3.4 — circular-shift randomization inference

- Status: Accepted
- Date: 2026-08-24

## Context

The Phase-8 placebo test permutes the shock **iid** across time. That destroys
the shock's serial correlation, so the placebo distribution does not respect the
data-generating design and its p-value can be anti-conservative. Tier 3.4 adds a
design-respecting randomization-inference test.

## Decision

`placebo.circular_shift_ri`: for a serially-correlated time-series treatment the
valid randomization is a **circular shift**. Each draw rotates the date-sorted
shock by a random offset (wrapping around), which preserves its autocovariance
exactly while breaking its alignment with the outcomes — the sharp-null
distribution for the timing. The headline panel LP is re-estimated at each draw.

Reported:

- **Per-horizon** RI p-values: the share of shifts with `|beta_h|` at least the
  observed (`+1` smoothing), at h = 0, 12, 24.
- A **joint** p-value on the `max|beta|` statistic across those horizons —
  family-wise valid, so it is not inflated by testing several horizons.

Stored as `randomization_inference`.

## Result (honest)

200 circular shifts of the BRW shock; the randomization distribution is centred
near zero (mean beta ~ 0), as a valid null requires.

| Horizon | actual beta | RI p-value |
|---|---|---|
| 0 | -0.025 | **0.055** (borderline) |
| 12 | -0.021 | 0.652 |
| 24 | -0.012 | 0.841 |
| **joint max\|beta\|** | — | **0.776** |

Under the design-valid test the **decision horizons are clearly not significant**
and the **joint family-wise sharp null is not rejected** (p = 0.78); only the
impact horizon (h = 0) is borderline. This confirms the correctly-signed but
not-robustly-significant headline with a randomization test that respects the
shock's serial dependence — the null stands.

## Consequences

- The headline now has a valid design-based p-value that respects the shock's
  serial dependence, alongside the analytic Driscoll-Kraay / exposure-robust /
  Conley bands. The iid placebo is kept for reference but superseded for
  inference on the shock timing.
- The joint statistic guards against reading significance into any single horizon.
