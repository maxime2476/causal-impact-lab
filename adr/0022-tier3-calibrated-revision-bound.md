# ADR-0022: Tier 3.1 — calibrated, correlated QCEW revision bound

- Status: Accepted
- Date: 2026-08-18

## Context

The Tier 3 roadmap listed "real QCEW point-in-time vintages" as a robustness
upgrade. That is **infeasible**: BLS does not archive state-by-industry QCEW
vintages (confirmed against the BLS QCEW data-availability documentation — the
"full history" BLS distributes is the current/revised data, not dated vintages).
Fabricating vintages would violate the project's real-data-only rule, so the slot
is reframed as a *revision bound* done honestly.

The existing bound (`revision_bound`) perturbs log employment with **iid** noise.
That is misleadingly reassuring: independent per-observation noise averages out
across ~1.5M cell-months, so the headline coefficient barely moves at any
magnitude (the stored width was ~0). It does not represent how QCEW actually
revises.

## Decision

- Add `correlated_revision_bound`: a **benchmark-step** revision model
  `r_{i,t} = b_{i,year(t)} + eta_{i,t}`, where `b` is a persistent per-cell,
  per-year level step (the annual benchmark) and `eta` is small idiosyncratic
  monthly noise. Persistent steps do not average out, and because they vary across
  years within a cell they are **not** absorbed by the cell fixed effect — so they
  actually move the estimator.
- Calibrate the step magnitude from data: `growth_discrepancy_sd` returns the
  standard deviation of the QCEW-vs-CES 12-month-growth discrepancy, a
  conservative revision scale (it also contains definitional differences between
  the two sources). `sigma_bench = sigma_g / sqrt(2)` so the benchmark component's
  year-over-year growth SD matches `sigma_g`.
- Keep `revision_bound` (iid) for the explicit contrast; store both.

## Result (honest)

Calibration: `sigma_g = 0.048` (QCEW-vs-CES 12-month-growth discrepancy SD) →
`sigma_bench = 0.034`. This is deliberately conservative — the CES-QCEW gap is
dominated by definitional differences and far exceeds true (near-census) QCEW
revisions. At h = 12 (actual beta = -0.0208):

| Revision model | beta range | width | SD |
|---|---|---|---|
| iid (old) | [-0.0212, -0.0205] | 0.0007 | — |
| **benchmark-step (correlated)** | **[-0.0254, -0.0193]** | **0.0062** | 0.0014 |

The correlated bound is **~9x wider** than the iid one, confirming that iid noise
was averaging out and gave a falsely tight bound. Even so, under this conservative
revision scale the headline coefficient **stays firmly negative and never crosses
zero** — it moves within roughly +/- 0.003 of the baseline -0.021. The
correctly-signed (not-BH-significant) headline is therefore robust to realistic
QCEW revisions.

## Consequences

- The revision robustness is now a real test rather than a near-zero-width
  artifact, and it is honest about the absence of true vintages.
- The macro confounders retain genuine ALFRED point-in-time vintages (strict PIT);
  only the QCEW panel lacks vintages, and that limitation is stated here and in
  `docs/data.md`.
