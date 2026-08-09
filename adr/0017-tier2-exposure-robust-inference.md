# ADR-0017: Tier 2.1 — exposure-robust (BHJ/AKM) inference

- Status: Accepted
- Date: 2026-08-08

## Context

The headline interacted panel LP estimates `beta_h` on the treatment
`exposure_k * shock_t`, where the exposure shifter `exposure_k` (the standardized
supersector interest-sensitivity) varies **only by supersector**. Inference so
far was Driscoll-Kraay (robust to cross-sectional and serial correlation over
time), which nonetheless treats cells that share a supersector as independent
draws. For a shift-share / exposure design that is exactly the dimension along
which residuals are correlated, so Driscoll-Kraay can understate uncertainty.

## Decision

- Add `run_panel_lp_exposure_robust`: identical point estimates, but the
  covariance is **two-way clustered on the exposure dimension (supersector) and on
  time**. The sector cluster is the Borusyak-Hull-Jaravel (2022) exposure-robust
  dimension (the shifter is common across all states in a supersector;
  Adao-Kolesar-Morales 2019 give an asymptotically equivalent variance). The time
  cluster is essential here because the aggregate shock `s_t` is common to every
  cell at `t`.
- Store `panel_lp_exposure_robust` and report the Driscoll-Kraay vs exposure-robust
  SE and whether any response horizon is BH-significant under it.
- Feasible because the Tier 0 move to 3-digit granularity yields 100 sector
  clusters (a handful of supersectors would have been too few).

## Result (honest)

The initial implementation clustered **one-way on the sector only** and produced
standard errors ~3× smaller than Driscoll-Kraay (median SE ratio 0.32), flipping
0 → 18 of 25 response horizons to BH-significant (|t| up to 8). That was a red
flag, not a discovery: one-way sector clustering **ignores the strong within-time
correlation** induced by the common aggregate shock. Triangulating the covariance
at h = 12 (β = −0.021) makes the artifact explicit:

| Covariance | SE (h=12) | t | Verdict |
|---|---|---|---|
| Driscoll-Kraay (default) | 0.046 | −0.45 | null |
| one-way **sector** (naive) | 0.020 | −1.04 | understated ~3× |
| one-way **state** | 0.007 | −6.8 | absurd (ignores shock) |
| **two-way sector × time** | 0.049 | −0.42 | **null, ≈ Driscoll-Kraay** |

The design-correct exposure-robust SE — two-way sector × time — lands essentially
on top of Driscoll-Kraay (median SE ratio 1.07, i.e. slightly *wider* on average).
The naive one-way variants that overturn the null do so only by discarding a
correlation dimension that the aggregate shock makes first-order.

Under the two-way SE the picture is nulls almost everywhere with a modest
short-horizon exception: **2 of 25 response horizons cross BH-FDR — only h = 2
and h = 3** (BH p ≈ 0.03, t ≈ −3.0), where the two-way SE is a little tighter than
Driscoll-Kraay. Every other horizon is null, and the **decision horizons h = 12
and h = 24 remain firmly null** (t ≈ −0.4 and −0.2).

## Verdict

The **pre-registered decision-horizon null is robust to proper (two-way)
exposure-robust inference** — it is not an artifact of the Driscoll-Kraay kernel.
Exposure-robust inference does sharpen the *early-horizon* (h = 2–3) relative
negatives to marginal BH significance, reported as such but distinct from the
registered h = 12/24 claim. A cautionary corollary, documented so it is not
rediscovered as a false positive: naive one-way clustering (sector- or state-only)
understates uncertainty ~3× in a design with a common aggregate shock and must not
be reported.

## Consequences

- The two-way exposure-robust covariance is available for every downstream
  consumer of the panel LP (`panel_lp_exposure_robust`). Driscoll-Kraay stays the
  default reported band; the two-way exposure-robust band is the
  design-consistent check, and it broadly agrees.
- The decision-horizon null survives the inference method shift-share critics
  would demand; the only new signal is a marginal early-horizon (h = 2–3) relative
  response, reported honestly and not elevated to the headline.
