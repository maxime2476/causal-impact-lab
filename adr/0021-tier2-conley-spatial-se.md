# ADR-0021: Tier 2.5 — Conley spatial (space-time HAC) standard errors

- Status: Accepted
- Date: 2026-08-15

## Context

The headline panel LP now carries Driscoll-Kraay (cross-sectional + serial) and
two-way exposure-robust (sector x time) standard errors. Neither models
**geographic** dependence: employment in cells belonging to nearby states
co-moves (regional shocks, migration, supply chains). Tier 2.5 adds Conley (1999)
spatial standard errors, combined with a serial kernel, as the geographic
robustness axis.

## Decision

- `cil.inference.conley`: a space-time HAC. The spatial kernel is a Bartlett
  weight `max(0, 1 - d/cutoff)` in great-circle distance between the two cells'
  **state** centroids (500 km cutoff); the serial kernel is Newey-West with
  bandwidth `h + 1`.
- Tractability: the spatial weight depends only on the state pair, so
  per-observation scores are aggregated to a `(state, time)` grid and the meat is
  `sum_{t,t'} K_time(|t-t'|) g_t' W g_{t'}` with `W` the 51x51 state kernel -- no
  sum over the ~1.5M cell pairs.
- `run_panel_lp_conley` reproduces the headline point estimates (Frisch-Waugh
  partialling of the controls and fixed effects) and reports the Conley SE; stored
  as `panel_lp_conley`.

## Result (honest)

The Conley SE at h = 12 (beta = -0.021) is **highly sensitive to the distance
cutoff**, and that sensitivity is the finding:

| Spatial cutoff | Conley SE | t |
|---|---|---|
| 200 km | 0.009 | -2.28 |
| 500 km | 0.011 | -1.89 |
| 1000 km | 0.015 | -1.35 |
| 3000 km | 0.028 | -0.76 |
| 100000 km (kernel ~ all-ones) | **0.041** | -0.50 |

As the kernel widens to admit full cross-sectional dependence, the Conley SE
**converges to Driscoll-Kraay** (0.041 vs 0.046) and the **null holds** — which
also validates the implementation. A short 500 km cutoff gives a ~4x tighter SE
and apparent significance (23/25 horizons BH at 500 km), but only by assuming
correlation vanishes beyond neighbouring states.

That assumption is wrong for this design: the dominant dependence is **not
geographic** but by **sector** — same-supersector cells across the whole country
share the `exposure_k * s_t` structure (and the cell-level Bayesian, ADR-0019,
found 99.9% of the response heterogeneity is between-sector). A purely spatial
kernel treats those distant same-sector cells as independent and so understates
the SE. This is the same trap documented in ADR-0017 for naive one-way
clustering: a single-dimension robust SE understates when it omits the dependence
dimension the design makes first-order.

## Verdict

The pre-registered null is **not overturned**. At cutoffs wide enough to respect
the design's cross-sectional dependence the Conley SE equals Driscoll-Kraay and
the decision-horizon null stands; the short-cutoff "significance" is a documented
distance-decay artifact, reported (via `conley_cutoff_sensitivity`) rather than
headlined.

## Consequences

- The headline null survives serial (Driscoll-Kraay), exposure-design (two-way
  sector x time, ADR-0017) and geographic (Conley, wide cutoff) robust inference.
- The cutoff-sensitivity table makes the spatial-only artifact transparent;
  centroids are coarse and the cutoff is a modelling choice, so Conley is a
  robustness axis, not a new point estimate.
