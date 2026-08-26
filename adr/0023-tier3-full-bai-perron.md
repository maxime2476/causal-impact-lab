# ADR-0023: Tier 3.2 — full Bai-Perron structural breaks

- Status: Accepted
- Date: 2026-08-20

## Context

Phase-8 break detection used PELT (`ruptures`), a fast *approximate* penalised
change-point search. It reports a break set but not the Bai-Perron machinery:
global-optimal segmentation, a principled choice of the number of breaks, or
confidence intervals for the break dates. Tier 3.2 upgrades to the full procedure.

## Decision

`breaks.bai_perron_full` on national employment growth:

1. **Exact global minimisation.** For each candidate break count `m = 0..M`,
   dynamic programming (`ruptures.Dynp`, `jump=1`) returns the segmentation that
   globally minimises the within-segment SSR — the Bai-Perron global optimiser,
   not a greedy/penalised approximation. The cost matrix is fit once and reused
   across `m`.
2. **Break-number selection by BIC.** `BIC(m) = n·log(SSR_m/n) + (2m+1)·log(n)`
   (`(m+1)` segment means + `m` break locations); the minimiser is selected. The
   full BIC path is stored (`break_selection`).
3. **Bootstrap break-date CIs.** A residual bootstrap (resample the
   fitted-model residuals, re-run the exact DP, take the 2.5/97.5 percentiles of
   each break date). This is self-contained and avoids hard-coding Bai's (1997)
   asymptotic CI constant — reported honestly as a bootstrap interval.

Stored as `structural_breaks` (`break_date`, `ci_low_date`, `ci_high_date`,
`delta`) and `break_selection`.

## Result (honest)

On national employment growth over the analysis window (1994-2020, n = 323):

| m breaks | SSR | BIC |
|---|---|---|
| **0** | 241.3 | **-88.4 (selected)** |
| 1 | 236.9 | -82.9 |
| 2 | 235.5 | -73.1 |
| 3 | 233.3 | -64.6 |

BIC is minimised at **zero breaks** and rises monotonically thereafter: the SSR
reduction from adding breaks (241 -> 236 -> 235) does not justify the extra
parameters. Under the full Bai-Perron/BIC procedure there is **no statistically
warranted structural break** in national employment growth over the study period
— the relationship is stable. The 2020 COVID collapse-and-rebound is a *transient
spike*, not a persistent mean-regime shift, so it is (correctly) not flagged as a
break; it is handled separately by the state-dependent LP (`state_dependent_irf`).
`structural_breaks` is therefore empty; the BIC path is in `break_selection`.

## Consequences

- Break detection is now a defensible Bai-Perron analysis (global optimum +
  model selection + dated uncertainty), not a single penalised guess.
- The bootstrap CI is an honest substitute for the asymptotic interval; it makes
  no distributional claim beyond resampling the estimated residuals.
