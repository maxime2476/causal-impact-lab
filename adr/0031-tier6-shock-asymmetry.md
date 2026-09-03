# ADR-0031: Tier 6.2 — shock asymmetry (sign and size)

- Status: Accepted
- Date: 2026-09-02

## Context

The headline estimates a single relative semi-elasticity `beta_h`, implicitly
assuming the response is symmetric in the shock's sign (tightening vs easing) and
linear in its size. Whether contractionary and expansionary shocks, or large and
small ones, have different relative effects is a natural scope question.

## Decision

`asymmetry.run_panel_lp_asymmetry` splits the interacted-LP treatment `E_i * s_t`
into two components estimated **jointly** in one two-way-FE regression, so each
gets its own coefficient:

- **sign**: `treat_a = E_i * max(s_t, 0)` (tightening), `treat_b = E_i * min(s_t,
  0)` (easing).
- **size**: split at the median `|s_t|` — `treat_a` on large shocks, `treat_b` on
  small ones.

A Wald test of `beta_a = beta_b` at each horizon (`diff`, its SE from the joint
covariance, and `p_diff` with BH-FDR) quantifies the asymmetry. Run at h = 0, 6,
12, 24 on the headline panel (`asymmetry_sign`, `asymmetry_size`).

## Result (honest)

**Sign.** The pooled negative relative effect comes more from *easing* than
*tightening*: at h = 12 the easing coefficient is −0.11 while the tightening
coefficient is near zero/positive (+0.08). The difference is only marginal
(`p_diff` 0.052 at h = 0, 0.067 at h = 12) and **does not survive BH-FDR** (BH p ≈
0.13). No robust sign asymmetry.

**Size.** The credible *large*-shock coefficient (−0.02 to −0.05) matches the
headline. The h = 0 large-vs-small difference is marginally BH-significant (BH p =
0.059), but it is driven by the **imprecise small-shock coefficient** (+0.25):
small shocks carry little identifying variation, so their estimate is very noisy
(standard errors blow up at longer horizons). This is not a credible structural
size asymmetry.

**Verdict.** Neither asymmetry survives false-discovery control; the headline's
single-`beta` linearity is not rejected. The marginal signals are consistent with
the imprecise, correctly-signed null rather than a genuine nonlinearity.

## Consequences

- The single-`beta` linearity of the headline is tested directly; any asymmetry
  is reported with its own uncertainty, and the joint fit makes the two
  coefficients comparable.
