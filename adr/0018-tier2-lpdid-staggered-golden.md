# ADR-0018: Tier 2.2 — LP-DiD staggered clean controls + known-DGP golden

- Status: Accepted
- Date: 2026-08-11

## Context

The LP-DiD port (`cil.estimators.lp_did`) used a **horizon-independent** clean
control condition: it dropped already-treated rows but kept every not-yet-treated
control. Under staggered adoption that is wrong at response horizons `h >= 1`: a
not-yet-treated control that adopts *inside* the outcome window `(t, t+h]` carries
its own treatment effect into the long difference `y_{t+h} - y_{t-1}`,
contaminating the ATT. The bug was invisible to the existing non-staggered
TWFE-equivalence test (single switch time, no in-window adoption).

## Decision

- Make the clean-control condition **horizon-dependent** (Dube-Girardi-Jorda-
  Taylor). At horizon `h`: the treated group is units newly treated at `t`
  (`d_treat == 1`); a clean control is untreated at `t` *and* still untreated at
  `t+h` (for a response horizon). For pre-treatment leads (`h < 0`) cleanliness
  only requires being untreated at `t`. Absorbing / staggered adoption is assumed.
- Add a **known-DGP staggered golden** (`tests/golden/test_lpdid_golden.py`,
  fixtures under `tests/golden/fixtures/lpdid/`): a deterministic panel with
  two-way fixed effects, three adoption cohorts, never-treated controls, and a
  prescribed dynamic effect path `tau_h = {0:-.2, 1:-.4, 2:-.6, 3:-.5, 4:-.4,
  5:-.3}` (no noise). The estimator must recover `tau_h` at each response horizon
  and 0 at the lead. This is **not** a Stata run — Stata cannot be executed here,
  so no Stata parity is claimed; `expected.csv` holds the DGP's known values, and
  a Stata cross-check can overwrite it later.

## Result

- Before the fix the golden DGP was recovered only at `h = 0` (`-0.20`); `h >= 1`
  drifted (`h=1` `-0.38` vs `-0.40`, `h=2` `-0.55` vs `-0.60`, worsening with
  horizon) with spurious non-zero SEs — the contamination signature.
- After the fix LP-DiD recovers the full path to **machine precision**
  (max abs error ~1e-15, SEs ~1e-16); the lead is 0 (no pre-trend). The
  non-staggered TWFE-equivalence unit test still passes.

## Consequences

- The staggered LP-DiD robustness estimator is now correct and covered by a
  committed golden that runs in the default suite. `lp_did` is used by the
  Goodman-Bacon diagnostic (unchanged public API); no stored headline result
  depends on it, so no downstream re-run is required.
- A future Stata `lpdid` cross-validation remains a drop-in (overwrite
  `expected.csv`), and is explicitly not claimed as done.
