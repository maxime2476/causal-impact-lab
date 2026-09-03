# ADR-0032: Tier 6.3 — sign-restriction SVAR

- Status: Accepted
- Date: 2026-09-02

## Context

The aggregate employment response was estimated two ways — a high-frequency-
instrument LP-IV (strong first stage but wrong-signed) and a narrative-shock LP
(right-signed but imprecise) — which disagree. A third, structurally different
identification is a natural check on whether the aggregate can be pinned down.

## Decision

`sign_svar.sign_restricted_svar`: a monetary VAR in the policy rate, log CPI, log
employment, and log industrial production (12 lags, 1994-2020), identified by
**sign restrictions** in the spirit of Uhlig (2005). Random orthogonal rotations
(Haar) are drawn; a rotation's candidate monetary column is accepted if, over
months 0-5, the **policy rate rises and the price level falls**. Output and
**employment are left unrestricted**, so the data — not the identifying scheme —
determine the employment response. The accepted draws' employment IRFs give the
identified set (median and 68% band); wired into `cil.estimators.aggregate`
(`sign_svar_irf`).

## Result (honest)

Acceptance rate 34%. The employment response to a one-SD contractionary shock is
**essentially indeterminate**: the median is near zero and wanders slightly
positive at the medium run (h = 0 +0.03%, h = 12 +0.10%, h = 24 −0.03%), and the
**68% band includes zero at every horizon** (e.g. h = 12: [−0.51, +0.67]).

Leaving employment unrestricted, the identified set spans zero — the data do not
pin down the sign. This third identification joins the wrong-signed HF LP-IV and
the imprecise narrative LP in **failing to identify the aggregate response**,
reinforcing that the aggregate effect is the assumption-dependent complement, not
the answer. The slightly-positive medium-run median echoes the price-puzzle-like
behaviour seen in the HF LP-IV.

## Consequences

- Three structurally different aggregate identifications (HF instrument,
  narrative, sign-restricted SVAR) all decline to pin down the employment
  response — a strong, honest statement about the fragility of the aggregate.
- The cleanly-identified relative design remains the headline.
