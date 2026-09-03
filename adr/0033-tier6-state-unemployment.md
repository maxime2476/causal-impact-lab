# ADR-0033: Tier 6.4 — state unemployment as a third outcome

- Status: Accepted
- Date: 2026-09-02

## Context

The relative design has been applied to employment (headline) and wages (Tier
6.1), both at the state x industry cell level. A third, conceptually independent
labour-market outcome — the **state unemployment rate** — is available at the
state level from BLS LAUS via FRED, and gives a further check on the honest null.

## Decision

- `cil.data.state_unemp` + `pipeline.ingest_state_unemployment`: fetch each
  state's seasonally-adjusted unemployment rate (FRED `{ABBR}UR`, e.g. `CAUR`),
  store `state_unemployment`, wired into `run`.
- `cil.estimators.unemployment`: the interacted panel LP at the **state** level.
  The outcome is the change in the state unemployment rate; the treatment is the
  state **Bartik exposure** `E_s = sum_k omega_{s,k} sigma_k` interacted with the
  national shock, with state and time fixed effects. It reuses `run_panel_lp` via
  its `outcome_col` argument (`unemployment_panel_lp_results`).

The **expected sign is positive** here (the opposite of employment): a
contractionary shock should raise unemployment *more* in exposed states.

## Result (honest)

On 51 states (30,957 state-months), the state-unemployment relative effect is
**correctly signed on impact and the short run** — a tightening raises
unemployment more in exposed states (h = 0: +0.32, declining through h ≈ 3) — but
the coefficient **turns negative at the medium/long run** (h = 12: −0.39, h = 24:
−0.49; only 7 of 25 horizons positive). Event-study leads are clean (max |t| =
1.3) and **no horizon is BH-significant**.

This is a **noisier, more mixed null** than the employment and wage outcomes,
which were negative at every horizon. The correct impact sign is reassuring; the
medium-run reversal is not statistically distinguishable from zero and is
consistent with an imprecise, sign-flipping response at a coarser (state) level of
aggregation with the state Bartik exposure. It neither strengthens nor contradicts
the headline — it is a third insignificant outcome.

## Consequences

- A third labour-market outcome, at a different level of aggregation (state, not
  cell) and with the state Bartik exposure, tests the headline story from another
  angle.
- The `run_panel_lp` `outcome_col` generalisation (Tier 6.1) is reused unchanged.
