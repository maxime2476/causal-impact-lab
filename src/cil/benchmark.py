"""Estimator benchmark on a fixed synthetic snapshot.

Catches **numerical** and **performance** regressions without network access or
the real ~1.5M-row panel. It builds a deterministic synthetic panel, runs the
headline panel-LP estimators (Driscoll-Kraay, exposure-robust, Conley), times
each, and compares key numeric outputs against a committed baseline
(``benchmarks/baseline.json``). Numeric drift beyond tolerance exits non-zero;
timings are reported (CI records them) but not hard-asserted, since wall-time is
machine-dependent.

    uv run python -m cil.benchmark                    # check against the baseline
    uv run python -m cil.benchmark --update-baseline  # rewrite the baseline
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import polars as pl

from cil.estimators.panel_lp import (
    PanelLPConfig,
    run_panel_lp,
    run_panel_lp_conley,
    run_panel_lp_exposure_robust,
)

BASELINE_PATH = Path(__file__).resolve().parents[2] / "benchmarks" / "baseline.json"
_SEED = 20260101
_REL_TOL = 0.05  # generous, to absorb cross-platform float differences
_ABS_TOL = 1e-4
_N_STATES = 40
_N_SECTORS = 15
_N_MONTHS = 156
_BETA_TRUE = -0.5


def synthetic_snapshot() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build the fixed deterministic ``(panel, exposure, shock)`` snapshot."""
    rng = np.random.default_rng(_SEED)
    states = [f"{s:02d}" for s in range(1, _N_STATES + 1)]
    sectors = [f"10{k}" for k in range(11, 11 + _N_SECTORS)]
    months = [dt.date(2000 + i // 12, i % 12 + 1, 1) for i in range(_N_MONTHS)]
    expo = dict(zip(sectors, np.linspace(-1.5, 1.5, _N_SECTORS), strict=True))
    s_t = {m: float(rng.normal()) for m in months}
    rows = []
    for st in states:
        for k in sectors:
            level = 0.0
            for m in months:
                level += _BETA_TRUE * expo[k] * s_t[m] + rng.normal(0, 0.2)
                rows.append((f"{st}_{k}", st, k, m, level))
    panel = pl.DataFrame(
        rows,
        schema=["unit_id", "state_fips", "supersector_code", "date", "log_employment"],
        orient="row",
    )
    exposure = pl.DataFrame(
        {"supersector_code": list(expo), "exposure": list(expo.values())}
    )
    shock = pl.DataFrame({"date": months, "shock": [s_t[m] for m in months]})
    return panel, exposure, shock


def run_benchmark() -> dict[str, dict[str, float]]:
    """Run the headline estimators on the snapshot; return metrics and timings."""
    panel, exposure, shock = synthetic_snapshot()
    cfg = PanelLPConfig(horizons=(0, 6, 12))
    cfg12 = PanelLPConfig(horizons=(12,))
    metrics: dict[str, float] = {}
    timings: dict[str, float] = {}

    def _timed(name: str, fn: Callable[[], pl.DataFrame]) -> pl.DataFrame:
        start = time.perf_counter()
        result = fn()
        timings[name] = time.perf_counter() - start
        return result

    dk = _timed(
        "panel_lp", lambda: run_panel_lp(panel, exposure, shock, cfg, shock_col="shock")
    )
    for h in (0, 6, 12):
        metrics[f"beta_h{h}"] = float(dk.filter(pl.col("horizon") == h)["beta"][0])
    metrics["se_h12_dk"] = float(dk.filter(pl.col("horizon") == 12)["se"][0])

    er = _timed(
        "exposure_robust",
        lambda: run_panel_lp_exposure_robust(
            panel, exposure, shock, cfg12, shock_col="shock"
        ),
    )
    metrics["se_h12_exposure_robust"] = float(er["se"][0])

    conley = _timed(
        "conley",
        lambda: run_panel_lp_conley(panel, exposure, shock, cfg12, shock_col="shock"),
    )
    metrics["se_h12_conley"] = float(conley["se"][0])

    return {"metrics": metrics, "timings": timings}


def compare_to_baseline(
    metrics: dict[str, float], baseline: dict[str, float]
) -> list[str]:
    """Return a list of drift messages for metrics that moved beyond tolerance."""
    drifts: list[str] = []
    for name, value in metrics.items():
        if name not in baseline:
            drifts.append(f"{name}: no baseline (current {value:.6g})")
            continue
        expected = baseline[name]
        if not math.isclose(value, expected, rel_tol=_REL_TOL, abs_tol=_ABS_TOL):
            drifts.append(f"{name}: {value:.6g} vs baseline {expected:.6g}")
    return drifts


def main() -> None:
    """Run the benchmark; check against or update the baseline (entry point)."""
    parser = argparse.ArgumentParser(description="Estimator regression benchmark.")
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Rewrite benchmarks/baseline.json from this run.",
    )
    args = parser.parse_args()

    result = run_benchmark()
    metrics, timings = result["metrics"], result["timings"]

    print("== timings (s) ==")
    for name, seconds in timings.items():
        print(f"  {name}: {seconds:.2f}")
    print("== metrics ==")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6g}")

    if args.update_baseline:
        BASELINE_PATH.parent.mkdir(parents=True, exist_ok=True)
        BASELINE_PATH.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
        print(f"baseline updated: {BASELINE_PATH}")
        return

    if not BASELINE_PATH.exists():
        raise SystemExit(f"No baseline at {BASELINE_PATH}; run --update-baseline.")
    baseline = json.loads(BASELINE_PATH.read_text())
    drifts = compare_to_baseline(metrics, baseline)
    if drifts:
        print("== NUMERICAL DRIFT ==")
        for drift in drifts:
            print(f"  {drift}")
        raise SystemExit(1)
    print("benchmark OK: no numerical drift.")


if __name__ == "__main__":
    main()
