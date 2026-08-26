"""End-to-end analysis orchestrator.

Runs the full analysis DAG in dependency order and times each stage:

1. ``data`` — ingest raw sources into the analysis tables (:mod:`cil.data`).
2. ``ces_reconciliation`` — CES supersector ingest + QCEW reconciliation (needed
   by the robustness revision bound).
3. ``shocks`` — build and diagnose the monetary-shock series.
4. ``aggregate`` — aggregate time-series LP, LP-IV variants, narrative-shock LP.
5. ``estimators`` — headline interacted panel LP + robust-inference variants +
   the Goodman-Bacon decomposition.
6. ``heterogeneity`` — DML CATE and CausalForest drivers.
7. ``bayesian`` — hierarchical LP (pooled + cell-level).
8. ``robustness`` — specification curve, placebo, breaks, revision bound,
   randomization inference.
9. ``report`` — export the app-asset CSVs.

Run the whole DAG, or a subset once the upstream tables exist::

    uv run python -m cil.orchestrate
    uv run python -m cil.orchestrate --stages estimators,robustness
    uv run python -m cil.orchestrate --list
"""

from __future__ import annotations

import argparse
import logging
import time
from collections.abc import Callable, Mapping, Sequence

from cil.config import Settings, get_settings
from cil.data import pipeline
from cil.estimators import aggregate, bayesian_build, heterogeneity
from cil.estimators import build as estimators_build
from cil.report import export as report_export
from cil.robustness import build as robustness_build
from cil.robustness import ces_reconciliation
from cil.shocks import build as shocks_build

logger = logging.getLogger("cil.orchestrate")

StageFn = Callable[[Settings], Mapping[str, float]]


def _data_stage(settings: Settings) -> Mapping[str, float]:
    return {key: float(value) for key, value in pipeline.run(settings).items()}


def _data_refresh_stage(settings: Settings) -> Mapping[str, float]:
    return {
        key: float(value)
        for key, value in pipeline.run(settings, force_refresh=True).items()
    }


def _aggregate_stage(settings: Settings) -> Mapping[str, float]:
    return {
        **aggregate.build_aggregate_irf(settings),
        **aggregate.build_narrative_shock_irf(settings),
    }


def _report_stage(settings: Settings) -> Mapping[str, float]:
    return {"assets_written": float(len(report_export.export_app_assets(settings)))}


#: Stages in dependency order.
STAGES: dict[str, StageFn] = {
    "data": _data_stage,
    "ces_reconciliation": ces_reconciliation.build_ces_reconciliation,
    "shocks": shocks_build.build_shocks,
    "aggregate": _aggregate_stage,
    "estimators": estimators_build.build_estimates,
    "heterogeneity": heterogeneity.build_heterogeneity,
    "bayesian": bayesian_build.build_bayesian,
    "robustness": robustness_build.build_robustness,
    "report": _report_stage,
}


def available_stages() -> list[str]:
    """Return the stage names in dependency order."""
    return list(STAGES)


def run_all(
    settings: Settings | None = None,
    *,
    stages: Sequence[str] | None = None,
    stage_map: Mapping[str, StageFn] | None = None,
) -> dict[str, dict[str, float]]:
    """Run the selected stages in dependency order, timing each.

    Parameters
    ----------
    settings
        Project settings.
    stages
        Subset of stage names to run (default: all, in order). Unknown names
        raise ``ValueError``. The given order is ignored; stages always run in
        the canonical dependency order.
    stage_map
        Stage registry (defaults to :data:`STAGES`); injectable for testing.

    Returns
    -------
    dict of str to dict
        Per stage: ``{"seconds": <duration>, **summary}``.
    """
    settings = settings or get_settings()
    registry = stage_map if stage_map is not None else STAGES
    selected = list(registry) if stages is None else list(stages)
    unknown = [name for name in selected if name not in registry]
    if unknown:
        msg = f"Unknown stage(s): {unknown}. Available: {list(registry)}"
        raise ValueError(msg)
    ordered = [name for name in registry if name in set(selected)]

    results: dict[str, dict[str, float]] = {}
    for name in ordered:
        logger.info("stage %s: start", name)
        start = time.perf_counter()
        summary = registry[name](settings)
        elapsed = time.perf_counter() - start
        logger.info("stage %s: done in %.1fs", name, elapsed)
        results[name] = {"seconds": elapsed, **summary}
    return results


def main() -> None:
    """CLI entry point for the orchestrator."""
    parser = argparse.ArgumentParser(description="Run the analysis DAG.")
    parser.add_argument(
        "--stages",
        help="Comma-separated subset of stages to run (default: all).",
    )
    parser.add_argument("--list", action="store_true", help="List the stages and exit.")
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-fetch raw data in the data stage (refresh path) instead of cache.",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    if args.list:
        for name in available_stages():
            print(name)
        return
    stages = args.stages.split(",") if args.stages else None
    registry = {**STAGES, "data": _data_refresh_stage} if args.refresh else None
    results = run_all(stages=stages, stage_map=registry)
    total = sum(stage["seconds"] for stage in results.values())
    for name, stage in results.items():
        print(f"{name}: {stage['seconds']:.1f}s")
    print(f"total: {total:.1f}s")


if __name__ == "__main__":
    main()
