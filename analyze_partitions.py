"""Student D — Correlation analysis for proxy metrics vs. accuracy drop.

Reads one or more ``partition_results.json`` files produced by
``aggregate_results.py`` and emits a single ``proxy_correlation.json`` with:

  * Spearman rank correlation between every proxy metric and the headline
    accuracy drop, computed per task.
  * Cross-task consistency: the same proxy's ρ across all included tasks.
  * Pareto frontier (lower latency, higher accuracy) per task.
  * Strategy comparison: max-TPU partition vs. best-actual partition vs. each
    proxy's pick.

The analysis layer does not require scipy; Spearman is implemented locally in
``src.analysis``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.analysis import (
    compare_partition_strategies,
    correlate_proxies,
    cross_task_consistency,
    pareto_frontier,
)
from src.io_utils import save_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run proxy-vs-accuracy correlation analysis across tasks.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more partition_results.json files (one per task).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Where to write the consolidated analysis JSON.",
    )
    return parser.parse_args()


def _load_results(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def main() -> None:
    args = parse_args()

    per_task: dict[str, list[dict[str, Any]]] = {}
    per_task_summaries: list[dict[str, Any]] = []
    pareto_per_task: dict[str, list[dict[str, Any]]] = {}
    strategy_per_task: dict[str, dict[str, Any]] = {}

    for path in args.results:
        payload = _load_results(path)
        records = payload.get("records") or []
        if not records:
            print(f"  {path}: no records, skipping")
            continue
        task_type = payload.get("task_type") or records[0]["task"]
        per_task.setdefault(task_type, []).extend(records)

        task_correlations = correlate_proxies(records)
        pareto = pareto_frontier(records)
        strategy = compare_partition_strategies(records, proxy_correlations=task_correlations)

        pareto_per_task[task_type] = pareto
        strategy_per_task[task_type] = strategy
        per_task_summaries.append(
            {
                "task_type": task_type,
                "task_name": payload.get("task_name"),
                "num_records": len(records),
                "proxy_correlations": task_correlations,
            }
        )

    consistency = cross_task_consistency(per_task)

    output = {
        "per_task": per_task_summaries,
        "cross_task_consistency": consistency,
        "pareto_frontier": pareto_per_task,
        "strategy_comparison": strategy_per_task,
    }
    save_json(args.output, output)

    print(f"Wrote analysis to: {args.output}")
    for summary in per_task_summaries:
        print(f"\n[{summary['task_type']}] proxies ranked by |spearman ρ|:")
        for entry in summary["proxy_correlations"][:5]:
            rho = entry["spearman_rho"]
            rho_str = "n/a" if rho is None else f"{rho:+.3f}"
            print(f"  {entry['proxy']:<28s} ρ={rho_str:>7s}  (n={entry['num_points']})")
    if consistency:
        print("\nCross-task consistency (proxies appearing in >=2 tasks):")
        multi = [entry for entry in consistency if entry["num_tasks"] >= 2]
        for entry in multi:
            min_abs = entry["min_abs_rho"]
            min_abs_str = "n/a" if min_abs is None else f"{min_abs:.3f}"
            print(f"  {entry['proxy']:<28s} min|ρ|={min_abs_str}  per-task={entry['per_task_rho']}")


if __name__ == "__main__":
    main()
