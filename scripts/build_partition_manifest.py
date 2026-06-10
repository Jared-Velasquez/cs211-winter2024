"""Build a partition manifest for Student D's aggregate_results.py.

This branch (Student A/C) stores each benchmarked candidate's data scattered under
``artifacts/<candidate_id>/results/`` rather than in the single manifest that
``aggregate_results.py`` expects. This script scans the candidate dirs for one task
and emits the manifest, normalizing latency out of the nested ``benchmark_summary.json``
(``tpu_timing.stats.t_*.mean_ms``) into the flat ``latency_ms`` object D wants.

It deliberately skips candidates with no ``results/outputs.npz`` (the CPU-only /
non-compiling candidates), since D re-evaluates from those saved predictions.

Usage::

    ./run_in_env.sh python scripts/build_partition_manifest.py --task b
    ./run_in_env.sh python scripts/build_partition_manifest.py --task c --output artifacts/manifest_task_c.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# task -> (config path, candidate-id prefix). Baselines are passed to
# aggregate_results.py via --baseline-results (the configs carry None).
TASKS: dict[str, dict[str, str]] = {
    "a": {"config": "configs/task_a_dlc.json", "prefix": "dlc_"},
    "b": {"config": "configs/task_b_detection.json", "prefix": "ssd_"},
    "c": {"config": "configs/task_c_segmentation.json", "prefix": "deeplab_"},
}

ARTIFACTS_ROOT = Path("artifacts")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a partition manifest from artifacts/<id>/results/.")
    parser.add_argument("--task", required=True, choices=sorted(TASKS), help="Task key: a (DLC), b (SSD), c (DeepLab).")
    parser.add_argument("--output", default=None, help="Manifest path. Default: artifacts/manifest_task_<task>.json.")
    return parser.parse_args()


def _normalize_latency(benchmark_summary: dict[str, Any]) -> dict[str, float] | None:
    """Pull the mean latency breakdown out of benchmark_summary.tpu_timing.stats."""
    stats = (benchmark_summary.get("tpu_timing") or {}).get("stats") or {}
    if not stats:
        return None  # cpu-only candidate: no hardware TPU latency
    mapping = {"tpu": "t_tpu", "transfer": "t_transfer", "cpu": "t_cpu", "total": "t_total"}
    latency: dict[str, float] = {}
    for out_key, stat_key in mapping.items():
        mean = (stats.get(stat_key) or {}).get("mean_ms")
        if mean is not None:
            latency[out_key] = float(mean)
    return latency or None


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_manifest(task: str) -> dict[str, Any]:
    spec = TASKS[task]
    prefix = spec["prefix"]

    candidates: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []

    for cand_dir in sorted(ARTIFACTS_ROOT.glob(f"{prefix}*")):
        if not cand_dir.is_dir():
            continue
        results = cand_dir / "results"
        predictions = results / "outputs.npz"
        if not predictions.exists():
            skipped.append({"candidate": cand_dir.name, "reason": "no outputs.npz (cpu-only / non-compiling)"})
            continue

        metadata = _read_json(cand_dir / "metadata.json") or {}
        benchmark = _read_json(results / "benchmark_summary.json") or {}

        proxy_path = results / "proxy_metrics.json"
        # Absolute paths so the manifest resolves correctly regardless of where it lives
        # (aggregate_results.py resolves relative paths against the manifest's own dir).
        entry: dict[str, Any] = {
            "partition_id": metadata.get("partition_id", cand_dir.name),
            "predictions": str(predictions.resolve()),
            "latency_ms": _normalize_latency(benchmark),
            "proxy_metrics": str(proxy_path.resolve()) if proxy_path.exists() else None,
            "num_tpu_ops": metadata.get("num_tpu_ops"),
            "num_cpu_ops": metadata.get("num_cpu_ops"),
            "candidate_id": cand_dir.name,
        }
        candidates.append(entry)

    # Mark the maximum-TPU candidate (most static TPU ops) for strategy comparison.
    tpu_op_counts = [c["num_tpu_ops"] for c in candidates if c.get("num_tpu_ops") is not None]
    max_tpu_ops = max(tpu_op_counts) if tpu_op_counts else None
    for c in candidates:
        c["is_max_tpu"] = c.get("num_tpu_ops") is not None and c["num_tpu_ops"] == max_tpu_ops

    return {
        "task_config": str(Path(spec["config"]).resolve()),
        "num_candidates": len(candidates),
        "skipped": skipped,
        "candidates": candidates,
    }


def main() -> None:
    args = parse_args()
    manifest = build_manifest(args.task)
    output = args.output or str(ARTIFACTS_ROOT / f"manifest_task_{args.task}.json")
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"Task {args.task}: {manifest['num_candidates']} candidates -> {output}")
    for c in manifest["candidates"]:
        lat = "latency" if c["latency_ms"] else "NO-latency"
        proxy = "proxy" if c["proxy_metrics"] else "NO-proxy"
        star = " *max_tpu" if c["is_max_tpu"] else ""
        print(f"  {c['partition_id']:34s} tpu_ops={c['num_tpu_ops']} {lat} {proxy}{star}")
    for s in manifest["skipped"]:
        print(f"  SKIP {s['candidate']}: {s['reason']}")


if __name__ == "__main__":
    main()
