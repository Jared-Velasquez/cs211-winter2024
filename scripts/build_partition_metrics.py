"""build_partition_metrics.py — Merge accuracy + latency breakdown + drift into one table.

Reads:
  * docs_student_c/partition_accuracy.json        (from evaluate_partitions.py)
  * each candidate's results/summary.json          (drift)
  * benchmark.py timing                            (latency breakdown, via plot_latency_drift)

Writes:
  * docs_student_c/partition_metrics.csv           one wide row per TPU candidate:
        identity + latency breakdown (t_tpu/t_transfer/t_cpu/t_total x mean/std/p95)
        + drift (output, boundary) + task accuracy scalars + baseline + drop-vs-baseline
  * docs_student_c/partition_per_class_iou.json    DeepLab per-class IoU (candidates + baseline)

Run via the repo env wrapper:
    ./run_in_env.sh python scripts/build_partition_metrics.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.plot_latency_drift import resolve_timing_breakdown  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
DOCS = REPO_ROOT / "docs_student_c"
ACC_JSON = DOCS / "partition_accuracy.json"

TASK_ORDER = ["dlc", "ssd", "deeplab"]
TIMING_COMPONENTS = ["t_tpu", "t_transfer", "t_cpu", "t_total"]
TIMING_STATS = ["mean_ms", "std_ms", "p95_ms"]
# Union of accuracy scalars across tasks (blank where not applicable).
ACC_SCALARS = ["pck_0.05", "pck_0.10", "pck_0.20", "rmse", "map_50", "map_50_95", "miou", "boundary_f1"]


def _drift(candidate_dir: Path) -> tuple[float | None, float | None]:
    summary = json.loads((candidate_dir / "results" / "summary.json").read_text())
    out = (summary.get("tpu_output_drift_vs_partitioned_cpu") or {}).get("mean_abs_diff")
    bnd = (summary.get("boundary_drift_tpu_dequantized_vs_float") or {}).get("mean_abs_diff")
    return out, bnd


def main() -> None:
    acc = json.loads(ACC_JSON.read_text())
    cands = sorted(acc["candidates"], key=lambda r: (TASK_ORDER.index(r["task"]), r["candidate"]))

    fields = ["task", "candidate", "num_frames"]
    fields += [f"{c}_{s.replace('_ms', '')}_ms" for c in TIMING_COMPONENTS for s in TIMING_STATS]
    fields += ["output_drift_mean", "boundary_drift_mean"]
    fields += ACC_SCALARS
    fields += [f"baseline_{m}" for m in ACC_SCALARS]
    fields += [f"drop_{m}" for m in ACC_SCALARS]

    csv_path = DOCS / "partition_metrics.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in cands:
            cdir = ARTIFACTS / r["candidate"]
            timing = resolve_timing_breakdown(cdir) or {}
            out_drift, bnd_drift = _drift(cdir)
            row = {"task": r["task"], "candidate": r["candidate"], "num_frames": r["num_frames"]}
            for c in TIMING_COMPONENTS:
                comp = timing.get(c, {})
                for s in TIMING_STATS:
                    row[f"{c}_{s.replace('_ms', '')}_ms"] = comp.get(s)
            row["output_drift_mean"] = out_drift
            row["boundary_drift_mean"] = bnd_drift
            for m in ACC_SCALARS:
                row[m] = r["accuracy_scalars"].get(m)
                row[f"baseline_{m}"] = r["baseline_scalars"].get(m)
                row[f"drop_{m}"] = r["drop_vs_baseline"].get(m)
            writer.writerow(row)
    print(f"Wrote {csv_path} ({len(cands)} candidates)")

    # Per-class IoU (DeepLab only) — too wide for the CSV.
    per_class = {"baseline": acc["baselines"].get("deeplab", {}).get("per_class_iou", {}),
                 "candidates": {}}
    for r in cands:
        if r["task"] == "deeplab":
            per_class["candidates"][r["candidate"]] = r["accuracy"].get("per_class_iou", {})
    pci_path = DOCS / "partition_per_class_iou.json"
    pci_path.write_text(json.dumps(per_class, indent=2))
    print(f"Wrote {pci_path}")


if __name__ == "__main__":
    main()
