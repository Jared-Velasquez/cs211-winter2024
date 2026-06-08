"""Regenerate corrected DLC pose accuracy + figures using the fixed evaluate_pose.

Re-evaluates every DLC partition candidate's TPU-hybrid predictions against
AP-10K ground truth, with the baseline matched per-candidate to the same frame
count (so accuracy drop is apples-to-apples). Writes a corrected JSON, a
markdown + CSV table, and pose figures (Pareto + accuracy bars).
"""
from __future__ import annotations

import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_loaders import iter_ap10k_pose_dataset
from src.evaluation import evaluate_pose

CFG = json.load(open("configs/task_a_dlc.json"))
OUT_KEY = "concat_1_0"  # evaluate_pose looks up config["output_tensors"][0] == concat_1:0
BASELINE_PRED = "artifacts/task_a/dlc_accuracy/full_graph_outputs.npz"
OUTDIR = "artifacts/task_a/dlc_accuracy/corrected_pose"
os.makedirs(OUTDIR, exist_ok=True)

samples = list(iter_ap10k_pose_dataset(
    images_dir=CFG["images_dir"],
    annotations_path=CFG["annotations_path"],
    resize=CFG["resize"],
    frame_limit=100,
    require_single_instance=CFG.get("require_single_instance", True),
    annotation_strategy=CFG.get("annotation_strategy", "largest_instance"),
))


def ev(arr, n):
    return evaluate_pose(samples[:n], {OUT_KEY: arr[:n]}, CFG)


base_preds = np.load(BASELINE_PRED)["concat_1_0"]
baseline_full = ev(base_preds, len(samples))

rows = []
for d in sorted(glob.glob("artifacts/dlc_split_*")):
    cid = os.path.basename(d)
    hy = np.load(os.path.join(d, "results", "hybrid_outputs.npz"))["output"]
    n = hy.shape[0]
    acc = ev(hy, n)
    base_n = ev(base_preds, n)  # baseline on the SAME frames -> fair drop
    summ = json.load(open(os.path.join(d, "results", "summary.json")))
    t_total = summ.get("hybrid_tpu_timing_ms", {}).get("mean", {}).get("t_total")
    meta = json.load(open(os.path.join(d, "metadata.json")))
    rows.append({
        "candidate": cid,
        "num_frames": n,
        "pck_0.05": acc["pck"]["0.05"],
        "pck_0.1": acc["pck"]["0.1"],
        "pck_0.2": acc["pck"]["0.2"],
        "rmse": acc["rmse"],
        "pck_0.1_drop_vs_baseline": base_n["pck"]["0.1"] - acc["pck"]["0.1"],
        "rmse_drop_vs_baseline": acc["rmse"] - base_n["rmse"],
        "baseline_matched_pck_0.1": base_n["pck"]["0.1"],
        "baseline_matched_rmse": base_n["rmse"],
        "latency_ms": t_total,
        "tpu_ops_mapped": meta.get("tpu_ops_mapped_edgetpu"),
        "num_tpu_ops": meta.get("num_tpu_ops"),
    })

out = {
    "note": "Corrected DLC pose accuracy after fixing the evaluate_pose x/y-swap + "
            "rescale bug. Drop columns use a baseline matched to each candidate's frame count.",
    "baseline_fp32": {
        "num_frames": len(samples),
        "pck": baseline_full["pck"],
        "rmse": baseline_full["rmse"],
    },
    "candidates": rows,
}
json.dump(out, open(os.path.join(OUTDIR, "partition_accuracy_dlc_corrected.json"), "w"), indent=2)

# ---- CSV ----
with open(os.path.join(OUTDIR, "dlc_pose_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- markdown table ----
md = ["| Candidate | frames | PCK@.05 | PCK@.10 | PCK@.20 | RMSE | PCK@.10 drop | Lat (ms) | TPU ops |",
      "|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["latency_ms"] or 1e9)):
    md.append("| {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:+.3f} | {:.1f} | {} |".format(
        r["candidate"], r["num_frames"], r["pck_0.05"], r["pck_0.1"], r["pck_0.2"],
        r["rmse"], r["pck_0.1_drop_vs_baseline"], r["latency_ms"], r["tpu_ops_mapped"]))
md.append("| **fp32 base** | {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | — | — | — |".format(
    len(samples), baseline_full["pck"]["0.05"], baseline_full["pck"]["0.1"],
    baseline_full["pck"]["0.2"], baseline_full["rmse"]))
open(os.path.join(OUTDIR, "dlc_pose_table.md"), "w").write("\n".join(md) + "\n")

# ---- Figure 1: Pareto (latency vs PCK@0.10) ----
pts = [r for r in rows if r["latency_ms"] is not None]
xs = [r["latency_ms"] for r in pts]
ys = [r["pck_0.1"] for r in pts]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(xs, ys, s=60, color="#1f77b4", zorder=3, label="partition candidate")
for r in pts:
    ax.annotate(r["candidate"].replace("dlc_split_", ""), (r["latency_ms"], r["pck_0.1"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
# Pareto frontier: min latency, max PCK
order = sorted(pts, key=lambda r: r["latency_ms"])
front, best = [], -1
for r in order:
    if r["pck_0.1"] > best:
        front.append(r); best = r["pck_0.1"]
ax.plot([r["latency_ms"] for r in front], [r["pck_0.1"] for r in front],
        color="#ff7f0e", marker="o", lw=2, zorder=2, label="Pareto frontier")
ax.axhline(baseline_full["pck"]["0.1"], ls="--", color="gray", label="fp32 baseline")
ax.set_xlabel("Hybrid latency t_total (ms)")
ax.set_ylabel("PCK@0.10")
ax.set_title("DLC pose: latency vs accuracy (corrected)")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "pareto_dlc_pose.png"), dpi=130)

# ---- Figure 2: PCK bars per candidate ----
labels = [r["candidate"].replace("dlc_split_", "") for r in rows]
x = np.arange(len(labels)); w = 0.25
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w, [r["pck_0.05"] for r in rows], w, label="PCK@.05")
ax.bar(x, [r["pck_0.1"] for r in rows], w, label="PCK@.10")
ax.bar(x + w, [r["pck_0.2"] for r in rows], w, label="PCK@.20")
for thr, key in [(0.05, "0.05"), (0.1, "0.1"), (0.2, "0.2")]:
    ax.axhline(baseline_full["pck"][key], ls=":", lw=1, color="gray")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("PCK"); ax.set_title("DLC pose PCK per candidate (dotted = fp32 baseline)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "dlc_pose_accuracy_bars.png"), dpi=130)

print("\n".join(md))
print(f"\nWrote outputs to {OUTDIR}/")
print("  - partition_accuracy_dlc_corrected.json")
print("  - dlc_pose_summary.csv")
print("  - dlc_pose_table.md")
print("  - pareto_dlc_pose.png")
print("  - dlc_pose_accuracy_bars.png")
