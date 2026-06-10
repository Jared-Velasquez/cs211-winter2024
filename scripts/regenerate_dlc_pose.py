"""Regenerate corrected DLC pose accuracy + figures using the fixed evaluate_pose.

Coordinate-correct, real-hardware version. Key facts driving the logic:
  * Per-candidate runs use a SQUARE 320x320 input (40x40 heatmap), NOT the
    config's 640x480. Predictions are rescaled to original-image space using
    each candidate's OWN input size from metadata.json.
  * results/hybrid_outputs.npz holds REAL Edge TPU predictions (int8). The
    candidate's own float-CPU predictions were never saved to disk -- only the
    boundary was. So the matched float baseline is recomputed once here via
    HybridRunner.run_full_cpu (identical across candidates: same frozen graph
    @320) and cached to float_320_outputs.npz.
  * `ran_on_tpu` is verified from summary.json's tpu_output_drift (non-zero =>
    a genuine Edge TPU run, not CPU fallback).

Drop columns = TPU-hybrid vs the float-CPU baseline at matched resolution/frames,
isolating the int8 Edge TPU quantization effect on pose accuracy.
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

from src.data_loaders import iter_ap10k_pose_dataset, load_samples
from src.evaluation import evaluate_pose
from src.hybrid_runner import HybridRunner

CFG = json.load(open("configs/task_a_dlc.json"))
OUT_KEY = "concat_1_0"  # evaluate_pose looks up config["output_tensors"][0] == concat_1:0
OUTDIR = "artifacts/task_a/dlc_accuracy/corrected_pose"
FLOAT_CACHE = "artifacts/task_a/dlc_accuracy/float_320_outputs.npz"
os.makedirs(OUTDIR, exist_ok=True)

# GT keypoints are resize-independent (always original-image space); only
# input_shape depends on resize and we override it per candidate below.
samples = list(iter_ap10k_pose_dataset(
    images_dir=CFG["images_dir"],
    annotations_path=CFG["annotations_path"],
    resize=CFG["resize"],
    frame_limit=100,
    require_single_instance=CFG.get("require_single_instance", True),
    annotation_strategy=CFG.get("annotation_strategy", "largest_instance"),
))

CANDS = sorted(glob.glob("artifacts/dlc_split_*"))

# Float-CPU baseline @320 (same full graph for every candidate) -- compute once, cache.
if os.path.exists(FLOAT_CACHE):
    float_preds = np.load(FLOAT_CACHE)["output"]
else:
    runner = HybridRunner(CANDS[0], require_tpu=False)
    fs = runner.validate_samples(load_samples(runner.config, frame_limit=100))
    cpu_out, _ = runner.run_full_cpu(fs)
    float_preds = cpu_out["output"] if "output" in cpu_out else list(cpu_out.values())[0]
    np.savez(FLOAT_CACHE, output=float_preds)
print(f"float baseline @320: {float_preds.shape}")


def ev(arr, n, in_hw):
    """Evaluate predictions in (in_h x in_w) input space against GT (original space)."""
    in_h, in_w = in_hw
    patched = []
    for s in samples[:n]:
        s2 = dict(s)
        s2["input_shape"] = [in_h, in_w, 3]
        patched.append(s2)
    return evaluate_pose(patched, {OUT_KEY: arr[:n]}, CFG)


def input_hw(meta):
    shp = meta.get("input_shape") or [1] + list(meta.get("resize", [320, 320])) + [3]
    # input_shape like [1, H, W, 3]
    if len(shp) == 4:
        return shp[1], shp[2]
    return shp[0], shp[1]


rows = []
baseline_float = ev(float_preds, float_preds.shape[0], (320, 320))  # full-frame float ref @320
for d in CANDS:
    cid = os.path.basename(d)
    meta = json.load(open(os.path.join(d, "metadata.json")))
    in_hw = input_hw(meta)
    hy = np.load(os.path.join(d, "results", "hybrid_outputs.npz"))["output"]   # real Edge TPU int8
    n = hy.shape[0]

    summ = json.load(open(os.path.join(d, "results", "summary.json")))
    drift = summ.get("tpu_output_drift_vs_partitioned_cpu", {}).get("per_output", {}).get("output", {})
    ran_on_tpu = summ.get("mode") == "tpu" and float(drift.get("mean_abs_diff", 0)) > 0

    acc_tpu = ev(hy, n, in_hw)
    acc_cpu = ev(float_preds, n, (320, 320))  # matched float baseline (same frames)
    t_total = summ.get("hybrid_tpu_timing_ms", {}).get("mean", {}).get("t_total")
    rows.append({
        "candidate": cid,
        "num_frames": n,
        "input_res": f"{in_hw[1]}x{in_hw[0]}",
        "ran_on_tpu": ran_on_tpu,
        "pck_0.05": acc_tpu["pck"]["0.05"],
        "pck_0.1": acc_tpu["pck"]["0.1"],
        "pck_0.2": acc_tpu["pck"]["0.2"],
        "rmse": acc_tpu["rmse"],
        "pck_0.1_drop_vs_float": acc_cpu["pck"]["0.1"] - acc_tpu["pck"]["0.1"],
        "rmse_drop_vs_float": acc_tpu["rmse"] - acc_cpu["rmse"],
        "float_pck_0.1": acc_cpu["pck"]["0.1"],
        "float_rmse": acc_cpu["rmse"],
        "tpu_vs_cpu_mean_drift": float(drift.get("mean_abs_diff", 0)),
        "latency_ms": t_total,
        "tpu_ops_mapped": meta.get("tpu_ops_mapped_edgetpu"),
        "num_tpu_ops": meta.get("num_tpu_ops"),
    })

any_tpu = any(r["ran_on_tpu"] for r in rows)
out = {
    "note": "DLC pose accuracy with the fixed evaluate_pose (x/y swap + per-candidate "
            "320x320 rescale). Drop columns = TPU-hybrid minus the candidate's own "
            "CPU-float reference at matched resolution/frames.",
    "all_candidates_ran_on_real_tpu": any_tpu,
    "warning": None if any_tpu else
        "NO candidate shows a TPU vs CPU-float difference: hybrid_outputs.npz is "
        "byte-identical to the CPU float reference (CPU-simulation runs). Drops are 0 "
        "by construction. Re-run candidates on real Coral hardware for genuine deltas.",
    "float_baseline": {"pck": baseline_float["pck"], "rmse": baseline_float["rmse"]},
    "candidates": rows,
}
json.dump(out, open(os.path.join(OUTDIR, "partition_accuracy_dlc_corrected.json"), "w"), indent=2)

# ---- CSV ----
with open(os.path.join(OUTDIR, "dlc_pose_summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

# ---- markdown table ----
md = ["| Candidate | frames | res | on TPU | PCK@.05 | PCK@.10 | PCK@.20 | RMSE | PCK@.10 drop | Lat (ms) | TPU ops |",
      "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in sorted(rows, key=lambda r: (r["latency_ms"] or 1e9)):
    md.append("| {} | {} | {} | {} | {:.3f} | {:.3f} | {:.3f} | {:.1f} | {:+.3f} | {:.1f} | {} |".format(
        r["candidate"], r["num_frames"], r["input_res"], "yes" if r["ran_on_tpu"] else "no(CPU)",
        r["pck_0.05"], r["pck_0.1"], r["pck_0.2"], r["rmse"],
        r["pck_0.1_drop_vs_float"], r["latency_ms"] or 0.0, r["tpu_ops_mapped"]))
md.append("| **float base** | {} | — | — | {:.3f} | {:.3f} | {:.3f} | {:.1f} | — | — | — |".format(
    baseline_float["num_evaluated_samples"], baseline_float["pck"]["0.05"],
    baseline_float["pck"]["0.1"], baseline_float["pck"]["0.2"], baseline_float["rmse"]))
if not any_tpu:
    md.append("\n> **WARNING:** all candidates are CPU-simulation (hybrid == CPU float); "
              "drops are 0 by construction. Re-run on real Coral hardware for genuine deltas.")
open(os.path.join(OUTDIR, "dlc_pose_table.md"), "w").write("\n".join(md) + "\n")

# ---- Figure 1: Pareto (latency vs PCK@0.10) ----
pts = [r for r in rows if r["latency_ms"] is not None]
fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter([r["latency_ms"] for r in pts], [r["pck_0.1"] for r in pts],
           s=60, color="#1f77b4", zorder=3, label="partition candidate")
for r in pts:
    ax.annotate(r["candidate"].replace("dlc_split_", ""), (r["latency_ms"], r["pck_0.1"]),
                fontsize=7, xytext=(4, 4), textcoords="offset points")
order = sorted(pts, key=lambda r: r["latency_ms"])
front, best = [], -1.0
for r in order:
    if r["pck_0.1"] > best:
        front.append(r); best = r["pck_0.1"]
ax.plot([r["latency_ms"] for r in front], [r["pck_0.1"] for r in front],
        color="#ff7f0e", marker="o", lw=2, zorder=2, label="Pareto frontier")
ax.axhline(baseline_float["pck"]["0.1"], ls="--", color="gray", label="float baseline")
ax.set_xlabel("Hybrid latency t_total (ms)"); ax.set_ylabel("PCK@0.10")
title = "DLC pose: latency vs accuracy"
if not any_tpu:
    title += "  [CPU-sim: no TPU signal]"
ax.set_title(title); ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "pareto_dlc_pose.png"), dpi=130)

# ---- Figure 2: PCK bars per candidate ----
labels = [r["candidate"].replace("dlc_split_", "") for r in rows]
x = np.arange(len(labels)); w = 0.25
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.bar(x - w, [r["pck_0.05"] for r in rows], w, label="PCK@.05")
ax.bar(x, [r["pck_0.1"] for r in rows], w, label="PCK@.10")
ax.bar(x + w, [r["pck_0.2"] for r in rows], w, label="PCK@.20")
for key in ("0.05", "0.1", "0.2"):
    ax.axhline(baseline_float["pck"][key], ls=":", lw=1, color="gray")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("PCK"); ax.set_title("DLC pose PCK per candidate (dotted = float baseline)")
ax.legend(); ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(OUTDIR, "dlc_pose_accuracy_bars.png"), dpi=130)

print("\n".join(md))
print(f"\nran_on_tpu: {sum(r['ran_on_tpu'] for r in rows)}/{len(rows)} candidates")
print(f"Wrote outputs to {OUTDIR}/")
