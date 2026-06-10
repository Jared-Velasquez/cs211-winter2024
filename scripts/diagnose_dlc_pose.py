"""Offline diagnostic for the DLC pose-evaluation coordinate mismatch.

Read-only: loads already-saved fp32 predictions (concat_1_0) and reloads AP-10K
ground truth via the existing loader, then reports RMSE / PCK under four
hypotheses to isolate which bug dominates. Touches no pipeline code.
"""
from __future__ import annotations

import json
import math
import os
import sys

sys.path.insert(0, os.getcwd())

import numpy as np

from src.data_loaders import iter_ap10k_pose_dataset
from src.evaluation import TASK_A_DLC_KEYPOINT_NAMES, TASK_A_AP10K_TO_DLC

CFG = json.load(open("configs/task_a_dlc.json"))
PRED_PATH = "artifacts/task_a/dlc_accuracy/full_graph_outputs.npz"
THRESHOLDS = [0.05, 0.10, 0.20]

preds = np.load(PRED_PATH)["concat_1_0"]  # (N, 39, 3) -> (x, y, conf) in 640x480 input space
samples = list(iter_ap10k_pose_dataset(
    images_dir=CFG["images_dir"],
    annotations_path=CFG["annotations_path"],
    resize=CFG["resize"],
    frame_limit=preds.shape[0],
    require_single_instance=CFG.get("require_single_instance", True),
    annotation_strategy=CFG.get("annotation_strategy", "largest_instance"),
))
N = min(len(samples), preds.shape[0])
print(f"predictions={preds.shape}  samples_loaded={len(samples)}  using N={N}")

dlc_index = {n: i for i, n in enumerate(TASK_A_DLC_KEYPOINT_NAMES)}


def collect(transform, pred_index_for):
    """Return (distances, correct_by_thr, per_joint_dists) for a given transform.

    transform(pred_xy, sx, sy) -> (x, y) in ORIGINAL image space.
    pred_index_for(gt_name) -> DLC index to read predictions from.
    """
    dists, per_joint = [], {}
    correct = {t: [] for t in THRESHOLDS}
    for s, pred in zip(samples[:N], preds[:N]):
        gt = s["gt_keypoints"]
        oh, ow = s["original_shape"][0], s["original_shape"][1]
        ih, iw = s["input_shape"][0], s["input_shape"][1]
        sx, sy = ow / iw, oh / ih
        bbox = s.get("bbox") or [0, 0, ow, oh]
        scale = max(float(bbox[2]), float(bbox[3]), 1.0)
        gname_idx = {n: i for i, n in enumerate(s["gt_keypoint_names"])}
        for gt_name in TASK_A_AP10K_TO_DLC:
            gi = gname_idx.get(gt_name)
            pi = pred_index_for(gt_name)
            if gi is None or pi is None:
                continue
            gx, gy, vis = gt[gi]
            if vis <= 0:
                continue
            px, py, _ = pred[pi]
            tx, ty = transform((px, py), sx, sy)
            d = float(math.hypot(tx - gx, ty - gy))
            dists.append(d)
            per_joint.setdefault(gt_name, []).append(d)
            for t in THRESHOLDS:
                correct[t].append(d <= t * scale)
    return dists, correct, per_joint


def summary(name, dists, correct):
    if not dists:
        print(f"{name:42s}  no points"); return
    rmse = math.sqrt(np.mean(np.square(dists)))
    pck = {t: float(np.mean(correct[t])) for t in THRESHOLDS}
    print(f"{name:42s}  RMSE={rmse:8.2f}  PCK@.05={pck[0.05]:.3f} "
          f"@.10={pck[0.10]:.3f} @.20={pck[0.20]:.3f}  n={len(dists)}")


mapped_index = lambda g: dlc_index.get(TASK_A_AP10K_TO_DLC[g])

# H1: current code (compare resized-space preds directly to original-space GT)
d, c, _ = collect(lambda p, sx, sy: (p[0], p[1]), mapped_index)
summary("H1 as-is (current pipeline)", d, c)

# H2: rescale predictions into original image space
d, c, perj = collect(lambda p, sx, sy: (p[0] * sx, p[1] * sy), mapped_index)
summary("H2 +rescale preds->original", d, c)

# H3: rescale + swap x/y (heatmap argmax row/col convention)
d, c, _ = collect(lambda p, sx, sy: (p[1] * sx, p[0] * sy), mapped_index)
summary("H3 +rescale +swap x/y", d, c)

# H4: rescale + per-joint best DLC index (brute force over all 39 parts)
print("\nH4 brute-force best DLC index per AP-10K joint (with rescale):")
best_dists = []
for gt_name in TASK_A_AP10K_TO_DLC:
    best = None
    for cand in range(len(TASK_A_DLC_KEYPOINT_NAMES)):
        d, _, _ = collect(lambda p, sx, sy: (p[0] * sx, p[1] * sy), lambda g, _c=cand: _c)
        # only the rows for THIS gt_name matter; recompute filtered
        # (collect mixes all joints; instead compute per-joint directly below)
        break
# direct per-joint computation (clearer than reusing collect)
for gt_name in TASK_A_AP10K_TO_DLC:
    rows = []
    for s in samples[:N]:
        gname_idx = {n: i for i, n in enumerate(s["gt_keypoint_names"])}
        gi = gname_idx.get(gt_name)
        if gi is None:
            continue
        rows.append(s)
    # gather per-candidate rmse
    cand_rmse = []
    for cand in range(len(TASK_A_DLC_KEYPOINT_NAMES)):
        dd = []
        for s, pred in zip(samples[:N], preds[:N]):
            gname_idx = {n: i for i, n in enumerate(s["gt_keypoint_names"])}
            gi = gname_idx.get(gt_name)
            if gi is None:
                continue
            gx, gy, vis = s["gt_keypoints"][gi]
            if vis <= 0:
                continue
            oh, ow = s["original_shape"][0], s["original_shape"][1]
            ih, iw = s["input_shape"][0], s["input_shape"][1]
            px, py, _ = pred[cand]
            dd.append(math.hypot(px * ow / iw - gx, py * oh / ih - gy))
        if dd:
            cand_rmse.append((math.sqrt(np.mean(np.square(dd))), cand, len(dd)))
    if not cand_rmse:
        continue
    cand_rmse.sort()
    best_rmse, best_cand, npts = cand_rmse[0]
    mapped = dlc_index.get(TASK_A_AP10K_TO_DLC[gt_name])
    mapped_rmse = next((r for r, c2, _ in cand_rmse if c2 == mapped), float("nan"))
    flag = "" if best_cand == mapped else "  <-- MAPPING DIFFERS"
    print(f"  {gt_name:16s} mapped={TASK_A_DLC_KEYPOINT_NAMES[mapped]:16s}"
          f"(rmse {mapped_rmse:7.1f})  best={TASK_A_DLC_KEYPOINT_NAMES[best_cand]:16s}"
          f"(rmse {best_rmse:7.1f}){flag}")
