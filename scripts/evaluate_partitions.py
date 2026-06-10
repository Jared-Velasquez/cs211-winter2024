"""evaluate_partitions.py — Objective task metrics for each partition candidate.

For every TPU candidate (those with a saved ``results/hybrid_outputs.npz``) this computes the
task-specific accuracy metrics requested for the final analysis:

  * DLC     (pose):          PCK@0.05, PCK@0.10, PCK@0.20, RMSE
  * SSD     (detection):     mAP@0.5, mAP@0.5:0.95
  * DeepLab (segmentation):  mIoU, per-class IoU, boundary F1

It also computes a **float32 baseline** per task on the candidates' actual base model (full-CPU
float path, no quantization) over the same frames, and the per-candidate **accuracy drop**
(baseline − candidate) on the matching frame subset.

Why a fresh baseline: the pre-existing ``artifacts/task_a/dlc_accuracy`` baseline is for
``snapshot-700000 @ 640x480``, but the DLC candidates partition ``snapshot-1000 @ 320x320`` —
a different model. Running the full float model through ``HybridRunner.run_full_cpu`` guarantees
the baseline uses the same model + preprocessing + frames as the candidates.

Candidate predictions come from ``hybrid_outputs.npz`` (unique frames, aligned to the summary's
``sample_ids``); no Coral hardware is needed. The DLC SavedModel suffix keys its single output
``output``; it is remapped to the expected ``concat_1_0``. SSD/DeepLab keys already match.

Outputs:
  * writes an ``accuracy`` block into each candidate's ``results/summary.json``
  * docs_student_c/partition_accuracy.json   — full results incl. baselines + per-class IoU
  * (the consolidated CSV that merges these with the latency breakdown is built by
     scripts/build_partition_metrics.py)

cpu-only candidates (4 DeepLab compile failures + ssd_split_at_box_predictor_biasadds) have no
saved TPU predictions and are skipped.

Run via the repo env wrapper:
    ./run_in_env.sh python scripts/evaluate_partitions.py [--frame-limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
ARTIFACTS = REPO_ROOT / "artifacts"
OUT_JSON = REPO_ROOT / "docs_student_c" / "partition_accuracy.json"

TASK_OF = lambda c: "dlc" if c.startswith("dlc") else "ssd" if c.startswith("ssd") else "deeplab"

# Keys that load_samples + evaluate_outputs read from a candidate's metadata.json.
CONFIG_KEYS = [
    "task_name", "task_type", "model_path", "input_tensor", "output_tensors",
    "data_loader", "resize", "input_shape", "input_dtype", "input_normalization",
    "cpu_graph_path", "images_dir", "annotations_path", "annotations_dir", "video_path",
    "require_single_instance", "annotation_strategy", "skip_missing_images", "pck_thresholds",
]


def _tensor_key(name: str) -> str:
    return name.replace(":", "_").replace("/", "_")


def _config_from_metadata(metadata: dict) -> dict:
    return {k: metadata[k] for k in CONFIG_KEYS if k in metadata}


def _load_outputs(npz_path: Path, output_tensors: list[str]) -> dict[str, np.ndarray]:
    """Load saved predictions, remapping a single-output SavedModel key to the expected key."""
    data = dict(np.load(npz_path))
    expected = [_tensor_key(t) for t in output_tensors]
    if len(expected) == 1 and expected[0] not in data and len(data) == 1:
        # DLC: SavedModel signature names the single output "output"; remap to concat_1_0.
        only_key = next(iter(data))
        data = {expected[0]: data[only_key]}
    return data


def _scalar_metrics(task: str, metrics: dict) -> dict:
    """Pull the flat (CSV-friendly) accuracy scalars for a task from an evaluate_outputs result."""
    if task == "dlc":
        pck = metrics.get("pck", {})
        return {
            "pck_0.05": pck.get("0.05"), "pck_0.10": pck.get("0.1"),
            "pck_0.20": pck.get("0.2"), "rmse": metrics.get("rmse"),
        }
    if task == "ssd":
        return {"map_50": metrics.get("map_50"), "map_50_95": metrics.get("map_50_95")}
    return {"miou": metrics.get("miou"), "boundary_f1": metrics.get("boundary_f1")}


def _candidate_records() -> list[tuple[str, Path, dict, dict]]:
    """(candidate_id, candidate_dir, metadata, summary) for each TPU candidate with predictions."""
    out = []
    for summary_path in sorted(ARTIFACTS.glob("*/results/summary.json")):
        summary = json.loads(summary_path.read_text())
        if summary.get("mode") != "tpu":
            continue
        cand_dir = summary_path.parent.parent
        if not (cand_dir / "results" / "hybrid_outputs.npz").exists():
            print(f"  skip {cand_dir.name}: no hybrid_outputs.npz")
            continue
        metadata = json.loads((cand_dir / "metadata.json").read_text())
        out.append((cand_dir.name, cand_dir, metadata, summary))
    return out


def _reload_samples(config: dict, frame_limit: int, expected_ids: list[str]):
    from src.data_loaders import load_samples

    samples = load_samples(config, frame_limit=frame_limit)
    got = [s["sample_id"] for s in samples]
    if expected_ids and got != expected_ids[: len(got)]:
        raise RuntimeError(
            f"Sample-id mismatch after reload (loader order changed?). "
            f"expected[:3]={expected_ids[:3]} got[:3]={got[:3]}"
        )
    return samples


def compute_baseline(task: str, cand_dir: Path, metadata: dict, frames: int) -> tuple[dict, dict]:
    """Full-CPU float32 baseline for a task: returns (per-frame baseline outputs, samples).

    Uses HybridRunner.run_full_cpu on the candidate's base model so the baseline shares the
    candidate's model + preprocessing. Returns outputs keyed by tensor name (matches evaluate).
    """
    from src.hybrid_runner import HybridRunner
    from src.data_loaders import load_samples

    runner = HybridRunner(str(cand_dir), require_tpu=False)
    samples = runner.validate_samples(load_samples(runner.config, frame_limit=frames))
    full_outputs, _ = runner.run_full_cpu(samples)
    return full_outputs, samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-limit", type=int, default=None,
                        help="Override frames per candidate (default: each candidate's num_frames).")
    args = parser.parse_args()

    from src.evaluation import evaluate_outputs

    candidates = _candidate_records()
    print(f"Evaluating {len(candidates)} TPU candidates.")

    # --- per-task baseline (computed once on the max frame count seen for that task) ---
    by_task: dict[str, list] = {}
    for rec in candidates:
        by_task.setdefault(TASK_OF(rec[0]), []).append(rec)

    baseline_outputs: dict[str, tuple[dict, list]] = {}
    baseline_metrics: dict[str, dict] = {}
    for task, recs in by_task.items():
        frames = max((args.frame_limit or s.get("num_frames") or 0) for _, _, _, s in recs)
        cand_id, cand_dir, metadata, _ = recs[0]
        print(f"[baseline:{task}] full-CPU float on {frames} frames via {cand_id} base model...")
        outs, samples = compute_baseline(task, cand_dir, metadata, frames)
        baseline_outputs[task] = (outs, samples)
        config = _config_from_metadata(metadata)
        baseline_metrics[task] = evaluate_outputs(config, samples, outs)
        print(f"   baseline {task}: {_scalar_metrics(task, baseline_metrics[task])}")

    # --- per-candidate accuracy + drop ---
    results: list[dict] = []
    for cand_id, cand_dir, metadata, summary in candidates:
        task = TASK_OF(cand_id)
        config = _config_from_metadata(metadata)
        frames = args.frame_limit or summary.get("num_frames")
        samples = _reload_samples(config, frames, summary.get("sample_ids", []))
        outputs = _load_outputs(cand_dir / "results" / "hybrid_outputs.npz", config["output_tensors"])

        metrics = evaluate_outputs(config, samples, outputs)
        cand_scalars = _scalar_metrics(task, metrics)

        # Baseline restricted to this candidate's frame subset (prefix of the task baseline).
        base_outs, base_samples = baseline_outputs[task]
        base_outs_n = {k: v[:frames] for k, v in base_outs.items()}
        base_metrics_n = evaluate_outputs(config, base_samples[:frames], base_outs_n)
        base_scalars = _scalar_metrics(task, base_metrics_n)
        drop = {
            k: (base_scalars[k] - cand_scalars[k])
            for k in cand_scalars
            if cand_scalars.get(k) is not None and base_scalars.get(k) is not None
        }

        record = {
            "candidate": cand_id, "task": task, "num_frames": frames,
            "accuracy": metrics, "accuracy_scalars": cand_scalars,
            "baseline_scalars": base_scalars, "drop_vs_baseline": drop,
        }
        results.append(record)
        print(f"[{cand_id}] {cand_scalars}")

        # write into the candidate's summary.json
        summary["accuracy"] = {
            "metrics": metrics,
            "baseline_scalars": base_scalars,
            "drop_vs_baseline": drop,
            "baseline_note": "Full-CPU float32 on the candidate's base model, matched frames.",
        }
        (cand_dir / "results" / "summary.json").write_text(json.dumps(summary, indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(
        {"baselines": baseline_metrics, "candidates": results}, indent=2
    ))
    print(f"\nWrote {OUT_JSON} ({len(results)} candidates, {len(baseline_metrics)} baselines)")


if __name__ == "__main__":
    main()
