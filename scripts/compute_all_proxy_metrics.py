"""compute_all_proxy_metrics.py — Student A, Weeks 7-8.

Compute boundary proxy metrics for every valid (partition, task) combination and
store them in the shared Results Record format.

For each candidate under artifacts/<candidate_id>/ that has BOTH boundary captures
saved by Student C's HybridRunner:

  * results/boundary_float_reference.npz   (float32 activations at the boundary)
  * results/boundary_tpu_dequantized.npz   (int8 values dequantized back to float)

we compute the five proxy metrics from the project plan (section 5, weeks 7-8):

  Boundary MSE | Boundary PSNR | Boundary KL Divergence
  Boundary Cosine Similarity | Activation Range Ratio

Candidates missing the TPU-dequantized capture are skipped with a logged reason
(these are the documented CPU-only / never-compiled partitions:
 ssd_split_at_box_predictor_biasadds, and the 4 DeepLab splits that exceed Edge
 TPU SRAM — see CLAUDE.md / docs_student_b/edgetpu_compile_failures.md).

Outputs:
  * artifacts/<candidate_id>/results/proxy_metrics.json   one Results Record per candidate
  * artifacts/proxy_metrics_summary.json                  aggregate of all candidates
  * artifacts/proxy_metrics_summary.csv                   one flat row per candidate (-> Student D)

Run via the repo env wrapper:
    ./run_in_env.sh python scripts/compute_all_proxy_metrics.py
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from compute_proxy_metrics import compute_tensor_metrics  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
FLOAT_NPZ = "boundary_float_reference.npz"
INT8_NPZ = "boundary_tpu_dequantized.npz"

# Shared Results Record proxy_metrics key names (section 4) + the extra range ratio.
# Maps the schema key -> the per-tensor metric key produced by compute_tensor_metrics.
SUMMARY_KEYS = {
    "boundary_mse": "boundary_mse",
    "boundary_psnr_db": "boundary_psnr_db",
    "boundary_kl_div": "boundary_kl_divergence",
    "boundary_cosine_sim": "boundary_cosine_similarity",
    "activation_range_ratio": "activation_range_ratio",
}

# Order candidates by task then id for stable output.
TASK_ORDER = {"dlc": 0, "ssd": 1, "deeplab": 2}


def _candidate_task(candidate_id: str) -> str:
    return candidate_id.split("_", 1)[0]


def _cross_tensor_mean(per_tensor: dict, metric_key: str) -> float | None:
    means = [
        t[metric_key]["mean"]
        for t in per_tensor.values()
        if t[metric_key]["mean"] is not None
    ]
    return float(np.mean(means)) if means else None


def _load_metadata(candidate_dir: Path) -> dict:
    meta_path = candidate_dir / "metadata.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text())


def process_candidate(candidate_dir: Path, num_bins: int = 256) -> dict:
    """Compute proxy metrics for one candidate; return its Results Record."""
    results_dir = candidate_dir / "results"
    f32 = np.load(results_dir / FLOAT_NPZ)
    i8 = np.load(results_dir / INT8_NPZ)

    shared = sorted(set(f32.files) & set(i8.files))
    if not shared:
        raise RuntimeError(f"no matching boundary tensors in {candidate_dir.name}")

    per_tensor: dict[str, dict] = {}
    for key in shared:
        per_tensor[key] = compute_tensor_metrics(f32[key], i8[key], num_bins=num_bins)

    meta = _load_metadata(candidate_dir)
    n_samples = next(iter(per_tensor.values()))["num_samples"]

    proxy_summary = {
        sk: _cross_tensor_mean(per_tensor, mk) for sk, mk in SUMMARY_KEYS.items()
    }

    return {
        "candidate_id": candidate_dir.name,
        "partition_id": meta.get("partition_id"),
        "model": meta.get("model"),
        "task": meta.get("task_type") or _candidate_task(candidate_dir.name),
        "quant_mode": meta.get("quant_mode"),
        "num_calibration_samples": n_samples,
        "boundary_tensors": shared,
        "boundary_tensor_shapes": meta.get("boundary_tensor_shapes"),
        "float32_boundary_path": str((results_dir / FLOAT_NPZ).resolve()),
        "int8_boundary_path": str((results_dir / INT8_NPZ).resolve()),
        # section 4 "proxy_metrics" block (cross-tensor means)
        "proxy_metrics": proxy_summary,
        # full per-tensor / per-sample detail for downstream re-analysis
        "per_tensor": per_tensor,
    }


def main() -> None:
    candidate_dirs = sorted(
        d for d in ARTIFACTS.iterdir()
        if d.is_dir() and (d / "metadata.json").exists()
    )

    records: list[dict] = []
    skipped: list[dict] = []

    for cdir in candidate_dirs:
        results_dir = cdir / "results"
        has_f32 = (results_dir / FLOAT_NPZ).exists()
        has_i8 = (results_dir / INT8_NPZ).exists()

        if not (has_f32 and has_i8):
            reason = (
                "no TPU-dequantized boundary capture "
                "(CPU-only / never-compiled candidate)"
            )
            if not has_f32:
                reason = "no float32 boundary capture"
            skipped.append({"candidate_id": cdir.name, "reason": reason})
            print(f"SKIP  {cdir.name:40s} {reason}")
            continue

        record = process_candidate(cdir)
        out_path = results_dir / "proxy_metrics.json"
        out_path.write_text(json.dumps(record, indent=2))
        records.append(record)
        pm = record["proxy_metrics"]
        print(
            f"OK    {cdir.name:40s} "
            f"MSE={pm['boundary_mse']:.4g} "
            f"PSNR={pm['boundary_psnr_db']:.1f}dB "
            f"KL={pm['boundary_kl_div']:.4g} "
            f"cos={pm['boundary_cosine_sim']:.5f} "
            f"range={pm['activation_range_ratio']:.4g}"
        )

    records.sort(key=lambda r: (TASK_ORDER.get(_candidate_task(r["candidate_id"]), 9),
                                r["candidate_id"]))

    # Aggregate JSON
    summary = {
        "num_candidates_computed": len(records),
        "num_skipped": len(skipped),
        "skipped": skipped,
        "metrics": list(SUMMARY_KEYS.keys()),
        "candidates": [
            {
                "candidate_id": r["candidate_id"],
                "task": r["task"],
                "partition_id": r["partition_id"],
                "num_calibration_samples": r["num_calibration_samples"],
                "num_boundary_tensors": len(r["boundary_tensors"]),
                "proxy_metrics": r["proxy_metrics"],
            }
            for r in records
        ],
    }
    summary_path = ARTIFACTS / "proxy_metrics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # Aggregate CSV (one flat row per candidate) for Student D's correlation analysis.
    csv_path = ARTIFACTS / "proxy_metrics_summary.csv"
    fields = ["task", "candidate_id", "partition_id", "num_calibration_samples",
              "num_boundary_tensors"] + list(SUMMARY_KEYS.keys())
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in records:
            row = {
                "task": r["task"],
                "candidate_id": r["candidate_id"],
                "partition_id": r["partition_id"],
                "num_calibration_samples": r["num_calibration_samples"],
                "num_boundary_tensors": len(r["boundary_tensors"]),
            }
            row.update(r["proxy_metrics"])
            writer.writerow(row)

    print(f"\nComputed proxy metrics for {len(records)} candidate(s); "
          f"skipped {len(skipped)}.")
    print(f"Wrote {summary_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
