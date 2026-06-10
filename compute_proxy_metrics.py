"""
Compute boundary proxy metrics between float32 and dequantized-int8 activations.

Metrics (per boundary tensor, averaged across calibration samples):
  - Boundary MSE               : mean squared error between float32 and int8 tensors
  - Boundary PSNR              : 10 * log10(max_val^2 / MSE)
  - Boundary KL Divergence     : KL(float32 || int8) on histogram distributions
  - Boundary Cosine Similarity : cosine similarity of flattened per-sample tensors
  - Activation Range Ratio     : (max-min) float32 / (max-min) int8

Usage:
  python compute_proxy_metrics.py \\
      --float32  artifacts/task_a/dlc/hybrid_float_boundary_outputs.npz \\
      --int8     artifacts/task_a/dlc/hybrid_tpu_boundary_dequantized.npz \\
      --output   artifacts/task_a/dlc/proxy_metrics.json \\
      --task     task_a_dlc
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Per-sample metric helpers
# ---------------------------------------------------------------------------

def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def _psnr(mse: float, max_val: float) -> float:
    if mse == 0.0:
        return float("inf")
    return 10.0 * math.log10((max_val ** 2) / mse)


def _kl_divergence(a: np.ndarray, b: np.ndarray, num_bins: int = 256) -> float:
    """KL(P_float32 || P_int8) using shared histogram edges."""
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    lo = min(a_flat.min(), b_flat.min())
    hi = max(a_flat.max(), b_flat.max())
    if lo == hi:
        return 0.0
    edges = np.linspace(lo, hi, num_bins + 1)
    p, _ = np.histogram(a_flat, bins=edges, density=False)
    q, _ = np.histogram(b_flat, bins=edges, density=False)
    p = p.astype(np.float64) + 1e-10
    q = q.astype(np.float64) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(np.sum(p * np.log(p / q)))


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.ravel().astype(np.float64)
    b_flat = b.ravel().astype(np.float64)
    norm_a = np.linalg.norm(a_flat)
    norm_b = np.linalg.norm(b_flat)
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a_flat, b_flat) / (norm_a * norm_b))


def _range_ratio(a: np.ndarray, b: np.ndarray) -> float:
    """(max-min) float32 / (max-min) int8. Returns inf if int8 range is zero."""
    range_a = float(a.max()) - float(a.min())
    range_b = float(b.max()) - float(b.min())
    if range_b == 0.0:
        return float("inf") if range_a != 0.0 else 1.0
    return range_a / range_b


# ---------------------------------------------------------------------------
# Per-tensor aggregation
# ---------------------------------------------------------------------------

def compute_tensor_metrics(
    float32_arr: np.ndarray,
    int8_arr: np.ndarray,
    num_bins: int = 256,
) -> dict:
    """
    Compute all proxy metrics for one boundary tensor across all samples.

    Both arrays are shape (N, ...) where N is the number of calibration samples.
    Returns per-sample lists and mean/std aggregates.
    """
    assert float32_arr.shape == int8_arr.shape, (
        f"Shape mismatch: {float32_arr.shape} vs {int8_arr.shape}"
    )
    n = float32_arr.shape[0]

    mse_vals, psnr_vals, kl_vals, cos_vals, range_vals = [], [], [], [], []

    # Use the global float32 max value for PSNR (consistent across samples)
    max_val = max(float(np.abs(float32_arr).max()), 1e-8)

    for i in range(n):
        a = float32_arr[i]
        b = int8_arr[i]
        mse = _mse(a, b)
        mse_vals.append(mse)
        psnr_vals.append(_psnr(mse, max_val))
        kl_vals.append(_kl_divergence(a, b, num_bins))
        cos_vals.append(_cosine_similarity(a, b))
        range_vals.append(_range_ratio(a, b))

    def _stats(vals: list[float]) -> dict:
        finite = [v for v in vals if math.isfinite(v)]
        return {
            "per_sample": vals,
            "mean": float(np.mean(finite)) if finite else None,
            "std": float(np.std(finite)) if finite else None,
        }

    return {
        "num_samples": n,
        "tensor_shape": list(float32_arr.shape),
        "float32_global_max_val": max_val,
        "boundary_mse": _stats(mse_vals),
        "boundary_psnr_db": _stats(psnr_vals),
        "boundary_kl_divergence": _stats(kl_vals),
        "boundary_cosine_similarity": _stats(cos_vals),
        "activation_range_ratio": _stats(range_vals),
    }


# ---------------------------------------------------------------------------
# Results Record format
# ---------------------------------------------------------------------------

def build_results_record(
    task_name: str,
    float32_path: str,
    int8_path: str,
    per_tensor: dict[str, dict],
) -> dict:
    """Wrap per-tensor results in the shared Results Record schema."""
    # Aggregate mean across tensors for a single summary number per metric
    def _cross_tensor_mean(metric_key: str) -> float | None:
        means = [
            t[metric_key]["mean"]
            for t in per_tensor.values()
            if t[metric_key]["mean"] is not None
        ]
        return float(np.mean(means)) if means else None

    return {
        "task_name": task_name,
        "float32_boundary_path": float32_path,
        "int8_boundary_path": int8_path,
        "num_boundary_tensors": len(per_tensor),
        "summary": {
            "boundary_mse_mean": _cross_tensor_mean("boundary_mse"),
            "boundary_psnr_db_mean": _cross_tensor_mean("boundary_psnr_db"),
            "boundary_kl_divergence_mean": _cross_tensor_mean("boundary_kl_divergence"),
            "boundary_cosine_similarity_mean": _cross_tensor_mean("boundary_cosine_similarity"),
            "activation_range_ratio_mean": _cross_tensor_mean("activation_range_ratio"),
        },
        "per_tensor": per_tensor,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute boundary proxy metrics.")
    p.add_argument(
        "--float32", required=True,
        help="Path to .npz with float32 boundary activations (keys = tensor names, shape N×...).",
    )
    p.add_argument(
        "--int8", required=True,
        help="Path to .npz with dequantized int8 boundary activations (same keys/shapes).",
    )
    p.add_argument(
        "--output", required=True,
        help="Destination .json path for the results record.",
    )
    p.add_argument(
        "--task", default="unknown",
        help="Task name label embedded in the results record.",
    )
    p.add_argument(
        "--bins", type=int, default=256,
        help="Number of histogram bins for KL divergence (default: 256).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    float32_path = Path(args.float32)
    int8_path = Path(args.int8)
    output_path = Path(args.output)

    if not float32_path.exists():
        raise FileNotFoundError(f"float32 file not found: {float32_path}")
    if not int8_path.exists():
        raise FileNotFoundError(f"int8 file not found: {int8_path}")

    print(f"Loading float32 activations: {float32_path}")
    f32 = np.load(float32_path)
    print(f"Loading int8 dequantized activations: {int8_path}")
    i8 = np.load(int8_path)

    keys_f32 = set(f32.files)
    keys_i8 = set(i8.files)
    shared_keys = sorted(keys_f32 & keys_i8)
    only_f32 = keys_f32 - keys_i8
    only_i8 = keys_i8 - keys_f32

    if only_f32:
        print(f"  WARNING: tensors only in float32 file (skipped): {only_f32}")
    if only_i8:
        print(f"  WARNING: tensors only in int8 file (skipped): {only_i8}")
    if not shared_keys:
        raise RuntimeError("No matching tensor keys found between the two .npz files.")

    print(f"\nComputing metrics for {len(shared_keys)} boundary tensor(s):")
    per_tensor: dict[str, dict] = {}
    for key in shared_keys:
        a = f32[key]
        b = i8[key]
        print(f"  [{key}]  shape={a.shape}  dtype={a.dtype}")
        metrics = compute_tensor_metrics(a, b, num_bins=args.bins)
        per_tensor[key] = metrics
        print(f"    MSE={metrics['boundary_mse']['mean']:.6f}")
        print(f"    PSNR={metrics['boundary_psnr_db']['mean']:.2f} dB")
        print(f"    KL={metrics['boundary_kl_divergence']['mean']:.6f}")
        print(f"    CosSim={metrics['boundary_cosine_similarity']['mean']:.6f}")
        print(f"    RangeRatio={metrics['activation_range_ratio']['mean']:.6f}")

    record = build_results_record(
        task_name=args.task,
        float32_path=str(float32_path.resolve()),
        int8_path=str(int8_path.resolve()),
        per_tensor=per_tensor,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fh:
        json.dump(record, fh, indent=2)

    print(f"\nResults record saved: {output_path}")
    print("\n=== Summary ===")
    for k, v in record["summary"].items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
