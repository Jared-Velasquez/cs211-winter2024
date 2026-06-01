"""benchmark.py — Multi-candidate hardware benchmarking sweep for Student C.

Runs every partition candidate for a given model family through the HybridRunner,
collecting per-candidate latency breakdowns, model outputs, and boundary tensors.

Two-phase execution per candidate:
  1. CPU sanity pass  — verifies that the partitioned CPU path matches the full
                        float32 model. Saves float32 boundary reference tensors.
  2. TPU timing pass  — runs the compiled Edge TPU prefix + CPU suffix for
                        --iterations total inferences, cycling through loaded
                        samples. Saves outputs and dequantized boundary tensors.

Outputs per candidate (artifacts/<id>/results/):
  benchmark_summary.json       timing stats + drift metrics + candidate metadata
  outputs.npz                  stacked model outputs (TPU pass, all iterations)
  boundary_tpu_dequantized.npz dequantized int8 boundary tensors (Student A input)
  boundary_float_reference.npz float32 boundary tensors from CPU prefix (Student A input)

Usage:
  # DLC — CPU sanity check only (fast, ~5 min total):
  python3 benchmark.py --model dlc --mode cpu-only --frame-limit 20

  # DLC — full sweep (CPU verify then 100-iteration TPU timing):
  python3 benchmark.py --model dlc --mode cpu-then-tpu --iterations 100 --frame-limit 50

  # Single candidate:
  python3 benchmark.py --candidate-dir artifacts/dlc_split_after_block1 --mode cpu-then-tpu

  # SSD / DeepLab (Weeks 7-8):
  python3 benchmark.py --model ssd --mode cpu-then-tpu --iterations 100 --frame-limit 50
"""
from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent

# Ordered by partition depth (shallow → deep) so results read naturally.
MODEL_CANDIDATES: dict[str, list[str]] = {
    "dlc": [
        "dlc_split_after_block1",
        "dlc_split_after_block2",
        "dlc_split_after_block3",
        "dlc_split_in_block4_unit1",
        "dlc_split_after_block4",
        "dlc_split_at_conv2d_transpose",
        "dlc_split_at_bias_add",
    ],
    "ssd": [
        "ssd_split_after_expanded_conv_5",
        "ssd_split_after_expanded_conv_10",
        "ssd_split_after_expanded_conv_13",
        "ssd_split_after_conv_1",
        "ssd_split_at_box_predictor_biasadds",
        "ssd_split_before_postprocessor",
    ],
    "deeplab": [
        "deeplab_split_after_expanded_conv_5",
        "deeplab_split_after_expanded_conv_10",
        "deeplab_split_after_expanded_conv_13",
        # The following 4 failed edgetpu_compiler (large activation tensors).
        # Run with --mode cpu-only to collect CPU-only latency baselines.
        "deeplab_split_after_expanded_conv_16",
        "deeplab_split_after_aspp",
        "deeplab_split_after_logits",
        "deeplab_split_after_resize",
    ],
}

# Warn if CPU split max absolute drift exceeds this.
CPU_DRIFT_WARN_THRESHOLD = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument(
        "--model",
        choices=["dlc", "ssd", "deeplab", "all"],
        help="Sweep all candidates for this model family.",
    )
    target.add_argument(
        "--candidate-dir",
        help="Path to a single candidate artifact directory.",
    )

    parser.add_argument(
        "--mode",
        choices=["cpu-only", "tpu-only", "cpu-then-tpu"],
        default="cpu-then-tpu",
        help=(
            "cpu-only: CPU sanity pass only (no Coral hardware required). "
            "tpu-only: TPU timing only (skips CPU verification). "
            "cpu-then-tpu: CPU verify first, then TPU timing (default)."
        ),
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
        help="TPU timing iterations per candidate. Samples are cycled if fewer are loaded. Default: 100.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Max unique samples to load per candidate. None = load all available.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for sweep-level summary (default: logs/benchmark/).",
    )
    parser.add_argument(
        "--skip-on-drift",
        action="store_true",
        help="Skip TPU pass for candidates whose CPU drift exceeds the warning threshold.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would run without executing.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    sv = sorted(values)
    idx = (p / 100.0) * (len(sv) - 1)
    lo, hi = int(idx), min(int(idx) + 1, len(sv) - 1)
    return sv[lo] + (idx - lo) * (sv[hi] - sv[lo])


def _timing_stats(timings: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    if not timings:
        return {}
    keys = list(timings[0].keys())
    stats: dict[str, dict[str, float]] = {}
    for key in keys:
        values = [t[key] for t in timings]
        stats[key] = {
            "mean_ms": float(np.mean(values)),
            "std_ms": float(np.std(values)),
            "p95_ms": _percentile(values, 95),
            "min_ms": float(np.min(values)),
            "max_ms": float(np.max(values)),
        }
    return stats


def _cycle_samples(samples: list[dict], n: int) -> list[dict]:
    """Return a list of exactly n samples by cycling through the available ones."""
    if not samples:
        return []
    return [samples[i % len(samples)] for i in range(n)]


# ---------------------------------------------------------------------------
# Per-candidate benchmark
# ---------------------------------------------------------------------------

def _run_cpu_sanity(runner, samples: list[dict]) -> dict[str, Any]:
    """Run full-CPU and partitioned-CPU passes; return drift metrics and boundary tensors."""
    from src.hybrid_runner import compare_named_outputs

    full_outputs, full_timings_ms = runner.run_full_cpu(samples)
    partitioned_outputs, partitioned_timings_ms, float_boundaries = runner.run_partitioned_cpu(samples)

    # run_full_cpu keys outputs via _tensor_name_to_key (e.g. "concat_1_0").
    # run_suffix_from_boundaries keys outputs via the SavedModel signature (e.g. "output").
    # Align by position — both iterate the same output_tensors list in order.
    if len(full_outputs) != len(partitioned_outputs):
        raise ValueError(
            f"Output count mismatch: full_cpu={len(full_outputs)}, partitioned={len(partitioned_outputs)}"
        )
    full_aligned = dict(zip(partitioned_outputs.keys(), full_outputs.values()))
    drift = compare_named_outputs(full_aligned, partitioned_outputs)

    return {
        "full_cpu_mean_ms": float(np.mean(full_timings_ms)),
        "prefix_mean_ms": float(np.mean(partitioned_timings_ms["prefix_ms"])),
        "suffix_mean_ms": float(np.mean(partitioned_timings_ms["suffix_ms"])),
        "max_abs_drift": float(drift["max_abs_diff"]),
        "mean_abs_drift": float(drift["mean_abs_diff"]),
        "drift_per_output": drift["per_output"],
        "drift_ok": bool(drift["max_abs_diff"] <= CPU_DRIFT_WARN_THRESHOLD),
        "_float_boundaries": float_boundaries,
    }


def run_candidate(
    candidate_dir: Path,
    mode: str,
    iterations: int,
    frame_limit: int | None,
    skip_on_drift: bool,
    dry_run: bool,
) -> dict[str, Any]:
    """Benchmark a single candidate. Returns a result dict for the sweep summary."""
    from src.data_loaders import load_samples
    from src.hybrid_runner import HybridRunner
    from src.io_utils import save_json, save_npz

    result: dict[str, Any] = {
        "candidate_dir": str(candidate_dir),
        "mode": mode,
        "iterations_requested": iterations,
        "frame_limit": frame_limit,
        "status": "pending",
    }

    if dry_run:
        result["status"] = "dry_run"
        print(f"  [DRY-RUN] {candidate_dir.name}  mode={mode}  iterations={iterations}")
        return result

    results_dir = candidate_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    require_tpu = mode in ("tpu-only", "cpu-then-tpu")
    try:
        runner = HybridRunner(str(candidate_dir), require_tpu=require_tpu)
    except Exception as exc:
        result["status"] = "failed_load"
        result["error"] = str(exc)
        print(f"  ERROR loading runner: {exc}")
        return result

    result.update(runner.candidate_summary_fields())

    try:
        samples = runner.validate_samples(load_samples(runner.config, frame_limit=frame_limit))
    except Exception as exc:
        result["status"] = "failed_data_load"
        result["error"] = str(exc)
        print(f"  ERROR loading data: {exc}")
        return result

    result["num_samples_loaded"] = len(samples)
    result["artifacts"] = {}
    cpu_drift_ok = True

    # --- CPU sanity pass ---
    if mode in ("cpu-only", "cpu-then-tpu"):
        print(f"    CPU sanity  ({len(samples)} samples)...", end="", flush=True)
        try:
            cpu = _run_cpu_sanity(runner, samples)
            float_boundaries = cpu.pop("_float_boundaries")
            result["cpu_sanity"] = cpu

            drift_flag = "" if cpu["drift_ok"] else f"  *** DRIFT {cpu['max_abs_drift']:.4f} > {CPU_DRIFT_WARN_THRESHOLD}"
            print(f"  max_drift={cpu['max_abs_drift']:.4f}{drift_flag}")
            if not cpu["drift_ok"]:
                print(f"    WARNING: CPU split drift exceeds threshold {CPU_DRIFT_WARN_THRESHOLD}")
                cpu_drift_ok = False

            ref_path = results_dir / "boundary_float_reference.npz"
            save_npz(ref_path, float_boundaries)
            result["artifacts"]["boundary_float_reference"] = str(ref_path)

        except Exception as exc:
            result["status"] = "failed_cpu_sanity"
            result["error"] = str(exc)
            print(f"\n  ERROR: {exc}")
            save_json(results_dir / "benchmark_summary.json", result)
            return result

    if mode == "cpu-only":
        result["status"] = "completed_cpu_only"
        save_json(results_dir / "benchmark_summary.json", result)
        return result

    # Free TF graph memory before loading PyCoral — running TF1 sessions + TF2 SavedModel
    # + PyCoral simultaneously OOMs on Raspberry Pi. The CPU sanity local vars (graph
    # sessions, full_outputs, prefix outputs) are out of scope here; force collection.
    gc.collect()

    # --- TPU timing pass ---
    if skip_on_drift and not cpu_drift_ok:
        result["status"] = "skipped_tpu_high_drift"
        print(f"    Skipping TPU pass (--skip-on-drift, drift={result['cpu_sanity']['max_abs_drift']:.4f})")
        save_json(results_dir / "benchmark_summary.json", result)
        return result

    iteration_samples = _cycle_samples(samples, iterations)
    print(f"    TPU timing  ({iterations} iterations over {len(samples)} unique samples)...", end="", flush=True)
    try:
        tpu_result = runner.run_hybrid_tpu(iteration_samples)
        timing_stats = _timing_stats(tpu_result["timings_ms"])
        result["tpu_timing"] = {
            "iterations_run": iterations,
            "stats": timing_stats,
        }
        print("")
        for key, stats in timing_stats.items():
            print(f"      {key}: mean={stats['mean_ms']:.1f}ms  std={stats['std_ms']:.1f}ms  p95={stats['p95_ms']:.1f}ms")

        out_path = results_dir / "outputs.npz"
        bnd_path = results_dir / "boundary_tpu_dequantized.npz"
        save_npz(out_path, tpu_result["outputs"])
        save_npz(bnd_path, tpu_result["boundary_outputs"])
        result["artifacts"]["tpu_outputs"] = str(out_path)
        result["artifacts"]["boundary_tpu_dequantized"] = str(bnd_path)
        result["status"] = "completed_tpu"

    except Exception as exc:
        result["status"] = "failed_tpu"
        result["error"] = str(exc)
        print(f"\n  ERROR: {exc}")

    save_json(results_dir / "benchmark_summary.json", result)
    return result


# ---------------------------------------------------------------------------
# Sweep driver
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    artifacts_dir = REPO_ROOT / "artifacts"
    output_dir = Path(args.output_dir) if args.output_dir else REPO_ROOT / "logs" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.candidate_dir:
        candidate_dirs = [Path(args.candidate_dir).resolve()]
        sweep_name = candidate_dirs[0].name
    else:
        models = list(MODEL_CANDIDATES.keys()) if args.model == "all" else [args.model]
        sweep_name = args.model
        candidate_dirs = []
        for model in models:
            for name in MODEL_CANDIDATES[model]:
                d = artifacts_dir / name
                if d.exists():
                    candidate_dirs.append(d)
                else:
                    print(f"WARNING: candidate directory not found: {d}")

    print(
        f"Benchmark sweep: {len(candidate_dirs)} candidate(s), "
        f"mode={args.mode}, iterations={args.iterations}"
        + (f", frame-limit={args.frame_limit}" if args.frame_limit else "")
    )

    sweep_results = []
    for i, candidate_dir in enumerate(candidate_dirs):
        print(f"\n[{i + 1}/{len(candidate_dirs)}] {candidate_dir.name}")
        result = run_candidate(
            candidate_dir=candidate_dir,
            mode=args.mode,
            iterations=args.iterations,
            frame_limit=args.frame_limit,
            skip_on_drift=args.skip_on_drift,
            dry_run=args.dry_run,
        )
        sweep_results.append(result)

    # Write sweep-level summary
    from src.io_utils import save_json

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    num_completed = sum(1 for r in sweep_results if "failed" not in r.get("status", "failed"))
    sweep_summary = {
        "sweep_name": sweep_name,
        "mode": args.mode,
        "iterations": args.iterations,
        "frame_limit": args.frame_limit,
        "timestamp": timestamp,
        "num_candidates": len(candidate_dirs),
        "num_completed": num_completed,
        "candidates": sweep_results,
    }
    summary_path = output_dir / f"{sweep_name}_sweep_{timestamp}.json"
    save_json(summary_path, sweep_summary)

    print(f"\n{'=' * 60}")
    print(f"Sweep complete: {num_completed}/{len(candidate_dirs)} succeeded")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
