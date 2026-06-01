from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a strict candidate-owned Edge TPU prefix plus TensorFlow CPU suffix execution."
    )
    parser.add_argument(
        "--candidate-dir",
        required=True,
        help="Candidate artifact directory containing metadata.json and candidate-owned runtime artifacts.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only the float prefix plus candidate CPU SavedModel suffix without loading PyCoral.",
    )
    return parser.parse_args()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _save_cpu_comparison(runner, samples: list[dict], compare_named_outputs) -> tuple[dict, dict, dict]:
    full_outputs, full_timings_ms = runner.run_full_cpu(samples)
    partitioned_outputs, partitioned_timings_ms, float_boundaries = runner.run_partitioned_cpu(samples)
    comparison = compare_named_outputs(full_outputs, partitioned_outputs)
    return (
        full_outputs,
        partitioned_outputs,
        {
            "full_cpu_latency_ms": {
                "per_frame": full_timings_ms,
                "mean": _mean(full_timings_ms),
            },
            "partitioned_cpu_latency_ms": {
                "prefix_per_frame": partitioned_timings_ms["prefix_ms"],
                "suffix_per_frame": partitioned_timings_ms["suffix_ms"],
                "prefix_mean": _mean(partitioned_timings_ms["prefix_ms"]),
                "suffix_mean": _mean(partitioned_timings_ms["suffix_ms"]),
            },
            "cpu_split_comparison": comparison,
            "float_boundary_outputs": float_boundaries,
        },
    )


def main() -> None:
    args = parse_args()

    from src.hybrid_runner import HybridRunner, compare_named_outputs, summarize_timing

    runner = HybridRunner(args.candidate_dir, require_tpu=not args.cpu_only)
    from src.data_loaders import load_samples
    from src.io_utils import save_json, save_npz

    samples = runner.validate_samples(load_samples(runner.config, frame_limit=args.frame_limit))

    _, partitioned_outputs, cpu_record = _save_cpu_comparison(runner, samples, compare_named_outputs)
    float_boundaries = cpu_record.pop("float_boundary_outputs")
    float_boundary_output_path = runner.results_path("boundary_float_reference.npz")
    save_npz(float_boundary_output_path, float_boundaries)

    if args.cpu_only:
        summary = {
            "mode": "cpu_only",
            "candidate_dir": str(runner.candidate_dir),
            "metadata_path": str(runner.metadata_path),
            "boundary_tensors": runner.boundary_tensors,
            "num_frames": len(samples),
            "sample_ids": [sample["sample_id"] for sample in samples],
            **runner.candidate_summary_fields(),
            **cpu_record,
            "boundary_shapes": {key: list(value.shape) for key, value in float_boundaries.items()},
            "output_artifacts": {
                "float_boundary_outputs": str(float_boundary_output_path),
            },
        }
        summary_path = runner.results_path("summary.json")
        save_json(summary_path, summary)
        print(f"Validated {len(samples)} frames in candidate CPU-only hybrid mode.")
        print(f"CPU split max abs diff: {summary['cpu_split_comparison']['max_abs_diff']}")
        print(f"Saved summary to: {summary_path}")
        return

    tpu_result = runner.run_hybrid_tpu(samples)
    tpu_outputs = tpu_result["outputs"]
    tpu_boundaries = tpu_result["boundary_outputs"]

    output_drift = compare_named_outputs(partitioned_outputs, tpu_outputs)
    boundary_drift = compare_named_outputs(float_boundaries, tpu_boundaries)

    tpu_output_path = runner.results_path("hybrid_outputs.npz")
    tpu_boundary_path = runner.results_path("boundary_tpu_dequantized.npz")
    save_npz(tpu_output_path, tpu_outputs)
    save_npz(tpu_boundary_path, tpu_boundaries)

    summary = {
        "mode": "tpu",
        "candidate_dir": str(runner.candidate_dir),
        "metadata_path": str(runner.metadata_path),
        "compiled_tflite_path": runner.config["tpu_edgetpu_path"],
        "boundary_tensors": runner.boundary_tensors,
        "num_frames": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        **runner.candidate_summary_fields(),
        **cpu_record,
        "tpu_output_drift_vs_partitioned_cpu": output_drift,
        "boundary_drift_tpu_dequantized_vs_float": boundary_drift,
        "hybrid_tpu_timing_ms": summarize_timing(tpu_result["timings_ms"]),
        "tflite_output_metadata": tpu_result["tflite_output_metadata"],
        "output_shapes": {key: list(value.shape) for key, value in tpu_outputs.items()},
        "boundary_shapes": {key: list(value.shape) for key, value in tpu_boundaries.items()},
        "output_artifacts": {
            "tpu_outputs": str(tpu_output_path),
            "tpu_boundary_dequantized": str(tpu_boundary_path),
            "float_boundary_outputs": str(float_boundary_output_path),
        },
    }
    summary_path = runner.results_path("summary.json")
    save_json(summary_path, summary)

    print(f"Ran {len(samples)} samples through candidate hybrid TPU execution.")
    print(f"CPU split max abs diff: {summary['cpu_split_comparison']['max_abs_diff']}")
    print(f"TPU output max abs drift: {output_drift['max_abs_diff']}")
    print(f"Boundary max abs drift: {boundary_drift['max_abs_diff']}")
    print(f"Saved summary to: {summary_path}")


if __name__ == "__main__":
    main()
