from __future__ import annotations

import argparse
from pathlib import Path

from src.config_utils import load_config
from src.data_loaders import load_samples
from src.hybrid_runner import HybridRunner, compare_named_outputs, summarize_timing
from src.io_utils import save_json, save_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run hybrid Edge TPU prefix plus TensorFlow CPU suffix execution.")
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--compiled-tflite",
        default=None,
        help="Path to the compiled Edge TPU TFLite prefix. Defaults to <artifacts_dir>/output_edgetpu.tflite.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    parser.add_argument(
        "--boundary-tensors",
        nargs="+",
        default=None,
        help="Override boundary tensor names for this run.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Run only the CPU split comparison path without loading PyCoral or a compiled TPU model.",
    )
    return parser.parse_args()


def _default_compiled_tflite_path(config: dict) -> str:
    return str(Path(config["artifacts_dir"]) / "output_edgetpu.tflite")


def _save_cpu_comparison(config: dict, runner: HybridRunner, samples: list[dict]) -> tuple[dict, dict, dict]:
    full_outputs, full_timings_ms = runner.run_full_cpu(samples)
    partitioned_outputs, partitioned_timings_ms, float_boundaries = runner.run_partitioned_cpu(samples)
    comparison = compare_named_outputs(full_outputs, partitioned_outputs)
    save_npz(f"{config['artifacts_dir']}/hybrid_float_boundary_outputs.npz", float_boundaries)
    return (
        full_outputs,
        partitioned_outputs,
        {
            "full_cpu_latency_ms": {
                "per_frame": full_timings_ms,
                "mean": sum(full_timings_ms) / len(full_timings_ms),
            },
            "partitioned_cpu_latency_ms": {
                "prefix_per_frame": partitioned_timings_ms["prefix_ms"],
                "suffix_per_frame": partitioned_timings_ms["suffix_ms"],
                "prefix_mean": sum(partitioned_timings_ms["prefix_ms"]) / len(partitioned_timings_ms["prefix_ms"]),
                "suffix_mean": sum(partitioned_timings_ms["suffix_ms"]) / len(partitioned_timings_ms["suffix_ms"]),
            },
            "cpu_split_comparison": comparison,
            "float_boundary_outputs": float_boundaries,
        },
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples = load_samples(config, frame_limit=args.frame_limit)
    runner = HybridRunner(config, boundary_tensors=args.boundary_tensors)

    full_outputs, partitioned_outputs, cpu_record = _save_cpu_comparison(config, runner, samples)
    float_boundaries = cpu_record.pop("float_boundary_outputs")

    if args.cpu_only:
        summary = {
            "task_name": config["task_name"],
            "mode": "cpu_only",
            "boundary_tensors": runner.boundary_tensors,
            "num_frames": len(samples),
            "sample_ids": [sample["sample_id"] for sample in samples],
            **cpu_record,
            "boundary_shapes": {key: list(value.shape) for key, value in float_boundaries.items()},
        }
        save_json(f"{config['artifacts_dir']}/hybrid_tpu_summary.json", summary)
        print(f"Validated {len(samples)} frames in CPU-only hybrid mode.")
        print(f"CPU split max abs diff: {summary['cpu_split_comparison']['max_abs_diff']}")
        print(f"Saved summary to: {config['artifacts_dir']}/hybrid_tpu_summary.json")
        return

    compiled_tflite_path = args.compiled_tflite or _default_compiled_tflite_path(config)
    tpu_result = runner.run_hybrid_tpu(samples, compiled_tflite_path=compiled_tflite_path)
    tpu_outputs = tpu_result["outputs"]
    tpu_boundaries = tpu_result["boundary_outputs"]

    output_drift = compare_named_outputs(partitioned_outputs, tpu_outputs)
    boundary_drift = compare_named_outputs(float_boundaries, tpu_boundaries)

    save_npz(f"{config['artifacts_dir']}/hybrid_tpu_outputs.npz", tpu_outputs)
    save_npz(f"{config['artifacts_dir']}/hybrid_tpu_boundary_dequantized.npz", tpu_boundaries)
    save_npz(f"{config['artifacts_dir']}/hybrid_float_boundary_outputs.npz", float_boundaries)

    summary = {
        "task_name": config["task_name"],
        "mode": "tpu",
        "compiled_tflite_path": compiled_tflite_path,
        "boundary_tensors": runner.boundary_tensors,
        "num_frames": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        **cpu_record,
        "tpu_output_drift_vs_partitioned_cpu": output_drift,
        "boundary_drift_tpu_dequantized_vs_float": boundary_drift,
        "hybrid_tpu_timing_ms": summarize_timing(tpu_result["timings_ms"]),
        "tflite_output_metadata": tpu_result["tflite_output_metadata"],
        "output_shapes": {key: list(value.shape) for key, value in tpu_outputs.items()},
        "boundary_shapes": {key: list(value.shape) for key, value in tpu_boundaries.items()},
    }
    save_json(f"{config['artifacts_dir']}/hybrid_tpu_summary.json", summary)

    print(f"Ran {len(samples)} samples through hybrid TPU execution.")
    print(f"CPU split max abs diff: {summary['cpu_split_comparison']['max_abs_diff']}")
    print(f"TPU output max abs drift: {output_drift['max_abs_diff']}")
    print(f"Boundary max abs drift: {boundary_drift['max_abs_diff']}")
    print(f"Saved summary to: {config['artifacts_dir']}/hybrid_tpu_summary.json")


if __name__ == "__main__":
    main()
