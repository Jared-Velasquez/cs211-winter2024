from __future__ import annotations

import argparse

from src.config_utils import load_config
from src.data_loaders import load_samples
from src.hybrid_runner import HybridRunner, compare_named_outputs
from src.io_utils import save_json, save_npz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a split graph in CPU-only mode.")
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of frames to evaluate.",
    )
    parser.add_argument(
        "--boundary-tensors",
        nargs="+",
        default=None,
        help="Override boundary tensor names for this validation run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples = load_samples(config, frame_limit=args.frame_limit)
    runner = HybridRunner(config, boundary_tensors=args.boundary_tensors)

    full_outputs, full_timings_ms = runner.run_full_cpu(samples)
    partitioned_outputs, partitioned_timings_ms, boundary_outputs = runner.run_partitioned_cpu(samples)
    comparison = compare_named_outputs(full_outputs, partitioned_outputs)

    base_dir = config["artifacts_dir"]
    save_npz(f"{base_dir}/partitioned_cpu_outputs.npz", partitioned_outputs)
    save_npz(f"{base_dir}/boundary_outputs.npz", boundary_outputs)
    save_json(
        f"{base_dir}/partitioned_cpu_summary.json",
        {
            "task_name": config["task_name"],
            "boundary_tensors": runner.boundary_tensors,
            "num_frames": len(samples),
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
            "comparison": comparison,
            "boundary_shapes": {key: list(value.shape) for key, value in boundary_outputs.items()},
        },
    )

    print(f"Validated {len(samples)} frames with partitioned CPU execution.")
    print(f"Max abs diff: {comparison['max_abs_diff']}")
    print(f"Saved summary to: {base_dir}/partitioned_cpu_summary.json")


if __name__ == "__main__":
    main()
