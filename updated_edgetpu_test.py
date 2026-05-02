from __future__ import annotations

import argparse
import time

import tensorflow as tf

from src.config_utils import get_boundary_tensors, load_config
from src.data_loaders import load_samples
from src.graph_utils import extract_prefix_graph_def, import_graph, load_graph_def
from src.io_utils import save_json, save_npz, stack_named_outputs, tensor_name_to_key

tf1 = tf.compat.v1


class HybridRunner:
    def __init__(self, config: dict, boundary_tensors: list[str] | None = None) -> None:
        self.config = config
        self.boundary_tensors = get_boundary_tensors(config, override=boundary_tensors)

    def run_full_cpu(self, samples: list[dict]) -> tuple[dict[str, tf.Tensor], list[float]]:
        graph = import_graph(load_graph_def(self.config["model_path"]))
        input_tensor = graph.get_tensor_by_name(self.config["input_tensor"])
        output_tensor_names = list(self.config["output_tensors"])
        output_tensors = [graph.get_tensor_by_name(name) for name in output_tensor_names]

        per_frame_outputs = []
        timings_ms = []
        with tf1.Session(graph=graph) as session:
            for sample in samples:
                start_time = time.time()
                values = session.run(output_tensors, feed_dict={input_tensor: sample["input"]})
                timings_ms.append((time.time() - start_time) * 1000.0)
                per_frame_outputs.append(
                    {
                        tensor_name_to_key(name): value
                        for name, value in zip(output_tensor_names, values)
                    }
                )
        return stack_named_outputs(per_frame_outputs), timings_ms

    def run_partitioned_cpu(self, samples: list[dict]) -> tuple[dict[str, tf.Tensor], dict[str, list[float]], dict[str, tf.Tensor]]:
        prefix_graph = import_graph(
            extract_prefix_graph_def(
                graph_path=self.config["model_path"],
                boundary_tensors=self.boundary_tensors,
                input_tensor_name=self.config["input_tensor"],
                fixed_input_shape=self.config.get("fixed_input_shape"),
            )
        )
        full_graph = import_graph(load_graph_def(self.config["model_path"]))

        prefix_input_tensor = prefix_graph.get_tensor_by_name(self.config["input_tensor"])
        boundary_output_tensors = [prefix_graph.get_tensor_by_name(name) for name in self.boundary_tensors]
        full_output_tensor_names = list(self.config["output_tensors"])
        full_output_tensors = [full_graph.get_tensor_by_name(name) for name in full_output_tensor_names]
        full_boundary_inputs = [full_graph.get_tensor_by_name(name) for name in self.boundary_tensors]

        per_frame_outputs = []
        per_frame_boundaries = []
        prefix_timings_ms = []
        suffix_timings_ms = []

        with tf1.Session(graph=prefix_graph) as prefix_session, tf1.Session(graph=full_graph) as suffix_session:
            for sample in samples:
                prefix_start = time.time()
                boundary_values = prefix_session.run(boundary_output_tensors, feed_dict={prefix_input_tensor: sample["input"]})
                prefix_timings_ms.append((time.time() - prefix_start) * 1000.0)

                suffix_start = time.time()
                feed_dict = {
                    boundary_tensor: value
                    for boundary_tensor, value in zip(full_boundary_inputs, boundary_values)
                }
                output_values = suffix_session.run(full_output_tensors, feed_dict=feed_dict)
                suffix_timings_ms.append((time.time() - suffix_start) * 1000.0)

                per_frame_outputs.append(
                    {
                        tensor_name_to_key(name): value
                        for name, value in zip(full_output_tensor_names, output_values)
                    }
                )
                per_frame_boundaries.append(
                    {
                        tensor_name_to_key(name): value
                        for name, value in zip(self.boundary_tensors, boundary_values)
                    }
                )

        return (
            stack_named_outputs(per_frame_outputs),
            {"prefix_ms": prefix_timings_ms, "suffix_ms": suffix_timings_ms},
            stack_named_outputs(per_frame_boundaries),
        )


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


def _compare_outputs(full_outputs: dict, partitioned_outputs: dict) -> dict:
    diffs = {}
    max_abs_diff = 0.0
    mean_abs_diff = 0.0
    for key, full_value in full_outputs.items():
        diff = abs(full_value - partitioned_outputs[key])
        diffs[key] = {
            "shape": list(full_value.shape),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
        }
        max_abs_diff = max(max_abs_diff, diffs[key]["max_abs_diff"])
        mean_abs_diff += diffs[key]["mean_abs_diff"]

    if diffs:
        mean_abs_diff /= len(diffs)

    return {
        "per_output": diffs,
        "max_abs_diff": max_abs_diff,
        "mean_abs_diff": mean_abs_diff,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    samples = load_samples(config, frame_limit=args.frame_limit)
    runner = HybridRunner(config, boundary_tensors=args.boundary_tensors)

    full_outputs, full_timings_ms = runner.run_full_cpu(samples)
    partitioned_outputs, partitioned_timings_ms, boundary_outputs = runner.run_partitioned_cpu(samples)
    comparison = _compare_outputs(full_outputs, partitioned_outputs)

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
