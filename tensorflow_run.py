from __future__ import annotations

import argparse
import time

import tensorflow as tf

from src.config_utils import load_config
from src.data_loaders import load_samples
from src.graph_utils import import_graph, load_graph_def
from src.io_utils import ensure_directory, save_json, save_npz, stack_named_outputs, tensor_name_to_key

tf1 = tf.compat.v1


def summarize_output_shapes(stacked_outputs: dict) -> dict:
    summary = {}
    for key, value in stacked_outputs.items():
        if getattr(value, "dtype", None) == object:
            summary[key] = [list(item.shape) for item in value]
        else:
            summary[key] = list(value.shape)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full float32 TensorFlow graph on CPU.")
    parser.add_argument(
        "--config",
        default="configs/task_a_dlc.json",
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    return parser.parse_args()


def run_full_graph(config: dict, frame_limit: int | None = None) -> dict:
    samples = load_samples(config, frame_limit=frame_limit)
    if not samples:
        raise RuntimeError("No input samples were loaded.")

    graph = import_graph(load_graph_def(config["model_path"]))
    input_tensor = graph.get_tensor_by_name(config["input_tensor"])
    output_tensor_names = list(config["output_tensors"])
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

    stacked_outputs = stack_named_outputs(per_frame_outputs)
    ensure_directory(config["artifacts_dir"])
    save_npz(config["baseline_output_path"], stacked_outputs)

    summary = {
        "task_name": config["task_name"],
        "task_type": config["task_type"],
        "model_path": config["model_path"],
        "data_source": config.get("video_path") or config.get("images_dir"),
        "num_frames": len(samples),
        "sample_ids": [sample["sample_id"] for sample in samples],
        "output_tensors": output_tensor_names,
        "output_shapes": summarize_output_shapes(stacked_outputs),
        "latency_ms": {
            "per_frame": timings_ms,
            "mean": sum(timings_ms) / len(timings_ms),
            "max": max(timings_ms),
            "min": min(timings_ms),
        },
    }
    save_json(config["baseline_summary_path"], summary)
    return {
        "summary": summary,
        "samples": samples,
        "outputs": stacked_outputs,
    }


def run_full_graph_summary(config: dict, frame_limit: int | None = None) -> dict:
    result = run_full_graph(config, frame_limit=frame_limit)
    return result["summary"]


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    summary = run_full_graph_summary(config, frame_limit=args.frame_limit)
    print(f"Processed {summary['num_frames']} frames.")
    print(f"Mean latency: {summary['latency_ms']['mean']:.2f} ms/frame")
    print(f"Saved outputs to: {config['baseline_output_path']}")


if __name__ == "__main__":
    main()
