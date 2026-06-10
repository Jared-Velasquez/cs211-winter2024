from __future__ import annotations

import argparse

from src.config_utils import load_config
from src.evaluation import evaluate_outputs
from src.io_utils import save_json
from tensorflow_run import run_full_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a float32 baseline and compute task-specific metrics.")
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the task config JSON.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    result = run_full_graph(config, frame_limit=args.frame_limit)

    compute_accuracy = bool(config.get("compute_accuracy", True))
    metrics = None
    if compute_accuracy:
        metrics = evaluate_outputs(config, result["samples"], result["outputs"])

    payload = {
        "task_name": config["task_name"],
        "task_type": config["task_type"],
        "model_path": config["model_path"],
        "data_source": config.get("video_path") or config.get("images_dir"),
        "baseline_mode": "fidelity_and_accuracy" if compute_accuracy else "fidelity_only",
        "inference_summary": result["summary"],
    }
    if compute_accuracy:
        payload["metrics"] = metrics
    if not compute_accuracy:
        payload["notes"] = [
            "This baseline is fidelity-only.",
            "The saved float32 predictions are the reference outputs for later drift comparisons.",
            "Task-specific labeled accuracy metrics were intentionally skipped for this task.",
        ]
    save_json(config["baseline_results_path"], payload)

    print(f"Saved baseline outputs to: {config['baseline_output_path']}")
    print(f"Saved baseline summary to: {config['baseline_summary_path']}")
    if compute_accuracy:
        print(f"Saved baseline metrics to: {config['baseline_results_path']}")
    else:
        print(f"Saved fidelity baseline record to: {config['baseline_results_path']}")


if __name__ == "__main__":
    main()
