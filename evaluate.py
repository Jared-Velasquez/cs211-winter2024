"""Student D — Evaluate saved predictions against ground truth.

This script is the disk-driven entry point for the evaluation step. Student C
writes per-frame predictions to a ``.npz`` file (one tensor per output tensor
name, batched along axis 0). This script:

    1. Loads the dataset samples for the task config (so it has ground truth
       labels in the same order Student C ran them).
    2. Loads the saved predictions and runs the task-specific metric in
       ``src.evaluation``.
    3. Optionally diffs against the float32 baseline ``baseline_results.json``
       to compute the per-metric and headline accuracy drop.
    4. Writes one JSON Results Record per evaluation, plus prints the headline
       metric.

Typical usage::

    ./run_in_env.sh python evaluate.py \
        --config configs/task_b_detection.json \
        --predictions artifacts/task_b/detection/hybrid_tpu_outputs.npz \
        --partition-id split_after_block11 \
        --output artifacts/task_b/detection/eval/split_after_block11.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.config_utils import load_config
from src.data_loaders import load_samples
from src.evaluation import evaluate_outputs
from src.io_utils import save_json
from src.results_records import build_results_record, flatten_accuracy, headline_metric_value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run task-specific accuracy metrics on saved predictions."
    )
    parser.add_argument("--config", required=True, help="Path to the task config JSON.")
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to a .npz file containing the predicted output tensors.",
    )
    parser.add_argument(
        "--partition-id",
        default="full_float_baseline",
        help="Identifier of the partition that produced these predictions.",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Restrict evaluation to the first N samples.",
    )
    parser.add_argument(
        "--baseline-results",
        default=None,
        help="Optional path to baseline_results.json. Defaults to the path in the config.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to save the Results Record JSON. Defaults to <artifacts_dir>/eval/<partition_id>.json.",
    )
    parser.add_argument(
        "--latency-json",
        default=None,
        help=(
            "Optional path to a JSON file containing a 'latency_ms' object "
            "(keys 'tpu', 'transfer', 'cpu', 'total')."
        ),
    )
    parser.add_argument(
        "--proxy-json",
        default=None,
        help="Optional path to a JSON file containing this candidate's proxy metrics.",
    )
    return parser.parse_args()


def _load_predictions(path: str) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return {key: archive[key] for key in archive.files}


def _load_baseline_accuracy(config: dict[str, Any], explicit_path: str | None) -> dict[str, float] | None:
    candidate_path = explicit_path or config.get("baseline_results_path")
    if not candidate_path:
        return None
    path = Path(candidate_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("metrics")
    if not raw:
        return None
    return flatten_accuracy(config["task_type"], raw)


def _truncate_outputs(outputs: dict[str, np.ndarray], frame_limit: int | None) -> dict[str, np.ndarray]:
    if frame_limit is None:
        return outputs
    truncated: dict[str, np.ndarray] = {}
    for key, value in outputs.items():
        truncated[key] = value[:frame_limit]
    return truncated


def _validate_alignment(samples: list[dict[str, Any]], outputs: dict[str, np.ndarray]) -> None:
    if not outputs:
        raise ValueError("No predictions were loaded from the npz file.")
    sample_count = len(samples)
    for key, value in outputs.items():
        if value.ndim == 0 or value.shape[0] < sample_count:
            raise ValueError(
                f"Predictions for tensor '{key}' have {value.shape} but {sample_count} samples were loaded."
            )


def _maybe_load_json(path: str | None) -> dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return payload


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    if not bool(config.get("compute_accuracy", True)):
        print(
            f"WARNING: task '{config['task_name']}' has compute_accuracy=false. "
            "Evaluation will still attempt to compute headline metrics from predictions, "
            "but they may be approximate (e.g. AP-10K-to-DLC mapping for pose)."
        )

    samples = load_samples(config, frame_limit=args.frame_limit)
    if not samples:
        raise RuntimeError("No samples were loaded for evaluation.")

    raw_outputs = _load_predictions(args.predictions)
    outputs = _truncate_outputs(raw_outputs, frame_limit=args.frame_limit)
    _validate_alignment(samples, outputs)

    raw_metrics = evaluate_outputs(config, samples, outputs)
    flat_accuracy = flatten_accuracy(config["task_type"], raw_metrics)

    baseline_accuracy = _load_baseline_accuracy(config, args.baseline_results)

    latency_payload = _maybe_load_json(args.latency_json) or {}
    latency_ms = latency_payload.get("latency_ms") if isinstance(latency_payload, dict) else None
    if latency_ms is None and isinstance(latency_payload, dict):
        # accept either a wrapped or flat layout
        latency_ms = latency_payload if {"tpu", "cpu", "total"} & latency_payload.keys() else None

    proxy_payload = _maybe_load_json(args.proxy_json) or {}
    proxy_metrics = proxy_payload.get("proxy_metrics") if isinstance(proxy_payload, dict) else None
    if proxy_metrics is None and isinstance(proxy_payload, dict):
        proxy_metrics = proxy_payload

    record = build_results_record(
        task_type=config["task_type"],
        partition_id=args.partition_id,
        accuracy=flat_accuracy,
        baseline_accuracy=baseline_accuracy,
        latency_ms=latency_ms,
        proxy_metrics=proxy_metrics,
        extra={
            "task_name": config["task_name"],
            "num_evaluated_samples": int(raw_metrics.get("num_evaluated_samples", len(samples))),
            "raw_metrics": raw_metrics,
            "predictions_path": str(Path(args.predictions).resolve()),
        },
    )

    output_path = args.output or str(Path(config["artifacts_dir"]) / "eval" / f"{args.partition_id}.json")
    save_json(output_path, record)

    headline_value = headline_metric_value(config["task_type"], flat_accuracy)
    headline_drop = (record.get("accuracy_drop") or {}).get("headline_drop")

    print(f"Evaluated partition '{args.partition_id}' on {len(samples)} samples.")
    if headline_value is not None:
        print(f"  {record['accuracy_drop'].get('headline_metric') if record.get('accuracy_drop') else config['task_type']}: {headline_value:.4f}")
    if headline_drop is not None:
        print(f"  headline drop vs baseline: {headline_drop:.4f}")
    print(f"  Saved record to: {output_path}")


if __name__ == "__main__":
    main()
