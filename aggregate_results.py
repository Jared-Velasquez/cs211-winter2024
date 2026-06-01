"""Student D — Aggregate per-candidate evaluations into one partition_results.json.

Inputs
------

A *partition manifest* (JSON) listing every partition candidate Student C has
benchmarked on hardware:

    {
      "task_config": "configs/task_b_detection.json",
      "candidates": [
        {
          "partition_id": "split_after_block11",
          "predictions": "artifacts/task_b/detection/runs/block11/predictions.npz",
          "latency_ms": {"tpu": 12.3, "transfer": 0.8, "cpu": 5.1, "total": 18.2},
          "proxy_metrics": {"boundary_mse": 0.0032, ...},
          "num_tpu_ops": 142,
          "num_cpu_ops": 38,
          "is_max_tpu": false
        },
        ...
      ]
    }

Each candidate's ``latency_ms`` and ``proxy_metrics`` may instead be a string
path pointing at a JSON file with the same structure as ``--latency-json`` /
``--proxy-json`` in ``evaluate.py``.

Outputs
-------

A single ``partition_results.json`` containing a list of Results Records (see
``src.results_records``), ready to feed into ``analyze_partitions.py`` and
``plot_partitions.py``.
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
from src.results_records import build_results_record, flatten_accuracy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate per-candidate evaluations into a single partition_results.json.")
    parser.add_argument("--manifest", required=True, help="Path to the partition manifest JSON.")
    parser.add_argument(
        "--config",
        default=None,
        help="Override task config path. Defaults to manifest['task_config'].",
    )
    parser.add_argument(
        "--frame-limit",
        type=int,
        default=None,
        help="Restrict each evaluation to the first N samples.",
    )
    parser.add_argument(
        "--baseline-results",
        default=None,
        help="Optional override for baseline_results.json.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Where to write partition_results.json. Defaults to <artifacts_dir>/partition_results.json.",
    )
    return parser.parse_args()


def _load_manifest(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_relative(manifest_path: Path, value: str | None) -> str | None:
    if value is None:
        return None
    candidate = Path(value)
    if candidate.is_absolute():
        return str(candidate)
    return str((manifest_path.parent / candidate).resolve())


def _maybe_load_inline_or_path(manifest_path: Path, value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        resolved = _resolve_relative(manifest_path, value)
        return json.loads(Path(resolved).read_text(encoding="utf-8"))
    raise TypeError(f"Expected dict or path, got {type(value).__name__}.")


def _load_predictions(path: str) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=True)
    return {key: archive[key] for key in archive.files}


def _truncate_outputs(outputs: dict[str, np.ndarray], frame_limit: int | None) -> dict[str, np.ndarray]:
    if frame_limit is None:
        return outputs
    return {key: value[:frame_limit] for key, value in outputs.items()}


def _load_baseline_accuracy(config: dict[str, Any], explicit_path: str | None) -> dict[str, float] | None:
    candidate_path = explicit_path or config.get("baseline_results_path")
    if not candidate_path:
        return None
    path = Path(candidate_path)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw_metrics = payload.get("metrics")
    if not raw_metrics:
        return None
    return flatten_accuracy(config["task_type"], raw_metrics)


def _normalize_latency(latency_payload: dict[str, Any] | None) -> dict[str, float] | None:
    if not latency_payload:
        return None
    if "latency_ms" in latency_payload and isinstance(latency_payload["latency_ms"], dict):
        return {key: float(value) for key, value in latency_payload["latency_ms"].items()}
    if {"tpu", "cpu", "total"} & latency_payload.keys():
        return {key: float(value) for key, value in latency_payload.items() if isinstance(value, (int, float))}
    return None


def _normalize_proxy(proxy_payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not proxy_payload:
        return None
    if "proxy_metrics" in proxy_payload and isinstance(proxy_payload["proxy_metrics"], dict):
        return dict(proxy_payload["proxy_metrics"])
    return dict(proxy_payload)


def evaluate_candidate(
    config: dict[str, Any],
    samples: list[dict[str, Any]],
    candidate: dict[str, Any],
    manifest_path: Path,
    baseline_accuracy: dict[str, float] | None,
    frame_limit: int | None,
) -> dict[str, Any]:
    predictions_path = _resolve_relative(manifest_path, candidate.get("predictions"))
    if not predictions_path:
        raise ValueError(
            f"Candidate '{candidate.get('partition_id')}' is missing a 'predictions' field."
        )

    raw_outputs = _load_predictions(predictions_path)
    outputs = _truncate_outputs(raw_outputs, frame_limit=frame_limit)
    raw_metrics = evaluate_outputs(config, samples, outputs)
    flat_accuracy = flatten_accuracy(config["task_type"], raw_metrics)

    latency_ms = _normalize_latency(_maybe_load_inline_or_path(manifest_path, candidate.get("latency_ms")))
    proxy_metrics = _normalize_proxy(_maybe_load_inline_or_path(manifest_path, candidate.get("proxy_metrics")))

    extra: dict[str, Any] = {
        "task_name": config["task_name"],
        "num_evaluated_samples": int(raw_metrics.get("num_evaluated_samples", len(samples))),
        "raw_metrics": raw_metrics,
        "predictions_path": predictions_path,
    }
    for passthrough_key in ("num_tpu_ops", "num_cpu_ops", "is_max_tpu", "boundary_tensor_shapes", "tpu_tflite_path", "cpu_graph_path"):
        if passthrough_key in candidate:
            extra[passthrough_key] = candidate[passthrough_key]

    return build_results_record(
        task_type=config["task_type"],
        partition_id=candidate["partition_id"],
        accuracy=flat_accuracy,
        baseline_accuracy=baseline_accuracy,
        latency_ms=latency_ms,
        proxy_metrics=proxy_metrics,
        extra=extra,
    )


def main() -> None:
    args = parse_args()
    manifest_path = Path(args.manifest).resolve()
    manifest = _load_manifest(str(manifest_path))

    config_path = args.config or manifest.get("task_config")
    if not config_path:
        raise ValueError("Manifest is missing 'task_config' and --config was not provided.")
    config_path = _resolve_relative(manifest_path, config_path)
    config = load_config(config_path)

    samples = load_samples(config, frame_limit=args.frame_limit)
    if not samples:
        raise RuntimeError("No samples were loaded for evaluation.")

    baseline_accuracy = _load_baseline_accuracy(config, args.baseline_results)

    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for candidate in manifest.get("candidates", []):
        try:
            record = evaluate_candidate(
                config=config,
                samples=samples,
                candidate=candidate,
                manifest_path=manifest_path,
                baseline_accuracy=baseline_accuracy,
                frame_limit=args.frame_limit,
            )
            records.append(record)
            print(f"  {record['partition_id']}: ok")
        except Exception as error:  # noqa: BLE001
            failures.append({"partition_id": candidate.get("partition_id"), "error": str(error)})
            print(f"  {candidate.get('partition_id')}: FAILED ({error})")

    output_path = args.output or str(Path(config["artifacts_dir"]) / "partition_results.json")
    save_json(
        output_path,
        {
            "task_name": config["task_name"],
            "task_type": config["task_type"],
            "num_records": len(records),
            "num_failures": len(failures),
            "failures": failures,
            "baseline_accuracy": baseline_accuracy,
            "records": records,
        },
    )
    print(f"Saved {len(records)} records to: {output_path}")


if __name__ == "__main__":
    main()
