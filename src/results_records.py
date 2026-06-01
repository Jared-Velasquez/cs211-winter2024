"""Shared schemas and helpers for Student D's Results Records.

A Results Record describes one (task, partition_candidate) measurement and follows
the format agreed at the end of Week 2 in `student_project_plan.md`:

    {
        "partition_id": "split_after_block3",
        "task": "pose_estimation",
        "latency_ms": {"tpu": 12.3, "transfer": 0.8, "cpu": 5.1, "total": 18.2},
        "accuracy": {"pck_0.05": 0.72, "pck_0.10": 0.88, "rmse": 4.3},
        "baseline_accuracy": {"pck_0.05": 0.91, ...},
        "proxy_metrics": {"boundary_mse": 0.0032, "boundary_kl_div": 0.145, ...}
    }

These helpers normalize the per-task accuracy dictionaries into flat, plottable
metrics, expose the "headline" accuracy metric per task, and compute an
``accuracy_drop`` summary that other downstream tools (correlation, plots) can
use without knowing about each task's specifics.
"""

from __future__ import annotations

from typing import Any


# Per-task headline metric used for Pareto plots and Spearman correlation against
# proxy metrics. "Higher is better" except where noted.
HEADLINE_METRIC: dict[str, str] = {
    "pose_estimation": "pck_0.10",
    "object_detection": "map_50",
    "semantic_segmentation": "miou",
}

# Direction of each metric: True if higher = better, False if lower = better.
METRIC_HIGHER_IS_BETTER: dict[str, bool] = {
    # Pose
    "pck_0.05": True,
    "pck_0.10": True,
    "pck_0.20": True,
    "rmse": False,
    # Detection
    "map_50_95": True,
    "map_50": True,
    # Segmentation
    "miou": True,
    "boundary_f1": True,
}


def flatten_accuracy(task_type: str, raw_accuracy: dict[str, Any]) -> dict[str, float]:
    """Flatten the nested accuracy dictionary returned by ``src.evaluation`` into
    a flat, scalar-only dict suitable for Results Records and analysis.

    The shapes returned by ``src.evaluation.evaluate_outputs`` differ per task,
    so the keys for nested values like ``pck`` (a dict by threshold) get
    collapsed into top-level keys like ``pck_0.05``.
    """
    flat: dict[str, float] = {}
    if not raw_accuracy:
        return flat

    if task_type == "pose_estimation":
        if "rmse" in raw_accuracy:
            flat["rmse"] = float(raw_accuracy["rmse"])
        for threshold, value in (raw_accuracy.get("pck") or {}).items():
            flat[f"pck_{threshold}"] = float(value)
    elif task_type == "object_detection":
        if "map_50" in raw_accuracy:
            flat["map_50"] = float(raw_accuracy["map_50"])
        if "map_50_95" in raw_accuracy:
            flat["map_50_95"] = float(raw_accuracy["map_50_95"])
    elif task_type == "semantic_segmentation":
        if "miou" in raw_accuracy:
            flat["miou"] = float(raw_accuracy["miou"])
        if "boundary_f1" in raw_accuracy:
            flat["boundary_f1"] = float(raw_accuracy["boundary_f1"])

    return flat


def headline_metric_name(task_type: str) -> str:
    if task_type not in HEADLINE_METRIC:
        raise ValueError(f"No headline metric registered for task_type '{task_type}'.")
    return HEADLINE_METRIC[task_type]


def headline_metric_value(task_type: str, accuracy: dict[str, float]) -> float | None:
    name = headline_metric_name(task_type)
    if name not in accuracy:
        return None
    return float(accuracy[name])


def metric_drop(metric_name: str, baseline: float, candidate: float) -> float:
    """Drop is signed so that "positive = candidate is worse than baseline".

    For metrics where higher is better (pck, mAP, miou) the drop is
    ``baseline - candidate``. For metrics where lower is better (RMSE) the drop
    is ``candidate - baseline``. Unknown metrics default to higher-is-better.
    """
    higher_is_better = METRIC_HIGHER_IS_BETTER.get(metric_name, True)
    if higher_is_better:
        return float(baseline) - float(candidate)
    return float(candidate) - float(baseline)


def compute_accuracy_drop(
    task_type: str,
    candidate_accuracy: dict[str, float],
    baseline_accuracy: dict[str, float] | None,
) -> dict[str, Any]:
    """Compute the per-metric drop and a single ``headline_drop`` scalar.

    ``headline_drop`` is the value used by analysis/plot tools.
    """
    if not baseline_accuracy:
        return {}

    per_metric: dict[str, float] = {}
    for metric_name, candidate_value in candidate_accuracy.items():
        if metric_name in baseline_accuracy:
            per_metric[metric_name] = metric_drop(
                metric_name,
                baseline_accuracy[metric_name],
                candidate_value,
            )

    headline_name = headline_metric_name(task_type)
    headline_drop = per_metric.get(headline_name)
    return {
        "per_metric": per_metric,
        "headline_metric": headline_name,
        "headline_drop": headline_drop,
    }


def build_results_record(
    *,
    task_type: str,
    partition_id: str,
    accuracy: dict[str, float],
    baseline_accuracy: dict[str, float] | None = None,
    latency_ms: dict[str, float] | None = None,
    proxy_metrics: dict[str, float] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "partition_id": partition_id,
        "task": task_type,
        "accuracy": dict(accuracy),
    }
    if baseline_accuracy:
        record["baseline_accuracy"] = dict(baseline_accuracy)
        record["accuracy_drop"] = compute_accuracy_drop(task_type, accuracy, baseline_accuracy)
    if latency_ms is not None:
        record["latency_ms"] = dict(latency_ms)
    if proxy_metrics is not None:
        # Keep only scalar values for the analysis/correlation workflow. Any
        # non-scalar metadata can ride along under ``proxy_metrics_extra``.
        scalar, non_scalar = _split_scalars(proxy_metrics)
        record["proxy_metrics"] = scalar
        if non_scalar:
            record["proxy_metrics_extra"] = non_scalar
    if extra:
        record.update(extra)
    return record


def _split_scalars(values: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    scalar: dict[str, float] = {}
    other: dict[str, Any] = {}
    for key, value in values.items():
        if isinstance(value, bool):
            other[key] = value
            continue
        if isinstance(value, (int, float)):
            scalar[key] = float(value)
            continue
        other[key] = value
    return scalar, other
