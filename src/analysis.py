"""Analysis utilities for Student D's Results Records.

The functions here are intentionally dependency-light: they take lists of
``ResultsRecord`` dicts (see ``src.results_records``) and produce structured
outputs used by ``analyze_partitions.py`` and ``plot_partitions.py``.

Spearman correlation is implemented locally so that the analysis layer does not
require ``scipy`` at runtime. ``scipy.stats.spearmanr`` is the gold standard but
this codebase already runs on environments without scipy installed (e.g. the
Coral host), so we keep this self-contained.
"""

from __future__ import annotations

import math
from typing import Any, Iterable


# --- Spearman correlation -------------------------------------------------- #


def _average_ranks(values: list[float]) -> list[float]:
    """Tied values get the average rank (the standard "fractional" ranking)."""
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        average_rank = (i + j) / 2.0 + 1.0  # 1-indexed average rank
        for k in range(i, j + 1):
            original_index, _ = indexed[k]
            ranks[original_index] = average_rank
        i = j + 1
    return ranks


def spearman_rho(xs: list[float], ys: list[float]) -> float | None:
    """Spearman rank correlation coefficient.

    Returns ``None`` if either series has zero variance (constant ranks) or if
    the series are too short to produce a meaningful coefficient.
    """
    if len(xs) != len(ys):
        raise ValueError("Spearman inputs must have equal length.")
    if len(xs) < 2:
        return None

    x_ranks = _average_ranks(xs)
    y_ranks = _average_ranks(ys)
    n = len(xs)

    mean_x = sum(x_ranks) / n
    mean_y = sum(y_ranks) / n
    cov = sum((rx - mean_x) * (ry - mean_y) for rx, ry in zip(x_ranks, y_ranks))
    var_x = sum((rx - mean_x) ** 2 for rx in x_ranks)
    var_y = sum((ry - mean_y) ** 2 for ry in y_ranks)
    if var_x == 0.0 or var_y == 0.0:
        return None
    return cov / math.sqrt(var_x * var_y)


# --- Filtering / extraction helpers ---------------------------------------- #


def _records_with_drop(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        record
        for record in records
        if record.get("accuracy_drop", {}).get("headline_drop") is not None
        and record.get("proxy_metrics")
    ]


def proxy_metric_names(records: Iterable[dict[str, Any]]) -> list[str]:
    seen: dict[str, None] = {}
    for record in records:
        for key in (record.get("proxy_metrics") or {}).keys():
            seen.setdefault(key, None)
    return list(seen.keys())


# --- Analysis routines ----------------------------------------------------- #


def correlate_proxies(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """For each proxy metric, compute Spearman ρ vs ``headline_drop``.

    A proxy that predicts accuracy drop well should have ρ close to +1 (proxy
    grows when accuracy drops) or -1 (proxy shrinks when accuracy drops).
    """
    valid = _records_with_drop(records)
    correlations: list[dict[str, Any]] = []

    for proxy in proxy_metric_names(valid):
        xs: list[float] = []
        ys: list[float] = []
        for record in valid:
            proxy_value = (record.get("proxy_metrics") or {}).get(proxy)
            drop = record["accuracy_drop"]["headline_drop"]
            if proxy_value is None or drop is None:
                continue
            xs.append(float(proxy_value))
            ys.append(float(drop))
        if len(xs) < 2:
            continue
        rho = spearman_rho(xs, ys)
        if rho is None:
            # Constant proxy series have no predictive power; skip them so they
            # don't pollute the ranked output or downstream strategy comparison.
            continue
        correlations.append(
            {
                "proxy": proxy,
                "spearman_rho": rho,
                "abs_spearman_rho": abs(rho),
                "num_points": len(xs),
            }
        )

    correlations.sort(key=lambda item: -item["abs_spearman_rho"])
    return correlations


def cross_task_consistency(records_by_task: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    """For each proxy that appears in multiple tasks, list per-task ρ side by side.

    Useful for answering "does the same proxy rank partitions correctly for all
    three tasks?".
    """
    per_task_correlations = {
        task: correlate_proxies(task_records)
        for task, task_records in records_by_task.items()
    }
    proxies: dict[str, dict[str, float | None]] = {}
    for task, correlations in per_task_correlations.items():
        for entry in correlations:
            proxies.setdefault(entry["proxy"], {})[task] = entry["spearman_rho"]

    return [
        {
            "proxy": proxy,
            "per_task_rho": per_task_rho,
            "num_tasks": len(per_task_rho),
            "min_abs_rho": min(
                (abs(value) for value in per_task_rho.values() if value is not None),
                default=None,
            ),
        }
        for proxy, per_task_rho in proxies.items()
    ]


# --- Pareto frontier ------------------------------------------------------- #


def pareto_frontier(
    records: Iterable[dict[str, Any]],
    *,
    latency_key: str = "total",
) -> list[dict[str, Any]]:
    """Return the records on the Pareto frontier of (lower latency, higher accuracy).

    Records without latency or headline accuracy are skipped.
    """
    points: list[dict[str, Any]] = []
    for record in records:
        latency = (record.get("latency_ms") or {}).get(latency_key)
        accuracy = record.get("accuracy") or {}
        headline_name = (record.get("accuracy_drop") or {}).get("headline_metric")
        if not headline_name:
            from src.results_records import headline_metric_name

            headline_name = headline_metric_name(record["task"])
        headline = accuracy.get(headline_name)
        if latency is None or headline is None:
            continue
        points.append(
            {
                "partition_id": record["partition_id"],
                "task": record["task"],
                "latency_ms": float(latency),
                "headline_metric": headline_name,
                "headline_value": float(headline),
            }
        )

    frontier: list[dict[str, Any]] = []
    points_sorted = sorted(points, key=lambda item: (item["latency_ms"], -item["headline_value"]))
    best_value = -math.inf
    for point in points_sorted:
        if point["headline_value"] > best_value:
            frontier.append(point)
            best_value = point["headline_value"]
    return frontier


# --- Strategy comparison --------------------------------------------------- #


def compare_partition_strategies(
    records: list[dict[str, Any]],
    proxy_correlations: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare three partition selection strategies on the headline metric.

    1. ``max_tpu`` — the candidate marked ``is_max_tpu`` in the manifest, or
       failing that, the candidate with the largest ``num_tpu_ops``.
    2. ``best_actual`` — the candidate with the highest headline accuracy.
    3. ``best_per_proxy`` — for each proxy metric, what headline accuracy would
       you have picked if you sorted by that proxy?
    """
    if not records:
        return {}

    task_type = records[0]["task"]
    from src.results_records import headline_metric_name

    headline_name = headline_metric_name(task_type)

    def _headline(record: dict[str, Any]) -> float | None:
        value = (record.get("accuracy") or {}).get(headline_name)
        return None if value is None else float(value)

    scored = [(record, _headline(record)) for record in records]
    scored = [(record, value) for record, value in scored if value is not None]
    if not scored:
        return {}

    summary: dict[str, Any] = {
        "task": task_type,
        "headline_metric": headline_name,
        "num_records": len(scored),
    }

    max_tpu = next((record for record, _ in scored if record.get("is_max_tpu")), None)
    if max_tpu is None:
        max_tpu = max(
            scored,
            key=lambda pair: pair[0].get("num_tpu_ops", -1),
        )[0]
    summary["max_tpu"] = {
        "partition_id": max_tpu["partition_id"],
        "headline_value": _headline(max_tpu),
    }

    best_actual_record, best_actual_value = max(scored, key=lambda pair: pair[1])
    summary["best_actual"] = {
        "partition_id": best_actual_record["partition_id"],
        "headline_value": best_actual_value,
    }

    if proxy_correlations is None:
        proxy_correlations = correlate_proxies(records)

    by_proxy: dict[str, Any] = {}
    for entry in proxy_correlations:
        proxy = entry["proxy"]
        rho = entry.get("spearman_rho")
        # If proxy positively correlates with drop, the smallest proxy value picks
        # the best partition. If it negatively correlates, the largest does.
        if rho is None:
            continue
        ascending = rho >= 0.0
        ranked = sorted(
            (
                (record, (record.get("proxy_metrics") or {}).get(proxy), value)
                for record, value in scored
                if (record.get("proxy_metrics") or {}).get(proxy) is not None
            ),
            key=lambda triple: triple[1],
            reverse=not ascending,
        )
        if not ranked:
            continue
        choice_record, _, choice_value = ranked[0]
        by_proxy[proxy] = {
            "partition_id": choice_record["partition_id"],
            "headline_value": choice_value,
            "rho": rho,
            "ascending": ascending,
        }
    summary["best_per_proxy"] = by_proxy

    return summary
