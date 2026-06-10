"""Student D — Plots and summary tables for the partition analysis.

Reads the same ``partition_results.json`` files as ``analyze_partitions.py``
and emits:

  * One Pareto frontier figure per task (latency vs. headline accuracy).
  * One combined proxy-vs-headline-drop scatter figure per proxy metric, with
    points colored by task.
  * A consolidated CSV summary table listing every Results Record's headline
    metric, latency, and proxy values side-by-side.

Matplotlib is imported lazily so that the module can still be imported in
analysis-only environments (e.g. for unit tests).
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from src.analysis import correlate_proxies, pareto_frontier, proxy_metric_names
from src.io_utils import ensure_directory
from src.results_records import headline_metric_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Pareto plots, scatter plots, and summary tables.")
    parser.add_argument(
        "--results",
        nargs="+",
        required=True,
        help="One or more partition_results.json files.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory to write figures and CSV into.")
    return parser.parse_args()


def _load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _records_by_task(results_paths: list[str]) -> dict[str, list[dict[str, Any]]]:
    by_task: dict[str, list[dict[str, Any]]] = {}
    for path in results_paths:
        payload = _load(path)
        records = payload.get("records") or []
        if not records:
            continue
        task_type = payload.get("task_type") or records[0]["task"]
        by_task.setdefault(task_type, []).extend(records)
    return by_task


def _plot_pareto(by_task: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[str] = []
    for task_type, records in by_task.items():
        headline = headline_metric_name(task_type)
        xs: list[float] = []
        ys: list[float] = []
        labels: list[str] = []
        for record in records:
            latency = (record.get("latency_ms") or {}).get("total")
            value = (record.get("accuracy") or {}).get(headline)
            if latency is None or value is None:
                continue
            xs.append(float(latency))
            ys.append(float(value))
            labels.append(record["partition_id"])

        if not xs:
            continue

        frontier = pareto_frontier(records)
        frontier_x = [point["latency_ms"] for point in frontier]
        frontier_y = [point["headline_value"] for point in frontier]

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(xs, ys, label="candidates", color="#4c72b0")
        if frontier_x:
            ax.plot(frontier_x, frontier_y, color="#dd8452", marker="o", label="Pareto frontier")
        for x, y, label in zip(xs, ys, labels):
            ax.annotate(label, (x, y), fontsize=7, alpha=0.7, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Total latency (ms)")
        ax.set_ylabel(f"{headline} (higher is better)")
        ax.set_title(f"Latency vs accuracy — {task_type}")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        path = output_dir / f"pareto_{task_type}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))

    return paths


def _plot_proxy_scatter(by_task: dict[str, list[dict[str, Any]]], output_dir: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    all_records = [record for records in by_task.values() for record in records]
    proxies = proxy_metric_names(all_records)
    if not proxies:
        return []

    task_colors = {
        "pose_estimation": "#4c72b0",
        "object_detection": "#55a868",
        "semantic_segmentation": "#c44e52",
    }

    paths: list[str] = []
    for proxy in proxies:
        fig, ax = plt.subplots(figsize=(7, 5))
        plotted_any = False
        for task_type, records in by_task.items():
            xs: list[float] = []
            ys: list[float] = []
            for record in records:
                proxy_value = (record.get("proxy_metrics") or {}).get(proxy)
                drop = (record.get("accuracy_drop") or {}).get("headline_drop")
                if proxy_value is None or drop is None:
                    continue
                xs.append(float(proxy_value))
                ys.append(float(drop))
            if not xs:
                continue
            color = task_colors.get(task_type, "gray")
            ax.scatter(xs, ys, label=task_type, color=color, alpha=0.8)
            plotted_any = True

        if not plotted_any:
            plt.close(fig)
            continue

        rho_lookup = {
            entry["proxy"]: entry["spearman_rho"]
            for entry in correlate_proxies(all_records)
        }
        rho = rho_lookup.get(proxy)
        rho_str = "n/a" if rho is None else f"{rho:+.3f}"
        ax.axhline(0.0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.set_xlabel(proxy)
        ax.set_ylabel("Headline accuracy drop (positive = worse)")
        ax.set_title(f"{proxy} vs. accuracy drop  (Spearman ρ = {rho_str})")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()

        safe_name = proxy.replace("/", "_").replace(" ", "_")
        path = output_dir / f"proxy_{safe_name}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        paths.append(str(path))
    return paths


def _summary_csv(by_task: dict[str, list[dict[str, Any]]], output_dir: Path) -> str:
    rows: list[dict[str, Any]] = []
    proxies: set[str] = set()
    for records in by_task.values():
        for record in records:
            proxies.update((record.get("proxy_metrics") or {}).keys())

    proxy_columns = sorted(proxies)
    fieldnames = [
        "task",
        "partition_id",
        "headline_metric",
        "headline_value",
        "headline_drop",
        "latency_total_ms",
        "latency_tpu_ms",
        "latency_cpu_ms",
        "latency_transfer_ms",
        "num_tpu_ops",
        "num_cpu_ops",
        "is_max_tpu",
        *proxy_columns,
    ]

    for task_type, records in by_task.items():
        headline_name = headline_metric_name(task_type)
        for record in records:
            accuracy = record.get("accuracy") or {}
            latency = record.get("latency_ms") or {}
            proxy_metrics = record.get("proxy_metrics") or {}
            row = {
                "task": task_type,
                "partition_id": record.get("partition_id"),
                "headline_metric": headline_name,
                "headline_value": accuracy.get(headline_name),
                "headline_drop": (record.get("accuracy_drop") or {}).get("headline_drop"),
                "latency_total_ms": latency.get("total"),
                "latency_tpu_ms": latency.get("tpu"),
                "latency_cpu_ms": latency.get("cpu"),
                "latency_transfer_ms": latency.get("transfer"),
                "num_tpu_ops": record.get("num_tpu_ops"),
                "num_cpu_ops": record.get("num_cpu_ops"),
                "is_max_tpu": record.get("is_max_tpu"),
            }
            for proxy in proxy_columns:
                row[proxy] = proxy_metrics.get(proxy)
            rows.append(row)

    csv_path = output_dir / "partition_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return str(csv_path)


def generate_outputs(results_paths: list[str], output_dir: str) -> dict[str, Any]:
    output_path = ensure_directory(output_dir)
    by_task = _records_by_task(results_paths)
    if not by_task:
        return {"pareto_figures": [], "proxy_figures": [], "summary_csv": None}

    pareto_paths = _plot_pareto(by_task, output_path)
    proxy_paths = _plot_proxy_scatter(by_task, output_path)
    csv_path = _summary_csv(by_task, output_path)
    return {
        "pareto_figures": pareto_paths,
        "proxy_figures": proxy_paths,
        "summary_csv": csv_path,
    }


def main() -> None:
    args = parse_args()
    outputs = generate_outputs(args.results, args.output_dir)
    if outputs["summary_csv"]:
        print(f"Summary CSV: {outputs['summary_csv']}")
    for path in outputs["pareto_figures"]:
        print(f"Pareto figure: {path}")
    for path in outputs["proxy_figures"]:
        print(f"Proxy scatter: {path}")


if __name__ == "__main__":
    main()
