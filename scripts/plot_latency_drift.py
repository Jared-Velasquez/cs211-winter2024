"""plot_latency_drift.py — Latency-vs-drift Pareto plots for partition candidates.

Combines two independently-collected metrics per partition candidate:

  * Latency  — authoritative 100-iteration end-to-end ``t_total`` mean (ms) from the
               Student C benchmark sweep (``benchmark.py``). Resolved from each
               candidate's ``results/benchmark_summary.json`` when it holds a TPU
               timing block, otherwise from the newest ``logs/benchmark/*_sweep_*.json``
               that recorded that candidate with timing (DLC's per-candidate files were
               overwritten by a later cpu-only run, so its timing lives only in the log).
  * Drift    — from each candidate's ``results/summary.json`` written by ``run_hybrid.py``:
                 - ``tpu_output_drift_vs_partitioned_cpu``       (final-output drift)
                 - ``boundary_drift_tpu_dequantized_vs_float``   (boundary proxy drift)
               ``mean_abs_diff`` is used (comparable across candidates within a task for the
               output metric; see the project notes for the boundary caveat).

Output: one figure per drift metric, each a 1x3 row of per-task panels (DLC / SSD / DeepLab).
x = t_total (ms, lower better), y = drift (lower better). The Pareto-optimal set
(minimise both) is connected with a step line and drawn filled; dominated points are hollow.
The "max-TPU subgraph" candidate (largest ``num_tpu_ops``) is starred — this is the strategy
the project asks us to test against the rest.

Only the 15 TPU candidates are plotted. The 5 cpu-only candidates (4 DeepLab compiler
failures + ssd_split_at_box_predictor_biasadds) have no TPU latency/drift and are listed in
each figure's caption.

Run via the repo env wrapper (matplotlib cache is pinned there):
    ./run_in_env.sh python scripts/plot_latency_drift.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = REPO_ROOT / "artifacts"
SWEEP_LOGS = REPO_ROOT / "logs" / "benchmark"
OUT_DIR = REPO_ROOT / "docs_student_c" / "figures"

TASK_ORDER = ["dlc", "ssd", "deeplab"]
TASK_TITLES = {
    "dlc": "Task A: DLC Pose",
    "ssd": "Task B: SSD Detection",
    "deeplab": "Task C: DeepLab Segmentation",
}

DRIFT_METRICS = [
    {
        "key": "tpu_output_drift_vs_partitioned_cpu",
        "stat": "mean_abs_diff",
        "ylabel": "Output drift (mean abs diff, TPU vs partitioned-CPU)",
        "title": "Latency vs Final-Output Drift",
        "filename": "pareto_latency_vs_output_drift.png",
    },
    {
        "key": "boundary_drift_tpu_dequantized_vs_float",
        "stat": "mean_abs_diff",
        "ylabel": "Boundary drift (mean abs diff, TPU-dequant vs float)",
        "title": "Latency vs Boundary Drift (proxy)",
        "filename": "pareto_latency_vs_boundary_drift.png",
    },
]


def _task_of(candidate_id: str) -> str:
    if candidate_id.startswith("dlc"):
        return "dlc"
    if candidate_id.startswith("ssd"):
        return "ssd"
    return "deeplab"


def _short(candidate_id: str) -> str:
    return candidate_id.split("_", 1)[1] if "_" in candidate_id else candidate_id


def _t_total_from_benchmark_summary(candidate_dir: Path) -> float | None:
    path = candidate_dir / "results" / "benchmark_summary.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    stats = data.get("tpu_timing", {}).get("stats", {}).get("t_total", {})
    return stats.get("mean_ms")


def _t_total_from_sweep_logs(candidate_id: str) -> float | None:
    """Newest sweep log that recorded this candidate with a TPU timing block."""
    best_mtime = -1.0
    best_value = None
    for log_path in SWEEP_LOGS.glob("*_sweep_*.json"):
        try:
            data = json.loads(log_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for cand in data.get("candidates", []):
            if Path(cand.get("candidate_dir", "")).name != candidate_id:
                continue
            stats = cand.get("tpu_timing", {}).get("stats", {}).get("t_total", {})
            value = stats.get("mean_ms")
            if value is None:
                continue
            mtime = log_path.stat().st_mtime
            if mtime > best_mtime:
                best_mtime, best_value = mtime, value
    return best_value


def resolve_latency(candidate_dir: Path) -> float | None:
    """Prefer per-candidate benchmark_summary, fall back to the newest sweep log."""
    value = _t_total_from_benchmark_summary(candidate_dir)
    if value is not None:
        return value
    return _t_total_from_sweep_logs(candidate_dir.name)


TIMING_COMPONENTS = ["t_tpu", "t_transfer", "t_cpu", "t_total"]
TIMING_STATS = ["mean_ms", "std_ms", "p95_ms", "min_ms", "max_ms"]


def resolve_timing_breakdown(candidate_dir: Path) -> dict | None:
    """Full per-component timing stats {component: {mean_ms, std_ms, p95_ms, ...}}.

    Prefer the per-candidate benchmark_summary's tpu_timing block; fall back to the newest
    sweep log that recorded this candidate with timing (DLC lives only in the sweep log).
    """
    bench = candidate_dir / "results" / "benchmark_summary.json"
    if bench.exists():
        stats = json.loads(bench.read_text()).get("tpu_timing", {}).get("stats")
        if stats:
            return stats
    best_mtime, best_stats = -1.0, None
    for log_path in SWEEP_LOGS.glob("*_sweep_*.json"):
        try:
            data = json.loads(log_path.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        for cand in data.get("candidates", []):
            if Path(cand.get("candidate_dir", "")).name != candidate_dir.name:
                continue
            stats = cand.get("tpu_timing", {}).get("stats")
            if stats and log_path.stat().st_mtime > best_mtime:
                best_mtime, best_stats = log_path.stat().st_mtime, stats
    return best_stats


def resolve_cpu_latency(candidate_dir: Path, summary: dict) -> float | None:
    """Pure-CPU latency for a cpu-only candidate: partitioned-CPU prefix + suffix (ms).

    This is the analogue of the TPU candidates' end-to-end ``t_total`` — the cost of running
    this same partition entirely on the CPU (no accelerator). Prefer the benchmark.py
    ``cpu_sanity`` block (same source as the TPU timing); fall back to the run_hybrid summary.
    """
    bench = candidate_dir / "results" / "benchmark_summary.json"
    if bench.exists():
        sanity = json.loads(bench.read_text()).get("cpu_sanity", {})
        pre, suf = sanity.get("prefix_mean_ms"), sanity.get("suffix_mean_ms")
        if pre is not None and suf is not None:
            return pre + suf
    pc = summary.get("partitioned_cpu_latency_ms", {})
    pre, suf = pc.get("prefix_mean"), pc.get("suffix_mean")
    if pre is not None and suf is not None:
        return pre + suf
    return None


def collect() -> list[dict]:
    """Build one record per candidate (TPU and cpu-only) with latency + both drift metrics.

    TPU candidates carry their measured drift. CPU-only candidates (no compiled/working TPU
    model) have no quantization, so their drift is 0 by construction (the float prefix->suffix
    split is lossless, ``cpu_split`` ~= 0); they are plotted as reference points and excluded
    from the Pareto frontier (pure-CPU latency is a different regime, no accelerator).
    """
    records = []
    for summary_path in sorted(glob.glob(str(ARTIFACTS / "*" / "results" / "summary.json"))):
        data = json.loads(Path(summary_path).read_text())
        mode = data.get("mode")
        candidate_id = Path(summary_path).parent.parent.name
        candidate_dir = ARTIFACTS / candidate_id
        is_cpu_only = mode != "tpu"

        if is_cpu_only:
            latency = resolve_cpu_latency(candidate_dir, data)
        else:
            latency = resolve_latency(candidate_dir)
        if latency is None:
            print(f"  WARNING: no latency found for {candidate_id}; skipping")
            continue

        record = {
            "candidate": candidate_id,
            "task": _task_of(candidate_id),
            "t_total_ms": latency,
            "num_tpu_ops": data.get("num_tpu_ops"),
            "is_cpu_only": is_cpu_only,
            "timing": None if is_cpu_only else resolve_timing_breakdown(candidate_dir),
        }
        for metric in DRIFT_METRICS:
            if is_cpu_only:
                record[metric["key"]] = 0.0  # no quantization -> no drift
            else:
                block = data.get(metric["key"]) or {}
                record[metric["key"]] = block.get(metric["stat"])
        records.append(record)
    return records


def pareto_mask(xs: list[float], ys: list[float]) -> list[bool]:
    """True where a point is Pareto-optimal (no other point has both x and y <=, one <)."""
    mask = []
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        dominated = any(
            j != i and xj <= xi and yj <= yi and (xj < xi or yj < yi)
            for j, (xj, yj) in enumerate(zip(xs, ys))
        )
        mask.append(not dominated)
    return mask


def plot_metric(records: list[dict], metric: dict) -> Path:
    key, ylabel, title = metric["key"], metric["ylabel"], metric["title"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, task in zip(axes, TASK_ORDER):
        task_rows = [r for r in records if r["task"] == task and r[key] is not None]
        tpu_rows = [r for r in task_rows if not r["is_cpu_only"]]
        cpu_rows = [r for r in task_rows if r["is_cpu_only"]]
        ax.set_title(TASK_TITLES[task], fontsize=12, fontweight="bold")
        ax.set_xlabel("End-to-end latency t_total (ms, 100-iter mean)")
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)
        if not task_rows:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
            continue

        # --- TPU candidates: Pareto frontier computed among these only ---
        xs = [r["t_total_ms"] for r in tpu_rows]
        ys = [r[key] for r in tpu_rows]
        opt = pareto_mask(xs, ys) if tpu_rows else []
        max_tpu_idx = (
            max(range(len(tpu_rows)), key=lambda i: tpu_rows[i]["num_tpu_ops"] or -1)
            if tpu_rows else -1
        )

        front = sorted((p for p, m in zip(zip(xs, ys), opt) if m), key=lambda p: p[0])
        if len(front) >= 2:
            ax.step(
                [p[0] for p in front], [p[1] for p in front],
                where="post", color="tab:green", alpha=0.6, lw=1.5, zorder=1,
            )

        for i, r in enumerate(tpu_rows):
            is_opt = opt[i]
            is_maxtpu = i == max_tpu_idx
            ax.scatter(
                xs[i], ys[i],
                s=170 if is_maxtpu else 90,
                marker="*" if is_maxtpu else "o",
                facecolor=("tab:green" if is_opt else "none"),
                edgecolor=("tab:red" if is_maxtpu else "tab:green" if is_opt else "tab:gray"),
                linewidths=1.8 if is_maxtpu else 1.2,
                zorder=3,
            )
            label = _short(r["candidate"])
            if is_maxtpu:
                label += "  (max-TPU)"
            ax.annotate(
                label, (xs[i], ys[i]),
                textcoords="offset points", xytext=(6, 4),
                fontsize=7.5, color="black",
            )

        # --- CPU-only candidates: blue squares at drift=0, pure-CPU latency, not on frontier.
        # These cluster (same x~full-CPU, y=0), so stagger labels vertically to stay readable.
        for j, r in enumerate(sorted(cpu_rows, key=lambda r: r["t_total_ms"])):
            ax.scatter(
                r["t_total_ms"], r[key],
                s=80, marker="s", facecolor="none",
                edgecolor="tab:blue", linewidths=1.3, zorder=2,
            )
            ax.annotate(
                _short(r["candidate"]), (r["t_total_ms"], r[key]),
                textcoords="offset points", xytext=(8, 6 + 11 * j),
                fontsize=7.5, color="tab:blue",
                arrowprops=dict(arrowstyle="-", color="tab:blue", lw=0.5, alpha=0.5),
            )

        # Legend proxy entries.
        handles = [
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="tab:green",
                       markeredgecolor="tab:green", markersize=8, label="Pareto-optimal (TPU)"),
            plt.Line2D([], [], marker="o", color="w", markerfacecolor="none",
                       markeredgecolor="tab:gray", markersize=8, label="dominated (TPU)"),
            plt.Line2D([], [], marker="*", color="w", markerfacecolor="none",
                       markeredgecolor="tab:red", markersize=12, label="max-TPU subgraph"),
            plt.Line2D([], [], marker="s", color="w", markerfacecolor="none",
                       markeredgecolor="tab:blue", markersize=8,
                       label="CPU-only (no TPU; drift=0, full-CPU latency)"),
        ]
        ax.legend(handles=handles, fontsize=8, loc="best")

    # excluded = (
    #     "Excluded (cpu-only, no TPU metrics): ssd_split_at_box_predictor_biasadds (TPU "
    #     "runtime fail), deeplab_split_after_{expanded_conv_16, aspp, logits, resize} "
    #     "(edgetpu compile failures)."
    # )
    fig.suptitle(
        f"{title}",
        fontsize=14, fontweight="bold",
    )
    # fig.text(0.5, 0.005, excluded, ha="center", fontsize=8, style="italic", color="dimgray")
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / metric["filename"]
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def write_csv(records: list[dict]) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "latency_drift_data.csv"
    base_fields = [
        "task", "candidate", "is_cpu_only", "t_total_ms", "num_tpu_ops",
        "tpu_output_drift_vs_partitioned_cpu", "boundary_drift_tpu_dequantized_vs_float",
    ]
    # Latency breakdown: each component x {mean, std, p95} (min/max kept in JSON, not CSV).
    breakdown_fields = [
        f"{c}_{s.replace('_ms', '')}_ms"
        for c in TIMING_COMPONENTS for s in ["mean_ms", "std_ms", "p95_ms"]
    ]
    fields = base_fields + breakdown_fields
    with open(out_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in sorted(records, key=lambda r: (TASK_ORDER.index(r["task"]), r["candidate"])):
            row = {k: r.get(k) for k in base_fields}
            timing = r.get("timing") or {}
            for c in TIMING_COMPONENTS:
                comp = timing.get(c, {})
                for s in ["mean_ms", "std_ms", "p95_ms"]:
                    row[f"{c}_{s.replace('_ms', '')}_ms"] = comp.get(s)
            writer.writerow(row)
    return out_path


def main() -> None:
    records = collect()
    n_tpu = sum(1 for r in records if not r["is_cpu_only"])
    n_cpu = sum(1 for r in records if r["is_cpu_only"])
    print(f"Collected {len(records)} candidates ({n_tpu} TPU + {n_cpu} cpu-only).")
    csv_path = write_csv(records)
    print(f"Wrote data table: {csv_path}")
    for metric in DRIFT_METRICS:
        out_path = plot_metric(records, metric)
        print(f"Wrote figure: {out_path}")


if __name__ == "__main__":
    main()
