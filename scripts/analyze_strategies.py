#!/usr/bin/env python3
"""Latency/accuracy analysis + Pareto + strategy comparison.

Pure-CPU analysis stage (no hardware). Reads the consolidated
docs_student_c/partition_metrics.csv (latency + accuracy + baselines) and each
candidate's artifacts/<id>/metadata.json (op counts), then emits:

  docs_student_c/analysis/table_{dlc,ssd,deeplab}.{csv,md}
  docs_student_c/analysis/figures/latency_vs_accuracy_{task}.png
  docs_student_c/analysis/figures/latency_vs_drop_{task}.png
  docs_student_c/analysis/strategy_comparison.{csv,md}
  docs_student_c/analysis/findings.md  (writeup is authored separately; this
                                        script prints the numbers it needs)

accuracy_drop convention (advisor): positive = accuracy lost.
  higher-is-better (PCK/mAP/mIoU/F1): drop = baseline - partition
  lower-is-better  (RMSE):            drop = partition - baseline
The CSV's precomputed drop_rmse uses baseline-partition (wrong sign), so RMSE
is recomputed here.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_IN = os.path.join(ROOT, "docs_student_c", "partition_metrics.csv")
ART = os.path.join(ROOT, "artifacts")
OUT = os.path.join(ROOT, "docs_student_c", "analysis")
FIG = os.path.join(OUT, "figures")
os.makedirs(FIG, exist_ok=True)

# Per-task config: objective metric (higher-is-better) used for the plots/Pareto.
TASK_CFG = {
    "dlc":     {"name": "DLC (pose)",         "obj": "pck_0.10",  "obj_label": "PCK@0.10"},
    "ssd":     {"name": "SSD (detection)",    "obj": "map_50_95", "obj_label": "mAP@0.5:0.95"},
    "deeplab": {"name": "DeepLab (segm.)",    "obj": "miou",      "obj_label": "mIoU"},
}

# Known max-TPU candidates (handoff sanity check).
EXPECTED_MAX_TPU = {
    "dlc": "dlc_split_at_bias_add",
    "ssd": "ssd_split_before_postprocessor",
    "deeplab": "deeplab_split_after_expanded_conv_13",
}

# Candidates with a float-path partition-correctness bug (CPU-split drift = 85.0).
# Their accuracy drops conflate a partition bug with int8 -> flag/exclude.
PARTITION_BUG = {"ssd_split_after_conv_1", "ssd_split_after_expanded_conv_13"}


def fnum(s):
    s = (s or "").strip()
    if s == "":
        return None
    return float(s)


def load_metadata(cand):
    p = os.path.join(ART, cand, "metadata.json")
    with open(p) as f:
        m = json.load(f)
    return m.get("num_tpu_ops"), m.get("tpu_ops_mapped_edgetpu")


def load_rows():
    rows = []
    with open(CSV_IN) as f:
        for r in csv.DictReader(f):
            task = r["task"]
            cfg = TASK_CFG[task]
            obj = fnum(r.get(cfg["obj"]))
            base = fnum(r.get("baseline_" + cfg["obj"]))
            if obj is None or base is None:
                continue
            ntpu, mapped = load_metadata(r["candidate"])
            row = {
                "task": task,
                "partition_id": r["candidate"],
                "t_total_ms": fnum(r["t_total_mean_ms"]),
                "objective_metric": obj,
                "baseline_metric": base,
                "accuracy_drop": base - obj,          # higher-is-better convention
                "num_tpu_ops": ntpu,
                "tpu_ops_mapped": mapped,
                "num_frames": int(r["num_frames"]),
                "buggy": r["candidate"] in PARTITION_BUG,
            }
            if task == "dlc":  # carry RMSE (lower-is-better) for the table
                rmse, brmse = fnum(r.get("rmse")), fnum(r.get("baseline_rmse"))
                row["rmse"] = rmse
                row["baseline_rmse"] = brmse
                row["rmse_drop"] = (rmse - brmse) if (rmse is not None and brmse is not None) else None
            rows.append(row)
    return rows


def pareto_optimal(points):
    """points: list of (latency, accuracy, id). Lower latency + higher accuracy = better.
    Returns set of ids that are non-dominated."""
    opt = set()
    for lat_i, acc_i, id_i in points:
        dominated = False
        for lat_j, acc_j, id_j in points:
            if id_j == id_i:
                continue
            # j dominates i if j is no worse on both and strictly better on one
            if lat_j <= lat_i and acc_j >= acc_i and (lat_j < lat_i or acc_j > acc_i):
                dominated = True
                break
        if not dominated:
            opt.add(id_i)
    return opt


def short(pid):
    return pid.replace("dlc_split_", "").replace("ssd_split_", "").replace("deeplab_split_", "")


# ---------------------------------------------------------------- tables
def write_tables(rows):
    for task, cfg in TASK_CFG.items():
        trows = sorted([r for r in rows if r["task"] == task], key=lambda x: x["t_total_ms"])
        base_cols = ["task", "partition_id", "t_total_ms", "objective_metric",
                     "baseline_metric", "accuracy_drop", "num_tpu_ops"]
        extra = ["rmse", "baseline_rmse", "rmse_drop"] if task == "dlc" else []
        cols = base_cols + extra + ["tpu_ops_mapped"]
        # csv
        with open(os.path.join(OUT, f"table_{task}.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            for r in trows:
                w.writerow(r)
        # md
        with open(os.path.join(OUT, f"table_{task}.md"), "w") as f:
            f.write(f"### {cfg['name']} — objective = {cfg['obj_label']}\n\n")
            hdr = ["partition_id", "t_total_ms", cfg["obj_label"], "baseline",
                   "accuracy_drop", "num_tpu_ops", "mapped"]
            if task == "dlc":
                hdr += ["rmse", "rmse_drop"]
            f.write("| " + " | ".join(hdr) + " |\n")
            f.write("|" + "|".join(["---"] * len(hdr)) + "|\n")
            for r in trows:
                flag = " ⚠️" if r["buggy"] else ""
                cells = [short(r["partition_id"]) + flag,
                         f"{r['t_total_ms']:.1f}",
                         f"{r['objective_metric']:.4f}",
                         f"{r['baseline_metric']:.4f}",
                         f"{r['accuracy_drop']:+.4f}",
                         str(r["num_tpu_ops"]),
                         str(r["tpu_ops_mapped"])]
                if task == "dlc":
                    cells += [f"{r['rmse']:.2f}", f"{r['rmse_drop']:+.3f}"]
                f.write("| " + " | ".join(cells) + " |\n")
        print(f"[table] {task}: {len(trows)} candidates")


# ---------------------------------------------------------------- plots
def plot_task(rows, task, mode):
    cfg = TASK_CFG[task]
    trows = [r for r in rows if r["task"] == task]
    pts = [(r["t_total_ms"], r["objective_metric"], r["partition_id"]) for r in trows]
    opt = pareto_optimal(pts)

    fig, ax = plt.subplots(figsize=(8, 6))
    for r in trows:
        x = r["t_total_ms"]
        y = r["accuracy_drop"] if mode == "drop" else r["objective_metric"]
        on_front = r["partition_id"] in opt
        color = "tab:red" if r["buggy"] else ("tab:blue" if on_front else "lightgray")
        marker = "o"
        ax.scatter(x, y, s=90, c=color,
                   edgecolors="black" if on_front else "gray",
                   linewidths=1.5 if on_front else 0.6,
                   zorder=3 if on_front else 2)
        lbl = short(r["partition_id"]) + (" ⚠️" if r["buggy"] else "")
        ax.annotate(lbl, (x, y), fontsize=7, xytext=(4, 4),
                    textcoords="offset points")

    if mode == "acc":
        # draw Pareto frontier line (sort optimal pts by latency)
        front = sorted([(r["t_total_ms"], r["objective_metric"])
                        for r in trows if r["partition_id"] in opt])
        if len(front) > 1:
            ax.plot([p[0] for p in front], [p[1] for p in front],
                    "--", c="tab:blue", lw=1.3, zorder=1, label="Pareto frontier")
            ax.legend(loc="best", fontsize=8)
        ax.set_ylabel(f"{cfg['obj_label']} (higher = better)")
        ax.set_title(f"{cfg['name']}: latency vs accuracy")
        fname = f"latency_vs_accuracy_{task}.png"
    else:
        ax.axhline(0, color="black", lw=0.6, ls=":")
        ax.set_ylabel(f"accuracy_drop in {cfg['obj_label']} (lower = better)")
        ax.set_title(f"{cfg['name']}: latency vs accuracy_drop  (lower-left = best)")
        fname = f"latency_vs_drop_{task}.png"

    ax.set_xlabel("t_total latency (ms)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, fname), dpi=130)
    plt.close(fig)
    print(f"[plot] {fname}  pareto={sorted(short(i) for i in opt)}")
    return opt


# ---------------------------------------------------------------- strategies
def strategy_comparison(rows, pareto_by_task):
    out = []
    for task, cfg in TASK_CFG.items():
        trows = [r for r in rows if r["task"] == task]
        # exclude partition-bug candidates from accuracy/Pareto strategy reasoning
        clean = [r for r in trows if not r["buggy"]]

        max_tpu = max(trows, key=lambda r: r["num_tpu_ops"])
        best_acc = min(clean, key=lambda r: r["accuracy_drop"])  # smallest drop = best
        # best pareto: among clean Pareto-optimal pts, pick best practical tradeoff:
        # smallest drop among the fastest half, else min (latency-normalized) score.
        opt = [r for r in clean if r["partition_id"] in pareto_by_task[task]]
        if not opt:
            opt = clean
        # practical pick: minimize latency*(1+|drop|) heuristic, favouring low both
        best_pareto = min(opt, key=lambda r: r["t_total_ms"] * (1 + abs(r["accuracy_drop"])))

        for strat, r in [("max_tpu", max_tpu), ("best_accuracy", best_acc),
                         ("best_pareto", best_pareto)]:
            out.append({
                "task": task,
                "strategy": strat,
                "selected_partition": r["partition_id"],
                "latency_ms": round(r["t_total_ms"], 1),
                "accuracy_metric": f"{cfg['obj_label']}={r['objective_metric']:.4f}",
                "accuracy_drop": round(r["accuracy_drop"], 4),
                "num_tpu_ops": r["num_tpu_ops"],
            })
        # sanity check vs handoff
        exp = EXPECTED_MAX_TPU[task]
        flag = "OK" if max_tpu["partition_id"] == exp else f"MISMATCH (expected {exp})"
        print(f"[strategy] {task}: max_tpu={short(max_tpu['partition_id'])} [{flag}] "
              f"best_acc={short(best_acc['partition_id'])} "
              f"best_pareto={short(best_pareto['partition_id'])}")

    cols = ["task", "strategy", "selected_partition", "latency_ms",
            "accuracy_metric", "accuracy_drop", "num_tpu_ops"]
    with open(os.path.join(OUT, "strategy_comparison.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    with open(os.path.join(OUT, "strategy_comparison.md"), "w") as f:
        f.write("### Strategy comparison\n\n")
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"] * len(cols)) + "|\n")
        for r in out:
            f.write("| " + " | ".join(str(r[c]) for c in cols) + " |\n")
    return out


def main():
    rows = load_rows()
    print(f"Loaded {len(rows)} TPU candidates with accuracy.")
    write_tables(rows)
    pareto_by_task = {}
    for task in TASK_CFG:
        pareto_by_task[task] = plot_task(rows, task, "acc")
        plot_task(rows, task, "drop")
    strategy_comparison(rows, pareto_by_task)
    print("Done. Outputs in docs_student_c/analysis/")


if __name__ == "__main__":
    main()
