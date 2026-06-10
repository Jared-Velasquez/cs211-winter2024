#!/usr/bin/env python3
"""Stacked-bar latency breakdown (t_tpu / t_transfer / t_cpu) per partition split.

Reads Student D's Results Records and plots, for each task, where inference time
goes as the split moves shallow -> deep. Shows the core latency mechanism: deeper
splits move work from the slow float CPU suffix onto the fast int8 TPU prefix.

    ./run_in_env.sh python scripts/plot_latency_breakdown.py
"""
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "figures/latency_breakdown"
RESULTS = {
    "Pose (DLC ResNet-50)": "docs_student_d/results/partition_results_task_a_dlc.json",
    "Detection (SSD MobileNet V2)": "docs_student_d/results/partition_results_task_b_detection.json",
    "Segmentation (DeepLab V3)": "docs_student_d/results/partition_results_task_c_segmentation.json",
}

# Shallow -> deep ordering per task (by how much of the graph runs on the TPU).
ORDER = {
    "Pose (DLC ResNet-50)": [
        "dlc_split_after_block1", "dlc_split_after_block2", "dlc_split_after_block3",
        "dlc_split_in_block4_unit1", "dlc_split_after_block4",
        "dlc_split_at_bias_add", "dlc_split_at_conv2d_transpose",
    ],
    "Detection (SSD MobileNet V2)": [
        "split_after_conv_1", "split_after_expanded_conv_5", "split_after_expanded_conv_10",
        "split_after_expanded_conv_13", "split_before_postprocessor",
    ],
    "Segmentation (DeepLab V3)": [
        "split_after_expanded_conv_5", "split_after_expanded_conv_10",
        "split_after_expanded_conv_13",
    ],
}

# Colorblind-friendly: TPU (fast) green, transfer orange, CPU (slow) blue.
COLORS = {"tpu": "#2ca02c", "transfer": "#ff7f0e", "cpu": "#1f77b4"}


def short(pid):
    return (pid.replace("dlc_split_", "").replace("split_", "")
            .replace("after_", "").replace("expanded_conv_", "exp_conv_"))


def load(path):
    d = json.load(open(path))
    return d if isinstance(d, list) else d.get("records", d.get("results", d))


def plot_task(ax, task, recs):
    by_id = {r["partition_id"]: r for r in recs}
    ids = [p for p in ORDER[task] if p in by_id]
    tpu = [by_id[p]["latency_ms"]["tpu"] for p in ids]
    trans = [by_id[p]["latency_ms"]["transfer"] for p in ids]
    cpu = [by_id[p]["latency_ms"]["cpu"] for p in ids]
    total = [by_id[p]["latency_ms"]["total"] for p in ids]
    x = range(len(ids))

    ax.bar(x, tpu, color=COLORS["tpu"], label="TPU (int8 prefix)")
    ax.bar(x, trans, bottom=tpu, color=COLORS["transfer"], label="Transfer (boundary)")
    ax.bar(x, cpu, bottom=[t + tr for t, tr in zip(tpu, trans)],
           color=COLORS["cpu"], label="CPU (float32 suffix)")

    for i, tot in enumerate(total):
        ax.text(i, tot, f"{tot:.0f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_title(task, fontsize=11, fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels([short(p) for p in ids], rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Latency (ms)")
    ax.margins(y=0.12)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlabel("split  (shallow → deep)", fontsize=9)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    data = {task: load(path) for task, path in RESULTS.items()}

    # Combined 3-panel figure.
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, task in zip(axes, RESULTS):
        plot_task(ax, task, data[task])
    axes[0].legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.suptitle("Latency breakdown per partition split: TPU → transfer → CPU",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    combined = os.path.join(OUT_DIR, "latency_breakdown_all_tasks.png")
    fig.savefig(combined, dpi=150)
    plt.close(fig)
    print("wrote", combined)

    # Per-task standalone figures.
    for task in RESULTS:
        fig, ax = plt.subplots(figsize=(7, 5))
        plot_task(ax, task, data[task])
        ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
        fig.tight_layout()
        slug = task.split(" ")[0].lower()
        out = os.path.join(OUT_DIR, f"latency_breakdown_{slug}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
