#!/usr/bin/env python3
"""Conceptual schematic of a TPU/CPU partition (the dataflow, not the results).

Draws the core mechanism every candidate in this project implements:

    input -> TPU int8 prefix -> dequantize boundary -> CPU float32 suffix -> output

It is deliberately a *structural* figure (no measured numbers) to sit next to the
latency-breakdown and Pareto plots in the report. Palette matches
scripts/plot_latency_breakdown.py: TPU green, transfer/boundary orange, CPU blue.

    ./run_in_env.sh python scripts/plot_split_schematic.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT_DIR = "figures"
OUT = os.path.join(OUT_DIR, "tpu_cpu_split_schematic.png")

# Same colorblind-friendly palette as the latency breakdown figure.
C_TPU = "#2ca02c"       # int8 prefix (fast)
C_TRANSFER = "#ff7f0e"  # boundary dequant + handoff
C_CPU = "#1f77b4"       # float32 suffix (slow)
C_IO = "#555555"        # input / output terminals


def box(ax, x, y, w, h, color, title, subtitle, text_color="white"):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=0, facecolor=color, alpha=0.95, zorder=2))
    ax.text(x + w / 2, y + h * 0.62, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=text_color, zorder=3)
    ax.text(x + w / 2, y + h * 0.30, subtitle, ha="center", va="center",
            fontsize=8.5, color=text_color, zorder=3)


def arrow(ax, x0, x1, y, label=None, color="#333333"):
    ax.add_patch(FancyArrowPatch(
        (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=18,
        linewidth=2.2, color=color, zorder=1))
    if label:
        ax.text((x0 + x1) / 2, y + 0.16, label, ha="center", va="bottom",
                fontsize=8, color=color, style="italic")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.set_xlim(0, 11.4)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    yb, h = 1.2, 1.3

    # Device backdrops: which physical chip runs each stage.
    ax.add_patch(FancyBboxPatch((0.95, 0.75), 2.6, 2.25,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.4, edgecolor=C_TPU, facecolor=C_TPU, alpha=0.06, zorder=0))
    ax.text(2.25, 2.82, "Edge TPU", ha="center", fontsize=9.5,
            fontweight="bold", color=C_TPU)

    ax.add_patch(FancyBboxPatch((4.55, 0.75), 5.5, 2.25,
                 boxstyle="round,pad=0.02,rounding_size=0.06",
                 linewidth=1.4, edgecolor=C_CPU, facecolor=C_CPU, alpha=0.06, zorder=0))
    ax.text(7.3, 2.82, "Host CPU", ha="center", fontsize=9.5,
            fontweight="bold", color=C_CPU)

    # Stage boxes.
    box(ax, 0.1, yb + 0.25, 0.85, 0.8, C_IO, "Input", "image", text_color="white")
    box(ax, 1.15, yb, 2.2, h, C_TPU, "Prefix subgraph", "int8  ·  t_tpu")
    box(ax, 4.7, yb, 1.7, h, C_TRANSFER, "Dequantize", "int8 -> float32")
    box(ax, 6.7, yb, 2.2, h, C_CPU, "Suffix subgraph", "float32  ·  t_cpu")
    box(ax, 9.2, yb + 0.25, 1.0, 0.8, C_IO, "Output", "pose / box / mask",
        text_color="white")

    # Arrows between stages. The TPU->CPU hop is the partition boundary.
    yc = yb + h / 2
    arrow(ax, 0.97, 1.13, yc)
    arrow(ax, 3.37, 4.68, yc, label="int8 boundary\ntensor", color=C_TRANSFER)
    arrow(ax, 6.42, 6.68, yc)
    arrow(ax, 8.92, 9.18, yc)

    # Boundary marker: the vertical cut where the graph is split.
    ax.axvline(4.05, ymin=0.12, ymax=0.86, color="#222222", linestyle="--",
               linewidth=1.4, zorder=4)
    ax.text(4.05, 0.42, "partition boundary\n(t_transfer)", ha="center",
            va="center", fontsize=8.5, fontweight="bold", color="#222222")

    ax.set_title("TPU / CPU model partition: prefix runs int8 on the Edge TPU, "
                 "suffix runs float32 on the CPU", fontsize=11.5, pad=12)

    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
