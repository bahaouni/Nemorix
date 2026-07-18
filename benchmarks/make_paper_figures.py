#!/usr/bin/env python3
"""Generate the three paper figures as PDF and PNG.

Figures produced:
    assets/fig_scaling.pdf    – SLA agents vs fleet size (the 4.4× story)
  assets/fig_ablation.pdf   – CXL contribution (latency decomposition)
  assets/fig_latency.pdf    – Resume latency percentile comparison

Run: python benchmarks/make_paper_figures.py
"""
from __future__ import annotations
import json
import os
import sys

# ------------------------------------------------------------------ deps
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
except ImportError:
    sys.exit("matplotlib / numpy required. Run: pip install matplotlib numpy")

# ------------------------------------------------------------------ output dir
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)

# ------------------------------------------------------------------ style
COLORS = {
    "no_offload": "#d62728",   # red
    "lru":        "#1f77b4",   # blue
    "semantic":   "#2ca02c",   # green
    "no_cxl":     "#ff7f0e",   # orange
}
LABELS = {
    "no_offload": "No-Offload",
    "lru":        "LRU + 4-tier",
    "semantic":   "Nemorix (Semantic)",
    "no_cxl":     "LRU, no CXL",
}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "lines.linewidth": 1.6,
    "lines.markersize": 5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 200,
})

# ================================================================== FIG 1
# SLA agents vs. fleet size
# ==================================================================

def fig_scaling():
    """Line chart: agents meeting SLA vs total agent count."""
    # Data from benchmarks (re-run after 36 GB/s CXL fix)
    agent_counts  = [10, 25, 50, 75, 100, 200, 500]
    no_offload    = [ 0,  0,  0,  0,   0,   0,   0]
    lru_sla       = [10, 25, 50, 75, 100, 200,  63]
    semantic_sla  = [10, 25, 50, 75, 100, 200, 276]

    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    ax.plot(agent_counts, no_offload,  color=COLORS["no_offload"], marker="x",
            linestyle="--", label=LABELS["no_offload"])
    ax.plot(agent_counts, lru_sla,     color=COLORS["lru"],        marker="s",
            linestyle="-",  label=LABELS["lru"])
    ax.plot(agent_counts, semantic_sla,color=COLORS["semantic"],   marker="o",
            linestyle="-",  label=LABELS["semantic"])

    # Perfect-SLA diagonal reference
    ax.plot(agent_counts, agent_counts, color="gray", linewidth=0.8,
            linestyle=":", label="Perfect SLA")

    # 4.4× annotation at 500
    ax.annotate("4.4×\nimprovement",
                xy=(500, 276), xytext=(350, 140),
                fontsize=7.5, color=COLORS["semantic"],
                arrowprops=dict(arrowstyle="->", color=COLORS["semantic"],
                                lw=1.1, connectionstyle="arc3,rad=0.25"))

    ax.set_xlabel("Number of agents")
    ax.set_ylabel("Agents under 200 ms SLA")
    ax.set_title("(a) SLA-meeting agents vs. fleet size", pad=4)
    ax.legend(loc="upper left", framealpha=0.8, edgecolor="none")
    ax.set_xticks(agent_counts)
    ax.set_xticklabels([str(x) for x in agent_counts], rotation=30, ha="right")
    ax.set_xlim(0, 520)
    ax.set_ylim(-5, 520)
    ax.grid(axis="y", linewidth=0.4, color="#dddddd")

    fig.tight_layout(pad=0.5)
    out = os.path.join(ASSETS, "fig_scaling.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ================================================================== FIG 2
# CXL ablation — latency contribution decomposition
# ==================================================================

def fig_ablation():
    """Horizontal bar chart: latency decomposition showing CXL contribution."""
    configs = [
        ("No-Offload\n(recompute)",     1205.3, COLORS["no_offload"]),
        ("LRU,\nno CXL",                  37.2, COLORS["no_cxl"]),
        ("LRU\n+ CXL",                    16.8, COLORS["lru"]),
        ("Nemorix\n+ CXL",                16.2, COLORS["semantic"]),
    ]
    labels = [c[0] for c in configs]
    values = [c[1] for c in configs]
    colors = [c[2] for c in configs]

    fig, ax = plt.subplots(figsize=(3.5, 2.5))

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, height=0.55, zorder=2)

    # Value labels
    for bar, val in zip(bars, values):
        x_pos = val + 10
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} ms", va="center", ha="left",
                fontsize=7.5, color="#333333")

    # Annotate the CXL speedup arrow
    ax.annotate("",
                xy=(16.8, 1.65), xytext=(37.2, 1.65),
                arrowprops=dict(arrowstyle="<->", color=COLORS["lru"], lw=1.3))
    ax.text(25.0, 1.35, "2.2× (CXL)", fontsize=7, color=COLORS["lru"],
            ha="center", va="top")

    ax.set_xscale("log")
    ax.set_xlim(5, 6000)
    ax.set_xlabel("Mean resume latency (ms, log scale)")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_title("(b) CXL contribution (50 agents, seed 42)", pad=4)
    ax.axvline(200, color="#999999", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(205, 3.55, "SLA\n200 ms", fontsize=6.5, color="#888888", va="top")
    ax.grid(axis="x", linewidth=0.4, color="#dddddd", zorder=0)
    ax.invert_yaxis()

    fig.tight_layout(pad=0.5)
    out = os.path.join(ASSETS, "fig_ablation.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ================================================================== FIG 3
# Resume latency percentile comparison (3 policies, 50 agents, 8 seeds)
# ==================================================================

def fig_latency():
    """Grouped bar chart: mean / P95 / P99 across policies (8-seed mean±std)."""

    # Robustness data (8 seeds, 50 agents) — from run_robustness.py
    data = {
        "no_offload": {
            "mean": (1151.1, 41.7),
            "p95":  (1552.7, 35.6),
            "p99":  (1608.8, 25.6),
        },
        "lru": {
            "mean": (16.4, 0.4),
            "p95":  (25.8, 0.4),
            "p99":  (27.2, 0.5),
        },
        "semantic": {
            "mean": (15.6, 0.5),
            "p95":  (25.7, 0.6),
            "p99":  (27.1, 0.6),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(5.5, 2.6),
                              gridspec_kw={"width_ratios": [1, 2.8]})

    metrics = ["mean", "p95", "p99"]
    metric_labels = ["Mean", "P95", "P99"]
    x = np.arange(len(metrics))
    width = 0.26

    # ---------- left panel: No-Offload alone (different scale)
    ax0 = axes[0]
    no_vals = [data["no_offload"][m][0] for m in metrics]
    no_errs = [data["no_offload"][m][1] for m in metrics]
    ax0.bar(x, no_vals, width=0.55, color=COLORS["no_offload"],
            yerr=no_errs, capsize=3, error_kw=dict(lw=1, capthick=1))
    ax0.set_xticks(x)
    ax0.set_xticklabels(metric_labels)
    ax0.set_ylabel("Latency (ms)")
    ax0.set_title("No-Offload", fontsize=9, pad=3)
    ax0.set_ylim(0, 1850)
    ax0.axhline(200, color="#999", linestyle="--", linewidth=0.8)
    ax0.text(2.4, 215, "SLA", fontsize=6.5, color="#888")
    ax0.grid(axis="y", linewidth=0.4, color="#dddddd")

    # ---------- right panel: LRU + Nemorix (zoomed scale)
    ax1 = axes[1]
    for i, (policy, offset) in enumerate([("lru", -width / 2), ("semantic", width / 2)]):
        vals = [data[policy][m][0] for m in metrics]
        errs = [data[policy][m][1] for m in metrics]
        ax1.bar(x + offset, vals, width=width,
                color=COLORS[policy], label=LABELS[policy],
                yerr=errs, capsize=3, error_kw=dict(lw=1, capthick=1))

    ax1.set_xticks(x)
    ax1.set_xticklabels(metric_labels)
    ax1.set_title("LRU vs. Nemorix (tiered)", fontsize=9, pad=3)
    ax1.set_ylim(0, 38)
    ax1.axhline(200 / 7, color="#999", linestyle="--", linewidth=0.8)   # hidden SLA at 28.5
    ax1.set_ylabel("Latency (ms)")
    ax1.legend(loc="upper left", framealpha=0.8, edgecolor="none")
    ax1.grid(axis="y", linewidth=0.4, color="#dddddd")

    fig.suptitle("(c) Resume latency: mean ± std over 8 seeds, 50 agents",
                 fontsize=9, y=1.01)
    fig.tight_layout(pad=0.5)
    out = os.path.join(ASSETS, "fig_latency.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {out}")


# ================================================================== MAIN
if __name__ == "__main__":
    print("Generating paper figures...")
    fig_scaling()
    fig_ablation()
    fig_latency()
    print("Done. Files written to assets/")
