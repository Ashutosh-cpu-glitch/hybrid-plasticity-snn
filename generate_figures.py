"""
generate_figures.py

Generates 7 figures for the report:
  1. Architecture diagram (schematic)
  2. Continual learning protocol (schematic)
  3. Forgetting comparison + ablation study (real data, bar chart with error bars)
  4. Gate evolution over tasks (real data, representative run)
  5. Task-1 accuracy trajectory across sequential training (real data)
  6. Fast vs. slow pathway isolated accuracy (real data)
  7. Replay mechanism (schematic)

Data is hardcoded from experiment logs (see REPORT.md) rather than
re-running training, since these are the final reported results.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import os

OUT_DIR = "figures"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.size": 11,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR_SLOW = "#4C72B0"
COLOR_FAST = "#DD8452"
COLOR_BASELINE = "#8C8C8C"
COLOR_HYBRID = "#55A868"
COLOR_ACCENT = "#C44E52"


# ---------------------------------------------------------------------
# Figure 1: Architecture diagram (schematic)
# ---------------------------------------------------------------------
def fig1_architecture():
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=10):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            linewidth=1.5, edgecolor="black",
                            facecolor=color, alpha=0.85)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, weight="bold", color="white")

    def arrow(x1, y1, x2, y2, label=""):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                             mutation_scale=15, linewidth=1.5, color="black")
        ax.add_patch(a)
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.15, label,
                     ha="center", fontsize=8, style="italic")

    box(0.5, 2.5, 1.8, 1, "Input\n(spike train)", "#666666")

    box(3, 4, 2.2, 1, "Slow Pathway\n(LIF + Backprop)", COLOR_SLOW)
    box(3, 1, 2.2, 1, "Fast Pathway\n(Delta-rule +\nReplay)", COLOR_FAST)

    box(6.3, 2.5, 1.4, 1, "Gate\n(\u03b1)", "#8172B2")

    box(8.3, 2.5, 1.3, 1, "Output\n(class)", "#666666")

    arrow(2.3, 3, 3, 4.3)
    arrow(2.3, 3, 3, 1.7)
    arrow(5.2, 4.5, 6.5, 3.3, "(1-\u03b1)")
    arrow(5.2, 1.5, 6.5, 2.7, "\u03b1")
    arrow(7.7, 3, 8.3, 3)

    ax.set_title("Figure 1: Hybrid SNN Architecture", fontsize=13, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig1_architecture.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Figure 2: Continual learning protocol (schematic)
# ---------------------------------------------------------------------
def fig2_protocol():
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")

    tasks = ["Task 1\n(0,1)", "Task 2\n(2,3)", "Task 3\n(4,5)",
             "Task 4\n(6,7)", "Task 5\n(8,9)"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2", "#CCB974"]

    for i, (t, c) in enumerate(zip(tasks, colors)):
        x = 0.5 + i * 2.3
        b = FancyBboxPatch((x, 1), 1.8, 1, boxstyle="round,pad=0.08",
                            linewidth=1.5, edgecolor="black",
                            facecolor=c, alpha=0.85)
        ax.add_patch(b)
        ax.text(x + 0.9, 1.5, t, ha="center", va="center",
                 fontsize=9, weight="bold", color="white")
        if i < len(tasks) - 1:
            a = FancyArrowPatch((x + 1.8, 1.5), (x + 2.3, 1.5),
                                 arrowstyle="-|>", mutation_scale=15,
                                 linewidth=1.5, color="black")
            ax.add_patch(a)

    ax.text(6, 2.5, "Sequential training, no access to old task data during training",
            ha="center", fontsize=9, style="italic")
    ax.set_title("Figure 2: Class-Incremental Continual Learning Protocol (Split-MNIST)",
                  fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig2_protocol.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Figure 3: Forgetting comparison + ablation (REAL DATA)
# ---------------------------------------------------------------------
def fig3_forgetting_ablation():
    variants = ["Baseline\n(backprop only)", "Hybrid\n(full)",
                "Hybrid\n(no replay)", "Hybrid\n(no gate)"]
    means = [0.9945, 0.5489, 0.6140, 0.5141]
    stds = [0.0003, 0.1739, 0.0202, 0.1118]
    colors = [COLOR_BASELINE, COLOR_HYBRID, "#DD8452", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(variants, means, yerr=stds, capsize=6, color=colors,
                   edgecolor="black", linewidth=1.2)
    for bar, m, s in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2, m + s + 0.02,
                 f"{m:.3f}", ha="center", fontsize=10, weight="bold")

    ax.set_ylabel("Average Forgetting (lower = better)")
    ax.set_ylim(0, 1.15)
    ax.set_title("Figure 3: Forgetting Comparison & Ablation Study\n(mean \u00b1 std over 5 seeds)",
                  fontsize=12, weight="bold")
    ax.axhline(0, color="black", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig3_forgetting_ablation.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Figure 4: Gate evolution over tasks (REAL DATA, representative run)
# ---------------------------------------------------------------------
def fig4_gate_evolution():
    tasks = [1, 2, 3, 4, 5]
    gate_values = [0.616, 0.565, 0.562, 0.562, 0.499]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(tasks, gate_values, marker="o", markersize=8, linewidth=2,
            color=COLOR_ACCENT)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1,
               label="\u03b1 = 0.5 (equal weight)")
    ax.set_xlabel("Task number (after training)")
    ax.set_ylabel("Gate value \u03b1 (0 = all slow, 1 = all fast)")
    ax.set_title("Figure 4: Gate Value Evolution Across Sequential Tasks\n(representative run)",
                  fontsize=12, weight="bold")
    ax.set_xticks(tasks)
    ax.set_ylim(0.4, 0.7)
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig4_gate_evolution.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Figure 5: Task-1 accuracy trajectory (REAL DATA)
# ---------------------------------------------------------------------
def fig5_task1_trajectory():
    tasks = [1, 2, 3, 4, 5]
    baseline_task1 = [0.999, 0.0, 0.0, 0.0, 0.0]
    hybrid_task1 = [1.0, 0.0, 0.0, 0.0, 0.964]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(tasks, baseline_task1, marker="s", markersize=8, linewidth=2,
            color=COLOR_BASELINE, label="Baseline")
    ax.plot(tasks, hybrid_task1, marker="o", markersize=8, linewidth=2,
            color=COLOR_HYBRID, label="Hybrid (full)")
    ax.set_xlabel("After training up to Task N")
    ax.set_ylabel("Task 1 accuracy")
    ax.set_title("Figure 5: Task 1 Accuracy as Later Tasks Are Learned\n(representative run)",
                  fontsize=12, weight="bold")
    ax.set_xticks(tasks)
    ax.set_ylim(-0.05, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    ax.annotate("Non-monotonic recovery\n(replay effect)", xy=(5, 0.964),
                xytext=(3.3, 0.75), fontsize=8, style="italic",
                arrowprops=dict(arrowstyle="->", color="gray"))
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig5_task1_trajectory.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Figure 6: Fast vs slow pathway isolated accuracy (REAL DATA)
# ---------------------------------------------------------------------
def fig6_fast_vs_slow():
    tasks = ["Task 1", "Task 2", "Task 3", "Task 4", "Task 5"]
    slow_acc = [0.000, 0.000, 0.000, 0.000, 0.491]
    fast_acc = [0.964, 0.136, 0.035, 0.000, 0.000]

    x = np.arange(len(tasks))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width / 2, slow_acc, width, label="Slow pathway only",
           color=COLOR_SLOW, edgecolor="black")
    ax.bar(x + width / 2, fast_acc, width, label="Fast pathway only",
           color=COLOR_FAST, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Accuracy (isolated pathway, after all 5 tasks)")
    ax.set_title("Figure 6: Isolated Fast vs Slow Pathway Accuracy\n(representative run)",
                  fontsize=12, weight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig6_fast_vs_slow.png", dpi=150)
    plt.close()


# ---------------------------------------------------------------------
# Figure 7: Replay mechanism (schematic)
# ---------------------------------------------------------------------
def fig7_replay_mechanism():
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, text, color, fontsize=9):
        b = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                            linewidth=1.5, edgecolor="black",
                            facecolor=color, alpha=0.85)
        ax.add_patch(b)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                 fontsize=fontsize, weight="bold", color="white")

    def arrow(x1, y1, x2, y2, style="-|>", color="black"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                             mutation_scale=15, linewidth=1.5, color=color)
        ax.add_patch(a)

    box(0.5, 3, 2.2, 1, "Task k training\ndata", COLOR_SLOW)
    box(0.5, 0.7, 2.2, 1, "Small exemplar\nbuffer (20/task)", "#CCB974")
    box(4, 3, 2.4, 1, "Fast pathway\n(current task)", COLOR_FAST)
    box(4, 0.7, 2.4, 1, "Fast pathway\n(REPLAY,\nold tasks)", COLOR_ACCENT)
    box(8, 1.85, 2.6, 1, "Fast pathway\nweights\n(shared)", "#8172B2")

    arrow(2.7, 3.5, 4, 3.5)
    arrow(2.7, 1.2, 4, 1.2)
    arrow(6.4, 3.4, 8, 2.6)
    arrow(6.4, 1.3, 8, 2.1)
    arrow(1.6, 3, 1.6, 1.7, style="-|>", color="gray")

    ax.text(1.6, 2.35, "store after\ntask ends", ha="center", fontsize=7,
            style="italic", color="gray")
    ax.text(6, 4.3, "Interleaved every few batches during new-task training",
            ha="center", fontsize=8, style="italic")

    ax.set_title("Figure 7: Replay Mechanism (Fast Pathway Only)",
                  fontsize=12, weight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/fig7_replay_mechanism.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    fig1_architecture()
    fig2_protocol()
    fig3_forgetting_ablation()
    fig4_gate_evolution()
    fig5_task1_trajectory()
    fig6_fast_vs_slow()
    fig7_replay_mechanism()
    print("All 7 figures generated in ./figures/")
