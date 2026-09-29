#!/usr/bin/env python3
"""
plot_catastrophic_forgetting.py

Visualizes accuracy and forgetting dynamics across sequential task training:
1. Accuracy degradation curves across Stage 1, 2, and 3.
2. Final catastrophic forgetting summary (Accuracy Lost).
3. Final checkpoint accuracies by task recency (1st learned vs 2nd vs 3rd/Last).
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Color palette mapped consistently across all tasks
TASK_COLORS = {
    "A": "#2b5c8f",  # Blue (Aquatic Mammals)
    "B": "#d95f02",  # Orange (Electronics)
    "C": "#2ca02c",  # Green (Vehicles)
}

RESULTS_DIR = Path("results/sequential")
OUTPUT_DIR = Path("figures/catastrophic_forgetting")


def load_all_sequential_metrics(results_dir: Path):
    """Load metrics.json from all sequential run directories."""
    all_data = {}
    if not results_dir.exists():
        print(f"Error: Directory {results_dir} not found.")
        return all_data

    for order_dir in sorted([p for p in results_dir.iterdir() if p.is_dir()]):
        metrics_file = order_dir / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                all_data[order_dir.name] = json.load(f)
    return all_data


def plot_evolution_grid(all_metrics: dict, output_dir: Path):
    """Plot 2x3 grid tracking accuracy drop over stages for each order."""
    orders = sorted(all_metrics.keys())
    if not orders:
        return

    sns.set_theme(style="whitegrid", font_scale=1.0)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    axes = axes.flatten()

    for idx, order_name in enumerate(orders):
        ax = axes[idx]
        data = all_metrics[order_name]
        order_sequence = data.get("order", order_name.split("_"))

        task_trajectories = {}
        for stage_info in data.get("stages", []):
            stage_idx = stage_info["stage"]
            for eval_entry in stage_info.get("evaluations", []):
                t = eval_entry["task"]
                acc = eval_entry["accuracy"] * 100.0
                task_trajectories.setdefault(t, []).append((stage_idx, acc))

        for task_name, points in task_trajectories.items():
            stages = [p[0] for p in points]
            accs = [p[1] for p in points]
            color = TASK_COLORS.get(task_name, "#333333")

            ax.plot(
                stages,
                accs,
                marker="o",
                linewidth=2.5,
                markersize=7,
                label=f"Task {task_name}",
                color=color,
            )

            for s, a in zip(stages, accs):
                ax.annotate(
                    f"{a:.1f}%",
                    (s, a),
                    textcoords="offset points",
                    xytext=(0, 7),
                    ha="center",
                    fontsize=8.5,
                    fontweight="semibold",
                )

        ax.axhline(20.0, linestyle="--", color="gray", alpha=0.6, label="Chance (20%)" if idx == 0 else "")
        ax.set_title(f"Order: {' → '.join(order_sequence)}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Stage 1", "Stage 2", "Stage 3"])
        ax.set_ylim(10, 95)
        ax.set_xlabel("Training Stage")
        if idx % 3 == 0:
            ax.set_ylabel("Test Accuracy (%)")

        if idx == 0:
            ax.legend(loc="lower left", framealpha=0.9)

    plt.suptitle("Test Accuracy Trajectories Across Sequential Stages", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    evolution_path = output_dir / "forgetting_evolution_all_orders.png"
    plt.savefig(evolution_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {evolution_path}")


def plot_forgetting_summary(all_metrics: dict, output_dir: Path):
    """Plot bar chart of absolute catastrophic forgetting per task."""
    records = []
    for order_name, data in all_metrics.items():
        order_sequence = data.get("order", order_name.split("_"))
        forgetting_dict = data.get("forgetting", {})
        for task, drop in forgetting_dict.items():
            task_position = order_sequence.index(task) + 1
            records.append({
                "Order": " → ".join(order_sequence),
                "Task": f"Task {task}",
                "Sequence Position": f"Position {task_position} in order",
                "Accuracy Drop (%)": drop * 100.0,
            })

    if not records:
        return

    df = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(12, 5.5))
    sns.barplot(
        data=df,
        x="Order",
        y="Accuracy Drop (%)",
        hue="Task",
        palette=[TASK_COLORS["A"], TASK_COLORS["B"], TASK_COLORS["C"]],
        ax=ax,
    )

    ax.set_title("Catastrophic Forgetting per Task Across All Sequences", fontsize=13, fontweight="bold", pad=15)
    ax.set_ylabel("Accuracy Lost (Initial % - Final %)", fontsize=11)
    ax.set_xlabel("Sequential Training Order", fontsize=11)
    ax.set_ylim(0, 40)
    plt.xticks(rotation=20)
    ax.legend(title="Evaluated Task", loc="upper right")

    for p in ax.patches:
        val = p.get_height()
        if val > 0.5:
            ax.annotate(
                f"{val:.1f}%",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8,
                fontweight="semibold",
            )

    plt.tight_layout()
    summary_path = output_dir / "catastrophic_forgetting_bar_summary.png"
    plt.savefig(summary_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {summary_path}")


def plot_final_checkpoint_accuracies(all_metrics: dict, output_dir: Path):
    """
    Extracts and plots final checkpoint (Stage 3) test accuracy for all tasks.
    Visualizes whether: Accuracy(Task 1) < Accuracy(Task 2) < Accuracy(Task 3).
    """
    records = []

    for order_name, data in all_metrics.items():
        order_sequence = data.get("order", order_name.split("_"))
        stages = data.get("stages", [])
        if not stages:
            continue

        # Stage 3 is the final model evaluated on all tasks
        final_stage = stages[-1]
        for eval_entry in final_stage.get("evaluations", []):
            task = eval_entry["task"]
            acc = eval_entry["accuracy"] * 100.0
            order_idx = order_sequence.index(task) + 1  # 1st, 2nd, or 3rd learned

            position_label = {
                1: "1st Learned Task",
                2: "2nd Learned Task",
                3: "3rd Learned Task (Last)",
            }[order_idx]

            records.append({
                "Order": " → ".join(order_sequence),
                "Task": f"Task {task}",
                "Raw Task": task,
                "Position Index": order_idx,
                "Learning Recency": position_label,
                "Final Test Accuracy (%)": acc,
            })

    if not records:
        return

    df = pd.DataFrame(records)

    # -------------------------------------------------------------
    # Graphic A: Grouped by Learning Recency (The Hypothesis Test)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    recency_palette = ["#9ecae1", "#4292c6", "#084594"]  # Light to Dark Blue

    sns.barplot(
        data=df,
        x="Order",
        y="Final Test Accuracy (%)",
        hue="Learning Recency",
        palette=recency_palette,
        ax=ax,
    )

    ax.axhline(20.0, linestyle="--", color="gray", alpha=0.7, label="Chance Level (20%)")
    ax.set_title(
        "Final Model Test Accuracy Grouped by Training Recency\n"
        "(Hypothesis: 1st Learned < 2nd Learned < 3rd Learned / Last)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=11)
    ax.set_xlabel("Sequential Training Order", fontsize=11)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    ax.legend(title="Training Recency", loc="upper left", framealpha=0.9)

    for p in ax.patches:
        val = p.get_height()
        if val > 1.0:
            ax.annotate(
                f"{val:.1f}%",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="semibold",
            )

    plt.tight_layout()
    recency_path = output_dir / "final_checkpoint_accuracies_by_recency.png"
    plt.savefig(recency_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {recency_path}")

    # -------------------------------------------------------------
    # Graphic B: Grouped by Explicit Task Identity (A, B, C)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(13, 6))
    sns.barplot(
        data=df,
        x="Order",
        y="Final Test Accuracy (%)",
        hue="Task",
        palette=[TASK_COLORS["A"], TASK_COLORS["B"], TASK_COLORS["C"]],
        ax=ax,
    )

    ax.axhline(20.0, linestyle="--", color="gray", alpha=0.7, label="Chance Level (20%)")
    ax.set_title(
        "Final Model Test Accuracy by Specific Task Identity (A, B, C)",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.set_ylabel("Final Test Accuracy (%)", fontsize=11)
    ax.set_xlabel("Sequential Training Order", fontsize=11)
    ax.set_ylim(0, 100)
    plt.xticks(rotation=20)
    ax.legend(title="Evaluated Task", loc="upper right", framealpha=0.9)

    for p in ax.patches:
        val = p.get_height()
        if val > 1.0:
            ax.annotate(
                f"{val:.1f}%",
                (p.get_x() + p.get_width() / 2.0, val),
                ha="center",
                va="center",
                xytext=(0, 6),
                textcoords="offset points",
                fontsize=8.5,
                fontweight="semibold",
            )

    plt.tight_layout()
    task_path = output_dir / "final_checkpoint_accuracies_by_task.png"
    plt.savefig(task_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {task_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_data = load_all_sequential_metrics(RESULTS_DIR)

    if not metrics_data:
        print(f"No metric files found in {RESULTS_DIR}. Please check the path.")
        return

    print(f"Found metrics for {len(metrics_data)} sequential orders: {list(metrics_data.keys())}")
    plot_evolution_grid(metrics_data, OUTPUT_DIR)
    plot_forgetting_summary(metrics_data, OUTPUT_DIR)
    plot_final_checkpoint_accuracies(metrics_data, OUTPUT_DIR)
    print("Visualizations complete.")


if __name__ == "__main__":
    main()