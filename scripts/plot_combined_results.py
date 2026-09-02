import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
except Exception:
    sns = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = PROJECT_ROOT / "results" / "combined" / "combined_analysis.json"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "combined"


def load_combined_results(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_accuracy_summary(data):
    orders = data.get("orders", [])
    total = max(len(orders), 1)

    method_defs = [
        ("Full model", "weight_distance", "full_model"),
        ("Backbone", "weight_distance", "backbone"),
        ("CKA", "representation", "cka_correct"),
        ("Feature drift", "representation", "drift_correct"),
    ]

    results = []
    for label, section, field in method_defs:
        correct = 0
        for order in orders:
            if section == "weight_distance":
                correct += int(order[section][field]["correct"])
            else:
                if field == "cka_correct":
                    correct += int(order[section]["cka_correct"])
                else:
                    correct += int(order[section]["drift_correct"])

        accuracy = 100.0 * correct / total
        results.append(
            {
                "label": label,
                "correct": correct,
                "accuracy": accuracy,
            }
        )

    return results


def plot_accuracy_summary(summary, path):
    labels = [item["label"] for item in summary]
    values = [item["accuracy"] for item in summary]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(labels, values, color=["#4C72B0", "#55A868", "#C44E52", "#8172B3"])
    plt.axhline(0, color="black", linewidth=0.7)
    plt.ylabel("Prediction accuracy (%)")
    plt.title("Last-task prediction accuracy by method")
    plt.ylim(0, 100)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_forgetting_by_order(data, path):
    orders = data.get("orders", [])
    tasks = sorted({task for order in orders for task in order.get("forgetting", {})})

    plt.figure(figsize=(9, 6))
    for order in orders:
        values = [order["forgetting"].get(task, 0.0) for task in tasks]
        plt.plot(tasks, values, marker="o", linewidth=2, label=f"{order['order_name']}")

    plt.axhline(0, color="black", linewidth=0.8, linestyle="--")
    plt.xlabel("Task")
    plt.ylabel("Forgetting")
    plt.title("Forgetting across tasks for each order")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_correctness_heatmap(data, path):
    orders = data.get("orders", [])
    order_names = [order["order_name"] for order in orders]

    method_names = ["Full model", "Backbone", "CKA", "Feature drift"]
    matrix = []

    for order in orders:
        row = [
            int(order["weight_distance"]["full_model"]["correct"]),
            int(order["weight_distance"]["backbone"]["correct"]),
            int(order["representation"]["cka_correct"]),
            int(order["representation"]["drift_correct"]),
        ]
        matrix.append(row)

    matrix = np.array(matrix)

    plt.figure(figsize=(8, 5.5))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=method_names,
            yticklabels=order_names,
            cbar=False,
            linewidths=0.5,
            linecolor="white",
        )
        plt.title("Prediction correctness by order and method")
        plt.xlabel("Method")
        plt.ylabel("Order")
    else:
        plt.imshow(matrix, cmap="Blues")
        plt.xticks(range(len(method_names)), method_names, rotation=20)
        plt.yticks(range(len(order_names)), order_names)
        plt.title("Prediction correctness by order and method")
        plt.xlabel("Method")
        plt.ylabel("Order")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(
                    j, i, int(matrix[i, j]), ha="center", va="center", color="black"
                )

    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


def get_loss_barrier_entries(data):
    entries = []
    for order in data.get("orders", []):
        barriers = order.get("loss_barrier", {}).get("barriers", {})
        if barriers:
            entries.append((order["order_name"], barriers))
    return entries


def plot_loss_barrier_curves(data, path):
    entries = get_loss_barrier_entries(data)
    if not entries:
        return False

    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    axes = axes.flatten()
    for axis, (order_name, barriers) in zip(axes, entries):
        for reference_name, task_barriers in barriers.items():
            for task, values in task_barriers.items():
                alphas = values.get("alphas", [])
                losses = values.get("losses", [])
                if alphas and losses:
                    axis.plot(
                        alphas,
                        losses,
                        marker="o",
                        markersize=2.5,
                        linewidth=1.2,
                        label=f"{reference_name}, task {task}",
                    )
        axis.set_title(order_name.replace("_", " -> "))
        axis.set_xlabel("Interpolation alpha")
        axis.grid(alpha=0.25)

    for axis in axes[: len(entries)]:
        axis.legend(fontsize=6, loc="best")
    for axis in axes:
        axis.set_ylabel("Test loss")
    figure.suptitle("Loss-barrier curves by sequential order")
    figure.tight_layout()
    figure.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return True


def plot_loss_barrier_summary(data, path):
    entries = get_loss_barrier_entries(data)
    if not entries:
        return False

    order_names = [item[0] for item in entries]
    references = sorted(
        {reference_name for _, barriers in entries for reference_name in barriers}
    )
    matrix = []
    for _, barriers in entries:
        row = []
        for reference_name in references:
            values = [
                task_data.get("barrier_height")
                for task_data in barriers.get(reference_name, {}).values()
                if task_data.get("barrier_height") is not None
            ]
            row.append(float(np.mean(values)) if values else np.nan)
        matrix.append(row)

    matrix = np.array(matrix)
    plt.figure(figsize=(9, 5.5))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="magma",
            xticklabels=references,
            yticklabels=order_names,
            linewidths=0.5,
            linecolor="white",
        )
    else:
        plt.imshow(matrix, cmap="magma", aspect="auto")
        plt.xticks(range(len(references)), references)
        plt.yticks(range(len(order_names)), order_names)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                if not np.isnan(matrix[row_index, column_index]):
                    plt.text(
                        column_index,
                        row_index,
                        f"{matrix[row_index, column_index]:.3f}",
                        ha="center",
                        va="center",
                    )
    plt.title("Mean loss-barrier height by order and reference")
    plt.xlabel("Reference model")
    plt.ylabel("Sequential order")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def get_jacobian_entries(data):
    entries = []
    for order in data.get("orders", []):
        jacobians = order.get("jacobian", {}).get("jacobians", {})
        sequential_entries = {
            key: value
            for key, value in jacobians.items()
            if key.startswith("Sequential-")
        }
        if sequential_entries:
            entries.append((order["order_name"], sequential_entries))
    return entries


def plot_jacobian_sensitivity(data, path):
    entries = get_jacobian_entries(data)
    if not entries:
        return False

    tasks = sorted(
        {
            result.get("task")
            for _, jacobians in entries
            for result in jacobians.values()
            if result.get("task")
        }
    )
    x = np.arange(len(entries))
    width = 0.8 / max(len(tasks), 1)
    plt.figure(figsize=(11, 6))
    for task_index, task in enumerate(tasks):
        values = []
        for _, jacobians in entries:
            task_results = [
                result for result in jacobians.values() if result.get("task") == task
            ]
            values.append(
                np.mean(
                    [
                        result.get("metrics", {}).get("mean_sensitivity", np.nan)
                        for result in task_results
                    ]
                )
            )
        plt.bar(x + task_index * width, values, width=width, label=f"Task {task}")

    plt.xticks(
        x + width * (len(tasks) - 1) / 2,
        [name.replace("_", " -> ") for name, _ in entries],
        rotation=25,
    )
    plt.ylabel("Mean Jacobian sensitivity")
    plt.title("Jacobian sensitivity by sequential order and task")
    plt.legend()
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def plot_jacobian_channels(data, path):
    entries = get_jacobian_entries(data)
    if not entries:
        return False

    rows = []
    labels = []
    for order_name, jacobians in entries:
        for task in sorted({result.get("task") for result in jacobians.values()}):
            task_results = [
                result for result in jacobians.values() if result.get("task") == task
            ]
            rows.append(
                [
                    np.mean(
                        [
                            result.get("channel_sensitivity", {}).get(channel, np.nan)
                            for result in task_results
                        ]
                    )
                    for channel in ["R", "G", "B"]
                ]
            )
            labels.append(f"{order_name} / {task}")

    matrix = np.array(rows)
    plt.figure(figsize=(8, max(4, len(labels) * 0.35)))
    if sns is not None:
        sns.heatmap(
            matrix,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=["R", "G", "B"],
            yticklabels=labels,
            linewidths=0.5,
            linecolor="white",
        )
    else:
        plt.imshow(matrix, cmap="YlGnBu", aspect="auto")
        plt.xticks(range(3), ["R", "G", "B"])
        plt.yticks(range(len(labels)), labels)
    plt.title("Jacobian channel sensitivity")
    plt.xlabel("Input channel")
    plt.ylabel("Order / task")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()
    return True


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"Combined result file not found: {RESULT_PATH}")

    data = load_combined_results(RESULT_PATH)

    summary = compute_accuracy_summary(data)
    plot_accuracy_summary(summary, OUTPUT_DIR / "method_accuracy_summary.png")
    plot_forgetting_by_order(data, OUTPUT_DIR / "forgetting_by_order.png")
    plot_correctness_heatmap(data, OUTPUT_DIR / "method_correctness_heatmap.png")

    optional_plots = [
        ("loss_barrier_curves.png", plot_loss_barrier_curves),
        ("loss_barrier_summary.png", plot_loss_barrier_summary),
        ("jacobian_sensitivity_by_order.png", plot_jacobian_sensitivity),
        ("jacobian_channel_sensitivity.png", plot_jacobian_channels),
    ]
    generated_optional = []
    for filename, plotter in optional_plots:
        if plotter(data, OUTPUT_DIR / filename):
            generated_optional.append(filename)

    print(f"Loaded combined analysis from: {RESULT_PATH}")
    print(f"Saved summary plots to: {OUTPUT_DIR}")
    print("Generated:")
    print("- method_accuracy_summary.png")
    print("- forgetting_by_order.png")
    print("- method_correctness_heatmap.png")
    if generated_optional:
        print("Optional advanced plots:")
        for filename in generated_optional:
            print(f"- {filename}")
    else:
        print(
            "Optional advanced plots skipped: no loss-barrier or Jacobian results found."
        )


if __name__ == "__main__":
    main()
