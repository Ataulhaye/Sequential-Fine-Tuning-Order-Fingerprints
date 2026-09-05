"""Create research-focused visualizations for sequential fine-tuning results."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results" / "combined" / "combined_analysis.json"
OUT = ROOT / "figures" / "combined"
TASKS = ("A", "B", "C")
ORDER_NAMES = ("A_B_C", "A_C_B", "B_A_C", "B_C_A", "C_A_B", "C_B_A")
METHODS = (
    ("Full-model weight distance", "full_model"),
    ("Backbone distance", "backbone"),
    ("CKA", "cka"),
    ("Feature drift", "drift"),
)
COLORS = {"A": "#2878B5", "B": "#E07B39", "C": "#3A9D5D"}


def label(order):
    return " -> ".join(order["order"])


def require(condition, message):
    if not condition:
        raise ValueError(f"Invalid combined analysis: {message}")


def load_validate(path):
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    orders = data.get("orders")
    require(
        isinstance(orders, list) and len(orders) == 6,
        "expected exactly 6 sequential orders",
    )
    require(
        tuple(order.get("order_name") for order in orders) == ORDER_NAMES,
        "orders must be the six expected permutations",
    )
    for order in orders:
        name, sequence = order["order_name"], order.get("order")
        require(
            isinstance(sequence, list)
            and len(sequence) == 3
            and set(sequence) == set(TASKS),
            f"{name}: invalid task order",
        )
        require(
            order.get("actual_last_task") == sequence[-1],
            f"{name}: actual last task does not match order",
        )
        require(
            set(order.get("forgetting", {})) == set(TASKS),
            f"{name}: forgetting needs A/B/C",
        )
        for variant in ("full_model", "backbone"):
            entry = order.get("weight_distance", {}).get(variant, {})
            require(
                set(entry.get("distances", {})) == set(TASKS)
                and entry.get("predicted_last_task") in TASKS,
                f"{name}: incomplete {variant} distance data",
            )
        rep = order.get("representation", {})
        require(
            set(rep.get("cka", {})) == set(TASKS)
            and set(rep.get("feature_drift", {})) == set(TASKS),
            f"{name}: incomplete representation data",
        )
        require(
            rep.get("predicted_last_task") in TASKS
            and rep.get("drift_predicted_last_task") in TASKS,
            f"{name}: missing representation prediction",
        )
        barriers = order.get("loss_barrier", {}).get("barriers", {})
        require(
            set(barriers) == {f"Single-{task}" for task in TASKS},
            f"{name}: incomplete loss barriers",
        )
        for reference, by_task in barriers.items():
            require(
                set(by_task) == set(TASKS), f"{name}: {reference} missing task barriers"
            )
            for task, entry in by_task.items():
                alphas, losses = entry.get("alphas"), entry.get("losses")
                require(
                    isinstance(alphas, list)
                    and len(alphas) > 1
                    and len(alphas) == len(losses),
                    f"{name}: {reference}/{task} alpha/loss mismatch",
                )
                require(
                    "barrier_height" in entry and "barrier_area" in entry,
                    f"{name}: {reference}/{task} missing barrier metrics",
                )
        jacobians = order.get("jacobian", {}).get("jacobians", {})
        for task in TASKS:
            sequential = [
                value
                for key, value in jacobians.items()
                if key.startswith("Sequential-") and value.get("task") == task
            ]
            require(
                len(sequential) == 1
                and "mean_sensitivity" in sequential[0].get("metrics", {}),
                f"{name}: missing sequential Jacobian task {task}",
            )
            require(
                set(sequential[0].get("channel_sensitivity", {})) == {"R", "G", "B"},
                f"{name}: incomplete RGB sensitivity for {task}",
            )
            single = jacobians.get(f"Single-{task}", {})
            require(
                single.get("task") == task
                and "mean_sensitivity" in single.get("metrics", {}),
                f"{name}: missing Single-{task} Jacobian",
            )
    print("Validation summary:")
    print(
        "Orders: 6/6\nWeight distance: complete\nRepresentation: complete\nLoss barrier: 6/6\nJacobian: 6/6"
    )
    return orders


def predicted(order, method):
    if method in ("full_model", "backbone"):
        return order["weight_distance"][method]["predicted_last_task"]
    return order["representation"][
        "predicted_last_task" if method == "cka" else "drift_predicted_last_task"
    ]


def save(fig, stem):
    names = []
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        name = f"{stem}.{suffix}"
        fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
        names.append(name)
    plt.close(fig)
    return names


def heatmap(ax, values, rows, columns, title, actual=None, cmap="YlGnBu"):
    image = ax.imshow(values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(columns)), columns, rotation=20, ha="right")
    ax.set_yticks(range(len(rows)), rows)
    ax.set_title(title)
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
            )
        if actual:
            col = TASKS.index(actual[row])
            ax.add_patch(
                Rectangle(
                    (col - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
            )
    return image


def plot_accuracy(orders):
    correct = [
        sum(predicted(order, method) == order["actual_last_task"] for order in orders)
        for _, method in METHODS
    ]
    values = np.array(correct) * 100 / 6
    fig, ax = plt.subplots(figsize=(9, 5.2))
    bars = ax.bar(
        [name for name, _ in METHODS],
        values,
        color=["#4C78A8", "#72B7B2", "#E45756", "#F2CF5B"],
    )
    ax.axhline(33.333, color="black", linestyle="--", label="Always-A baseline: 33.3%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Last-task prediction accuracy (%)")
    ax.set_title("Last-task prediction accuracy")
    ax.legend()
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%",
            ha="center",
            fontweight="bold",
        )
    return save(fig, "last_task_prediction_accuracy")


def plot_prediction_matrix(orders):
    columns = ["Actual last task"] + [name for name, _ in METHODS]
    labels = [
        [order["actual_last_task"]]
        + [predicted(order, method) for _, method in METHODS]
        for order in orders
    ]
    values = np.array(
        [[1] + [int(value == row[0]) for value in row[1:]] for row in labels]
    )
    fig, ax = plt.subplots(figsize=(11, 5.6))
    ax.imshow(
        values,
        cmap=ListedColormap(["#F4B6B2", "#B8E0C2"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    ax.set_xticks(range(5), columns, rotation=20, ha="right")
    ax.set_yticks(range(6), [label(order) for order in orders])
    ax.set_title("Predicted last task by sequential order")
    for row in range(6):
        for col in range(5):
            ax.text(
                col, row, labels[row][col], ha="center", va="center", fontweight="bold"
            )
    return save(fig, "prediction_matrix")


def plot_weight(orders):
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(6)
    width = 0.23
    for ax, variant, title in zip(
        axes,
        ("full_model", "backbone"),
        ("Full-model distance", "Backbone-only distance"),
    ):
        for offset, task in enumerate(TASKS):
            bars = ax.bar(
                x + (offset - 1) * width,
                [
                    order["weight_distance"][variant]["distances"][task]
                    for order in orders
                ],
                width,
                label=f"Single-{task}",
                color=COLORS[task],
            )
            for index, bar in enumerate(bars):
                if orders[index]["actual_last_task"] == task:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)
        ax.set_title(title)
        ax.set_ylabel("Weight distance")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(ncol=3)
    axes[-1].set_xticks(x, [label(order) for order in orders], rotation=18, ha="right")
    fig.suptitle(
        "Distance to single-task references (black outline = actual last task)", y=1.01
    )
    return save(fig, "weight_distance_scores")


def plot_representation(orders, key, title, stem, higher):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharey=True)
    for ax, order in zip(axes.flat, orders):
        values = [order["representation"][key][task] for task in TASKS]
        bars = ax.bar(TASKS, values, color=[COLORS[task] for task in TASKS])
        last = TASKS.index(order["actual_last_task"])
        bars[last].set_edgecolor("black")
        bars[last].set_linewidth(2.3)
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(f"{label(order)} (last: {order['actual_last_task']})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    fig.suptitle(
        f"{title} by final sequential model ({'higher' if higher else 'lower'} = greater similarity; black = actual last task)",
        y=1.01,
    )
    return save(fig, stem)


def plot_agreement(orders):
    text = [[predicted(order, method) for _, method in METHODS] for order in orders]
    values = np.array([[TASKS.index(value) for value in row] for row in text])
    fig, ax = plt.subplots(figsize=(8.5, 5.4))
    ax.imshow(
        values,
        cmap=ListedColormap([COLORS[task] for task in TASKS]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    ax.set_xticks(range(4), [name for name, _ in METHODS], rotation=20, ha="right")
    ax.set_yticks(
        range(6),
        [f"{label(order)} (actual: {order['actual_last_task']})" for order in orders],
    )
    ax.set_title("Method agreement on predicted last task")
    for row in range(6):
        for col in range(4):
            ax.text(
                col, row, text[row][col], ha="center", va="center", fontweight="bold"
            )
    return save(fig, "method_agreement")


def plot_forgetting(orders):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, order in zip(axes.flat, orders):
        bars = ax.bar(
            TASKS,
            [order["forgetting"][task] for task in TASKS],
            color=[COLORS[task] for task in TASKS],
        )
        last = TASKS.index(order["actual_last_task"])
        bars[last].set_edgecolor("black")
        bars[last].set_linewidth(2.3)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(f"{label(order)} (last: {order['actual_last_task']})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Forgetting")
    axes[1, 0].set_ylabel("Forgetting")
    fig.suptitle(
        "Forgetting by order and task (black outline = actual last task)", y=1.01
    )
    return save(fig, "forgetting_by_order")


def plot_loss(orders, metric=None):
    if metric:
        values = np.array(
            [
                [
                    order["loss_barrier"]["barriers"][f"Single-{reference}"][
                        order["actual_last_task"]
                    ][metric]
                    for reference in TASKS
                ]
                for order in orders
            ]
        )
        fig, ax = plt.subplots(figsize=(8, 5.5))
        image = heatmap(
            ax,
            values,
            [label(order) for order in orders],
            [f"Single-{task}" for task in TASKS],
            f"{metric.replace('_', ' ').title()} on actual last task",
            [order["actual_last_task"] for order in orders],
            "magma",
        )
        fig.colorbar(image, ax=ax, label=metric.replace("_", " "))
        return save(fig, f"loss_barrier_{metric.split('_')[1]}")
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, order in zip(axes.flat, orders):
        actual = order["actual_last_task"]
        for reference in TASKS:
            entry = order["loss_barrier"]["barriers"][f"Single-{reference}"][actual]
            ax.plot(
                entry["alphas"],
                entry["losses"],
                marker="o",
                markersize=3,
                linewidth=1.7,
                color=COLORS[reference],
                label=f"Single-{reference}",
            )
        ax.axvline(0, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_title(f"{label(order)}; evaluate task {actual}", fontsize=10)
        ax.set_xlabel("Alpha (0 = reference, 1 = sequential)")
        ax.set_ylabel("Loss")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("Loss-barrier curves on the actual last task", y=1.01)
    return save(fig, "loss_barrier_curves")


def sequential(order, task):
    return next(
        value
        for key, value in order["jacobian"]["jacobians"].items()
        if key.startswith("Sequential-") and value["task"] == task
    )


def plot_jacobian(orders):
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey=True)
    for ax, order in zip(axes.flat, orders):
        bars = ax.bar(
            TASKS,
            [sequential(order, task)["metrics"]["mean_sensitivity"] for task in TASKS],
            color=[COLORS[task] for task in TASKS],
        )
        last = TASKS.index(order["actual_last_task"])
        bars[last].set_edgecolor("black")
        bars[last].set_linewidth(2.3)
        ax.set_title(f"{label(order)} (last: {order['actual_last_task']})", fontsize=10)
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("Mean sensitivity")
    axes[1, 0].set_ylabel("Mean sensitivity")
    fig.suptitle(
        "Sequential-model Jacobian sensitivity (black outline = actual last task)",
        y=1.01,
    )
    return save(fig, "jacobian_sensitivity_by_order")


def plot_jacobian_references(orders):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    x = np.arange(6)
    for ax, task in zip(axes, TASKS):
        ax.bar(
            x,
            [
                sequential(order, task)["metrics"]["mean_sensitivity"]
                for order in orders
            ],
            color=COLORS[task],
            label=f"Sequential: task {task}",
        )
        for reference in TASKS:
            ax.plot(
                x,
                [
                    order["jacobian"]["jacobians"][f"Single-{reference}"]["metrics"][
                        "mean_sensitivity"
                    ]
                    for order in orders
                ],
                marker="o",
                linewidth=1.4,
                color=COLORS[reference],
                label=f"Single-{reference}",
            )
        ax.set_xticks(
            x, [label(order) for order in orders], rotation=45, ha="right", fontsize=8
        )
        ax.set_title(f"Evaluation task {task}")
        ax.set_ylabel("Mean sensitivity")
        ax.grid(axis="y", alpha=0.25)
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "Sequential model compared with single-task Jacobian references", y=1.03
    )
    return save(fig, "jacobian_vs_single_references")


def plot_channels(orders):
    values, rows = [], []
    for order in orders:
        for task in TASKS:
            channels = sequential(order, task)["channel_sensitivity"]
            values.append([channels[channel] for channel in ("R", "G", "B")])
            rows.append(f"{label(order)} / {task}")
    fig, ax = plt.subplots(figsize=(7, 9))
    image = heatmap(
        ax,
        np.array(values),
        rows,
        ["R", "G", "B"],
        "Sequential-model RGB channel sensitivity",
        cmap="YlGnBu",
    )
    fig.colorbar(image, ax=ax, label="Channel sensitivity")
    return save(fig, "jacobian_channel_sensitivity")


def plot_summary(orders):
    correct = [
        sum(predicted(order, method) == order["actual_last_task"] for order in orders)
        for _, method in METHODS
    ]
    values = np.array(correct) * 100 / 6
    distribution = [
        [sum(predicted(order, method) == task for order in orders) for task in TASKS]
        for _, method in METHODS
    ]
    fig, (ax, table_ax) = plt.subplots(
        1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1.25, 1]}
    )
    bars = ax.bar(
        [name for name, _ in METHODS],
        values,
        color=["#4C78A8", "#72B7B2", "#E45756", "#F2CF5B"],
    )
    ax.axhline(33.333, color="black", linestyle="--", label="Always-A baseline: 33.3%")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Last-task fingerprinting")
    ax.legend(fontsize=8)
    for bar, value, count in zip(bars, values, correct):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 2,
            f"{value:.1f}%\n({count}/6)",
            ha="center",
            fontsize=9,
        )
    table_ax.axis("off")
    table_ax.set_title("Prediction distribution")
    table_ax.table(
        cellText=[[f"{row[0]} / {row[1]} / {row[2]}"] for row in distribution],
        colLabels=["A / B / C predictions"],
        rowLabels=[name for name, _ in METHODS],
        cellLoc="center",
        loc="center",
    )
    return save(fig, "research_summary")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not RESULT.exists():
        raise FileNotFoundError(f"Combined result file not found: {RESULT}")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    orders = load_validate(RESULT)
    generated = []
    generated += plot_accuracy(orders)
    generated += plot_prediction_matrix(orders)
    generated += plot_weight(orders)
    generated += plot_representation(
        orders, "cka", "CKA similarity", "cka_similarity", True
    )
    generated += plot_representation(
        orders, "feature_drift", "Feature drift", "feature_drift", False
    )
    generated += plot_agreement(orders)
    generated += plot_forgetting(orders)
    generated += plot_loss(orders)
    generated += plot_loss(orders, "barrier_height")
    generated += plot_loss(orders, "barrier_area")
    generated += plot_jacobian(orders)
    generated += plot_jacobian_references(orders)
    generated += plot_channels(orders)
    generated += plot_summary(orders)
    print(f"Generated {len(generated)} research figure files successfully.")
    print("Generated filenames:")
    for filename in generated:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
