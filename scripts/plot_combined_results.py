"""Create research-focused visualizations from completed combined analysis results."""

import json
import sys
from math import ceil
from pathlib import Path

import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch, Rectangle

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sequential_finetuning.config import load_config

ROOT = PROJECT_ROOT
config = load_config(ROOT / "configs" / "experiment.yaml")
probe_enabled = config["representation"]["probe"]["enabled"]
if not isinstance(probe_enabled, bool):
    raise ValueError("representation.probe.enabled must be true or false.")
representation_mode = "probe" if probe_enabled else "all_test"
RESULT_ROOT = Path(config["paths"]["results"])
if not RESULT_ROOT.is_absolute():
    RESULT_ROOT = ROOT / RESULT_ROOT
RESULT = RESULT_ROOT / "combined" / representation_mode / "combined_analysis.json"
WEIGHT_MATCHING_RESULT = RESULT_ROOT / "weight_matching" / "weight_matching.json"
if not RESULT.exists() and representation_mode == "probe":
    RESULT = ROOT / "results" / "combined" / "combined_analysis.json"

REPRESENTATION_SUFFIX = f"_{representation_mode}"
OUT = ROOT / "figures"
METHODS = (
    ("Full-model weight distance", "full_model", "lower"),
    ("Backbone distance", "backbone", "lower"),
    ("CKA", "cka", "higher"),
    ("Feature drift", "drift", "lower"),
)
PALETTE = ("#2878B5", "#E07B39", "#3A9D5D", "#B15D8D", "#8A6FB3", "#A76D35")
METHOD_COLORS = ("#4C78A8", "#72B7B2", "#E45756", "#F2CF5B")


def require(condition, message):
    if not condition:
        raise ValueError(f"Invalid combined analysis: {message}")


def order_label(order):
    return " -> ".join(order["order"])


def format_evaluation_set_label(evaluation_set):
    mode = evaluation_set.get("mode", representation_mode)
    details = [
        f"{evaluation_set[key]} {label}"
        for key, label in (("num_samples", "samples"), ("num_classes", "classes"))
        if key in evaluation_set
    ]
    suffix = f" ({', '.join(details)})" if details else ""
    return f"representation evaluation set: {mode}{suffix}"


def get_orders(results):
    orders = results.get("orders")
    require(
        isinstance(orders, list) and orders, "expected at least one sequential order"
    )
    return orders


def get_tasks(orders):
    tasks = list(orders[0].get("order", []))
    require(
        len(tasks) >= 2 and len(tasks) == len(set(tasks)),
        "first order must contain at least two unique tasks",
    )
    return tasks


def reference_name(task):
    return f"Single-{task}"


def task_colors(tasks):
    return {task: PALETTE[index % len(PALETTE)] for index, task in enumerate(tasks)}


def panel_axes(count, width=4.5, height=3.5, maximum_columns=3, sharey=False):
    columns = min(count, maximum_columns)
    rows = ceil(count / columns)
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(width * columns, height * rows),
        squeeze=False,
        sharey=sharey,
    )
    for axis in axes.flat[count:]:
        axis.set_visible(False)
    return figure, list(axes.flat[:count])


def load_validate(path):
    with path.open(encoding="utf-8") as file:
        results = json.load(file)
    orders = get_orders(results)
    evaluation_set = results.get("representation_evaluation_set", {})
    tasks = get_tasks(orders)
    expected_references = {reference_name(task) for task in tasks}

    for order in orders:
        name = order.get("order_name", "<unnamed>")
        sequence = order.get("order")
        require(
            isinstance(sequence, list)
            and len(sequence) >= 2
            and len(sequence) == len(tasks)
            and set(sequence) == set(tasks),
            f"{name}: invalid task order",
        )
        require(
            order.get("actual_last_task") == sequence[-1],
            f"{name}: actual_last_task must equal order[-1]",
        )
        require(
            set(order.get("forgetting", {})) == set(tasks),
            f"{name}: forgetting must contain all discovered tasks",
        )
        for variant in ("full_model", "backbone"):
            entry = order.get("weight_distance", {}).get(variant, {})
            require(
                set(entry.get("distances", {})) == set(tasks),
                f"{name}: {variant} distances must contain all tasks",
            )
            require(
                entry.get("predicted_last_task") in tasks,
                f"{name}: {variant} prediction missing",
            )
        representation = order.get("representation", {})
        require(
            set(representation.get("cka", {})) == set(tasks),
            f"{name}: CKA must contain all tasks",
        )
        require(
            set(representation.get("feature_drift", {})) == set(tasks),
            f"{name}: feature drift must contain all tasks",
        )
        require(
            representation.get("predicted_last_task") in tasks
            and representation.get("drift_predicted_last_task") in tasks,
            f"{name}: representation prediction missing",
        )

        fisher = order.get("fisher")
        if fisher is not None:
            require(
                set(fisher.get("scores", {})) == {"fisher_total", "fisher_backbone"},
                f"{name}: Fisher scores must contain total and backbone metrics",
            )
            require(
                set(fisher.get("predictions", {})) == {"fisher_total", "fisher_backbone"},
                f"{name}: Fisher predictions must contain total and backbone metrics",
            )
            for metric in ("fisher_total", "fisher_backbone"):
                require(
                    set(fisher["scores"][metric]) == set(tasks),
                    f"{name}: {metric} scores must contain all tasks",
                )
                hypotheses = fisher["predictions"][metric]
                require(
                    set(hypotheses) == {"high_is_recent", "low_is_recent"},
                    f"{name}: {metric} must contain both Fisher hypotheses",
                )
                for hypothesis, prediction in hypotheses.items():
                    require(
                        prediction.get("predicted_last_task") in tasks
                        and set(prediction.get("predicted_order", [])) == set(tasks),
                        f"{name}: invalid {metric}/{hypothesis} prediction",
                    )

        matching = order.get("weight_matching")
        if matching is not None:
            predictions = matching.get("predictions", {})
            required_predictions = {
                "matched_full_model_l2",
                "matched_backbone_l2",
                "unmatched_full_model_l2",
                "matched_full_model_cosine",
                "matched_backbone_cosine",
                "unmatched_full_model_cosine",
            }
            require(
                set(predictions) == required_predictions,
                f"{name}: incomplete weight-matching predictions",
            )
            require(
                set(matching.get("comparisons", {})) == set(tasks),
                f"{name}: weight matching must compare every task",
            )
            for metric, prediction in predictions.items():
                require(
                    set(prediction.get("values", {})) == set(tasks)
                    and prediction.get("predicted_last_task") in tasks,
                    f"{name}: invalid weight-matching prediction {metric}",
                )
            for task, comparison in matching["comparisons"].items():
                require(
                    comparison.get("verification", {}).get("passed") is True,
                    f"{name}/{task}: output-invariance verification failed",
                )

        barriers = order.get("loss_barrier")
        if barriers is not None:
            barriers = barriers.get("barriers", {})
            require(
                set(barriers) == expected_references,
                f"{name}: loss barriers must contain each single-task reference",
            )
            for reference, by_task in barriers.items():
                require(
                    set(by_task) == set(tasks),
                    f"{name}: {reference} barriers must contain all tasks",
                )
                for task, entry in by_task.items():
                    alphas, losses = entry.get("alphas"), entry.get("losses")
                    require(
                        isinstance(alphas, list)
                        and len(alphas) > 1
                        and len(alphas) == len(losses),
                        f"{name}: {reference}/{task} alpha and loss arrays must match",
                    )
                    require(
                        "barrier_height" in entry and "barrier_area" in entry,
                        f"{name}: {reference}/{task} barrier metrics missing",
                    )

        jacobians = order.get("jacobian")
        if jacobians is not None:
            jacobians = jacobians.get("jacobians", {})

            require(isinstance(jacobians, dict), f"{name}: Jacobian data missing")
            for task in tasks:
                sequential = [
                    entry
                    for key, entry in jacobians.items()
                    if key.startswith("Sequential-") and entry.get("task") == task
                ]
                require(
                    len(sequential) == 1
                    and "mean_sensitivity" in sequential[0].get("metrics", {}),
                    f"{name}: missing sequential Jacobian entry for {task}",
                )
                require(
                    set(sequential[0].get("channel_sensitivity", {}))
                    == {"R", "G", "B"},
                    f"{name}: RGB channel sensitivity missing for {task}",
                )
                single = jacobians.get(reference_name(task), {})
                require(
                    single.get("task") == task
                    and "mean_sensitivity" in single.get("metrics", {}),
                    f"{name}: missing {reference_name(task)} Jacobian entry",
                )

    print("Validation summary:")
    print(f"Orders: {len(orders)}/{len(orders)}")
    print(f"Tasks: {', '.join(tasks)}")
    print("Weight distance: complete")
    print("Representation: complete")
    print(f"Loss barrier: {len(orders)}/{len(orders)}")
    print(f"Jacobian: {sum(order.get('jacobian') is not None for order in orders)}/{len(orders)}")
    print(f"Fisher: {sum(order.get('fisher') is not None for order in orders)}/{len(orders)}")
    print(
        "Weight matching: "
        f"{sum(order.get('weight_matching') is not None for order in orders)}/{len(orders)}"
    )
    return orders, tasks, evaluation_set


def predicted(order, method):
    if method in ("full_model", "backbone"):
        return order["weight_distance"][method]["predicted_last_task"]
    key = "predicted_last_task" if method == "cka" else "drift_predicted_last_task"
    return order["representation"][key]


def prediction_experiments(orders):
    """Return every available last-task experiment and its stored predictions."""
    experiments = [
        (name, [predicted(order, method) for order in orders])
        for name, method, _ in METHODS
    ]

    if all(isinstance(order.get("fisher"), dict) for order in orders):
        fisher_specs = (
            ("Fisher total: high = recent", "fisher_total", "high_is_recent"),
            ("Fisher total: low = recent", "fisher_total", "low_is_recent"),
            ("Fisher backbone: high = recent", "fisher_backbone", "high_is_recent"),
            ("Fisher backbone: low = recent", "fisher_backbone", "low_is_recent"),
        )
        experiments.extend(
            (
                name,
                [
                    order["fisher"]["predictions"][metric][hypothesis][
                        "predicted_last_task"
                    ]
                    for order in orders
                ],
            )
            for name, metric, hypothesis in fisher_specs
        )

    if all(isinstance(order.get("weight_matching"), dict) for order in orders):
        matching_specs = (
            ("Matched full L2", "matched_full_model_l2"),
            ("Matched backbone L2", "matched_backbone_l2"),
            ("Unmatched full L2", "unmatched_full_model_l2"),
            ("Matched full cosine", "matched_full_model_cosine"),
            ("Matched backbone cosine", "matched_backbone_cosine"),
            ("Unmatched full cosine", "unmatched_full_model_cosine"),
        )
        experiments.extend(
            (
                name,
                [
                    order["weight_matching"]["predictions"][metric][
                        "predicted_last_task"
                    ]
                    for order in orders
                ],
            )
            for name, metric in matching_specs
        )

    return experiments


def accuracy_counts(orders):
    return [
        sum(value == order["actual_last_task"] for value, order in zip(predictions, orders))
        for _, predictions in prediction_experiments(orders)
    ]


def baseline(orders, tasks):
    task = "A" if "A" in tasks else tasks[0]
    accuracy = (
        sum(order["actual_last_task"] == task for order in orders) * 100 / len(orders)
    )
    return task, accuracy


def save(figure, stem, mode_specific=False):
    figure.tight_layout()
    filenames = []
    # The active evaluation mode is written in each mode-dependent figure title.
    # Use one stable filename so documentation needs only one path.
    suffix_part = ""
    # , "pdf"
    for suffix in ["png"]:
        filename = f"{stem}{suffix_part}.{suffix}"
        path = OUT / filename
        if path.exists():
            path.unlink()
        figure.savefig(path, dpi=300, bbox_inches="tight")
        filenames.append(filename)
    plt.close(figure)
    return filenames


def heatmap(
    axis,
    values,
    rows,
    columns,
    title,
    actual_tasks=None,
    cmap="YlGnBu",
    text_color=None,
):
    image = axis.imshow(values, aspect="auto", cmap=cmap)
    axis.set_xticks(range(len(columns)), columns, rotation=20, ha="right")
    axis.set_yticks(range(len(rows)), rows)
    axis.set_title(title)
    norm = image.norm

    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            value = values[row, column]
            if text_color is None:
                color = "white" if norm(value) < 0.7 else "black"
            else:
                color = text_color
            axis.text(
                column,
                row,
                f"{value:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color=color,
            )
        if actual_tasks is not None:
            column = (
                columns.index(reference_name(actual_tasks[row]))
                if columns[0].startswith("Single-")
                else columns.index(actual_tasks[row])
            )
            axis.add_patch(
                Rectangle(
                    (column - 0.5, row - 0.5),
                    1,
                    1,
                    fill=False,
                    edgecolor="black",
                    linewidth=2,
                )
            )
    return image


def plot_accuracy(orders, tasks, evaluation_set):
    experiments = prediction_experiments(orders)
    names = [name for name, _ in experiments]
    correct = accuracy_counts(orders)
    values = np.array(correct) * 100 / len(orders)
    baseline_task, baseline_accuracy = baseline(orders, tasks)
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiments)))
    figure, axis = plt.subplots(figsize=(max(12, len(experiments) * 0.9), 6.5))
    bars = axis.bar(names, values, color=colors)
    axis.axhline(
        baseline_accuracy,
        color="black",
        linestyle="--",
        label=f"Always-{baseline_task} baseline: {baseline_accuracy:.1f}%",
    )
    axis.set_ylim(0, 108)
    axis.set_ylabel("Last-task prediction accuracy (%)")
    axis.set_title(
        f"Last-task prediction accuracy ({format_evaluation_set_label(evaluation_set)})"
    )
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.legend()
    for bar, value, count in zip(bars, values, correct):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%\n({count}/{len(orders)})",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )
    figure.tight_layout()
    return save(figure, "last_task_prediction_accuracy", mode_specific=True)


def plot_prediction_matrix(orders, evaluation_set):
    experiments = prediction_experiments(orders)
    columns = [name.replace(": ", "\n") for name, _ in experiments]
    labels = [
        [predictions[row] for _, predictions in experiments]
        for row in range(len(orders))
    ]
    values = np.array(
        [
            [int(value == order["actual_last_task"]) for value in row]
            for order, row in zip(orders, labels)
        ]
    )
    figure, axis = plt.subplots(
        figsize=(max(11, 1.35 * len(columns)), max(4.5, len(orders) * 0.65 + 1.5))
    )
    axis.imshow(
        values,
        cmap=ListedColormap(["#F4B6B2", "#B8E0C2"]),
        vmin=0,
        vmax=1,
        aspect="auto",
    )
    axis.set_xticks(range(len(columns)), columns, rotation=30, ha="right")
    axis.set_yticks(
        range(len(orders)),
        [f"{order_label(order)} (actual last: {order['actual_last_task']})" for order in orders],
    )
    axis.set_title(
        f"Predicted last task by method ({format_evaluation_set_label(evaluation_set)})"
    )
    for row, labels_row in enumerate(labels):
        for column, value in enumerate(labels_row):
            axis.text(column, row, value, ha="center", va="center", fontweight="bold")
    figure.tight_layout()
    return save(figure, "prediction_matrix", mode_specific=True)


def plot_weight(orders, tasks, colors):
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(orders))
    width = 0.8 / len(tasks)
    for axis, variant, title in zip(
        axes,
        ("full_model", "backbone"),
        ("Full-model distance", "Backbone-only distance"),
    ):
        for index, task in enumerate(tasks):
            bars = axis.bar(
                x + (index - (len(tasks) - 1) / 2) * width,
                [
                    order["weight_distance"][variant]["distances"][task]
                    for order in orders
                ],
                width,
                label=reference_name(task),
                color=colors[task],
            )
            for row, bar in enumerate(bars):
                if orders[row]["actual_last_task"] == task:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)
        axis.set_title(title)
        axis.set_ylabel("Weight distance")
        axis.grid(axis="y", alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        handles.append(Patch(facecolor="white", edgecolor="black", linewidth=2))
        labels.append("Black outline = actual-last-task reference")
        axis.legend(handles, labels, ncol=min(len(labels), 4), fontsize=8)
    axes[-1].set_xticks(
        x, [order_label(order) for order in orders], rotation=18, ha="right"
    )
    figure.suptitle(
        "Distance from each final sequential model to single-task references",
        y=1.01,
    )
    return save(figure, "weight_distance_scores")


def plot_representation(
    orders, tasks, colors, key, title, stem, direction, evaluation_set
):
    figure, axes = panel_axes(len(orders), sharey=True)
    for axis, order in zip(axes, orders):
        values = [order["representation"][key][task] for task in tasks]
        bars = axis.bar(tasks, values, color=[colors[task] for task in tasks])
        last = tasks.index(order["actual_last_task"])
        bars[last].set_edgecolor("black")
        bars[last].set_linewidth(2.3)
        for bar, value in zip(bars, values):
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        axis.set_title(
            f"{order_label(order)} (actual last: {order['actual_last_task']})",
            fontsize=10,
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Score")
    figure.suptitle(
        f"{title}: final sequential model vs single-task references ({direction} = greater similarity; black outline = actual-last-task reference)\n"
        f"{format_evaluation_set_label(evaluation_set).capitalize()}",
        y=1.03,
    )
    return save(figure, stem, mode_specific=True)


def plot_forgetting(orders, tasks, colors):
    figure, axes = panel_axes(len(orders), sharey=True)
    for axis, order in zip(axes, orders):
        bars = axis.bar(
            tasks,
            [order["forgetting"][task] for task in tasks],
            color=[colors[task] for task in tasks],
        )
        last = tasks.index(order["actual_last_task"])
        bars[last].set_edgecolor("black")
        bars[last].set_linewidth(2.3)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(
            f"{order_label(order)} (actual last: {order['actual_last_task']})",
            fontsize=10,
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Forgetting")
    figure.suptitle(
        "Forgetting by order and task (black outline = actual last task)", y=1.01
    )
    return save(figure, "forgetting_by_order")


def plot_loss(orders, tasks, colors, metric=None):
    references = [reference_name(task) for task in tasks]
    if metric:
        values = np.array(
            [
                [
                    order["loss_barrier"]["barriers"][reference][
                        order["actual_last_task"]
                    ][metric]
                    for reference in references
                ]
                for order in orders
            ]
        )
        figure, axis = plt.subplots(figsize=(8, max(4.5, len(orders) * 0.65 + 1.5)))
        image = heatmap(
            axis,
            values,
            [order_label(order) for order in orders],
            references,
            f"{metric.replace('_', ' ').title()} for reference ↔ final sequential interpolation, evaluated on actual last task",
            [order["actual_last_task"] for order in orders],
            "magma",
        )
        figure.colorbar(image, ax=axis, label=metric.replace("_", " "))
        return save(figure, f"loss_barrier_{metric.split('_')[1]}")
    figure, axes = panel_axes(len(orders))
    for axis, order in zip(axes, orders):
        actual = order["actual_last_task"]
        for task in tasks:
            entry = order["loss_barrier"]["barriers"][reference_name(task)][actual]
            axis.plot(
                entry["alphas"],
                entry["losses"],
                marker="o",
                markersize=3,
                linewidth=1.7,
                color=colors[task],
                label=reference_name(task),
            )
        axis.axvline(0, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(1, color="black", linestyle="--", linewidth=0.8)
        axis.set_title(f"{order_label(order)}; actual last task: {actual}", fontsize=10)
        axis.set_xlabel("Alpha (0 = reference, 1 = sequential)")
        axis.set_ylabel("Loss")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, title="Reference model")
    figure.suptitle(
        "Loss interpolation: each Single-task reference ↔ the final sequential model",
        y=1.01,
    )
    return save(figure, "loss_barrier_curves")


def sequential_jacobian(order, task):
    return next(
        entry
        for key, entry in order["jacobian"]["jacobians"].items()
        if key.startswith("Sequential-") and entry["task"] == task
    )


def plot_jacobian(orders, tasks, colors):
    figure, axes = panel_axes(len(orders), sharey=True)
    for axis, order in zip(axes, orders):
        bars = axis.bar(
            tasks,
            [
                sequential_jacobian(order, task)["metrics"]["mean_sensitivity"]
                for task in tasks
            ],
            color=[colors[task] for task in tasks],
        )
        last = tasks.index(order["actual_last_task"])
        bars[last].set_edgecolor("black")
        bars[last].set_linewidth(2.3)
        axis.set_title(
            f"{order_label(order)} (actual last: {order['actual_last_task']})",
            fontsize=10,
        )
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel("Mean sensitivity")
    figure.suptitle(
        "Final sequential model sensitivity by evaluated task (black outline = actual last task)",
        y=1.01,
    )
    return save(figure, "jacobian_sensitivity_by_order")


def plot_jacobian_references(orders, tasks, colors):
    figure, axes = panel_axes(
        len(tasks), width=4.8, height=4.8, maximum_columns=len(tasks)
    )
    x = np.arange(len(orders))
    for axis, task in zip(axes, tasks):
        axis.bar(
            x,
            [
                sequential_jacobian(order, task)["metrics"]["mean_sensitivity"]
                for order in orders
            ],
            color=colors[task],
            label=f"Sequential: task {task}",
        )
        for reference in tasks:
            axis.plot(
                x,
                [
                    order["jacobian"]["jacobians"][reference_name(reference)][
                        "metrics"
                    ]["mean_sensitivity"]
                    for order in orders
                ],
                marker="o",
                linewidth=1.4,
                color=colors[reference],
                label=f"{reference_name(reference)} (own task {reference})",
            )
        axis.set_xticks(
            x,
            [order_label(order) for order in orders],
            rotation=45,
            ha="right",
            fontsize=8,
        )
        axis.set_title(f"Sequential models evaluated on task {task}")
        axis.set_ylabel("Mean sensitivity")
        axis.grid(axis="y", alpha=0.25)
    axes[-1].legend(fontsize=8)
    figure.suptitle(
        "Mean Jacobian sensitivity: sequential task evaluations and single-reference own-task baselines",
        y=1.03,
    )
    return save(figure, "jacobian_vs_single_references")


def plot_channels(orders, tasks):
    values, rows = [], []
    for order in orders:
        for task in tasks:
            channels = sequential_jacobian(order, task)["channel_sensitivity"]
            values.append([channels[channel] for channel in ("R", "G", "B")])
            rows.append(f"{order_label(order)} / {task}")
    figure, axis = plt.subplots(figsize=(7, max(5, len(rows) * 0.35 + 2)))
    image = heatmap(
        axis,
        np.array(values),
        rows,
        ["R", "G", "B"],
        "Sequential-model RGB channel sensitivity",
        text_color="black",
    )
    figure.colorbar(image, ax=axis, label="Channel sensitivity")
    return save(figure, "jacobian_channel_sensitivity")


def plot_fisher(orders, tasks, colors):
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    x = np.arange(len(orders))
    width = 0.8 / len(tasks)
    for axis, metric, title in zip(
        axes,
        ("fisher_total", "fisher_backbone"),
        ("Total Fisher trace", "Backbone Fisher trace"),
    ):
        for index, task in enumerate(tasks):
            bars = axis.bar(
                x + (index - (len(tasks) - 1) / 2) * width,
                [order["fisher"]["scores"][metric][task] for order in orders],
                width,
                color=colors[task],
                label=f"Task {task}",
            )
            for row, bar in enumerate(bars):
                if orders[row]["actual_last_task"] == task:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)
        axis.set_title(title)
        axis.set_ylabel("Fisher trace")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(fontsize=8, ncol=len(tasks))
    axes[-1].set_xticks(x, [order_label(order) for order in orders], rotation=18, ha="right")
    figure.suptitle("Fisher information by sequential order and evaluation task\nBlack outline = actual last task", y=1.02)
    return save(figure, "fisher_information_scores")


def plot_fisher_hypotheses(orders):
    labels, values, counts = [], [], []
    for metric, metric_label in (("fisher_total", "Total"), ("fisher_backbone", "Backbone")):
        for hypothesis, hypothesis_label in (("high_is_recent", "High = recent"), ("low_is_recent", "Low = recent")):
            count = sum(
                order["fisher"]["predictions"][metric][hypothesis]["last_task_correct"]
                for order in orders
            )
            labels.append(f"{metric_label}\n{hypothesis_label}")
            counts.append(count)
            values.append(100.0 * count / len(orders))
    figure, axis = plt.subplots(figsize=(9, 5))
    bars = axis.bar(labels, values, color=("#D95F02", "#1B9E77", "#D95F02", "#1B9E77"))
    axis.set_ylim(0, 100)
    axis.set_ylabel("Last-task prediction accuracy (%)")
    axis.set_title("Fisher recency hypotheses")
    axis.grid(axis="y", alpha=0.25)
    for bar, value, count in zip(bars, values, counts):
        axis.text(bar.get_x() + bar.get_width() / 2, value + 2, f"{value:.1f}%\n({count}/{len(orders)})", ha="center")
    return save(figure, "fisher_hypothesis_accuracy")


def plot_weight_matching_scores(orders, tasks, colors):
    specifications = (
        ("matched_full_model_l2", "Matched full-model L2", "lower"),
        ("matched_backbone_l2", "Matched backbone L2", "lower"),
        ("matched_full_model_cosine", "Matched full-model cosine", "higher"),
        ("matched_backbone_cosine", "Matched backbone cosine", "higher"),
    )
    figure, axes = panel_axes(len(specifications), width=5.3, height=4.1, maximum_columns=2)
    x = np.arange(len(orders))
    width = 0.8 / len(tasks)
    for axis, (metric, title, direction) in zip(axes, specifications):
        for index, task in enumerate(tasks):
            bars = axis.bar(
                x + (index - (len(tasks) - 1) / 2) * width,
                [order["weight_matching"]["predictions"][metric]["values"][task] for order in orders],
                width,
                color=colors[task],
                label=f"Task {task}",
            )
            for row, bar in enumerate(bars):
                if orders[row]["actual_last_task"] == task:
                    bar.set_edgecolor("black")
                    bar.set_linewidth(2)
        axis.set_title(f"{title} ({direction} predicts recent)")
        axis.set_xticks(x, [order_label(order) for order in orders], rotation=35, ha="right", fontsize=8)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=len(tasks), fontsize=8)
    figure.suptitle("Weight-matching task scores\nBlack outline = actual last task", y=1.02)
    return save(figure, "weight_matching_scores")


def plot_weight_matching_effect(orders, tasks):
    rows = [f"{order_label(order)} / {task}" for order in orders for task in tasks]
    l2_before, l2_after, cosine_before, cosine_after = [], [], [], []
    for order in orders:
        for task in tasks:
            comparison = order["weight_matching"]["comparisons"][task]
            l2_before.append(comparison["l2"]["full_model"]["before"])
            l2_after.append(comparison["l2"]["full_model"]["after"])
            cosine_before.append(comparison["cosine"]["full_model"]["before"])
            cosine_after.append(comparison["cosine"]["full_model"]["after"])
    figure, axes = plt.subplots(1, 2, figsize=(13, max(5, len(rows) * 0.3)))
    y = np.arange(len(rows))
    axes[0].scatter(l2_before, y, label="Before", marker="o")
    axes[0].scatter(l2_after, y, label="After", marker="x")
    axes[0].set_title("Full-model L2 before and after matching")
    axes[0].set_xlabel("L2 distance")
    axes[1].scatter(cosine_before, y, label="Before", marker="o")
    axes[1].scatter(cosine_after, y, label="After", marker="x")
    axes[1].set_title("Full-model cosine before and after matching")
    axes[1].set_xlabel("Cosine similarity")
    for axis in axes:
        axis.set_yticks(y, rows, fontsize=7)
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Effect of permutation matching")
    return save(figure, "weight_matching_before_after")


def load_raw_weight_matching():
    if not WEIGHT_MATCHING_RESULT.exists():
        return None
    with WEIGHT_MATCHING_RESULT.open(encoding="utf-8") as file:
        return json.load(file)


def plot_weight_matching_layers(raw_results, tasks):
    raw_orders = raw_results.get("orders", [])
    if not raw_orders:
        return []
    layers = list(raw_orders[0]["comparisons"][tasks[0]]["per_layer_cosine_after"])
    rows, values = [], []
    for order in raw_orders:
        for task in tasks:
            layer_values = order["comparisons"][task]["per_layer_cosine_after"]
            rows.append(f"{' -> '.join(order['order'])} / {task}")
            values.append([np.nan if layer_values[layer] is None else layer_values[layer] for layer in layers])
    figure, axis = plt.subplots(figsize=(max(14, len(layers) * 0.34), max(6, len(rows) * 0.35)))
    image = axis.imshow(np.asarray(values, dtype=float), aspect="auto", cmap="viridis", vmin=-1, vmax=1)
    axis.set_xticks(range(len(layers)), layers, rotation=90, fontsize=7)
    axis.set_yticks(range(len(rows)), rows, fontsize=7)
    axis.set_title("Per-layer cosine after weight matching")
    figure.colorbar(image, ax=axis, label="Cosine similarity")
    return save(figure, "weight_matching_per_layer_cosine")


def plot_summary(orders, tasks, evaluation_set):
    experiments = prediction_experiments(orders)
    names = [name for name, _ in experiments]
    correct = accuracy_counts(orders)
    values = np.array(correct) * 100 / len(orders)
    baseline_task, baseline_accuracy = baseline(orders, tasks)
    distribution = [
        [sum(value == task for value in predictions) for task in tasks]
        for _, predictions in experiments
    ]
    colors = plt.cm.tab20(np.linspace(0, 1, len(experiments)))
    figure, (axis, table_axis) = plt.subplots(
        1,
        2,
        figsize=(max(16, len(experiments) * 1.15), max(7, len(experiments) * 0.42)),
        gridspec_kw={"width_ratios": [1.7, 1]},
    )
    bars = axis.bar(names, values, color=colors)
    axis.axhline(
        baseline_accuracy,
        color="black",
        linestyle="--",
        label=f"Always-{baseline_task} baseline: {baseline_accuracy:.1f}%",
    )
    axis.set_ylim(0, 108)
    axis.set_ylabel("Accuracy (%)")
    axis.set_title(
        f"Last-task fingerprinting ({format_evaluation_set_label(evaluation_set)})"
    )
    axis.tick_params(axis="x", rotation=35)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.legend(fontsize=8)
    for bar, value, count in zip(bars, values, correct):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.1f}%\n({count}/{len(orders)})",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    table_axis.axis("off")
    table_axis.set_title("Prediction distribution")
    table = table_axis.table(
        cellText=distribution,
        colLabels=[f"Predicted {task}" for task in tasks],
        rowLabels=names,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.2)
    figure.tight_layout()
    return save(figure, "research_summary", mode_specific=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if not RESULT.exists():
        raise FileNotFoundError(f"Combined result file not found: {RESULT}")
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 12})
    orders, tasks, evaluation_set = load_validate(RESULT)
    colors = task_colors(tasks)
    generated = []
    generated += plot_accuracy(orders, tasks, evaluation_set)
    generated += plot_prediction_matrix(orders, evaluation_set)
    generated += plot_weight(orders, tasks, colors)
    generated += plot_representation(
        orders,
        tasks,
        colors,
        "cka",
        "CKA similarity",
        "cka_similarity",
        "higher",
        evaluation_set,
    )
    generated += plot_representation(
        orders,
        tasks,
        colors,
        "feature_drift",
        "Feature drift",
        "feature_drift",
        "lower",
        evaluation_set,
    )
    generated += plot_forgetting(orders, tasks, colors)
    # Standard loss plots, if regular loss data exists.
    if any("loss" in order for order in orders):
        generated += plot_loss(orders, tasks, colors)
    else:
        print("Skipping loss plots: loss results are not available.")

    # --------------------------------------------------------
    # Optional loss-barrier figures
    # --------------------------------------------------------

    has_loss_barrier = all(
        isinstance(order.get("loss_barrier"), dict)
        and isinstance(order["loss_barrier"].get("barriers"), dict)
        and bool(order["loss_barrier"]["barriers"])
        for order in orders
    )

    if has_loss_barrier:
        generated += plot_loss(
            orders,
            tasks,
            colors,
            "barrier_height",
        )

        generated += plot_loss(
            orders,
            tasks,
            colors,
            "barrier_area",
        )
    else:
        print(
            "Skipping loss-barrier plots: "
            "loss-barrier results are not available for all orders."
        )

    # --------------------------------------------------------
    # Optional Jacobian and channel-sensitivity figures
    # --------------------------------------------------------

    has_jacobian = all(
        isinstance(order.get("jacobian"), dict)
        and isinstance(order["jacobian"].get("jacobians"), dict)
        and bool(order["jacobian"]["jacobians"])
        for order in orders
    )

    if has_jacobian:
        generated += plot_jacobian(
            orders,
            tasks,
            colors,
        )

        generated += plot_jacobian_references(
            orders,
            tasks,
            colors,
        )

        generated += plot_channels(
            orders,
            tasks,
        )
    else:
        print(
            "Skipping Jacobian and channel-sensitivity plots: "
            "Jacobian results are not available for all orders."
        )

    has_fisher = all(isinstance(order.get("fisher"), dict) for order in orders)
    if has_fisher:
        generated += plot_fisher(orders, tasks, colors)
        generated += plot_fisher_hypotheses(orders)
    else:
        print("Skipping Fisher plots: Fisher results are not available for all orders.")

    has_weight_matching = all(
        isinstance(order.get("weight_matching"), dict) for order in orders
    )
    if has_weight_matching:
        generated += plot_weight_matching_scores(orders, tasks, colors)
        generated += plot_weight_matching_effect(orders, tasks)
        raw_weight_matching = load_raw_weight_matching()
        if raw_weight_matching is not None:
            generated += plot_weight_matching_layers(raw_weight_matching, tasks)
        else:
            print("Skipping per-layer cosine plot: raw weight-matching JSON not found.")
    else:
        print(
            "Skipping weight-matching plots: "
            "weight-matching results are not available for all orders."
        )

    generated += plot_summary(
        orders,
        tasks,
        evaluation_set,
    )

    print(f"Figures use {format_evaluation_set_label(evaluation_set)}")
    print(f"Output directory: {OUT}")
    print(f"Generated {len(generated)} research figure files successfully.")
    print("Generated filenames:")
    for filename in generated:
        print(f"- {filename}")


if __name__ == "__main__":
    main()
