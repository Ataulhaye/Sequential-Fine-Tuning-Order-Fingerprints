import json
import sys
from pathlib import Path

import torch

# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from sequential_finetuning.analysis.weight_distance import (
    calculate_all_backbone_distances,
    calculate_all_weight_distances,
    predict_closest_task,
)
from sequential_finetuning.checkpoint import (
    load_model_from_checkpoint,
)
from sequential_finetuning.config import (
    load_config,
)
from sequential_finetuning.training_utils import (
    get_device,
)

# ============================================================
# Load single-task reference models
# ============================================================


def load_single_task_models(
    config,
    device,
):
    """
    Load the three single-task reference checkpoints.

    Returns
    -------
    Dict[str, nn.Module]
        Models for tasks A, B, and C.
    """

    single_checkpoint_root = Path(config["paths"]["single_checkpoints"])

    single_models = {}

    for task in config["tasks"]:

        checkpoint_path = single_checkpoint_root / f"task_{task}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Single-task checkpoint not found:\n" f"{checkpoint_path}"
            )

        print(f"Loading single-task checkpoint {task}:")
        print(checkpoint_path)

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            num_classes_per_task=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
        )

        print(f"Checkpoint task: " f"{checkpoint.get('task')}")

        single_models[task] = model

    return single_models


# ============================================================
# Analyze one sequential order
# ============================================================


def analyze_sequential_order(
    order,
    single_models,
    config,
    device,
):
    """
    Compare the final checkpoint of one sequential order
    against all single-task checkpoints.
    """

    order_name = "_".join(order)

    checkpoint_root = Path(config["paths"]["sequential_checkpoints"])

    checkpoint_path = checkpoint_root / order_name / f"{order_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Final sequential checkpoint not found:\n" f"{checkpoint_path}"
        )

    print()
    print("=" * 70)
    print(f"WEIGHT DISTANCE: " f"{' -> '.join(order)}")
    print("=" * 70)

    print(f"Final checkpoint:\n" f"{checkpoint_path}")

    sequential_model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    actual_last_task = order[-1]

    print(f"Actual last task: " f"{actual_last_task}")

    print(f"Checkpoint task: " f"{checkpoint.get('task')}")

    print(f"Checkpoint order: " f"{checkpoint.get('order')}")

    # --------------------------------------------------------
    # Calculate distances
    # --------------------------------------------------------

    full_model_distances = calculate_all_weight_distances(
        sequential_model=sequential_model,
        single_task_models=single_models,
    )

    backbone_distances = calculate_all_backbone_distances(
        sequential_model=sequential_model,
        single_task_models=single_models,
    )

    full_model_prediction = predict_closest_task(full_model_distances)

    backbone_prediction = predict_closest_task(backbone_distances)

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print()
    print("FULL-MODEL L2 DISTANCES")
    print("-" * 70)

    for task, distance in full_model_distances.items():

        marker = ""

        if task == full_model_prediction:
            marker = "  <-- closest"

        print(f"Single-{task}: " f"{distance:.6f}" f"{marker}")

    print()
    print(f"Full-model prediction: " f"{full_model_prediction}")

    print()
    print("BACKBONE-ONLY L2 DISTANCES")
    print("-" * 70)

    for task, distance in backbone_distances.items():

        marker = ""

        if task == backbone_prediction:
            marker = "  <-- closest"

        print(f"Single-{task}: " f"{distance:.6f}" f"{marker}")

    print()
    print(f"Backbone prediction: " f"{backbone_prediction}")

    print()
    print(f"Actual last task: " f"{actual_last_task}")

    full_model_correct = full_model_prediction == actual_last_task

    backbone_correct = backbone_prediction == actual_last_task

    print(f"Full-model correct: " f"{full_model_correct}")

    print(f"Backbone correct:   " f"{backbone_correct}")

    # --------------------------------------------------------
    # Release sequential model
    # --------------------------------------------------------

    del sequential_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "order": list(order),
        "actual_last_task": actual_last_task,
        "full_model": {
            "distances": full_model_distances,
            "predicted_last_task": full_model_prediction,
            "correct": full_model_correct,
        },
        "backbone": {
            "distances": backbone_distances,
            "predicted_last_task": backbone_prediction,
            "correct": backbone_correct,
        },
        "checkpoint": str(checkpoint_path),
    }


# ============================================================
# Main
# ============================================================


def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config("configs/experiment.yaml")

    sequential_orders = config["experiment"]["sequential_orders"]

    if not sequential_orders:
        raise ValueError("No sequential orders found in " "configs/experiment.yaml")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print("=" * 70)
    print("WEIGHT DISTANCE ANALYSIS")
    print("=" * 70)

    print(f"Number of orders: " f"{len(sequential_orders)}")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Load reference models ONCE
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING SINGLE-TASK REFERENCE MODELS")
    print("=" * 70)

    single_models = load_single_task_models(
        config=config,
        device=device,
    )

    # --------------------------------------------------------
    # Analyze every sequential order
    # --------------------------------------------------------

    all_results = []

    for experiment_index, order in enumerate(
        sequential_orders,
        start=1,
    ):

        print()
        print()
        print("#" * 70)

        print(f"ANALYSIS " f"{experiment_index}/" f"{len(sequential_orders)}")

        print(f"Order: " f"{' -> '.join(order)}")

        print("#" * 70)

        result = analyze_sequential_order(
            order=order,
            single_models=single_models,
            config=config,
            device=device,
        )

        all_results.append(result)

    # --------------------------------------------------------
    # Overall prediction accuracy
    # --------------------------------------------------------

    total_predictions = len(all_results)

    full_model_correct_predictions = sum(
        result["full_model"]["correct"] for result in all_results
    )

    backbone_correct_predictions = sum(
        result["backbone"]["correct"] for result in all_results
    )

    full_model_accuracy = (
        full_model_correct_predictions / total_predictions
        if total_predictions > 0
        else 0.0
    )

    backbone_accuracy = (
        backbone_correct_predictions / total_predictions
        if total_predictions > 0
        else 0.0
    )
    # --------------------------------------------------------
    # Print summary
    # --------------------------------------------------------

    print()
    print()
    print("#" * 70)
    print("WEIGHT DISTANCE SUMMARY")
    print("#" * 70)

    print()

    for result in all_results:

        order = result["order"]

        print(f"Order: " f"{' -> '.join(order)}")

        print(f"Actual last: " f"{result['actual_last_task']}")

        print(
            f"Full-model correct predictions: "
            f"{full_model_correct_predictions}/"
            f"{total_predictions}"
        )

        print(
            f"Full-model last-task prediction accuracy: "
            f"{100.0 * full_model_accuracy:.2f}%"
        )

        print(
            f"Backbone correct predictions: "
            f"{backbone_correct_predictions}/"
            f"{total_predictions}"
        )

        print(
            f"Backbone last-task prediction accuracy: "
            f"{100.0 * backbone_accuracy:.2f}%"
        )

        print()

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    result_root = Path(config["paths"]["results"])

    result_dir = result_root / "weight_distance"

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "analysis": "weight_distance",
        "metrics": [
            "full_model_l2",
            "backbone_l2",
        ],
        "orders": all_results,
        "summary": {
            "total_predictions": total_predictions,
            "full_model": {
                "correct_predictions": (full_model_correct_predictions),
                "prediction_accuracy": (full_model_accuracy),
            },
            "backbone": {
                "correct_predictions": (backbone_correct_predictions),
                "prediction_accuracy": (backbone_accuracy),
            },
        },
    }

    results_path = result_dir / "weight_distance.json"

    with open(
        results_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=2,
        )

    print()
    print("Results saved to:")
    print(results_path)

    # --------------------------------------------------------
    # Release reference models
    # --------------------------------------------------------

    for model in single_models.values():
        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
