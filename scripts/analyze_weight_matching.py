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

from sequential_finetuning.analysis.weight_matching import (
    align_state_dict,
    parameter_keys,
    per_layer_task_vector_cosine_similarity,
    predict_maximum,
    predict_minimum,
    resnet18_multitask_permutation_spec,
    state_dict_l2_distance,
    task_vector_cosine_similarity,
    verify_functional_equivalence,
)
from sequential_finetuning.checkpoint import (
    load_initialization_checkpoint,
    load_model_from_checkpoint,
)
from sequential_finetuning.config import (
    load_config,
)
from sequential_finetuning.dataset import (
    create_task_dataset,
    get_test_transform,
)
from sequential_finetuning.model import ResNet18MultiTask
from sequential_finetuning.training_utils import (
    get_device,
)

# ============================================================
# Helpers
# ============================================================


def to_cpu_state_dict(model):
    """Detached CPU copy of a model state dictionary."""

    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def load_single_task_states(config, device):
    """Load the three single-task reference checkpoints."""

    checkpoint_root = Path(config["paths"]["single_checkpoints"])

    states = {}

    for task in config["tasks"]:

        checkpoint_path = checkpoint_root / f"task_{task}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                "Single-task checkpoint not found:\n" f"{checkpoint_path}"
            )

        print(f"Loading single-task checkpoint {task}: {checkpoint_path}")

        model, _ = load_model_from_checkpoint(
            checkpoint_path=checkpoint_path,
            device=device,
            num_classes_per_task=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
        )

        states[task] = to_cpu_state_dict(model)

        del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return states


def load_initial_state(config):
    """Load the shared initialization state theta_init."""

    initialization_path = Path(config["paths"]["initialization_checkpoint"])

    if not initialization_path.exists():
        raise FileNotFoundError(
            "Initialization checkpoint not found:\n"
            f"{initialization_path}\n"
            "Run scripts/create_base_model.py first."
        )

    checkpoint = load_initialization_checkpoint(
        initialization_path,
        map_location="cpu",
    )

    return {
        key: value.detach().cpu().clone()
        for key, value in checkpoint["model_state_dict"].items()
    }


def create_verification_images(config, weight_matching_config, device):
    """A fixed batch of test images used for the invariance check."""

    first_task = list(config["tasks"])[0]

    dataset = create_task_dataset(
        task_name=first_task,
        tasks_config=config,
        root=config["dataset"]["root"],
        train=False,
    )

    dataset.base_dataset.transform = get_test_transform()

    num_images = min(
        weight_matching_config["verification_samples"],
        len(dataset),
    )

    images = torch.stack([dataset[index][0] for index in range(num_images)])

    return images.to(device)


def build_model_from_state(state_dict, config, device):
    """Instantiate a model holding the given state."""

    model = ResNet18MultiTask(
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    model.load_state_dict(state_dict)

    return model.to(device).eval()


# ============================================================
# Analyze one sequential order
# ============================================================


def analyze_sequential_order(
    order,
    single_states,
    initial_state,
    verification_images,
    permutation_spec,
    config,
    weight_matching_config,
    device,
):
    """Align one final checkpoint to every single-task reference."""

    order_name = "_".join(order)

    checkpoint_path = (
        Path(config["paths"]["sequential_checkpoints"])
        / order_name
        / f"{order_name}.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "Final sequential checkpoint not found:\n" f"{checkpoint_path}"
        )

    print()
    print("=" * 70)
    print(f"WEIGHT MATCHING: {' -> '.join(order)}")
    print("=" * 70)

    print(f"Final checkpoint:\n{checkpoint_path}")

    sequential_model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    print(f"Checkpoint order: {checkpoint.get('order')}")

    sequential_state = to_cpu_state_dict(sequential_model)

    full_keys = parameter_keys(sequential_model, backbone_only=False)
    backbone_keys = parameter_keys(sequential_model, backbone_only=True)

    comparisons = {}

    for task, reference_state in single_states.items():

        print()
        print(f"Aligning final checkpoint to Single-{task}...")

        aligned_state, _, info = align_state_dict(
            reference_state=reference_state,
            target_state=sequential_state,
            spec=permutation_spec,
            max_iterations=weight_matching_config["max_iterations"],
            seed=weight_matching_config["seed"],
            verbose=weight_matching_config.get("verbose", False),
        )

        print(
            f"  iterations: {info['iterations']}, "
            f"converged: {info['converged']}, "
            f"identity permutation: {info['is_identity']}"
        )

        # ----------------------------------------------------
        # Functional equivalence check
        # ----------------------------------------------------

        aligned_model = build_model_from_state(aligned_state, config, device)

        differences = verify_functional_equivalence(
            original_model=sequential_model,
            permuted_model=aligned_model,
            images=verification_images,
            tasks=list(config["tasks"]),
        )

        verification_passed = (
            differences["max"] <= weight_matching_config["verification_tolerance"]
        )

        print(
            f"  max logit difference after permutation: "
            f"{differences['max']:.3e} (passed: {verification_passed})"
        )

        if not verification_passed:
            raise RuntimeError(
                "Weight matching changed the network outputs for "
                f"order {order_name} and reference Single-{task}: "
                f"max logit difference {differences['max']:.3e} exceeds "
                f"tolerance {weight_matching_config['verification_tolerance']:.3e}."
            )

        del aligned_model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # ----------------------------------------------------
        # Distances and directions
        # ----------------------------------------------------

        distances = {
            "full_model": {
                "before": state_dict_l2_distance(
                    sequential_state, reference_state, full_keys
                ),
                "after": state_dict_l2_distance(
                    aligned_state, reference_state, full_keys
                ),
            },
            "backbone": {
                "before": state_dict_l2_distance(
                    sequential_state, reference_state, backbone_keys
                ),
                "after": state_dict_l2_distance(
                    aligned_state, reference_state, backbone_keys
                ),
            },
        }

        cosines = {
            "full_model": {
                "before": task_vector_cosine_similarity(
                    final_state=sequential_state,
                    reference_state=reference_state,
                    initial_state=initial_state,
                    keys=full_keys,
                ),
                "after": task_vector_cosine_similarity(
                    final_state=aligned_state,
                    reference_state=reference_state,
                    initial_state=initial_state,
                    keys=full_keys,
                ),
            },
            "backbone": {
                "before": task_vector_cosine_similarity(
                    final_state=sequential_state,
                    reference_state=reference_state,
                    initial_state=initial_state,
                    keys=backbone_keys,
                ),
                "after": task_vector_cosine_similarity(
                    final_state=aligned_state,
                    reference_state=reference_state,
                    initial_state=initial_state,
                    keys=backbone_keys,
                ),
            },
        }

        per_layer_cosine = per_layer_task_vector_cosine_similarity(
            final_state=aligned_state,
            reference_state=reference_state,
            initial_state=initial_state,
            keys=full_keys,
        )

        print(
            f"  full-model L2: {distances['full_model']['before']:.6f} -> "
            f"{distances['full_model']['after']:.6f}"
        )
        print(
            f"  backbone  L2: {distances['backbone']['before']:.6f} -> "
            f"{distances['backbone']['after']:.6f}"
        )
        print(
            f"  cosine (full-model): {cosines['full_model']['before']:.6f} -> "
            f"{cosines['full_model']['after']:.6f}"
        )

        comparisons[task] = {
            "l2": distances,
            "cosine": cosines,
            "per_layer_cosine_after": per_layer_cosine,
            "permutation": info,
            "verification": {
                "max_logit_difference": differences["max"],
                "per_task_logit_difference": {
                    key: value for key, value in differences.items() if key != "max"
                },
                "tolerance": weight_matching_config["verification_tolerance"],
                "passed": verification_passed,
                "num_images": int(verification_images.shape[0]),
            },
        }

    del sequential_model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    actual_last_task = order[-1]

    metric_values = {
        "matched_full_model_l2": {
            task: comparisons[task]["l2"]["full_model"]["after"] for task in comparisons
        },
        "matched_backbone_l2": {
            task: comparisons[task]["l2"]["backbone"]["after"] for task in comparisons
        },
        "unmatched_full_model_l2": {
            task: comparisons[task]["l2"]["full_model"]["before"]
            for task in comparisons
        },
        "matched_full_model_cosine": {
            task: comparisons[task]["cosine"]["full_model"]["after"]
            for task in comparisons
        },
        "matched_backbone_cosine": {
            task: comparisons[task]["cosine"]["backbone"]["after"]
            for task in comparisons
        },
        "unmatched_full_model_cosine": {
            task: comparisons[task]["cosine"]["full_model"]["before"]
            for task in comparisons
        },
    }

    predictions = {}

    for metric, values in metric_values.items():

        predicted = (
            predict_minimum(values)
            if metric.endswith("l2")
            else predict_maximum(values)
        )

        predictions[metric] = {
            "values": values,
            "predicted_last_task": predicted,
            "correct": predicted == actual_last_task,
        }

    print()
    print(f"Actual last task: {actual_last_task}")

    for metric, prediction in predictions.items():
        print(
            f"{metric:>28}: {prediction['predicted_last_task']} "
            f"(correct: {prediction['correct']})"
        )

    return {
        "order": list(order),
        "actual_last_task": actual_last_task,
        "checkpoint": str(checkpoint_path),
        "comparisons": comparisons,
        "predictions": predictions,
    }


# ============================================================
# Main
# ============================================================


def main():

    config = load_config("configs/experiment.yaml")

    weight_matching_config = config["analysis"]["weight_matching"]

    sequential_orders = config["experiment"]["sequential_orders"]

    if not sequential_orders:
        raise ValueError("No sequential orders found in configs/experiment.yaml")

    device = get_device()

    print("=" * 70)
    print("WEIGHT MATCHING ANALYSIS (GIT RE-BASIN)")
    print("=" * 70)

    print(f"Number of orders: {len(sequential_orders)}")
    print(
        f"Max coordinate-descent iterations: {weight_matching_config['max_iterations']}"
    )

    print()
    print("=" * 70)
    print("LOADING REFERENCE STATES")
    print("=" * 70)

    single_states = load_single_task_states(config=config, device=device)

    initial_state = load_initial_state(config)

    verification_images = create_verification_images(
        config=config,
        weight_matching_config=weight_matching_config,
        device=device,
    )

    print(f"Verification images: {verification_images.shape[0]}")

    permutation_spec = resnet18_multitask_permutation_spec(
        task_names=list(config["tasks"]),
    )

    print(f"Permutation groups: {len(permutation_spec.perm_to_axes)}")

    all_results = []

    for experiment_index, order in enumerate(sequential_orders, start=1):

        print()
        print("#" * 70)
        print(f"ANALYSIS {experiment_index}/{len(sequential_orders)}")
        print(f"Order: {' -> '.join(order)}")
        print("#" * 70)

        all_results.append(
            analyze_sequential_order(
                order=order,
                single_states=single_states,
                initial_state=initial_state,
                verification_images=verification_images,
                permutation_spec=permutation_spec,
                config=config,
                weight_matching_config=weight_matching_config,
                device=device,
            )
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(all_results)

    metrics = list(all_results[0]["predictions"])

    summary = {}

    for metric in metrics:

        correct = sum(
            int(result["predictions"][metric]["correct"]) for result in all_results
        )

        summary[metric] = {
            "correct_predictions": correct,
            "prediction_accuracy": correct / total,
        }

    print()
    print("#" * 70)
    print("LAST-TASK PREDICTION ACCURACY")
    print("#" * 70)

    for metric, values in summary.items():
        print(
            f"{metric:>28}: {values['correct_predictions']}/{total} "
            f"({100.0 * values['prediction_accuracy']:.2f}%)"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = Path(config["paths"]["results"]) / "weight_matching"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "weight_matching.json"

    results = {
        "analysis": "weight_matching",
        "algorithm": "git_re_basin_weight_matching",
        "initialization_checkpoint": config["paths"]["initialization_checkpoint"],
        "orders": all_results,
        "summary": {
            "total_predictions": total,
            "metrics": summary,
        },
    }

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2)

    print()
    print("Results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()
