"""
Analyze Jacobian Sensitivity

Computes input-output Jacobian sensitivity on task probe images to understand
how sensitive model predictions are to input perturbations.

This reveals task-specific sensitivity patterns and which input features
most strongly influence predictions.
"""

import json
from pathlib import Path

import torch

from sequential_finetuning.analysis.jacobian import (
    compute_channel_sensitivity,
    compute_jacobian_batch,
    compute_jacobian_sensitivity_metrics,
    compute_spatial_sensitivity,
)
from sequential_finetuning.checkpoint import load_model_from_checkpoint
from sequential_finetuning.config import load_config
from sequential_finetuning.dataset import create_task_dataloader
from sequential_finetuning.training_utils import get_device, print_device_info


def analyze_jacobian_for_model(
    model_name: str,
    checkpoint_path: Path,
    config,
    device,
    task_dataloader,
    task: str,
):
    """
    Analyze Jacobian sensitivity for one model on one task.
    """

    print(f"  Computing Jacobian for {model_name}...")

    # Load model
    model, _ = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    model.eval()

    # Collect a subset of images from the dataloader
    images_list = []
    labels_list = []
    max_samples = config.get("analysis", {}).get("jacobian", {}).get("max_samples", 100)
    if max_samples < 1:
        raise ValueError("analysis.jacobian.max_samples must be at least 1.")

    for images, labels in task_dataloader:
        images_list.append(images)
        labels_list.append(labels)

        if sum(batch.shape[0] for batch in images_list) >= max_samples:
            break

    images_batch = torch.cat(images_list)[:max_samples].to(device)

    # Compute Jacobians
    jacobians, pred_classes = compute_jacobian_batch(
        model=model,
        images=images_batch,
        task=task,
        device=device,
        compute_class_jacobian=True,
    )

    # Compute metrics
    metrics = compute_jacobian_sensitivity_metrics(jacobians)

    # Compute spatial and channel sensitivity
    spatial_sensitivity = compute_spatial_sensitivity(jacobians)
    channel_sensitivity = compute_channel_sensitivity(jacobians)

    return {
        "model": model_name,
        "task": task,
        "num_samples": images_batch.shape[0],
        "metrics": metrics,
        "spatial_sensitivity_mean": float(spatial_sensitivity.mean().item()),
        "channel_sensitivity": {
            "R": float(channel_sensitivity[0].item()),
            "G": float(channel_sensitivity[1].item()),
            "B": float(channel_sensitivity[2].item()),
        },
    }


def analyze_jacobian_for_order(
    order,
    config,
    device,
):
    """
    Analyze Jacobian sensitivity for one sequential order and all single-task references.
    """

    print()
    print("#" * 70)
    print(f"JACOBIAN SENSITIVITY ANALYSIS: {' → '.join(order)}")
    print("#" * 70)

    order_name = "_".join(order)

    results = {
        "order": order,
        "jacobians": {},
    }

    dataset_root = config["dataset"]["root"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]

    # --------------------------------------------------------
    # Analyze single-task references
    # --------------------------------------------------------

    print("\nSingle-task reference models:")

    for ref_task in config["tasks"]:

        print()
        print(f"  Single-{ref_task}:")

        ref_checkpoint = (
            Path(config["paths"]["single_checkpoints"]) / f"task_{ref_task}.pt"
        )

        if not ref_checkpoint.exists():
            print(f"    Checkpoint not found: {ref_checkpoint}")
            continue

        # Analyze on task data
        test_loader = create_task_dataloader(
            task_name=ref_task,
            root=dataset_root,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        result = analyze_jacobian_for_model(
            model_name=f"Single-{ref_task}",
            checkpoint_path=ref_checkpoint,
            config=config,
            device=device,
            task_dataloader=test_loader,
            task=ref_task,
        )

        results["jacobians"][f"Single-{ref_task}"] = result

    # --------------------------------------------------------
    # Analyze sequential model on each task
    # --------------------------------------------------------

    print()
    print("Sequential model:")

    seq_checkpoint = (
        Path(config["paths"]["sequential_checkpoints"])
        / order_name
        / f"{order_name}.pt"
    )

    if not seq_checkpoint.exists():
        print(f"Checkpoint not found: {seq_checkpoint}")
        return None

    for task in order:

        print()
        print(f"  Task {task}:")

        test_loader = create_task_dataloader(
            task_name=task,
            root=dataset_root,
            train=False,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        result = analyze_jacobian_for_model(
            model_name=f"Sequential-{order_name}",
            checkpoint_path=seq_checkpoint,
            config=config,
            device=device,
            task_dataloader=test_loader,
            task=task,
        )

        key = f"Sequential-{order_name}-{task}"
        results["jacobians"][key] = result

    return results


def main():

    print()
    print("=" * 70)
    print("JACOBIAN SENSITIVITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config("configs/experiment.yaml")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print()
    print_device_info(device)

    print()
    print("Note: Computing Jacobians is computationally intensive.")
    max_samples = config["analysis"]["jacobian"]["max_samples"]
    print(f"Each model requires up to {max_samples} backward passes per task.")
    print()

    # --------------------------------------------------------
    # Analyze all sequential orders
    # --------------------------------------------------------

    all_results = {}

    for order in config["experiment"]["sequential_orders"]:

        result = analyze_jacobian_for_order(order, config, device)

        if result is not None:
            order_name = "_".join(result["order"])
            all_results[order_name] = result

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = (
        Path(config["paths"]["results"]) / "jacobian" / "jacobian_analysis.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print()
    print("=" * 70)
    print("JACOBIAN ANALYSIS COMPLETE")
    print("=" * 70)
    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
