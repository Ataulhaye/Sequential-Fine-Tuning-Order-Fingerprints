"""
Analyze Loss-Barrier Curves

Measures the "smoothness" of the optimization landscape between sequential
and single-task reference models by interpolating in weight space and
evaluating loss at intermediate points.

A low barrier indicates similar loss landscapes.
A high barrier indicates different local minima.
"""

import json
from pathlib import Path

from sequential_finetuning.analysis.loss_barrier import (
    compute_barrier_area,
    compute_barrier_height,
    evaluate_loss_at_interpolation,
    get_model_weights_dict,
)
from sequential_finetuning.checkpoint import load_model_from_checkpoint
from sequential_finetuning.config import load_config
from sequential_finetuning.dataset import create_task_dataloader
from sequential_finetuning.training_utils import get_device, print_device_info


def analyze_loss_barrier_for_order(
    order,
    config,
    device,
):
    """
    Analyze loss barrier for one sequential order against all single-task references.
    """

    print()
    print("#" * 70)
    print(f"LOSS BARRIER ANALYSIS: {' → '.join(order)}")
    print("#" * 70)

    # --------------------------------------------------------
    # Load sequential model (final checkpoint)
    # --------------------------------------------------------

    order_name = "_".join(order)
    sequential_checkpoint = (
        Path(config["paths"]["sequential_checkpoints"])
        / order_name
        / f"{order_name}.pt"
    )

    if not sequential_checkpoint.exists():
        print(f"Checkpoint not found: {sequential_checkpoint}")
        return None

    sequential_model, _ = load_model_from_checkpoint(
        checkpoint_path=sequential_checkpoint,
        device=device,
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    seq_weights = get_model_weights_dict(sequential_model)

    print(f"Loaded sequential model: {sequential_checkpoint}")

    # --------------------------------------------------------
    # Load single-task references
    # --------------------------------------------------------

    results = {
        "order": order,
        "barriers": {},
    }

    dataset_root = config["dataset"]["root"]
    batch_size = config["training"]["batch_size"]
    num_workers = config["training"]["num_workers"]
    num_points = (
        config.get("analysis", {}).get("loss_barrier", {}).get("num_points", 11)
    )
    if num_points < 2:
        raise ValueError("analysis.loss_barrier.num_points must be at least 2.")

    for ref_task in config["tasks"]:

        print()
        print("=" * 70)
        print(f"Barrier: Sequential({' → '.join(order)}) ↔ Single-{ref_task}")
        print("=" * 70)

        # Load reference model
        ref_checkpoint = (
            Path(config["paths"]["single_checkpoints"]) / f"task_{ref_task}.pt"
        )

        if not ref_checkpoint.exists():
            print(f"Checkpoint not found: {ref_checkpoint}")
            continue

        ref_model, _ = load_model_from_checkpoint(
            checkpoint_path=ref_checkpoint,
            device=device,
            num_classes_per_task=config["model"]["num_classes"],
            pretrained=config["model"]["pretrained"],
        )

        ref_weights = get_model_weights_dict(ref_model)

        print(f"Loaded reference model: {ref_checkpoint}")

        # ------------------------------------------------
        # For each task in the sequential order,
        # compute loss barrier on that task's data
        # ------------------------------------------------

        task_barriers = {}

        for task_idx, task in enumerate(order):

            print()
            print(f"  Task {task}:")

            # Load data for this task
            test_loader = create_task_dataloader(
                task_name=task,
                root=dataset_root,
                train=False,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            # Evaluate loss at interpolation points
            alphas, losses = evaluate_loss_at_interpolation(
                model=sequential_model,
                weights_A=ref_weights,
                weights_B=seq_weights,
                dataloader=test_loader,
                device=device,
                task=task,
                num_points=num_points,
            )

            barrier_height = compute_barrier_height(alphas, losses)
            barrier_area = compute_barrier_area(alphas, losses)

            print(f"    Barrier height: {barrier_height:.6f}")
            print(f"    Barrier area:   {barrier_area:.6f}")
            print(f"    Loss range:     [{min(losses):.6f}, {max(losses):.6f}]")

            task_barriers[task] = {
                "alphas": alphas,
                "losses": losses,
                "barrier_height": barrier_height,
                "barrier_area": barrier_area,
                "loss_at_ref": losses[0],
                "loss_at_seq": losses[-1],
                "min_loss": min(losses),
                "max_loss": max(losses),
            }

        results["barriers"][f"Single-{ref_task}"] = task_barriers

    return results


def main():

    print()
    print("=" * 70)
    print("LOSS-BARRIER ANALYSIS")
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

    # --------------------------------------------------------
    # Analyze all sequential orders
    # --------------------------------------------------------

    all_results = {}

    for order in config["experiment"]["sequential_orders"]:

        result = analyze_loss_barrier_for_order(order, config, device)

        if result is not None:
            order_name = "_".join(result["order"])
            all_results[order_name] = result

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    output_path = (
        Path(config["paths"]["results"]) / "loss_barrier" / "loss_barrier.json"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print()
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
