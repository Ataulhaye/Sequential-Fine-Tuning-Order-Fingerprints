import json
import sys
from pathlib import Path

import torch

from sequential_finetuning.training_utils import get_device

# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from sequential_finetuning.checkpoint import (
    load_checkpoint,
    load_model_from_checkpoint,
)
from sequential_finetuning.config import (
    load_config,
)
from sequential_finetuning.dataset import (
    create_task_dataloader,
)
from sequential_finetuning.evaluation import (
    evaluate,
)
from sequential_finetuning.forgetting import (
    calculate_forgetting,
)
from sequential_finetuning.model import (
    ResNet18MultiTask,
)

# ============================================================
# Evaluate one sequential order
# ============================================================


def evaluate_sequential_order(
    order,
    config,
    device,
):
    """
    Evaluate all checkpoints belonging to one sequential order.

    Example:

        A → B → C

    evaluates:

        A.pt
        A_B.pt
        A_B_C.pt
    """

    print()
    print("#" * 70)
    print(f"EVALUATING ORDER: {' → '.join(order)}")
    print("#" * 70)

    dataset_root = config["dataset"]["root"]

    batch_size = config["training"]["batch_size"]

    num_workers = config["training"]["num_workers"]

    checkpoint_root = Path(config["paths"]["sequential_checkpoints"])

    result_root = Path(config["paths"]["results"])

    order_name = "_".join(order)

    checkpoint_dir = checkpoint_root / order_name

    if not checkpoint_dir.exists():

        raise FileNotFoundError("Checkpoint directory not found:\n" f"{checkpoint_dir}")

    stage_results = []

    # ========================================================
    # Evaluate every stage
    # ========================================================

    for stage_index in range(
        1,
        len(order) + 1,
    ):

        task_history = order[:stage_index]

        checkpoint_name = f"{'_'.join(task_history)}.pt"

        checkpoint_path = checkpoint_dir / checkpoint_name

        print()
        print("=" * 70)

        print(f"STAGE {stage_index}: " f"{' → '.join(task_history)}")

        print("=" * 70)

        if not checkpoint_path.exists():

            raise FileNotFoundError("Checkpoint not found:\n" f"{checkpoint_path}")

        print(f"Checkpoint:" f"\n{checkpoint_path}")

        # ----------------------------------------------------
        # Load model
        # ----------------------------------------------------

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path,
            device,
        )

        print(f"Checkpoint task: " f"{checkpoint.get('task')}")

        print(f"Checkpoint order: " f"{checkpoint.get('order')}")

        # ----------------------------------------------------
        # Evaluate every task learned so far
        # ----------------------------------------------------

        evaluations = []

        for task in task_history:

            print()
            print(f"Evaluating Task {task}...")

            test_loader = create_task_dataloader(
                task_name=task,
                root=dataset_root,
                train=False,
                batch_size=batch_size,
                num_workers=num_workers,
            )

            metrics = evaluate(
                model=model,
                dataloader=test_loader,
                device=device,
                task=task,
            )

            print(
                f"Task {task}: "
                f"loss={metrics['loss']:.4f}, "
                f"accuracy="
                f"{100.0 * metrics['accuracy']:.2f}%"
            )

            evaluations.append(
                {
                    "task": task,
                    "loss": metrics["loss"],
                    "accuracy": metrics["accuracy"],
                }
            )

        # ----------------------------------------------------
        # Store stage results
        # ----------------------------------------------------

        stage_results.append(
            {
                "stage": stage_index,
                "learned_task": task_history[-1],
                "task_history": list(task_history),
                "checkpoint": str(checkpoint_path),
                "evaluations": evaluations,
            }
        )

        # ----------------------------------------------------
        # Release model
        # ----------------------------------------------------

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================
    # Forgetting
    # ========================================================

    forgetting = calculate_forgetting(stage_results)

    print()
    print("=" * 70)
    print("FORGETTING SUMMARY")
    print("=" * 70)

    for task, value in forgetting.items():

        print(f"Task {task}: " f"{100.0 * value:.2f} " f"percentage points")

    # ========================================================
    # Save results
    # ========================================================

    result_dir = result_root / "sequential" / order_name

    result_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results = {
        "order": list(order),
        "stages": stage_results,
        "forgetting": forgetting,
    }

    results_path = result_dir / "metrics.json"

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
    print(f"Results saved to:")
    print(results_path)

    return results


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
    print("SEQUENTIAL CHECKPOINT EVALUATION")
    print("=" * 70)

    print(f"Number of orders: " f"{len(sequential_orders)}")

    print(f"Device: {device}")

    # --------------------------------------------------------
    # Evaluate all configured orders
    # --------------------------------------------------------

    all_results = []

    for experiment_index, order in enumerate(
        sequential_orders,
        start=1,
    ):

        print()
        print()
        print("#" * 70)

        print(f"EVALUATION " f"{experiment_index}/" f"{len(sequential_orders)}")

        print(f"Order: {' → '.join(order)}")

        print("#" * 70)

        result = evaluate_sequential_order(
            order=order,
            config=config,
            device=device,
        )

        all_results.append(result)

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print()
    print()
    print("#" * 70)
    print("ALL SEQUENTIAL EVALUATIONS COMPLETE")
    print("#" * 70)

    for result in all_results:

        order = result["order"]

        forgetting = result["forgetting"]

        print()
        print(f"Order: {' → '.join(order)}")

        print(f"Last task: {order[-1]}")

        print(
            "Forgetting: "
            + ", ".join(
                [
                    f"{task}=" f"{100.0 * value:.2f} pp"
                    for task, value in forgetting.items()
                ]
            )
        )

    print()
    print(f"Completed " f"{len(all_results)} " f"evaluations.")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()
