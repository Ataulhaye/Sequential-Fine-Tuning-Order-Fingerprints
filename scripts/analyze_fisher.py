import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

# ============================================================
# Make project root importable
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# Project imports
# ============================================================

from sequential_finetuning.analysis.fisher import (
    HYPOTHESES,
    compute_fisher_trace,
    evaluate_fisher_predictions,
    select_sample_indices,
)
from sequential_finetuning.checkpoint import (
    load_model_from_checkpoint,
)
from sequential_finetuning.config import (
    load_config,
)
from sequential_finetuning.dataset import (
    create_task_dataset,
    get_test_transform,
)
from sequential_finetuning.training_utils import (
    get_device,
)

# ============================================================
# Fisher evaluation data
# ============================================================


def create_fisher_loaders(config, fisher_config):
    """
    Build one fixed sample subset per task.

    The same images are reused for every checkpoint, so Fisher traces are
    directly comparable across orders.
    """

    use_train_split = fisher_config["split"] == "train"

    loaders = {}
    metadata = {}

    for task in config["tasks"]:

        dataset = create_task_dataset(
            task_name=task,
            tasks_config=config,
            root=config["dataset"]["root"],
            train=use_train_split,
        )

        # Deterministic preprocessing, also for the train split, so that the
        # Fisher estimate is reproducible.
        dataset.base_dataset.transform = get_test_transform()

        indices = select_sample_indices(
            dataset_size=len(dataset),
            num_samples=fisher_config["num_samples"],
            seed=fisher_config["seed"],
        )

        loaders[task] = DataLoader(
            Subset(dataset, indices),
            batch_size=fisher_config["batch_size"],
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        metadata[task] = {
            "available_samples": len(dataset),
            "used_samples": len(indices),
        }

        print(
            f"Task {task}: {len(indices)} of {len(dataset)} "
            f"{fisher_config['split']} samples selected."
        )

    return loaders, metadata


# ============================================================
# Analyze one sequential order
# ============================================================


def analyze_sequential_order(
    order,
    loaders,
    config,
    fisher_config,
    device,
):
    """Estimate F_A, F_B and F_C for the final checkpoint of one order."""

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
    print(f"FISHER INFORMATION: {' -> '.join(order)}")
    print("=" * 70)

    print(f"Final checkpoint:\n{checkpoint_path}")

    model, checkpoint = load_model_from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=device,
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )

    print(f"Checkpoint order: {checkpoint.get('order')}")

    traces = {}

    for task in config["tasks"]:

        result = compute_fisher_trace(
            model=model,
            dataloader=loaders[task],
            task=task,
            device=device,
            seed=fisher_config["seed"],
        )

        traces[task] = result

        print(
            f"F_{task} = {result['fisher_total']:.6e}  "
            f"(backbone {result['fisher_backbone']:.6e}, "
            f"head {result['fisher_head']:.6e}, "
            f"n = {result['num_samples']})"
        )

    total_scores = {task: traces[task]["fisher_total"] for task in traces}
    backbone_scores = {task: traces[task]["fisher_backbone"] for task in traces}

    predictions = {
        "fisher_total": evaluate_fisher_predictions(total_scores, order),
        "fisher_backbone": evaluate_fisher_predictions(backbone_scores, order),
    }

    print()
    print(f"Actual order: {' -> '.join(order)}")

    for hypothesis in HYPOTHESES:
        guess = predictions["fisher_total"][hypothesis]
        print(
            f"{hypothesis:>15}: {' -> '.join(guess['predicted_order'])}  "
            f"(order correct: {guess['order_correct']}, "
            f"last task correct: {guess['last_task_correct']})"
        )

    del model

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "order": list(order),
        "actual_last_task": order[-1],
        "checkpoint": str(checkpoint_path),
        "fisher": traces,
        "scores": {
            "fisher_total": total_scores,
            "fisher_backbone": backbone_scores,
        },
        "predictions": predictions,
    }


# ============================================================
# Main
# ============================================================


def main():

    config = load_config("configs/experiment.yaml")

    fisher_config = config["analysis"]["fisher"]

    if fisher_config["split"] not in ("train", "test"):
        raise ValueError("analysis.fisher.split must be 'train' or 'test'.")

    sequential_orders = config["experiment"]["sequential_orders"]

    if not sequential_orders:
        raise ValueError("No sequential orders found in configs/experiment.yaml")

    device = get_device()

    print("=" * 70)
    print("FISHER INFORMATION ANALYSIS")
    print("=" * 70)

    print(f"Samples per task: {fisher_config['num_samples']}")
    print(f"Split: {fisher_config['split']}")
    print(f"Sampling seed: {fisher_config['seed']}")
    print(f"Number of orders: {len(sequential_orders)}")

    print()
    print("Preparing fixed Fisher sample sets...")

    loaders, sample_metadata = create_fisher_loaders(
        config=config,
        fisher_config=fisher_config,
    )

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
                loaders=loaders,
                config=config,
                fisher_config=fisher_config,
                device=device,
            )
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    total = len(all_results)

    summary = {}

    for metric in ("fisher_total", "fisher_backbone"):

        summary[metric] = {}

        for hypothesis in HYPOTHESES:

            order_correct = sum(
                int(result["predictions"][metric][hypothesis]["order_correct"])
                for result in all_results
            )

            last_task_correct = sum(
                int(result["predictions"][metric][hypothesis]["last_task_correct"])
                for result in all_results
            )

            summary[metric][hypothesis] = {
                "correct_orders": order_correct,
                "order_accuracy": order_correct / total,
                "correct_last_tasks": last_task_correct,
                "last_task_accuracy": last_task_correct / total,
            }

    print()
    print("#" * 70)
    print("FISHER PREDICTION SUMMARY")
    print("#" * 70)

    for metric, hypotheses in summary.items():
        print()
        print(metric)
        print("-" * 70)
        for hypothesis, values in hypotheses.items():
            print(
                f"{hypothesis:>15}: "
                f"full order {values['correct_orders']}/{total} "
                f"({100.0 * values['order_accuracy']:.2f}%), "
                f"last task {values['correct_last_tasks']}/{total} "
                f"({100.0 * values['last_task_accuracy']:.2f}%)"
            )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_dir = Path(config["paths"]["results"]) / "fisher"

    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "fisher.json"

    results = {
        "analysis": "fisher_information",
        "estimator": "true_fisher_sampled_labels",
        "samples": {
            "split": fisher_config["split"],
            "requested_per_task": fisher_config["num_samples"],
            "seed": fisher_config["seed"],
            "per_task": sample_metadata,
        },
        "hypotheses": list(HYPOTHESES),
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
