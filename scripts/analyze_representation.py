import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sequential_finetuning.checkpoint import load_checkpoint, load_model_from_checkpoint
from sequential_finetuning.config import load_config
from sequential_finetuning.dataset import create_probe_loader
from sequential_finetuning.model import ResNet18MultiTask
from sequential_finetuning.probe import load_probe
from sequential_finetuning.representation import (
    compare_representations,
    extract_features,
)
from sequential_finetuning.training_utils import get_device


def load_single_task_features(config, device, probe_loader):
    checkpoint_root = Path(config["paths"]["single_checkpoints"])
    single_features = {}

    print("\n" + "=" * 70)
    print("LOADING SINGLE-TASK REPRESENTATIONS")
    print("=" * 70)

    for task in ["A", "B", "C"]:
        checkpoint_path = checkpoint_root / f"task_{task}.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Single-task checkpoint not found: {checkpoint_path}"
            )

        print(f"\nLoading Single-{task}\nCheckpoint: {checkpoint_path}")

        model, _ = load_model_from_checkpoint(checkpoint_path, device)
        features = extract_features(model=model, dataloader=probe_loader, device=device)

        print(f"Feature shape: {tuple(features.shape)}")
        single_features[task] = features

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return single_features


def analyze_order(order, config, device, probe_loader, single_features):
    order_name = "_".join(order)
    checkpoint_root = Path(config["paths"]["sequential_checkpoints"])
    checkpoint_path = checkpoint_root / order_name / f"{order_name}.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Sequential checkpoint not found: {checkpoint_path}")

    print("\n\n" + "#" * 70)
    print(f"REPRESENTATION ANALYSIS: {' -> '.join(order)}")
    print("#" * 70)

    model, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    sequential_features = extract_features(
        model=model, dataloader=probe_loader, device=device
    )

    comparisons = {}
    print("\nRepresentation similarities:\n" + "-" * 70)

    tasks = list(config["tasks"].keys())

    for task in tasks:
        metrics = compare_representations(
            reference_features=sequential_features,
            comparison_features=single_features[task],
        )
        comparisons[task] = metrics
        print(
            f"Single-{task}: CKA={metrics['cka']:.6f}, drift={metrics['feature_drift']:.6f}"
        )

    predicted_by_cka = max(comparisons, key=lambda t: comparisons[t]["cka"])
    predicted_by_drift = min(comparisons, key=lambda t: comparisons[t]["feature_drift"])
    actual_last = order[-1]

    cka_correct = predicted_by_cka == actual_last
    drift_correct = predicted_by_drift == actual_last

    print(f"\nCKA prediction:       {predicted_by_cka}")
    print(f"Feature drift pred:   {predicted_by_drift}")
    print(f"Actual last task:     {actual_last}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "order": list(order),
        "actual_last_task": actual_last,
        "checkpoint": str(checkpoint_path),
        "comparisons": comparisons,
        "prediction": {"cka": predicted_by_cka, "feature_drift": predicted_by_drift},
        "correct": {"cka": cka_correct, "feature_drift": drift_correct},
    }


def main():
    config = load_config("configs/experiment.yaml")
    device = get_device()

    probe_path = Path(config["paths"]["results"]) / "representation" / "probe_set.json"
    probe = load_probe(probe_path)

    # Standardized loader instantiation
    probe_loader = create_probe_loader(
        dataset_root=config["dataset"]["root"],
        probe_indices=probe["all_indices"],
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    single_features = load_single_task_features(config, device, probe_loader)

    all_results = []
    for order in config["experiment"]["sequential_orders"]:
        result = analyze_order(order, config, device, probe_loader, single_features)
        all_results.append(result)

    # Save output summary
    result_dir = Path(config["paths"]["results"]) / "representation"
    result_dir.mkdir(parents=True, exist_ok=True)

    results = {
        "probe": {"num_samples": probe["num_samples"], "seed": probe["seed"]},
        "orders": all_results,
    }

    with open(result_dir / "representation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
