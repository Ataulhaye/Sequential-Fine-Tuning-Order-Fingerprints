import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sequential_finetuning.checkpoint import load_model_from_checkpoint
from sequential_finetuning.config import load_config
from sequential_finetuning.dataset import (
    create_all_test_loader,
    create_probe_loader,
    get_project_classes,
)
from sequential_finetuning.probe import load_probe
from sequential_finetuning.representation import (
    compare_representations,
    extract_features,
)
from sequential_finetuning.training_utils import get_device


def main():

    config = load_config("configs/experiment.yaml")

    device = get_device()

    print("=" * 70)
    print("REPRESENTATION ANALYSIS TEST")
    print("=" * 70)

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Load representation evaluation set (probe or all_test)
    # --------------------------------------------------------

    probe_enabled = config["representation"]["probe"]["enabled"]
    if not isinstance(probe_enabled, bool):
        raise ValueError("representation.probe.enabled must be true or false.")
    representation_mode = "probe" if probe_enabled else "all_test"

    print()
    print(f"Representation evaluation set mode: {representation_mode}")

    if representation_mode == "probe":
        probe_path = (
            Path(config["paths"]["results"]) / "representation" / "probe_set.json"
        )

        probe = load_probe(probe_path)

        probe_indices = probe["all_indices"]

        print(f"  Samples: {len(probe_indices)}")
        print(f"  Seed:    {probe['seed']}")

        probe_loader = create_probe_loader(
            dataset_root=config["dataset"]["root"],
            probe_indices=probe_indices,
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
        )
    else:
        probe_loader, all_test_indices = create_all_test_loader(
            dataset_root=config["dataset"]["root"],
            classes=get_project_classes(config["tasks"]),
            batch_size=config["training"]["batch_size"],
            num_workers=config["training"]["num_workers"],
        )

        print(f"  Samples: {len(all_test_indices)}")

    # --------------------------------------------------------
    # Single-task checkpoints
    # --------------------------------------------------------

    checkpoint_root = Path(config["paths"]["single_checkpoints"])

    checkpoints = {
        "A": checkpoint_root / "task_A.pt",
        "B": checkpoint_root / "task_B.pt",
        "C": checkpoint_root / "task_C.pt",
    }

    representations = {}

    # --------------------------------------------------------
    # Extract features
    # --------------------------------------------------------

    for task, checkpoint_path in checkpoints.items():

        print()
        print("-" * 70)
        print(f"Loading Single-{task}")
        print("-" * 70)

        model, checkpoint = load_model_from_checkpoint(
            checkpoint_path,
            device,
        )

        features = extract_features(
            model=model,
            dataloader=probe_loader,
            device=device,
        )

        representations[task] = features

        print(f"Feature shape: {tuple(features.shape)}")

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --------------------------------------------------------
    # Basic sanity checks
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("PAIRWISE SINGLE-TASK REPRESENTATION CHECK")
    print("=" * 70)

    for first, second in [
        ("A", "B"),
        ("A", "C"),
        ("B", "C"),
    ]:

        metrics = compare_representations(
            representations[first],
            representations[second],
        )

        print()
        print(f"Single-{first} vs Single-{second}")

        print(f"CKA:           {metrics['cka']:.6f}")

        print(f"Feature drift: {metrics['feature_drift']:.6f}")

    print()
    print("=" * 70)
    print("REPRESENTATION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
