"""Create and verify the canonical model initialization checkpoint."""

import torch

from sequential_finetuning.checkpoint import (
    compare_state_dicts,
    create_model_from_initialization,
    ensure_initialization_checkpoint,
    load_initialization_checkpoint,
)
from sequential_finetuning.config import load_config
from sequential_finetuning.seed import set_seed
from sequential_finetuning.training_utils import get_device


def main():
    config = load_config("configs/experiment.yaml")
    set_seed(config["seed"])
    device = get_device()
    initialization_path = config["paths"]["initialization_checkpoint"]

    checkpoint_path = ensure_initialization_checkpoint(
        path=initialization_path,
        config=config,
        device=device,
    )
    checkpoint = load_initialization_checkpoint(
        checkpoint_path,
        map_location=device,
    )
    model = create_model_from_initialization(
        initialization_path=checkpoint_path,
        device=device,
        num_classes_per_task=config["model"]["num_classes"],
        pretrained=config["model"]["pretrained"],
    )
    comparison = compare_state_dicts(
        checkpoint["model_state_dict"],
        model.state_dict(),
    )
    if not comparison["identical"]:
        raise RuntimeError(
            "Canonical initialization checkpoint failed exact reload verification"
        )

    print(f"Base model checkpoint: {checkpoint_path}")
    print(f"Initialization ID: {checkpoint['initialization_id']}")
    print(f"Epoch: {checkpoint['epoch']}")
    print(f"Exact reload: {comparison['identical']}")
    print(f"Maximum parameter difference: {comparison['max_absolute_difference']}")


if __name__ == "__main__":
    main()
