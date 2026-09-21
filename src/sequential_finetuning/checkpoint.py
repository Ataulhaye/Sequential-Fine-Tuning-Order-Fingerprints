from pathlib import Path

import torch

from sequential_finetuning.model import ResNet18MultiTask


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    task=None,
    order=None,
    metrics=None,
    config=None,
    initialization_id="base_model",
    parent_checkpoint=None,
):
    """
    Save a complete experiment checkpoint.

    The checkpoint contains the model state as well as
    enough metadata to understand how it was produced.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": epoch,
        "task": task,
        "order": order,
        "metrics": metrics,
        "config": config,
        "initialization_id": initialization_id,
        "parent_checkpoint": parent_checkpoint,
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(
        checkpoint,
        path,
    )


def save_initialization_checkpoint(
    path,
    model,
    config=None,
    initialization_id="base_model",
):
    """Save the canonical model state before any task training."""

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "initialization_id": initialization_id,
            "parent_checkpoint": None,
            "task": None,
            "order": [],
            "epoch": 0,
            "metrics": None,
            "config": config,
        },
        path,
    )


def load_initialization_checkpoint(path, map_location=None):
    """Load the canonical initialization checkpoint metadata and state."""

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    if checkpoint.get("initialization_id") != "base_model":
        raise ValueError(f"Invalid initialization checkpoint: {path}")

    return checkpoint


def create_model_from_initialization(
    initialization_path,
    device,
    num_classes_per_task=5,
    pretrained=False,
):
    """Create a model and load the exact canonical initialization state."""

    model = ResNet18MultiTask(
        num_classes_per_task=num_classes_per_task,
        pretrained=pretrained,
    )
    checkpoint = load_initialization_checkpoint(
        initialization_path,
        map_location=device,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    return model.to(device)


def ensure_initialization_checkpoint(path, config, device):
    """Create the canonical initialization checkpoint if it does not exist."""

    path = Path(path)
    if not path.exists():
        model = ResNet18MultiTask(
            num_classes_per_task=5,
            pretrained=config["model"]["pretrained"],
        ).to(device)
        save_initialization_checkpoint(
            path=path,
            model=model,
            config=config,
        )
    return path


def compare_state_dicts(first, second):
    """Return exact tensor difference statistics for two state dictionaries."""

    if first.keys() != second.keys():
        raise ValueError("State dictionaries have different keys")

    max_difference = 0.0
    total_difference = 0.0
    differing_parameters = 0

    for key in first:
        difference = (first[key].detach().cpu() - second[key].detach().cpu()).abs()
        parameter_max = difference.max().item() if difference.numel() else 0.0
        max_difference = max(max_difference, parameter_max)
        total_difference += difference.sum().item()
        if parameter_max != 0.0:
            differing_parameters += 1

    return {
        "max_absolute_difference": max_difference,
        "mean_absolute_difference": total_difference
        / max(sum(value.numel() for value in first.values()), 1),
        "differing_parameters": differing_parameters,
        "identical": max_difference == 0.0,
    }


def load_checkpoint(
    path,
    model,
    optimizer=None,
    map_location=None,
):
    """
    Load a checkpoint into a model.

    Returns
    -------
    checkpoint : dict
        Complete checkpoint metadata.
    """

    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return checkpoint


def load_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    num_classes_per_task: int = 5,
    pretrained: bool = False,
):
    """
    Instantiates a fresh ResNet18MultiTask model, loads saved checkpoint weights,
    transfers it to the target device, and sets it to eval mode.
    """
    model = ResNet18MultiTask(
        num_classes_per_task=num_classes_per_task,
        pretrained=pretrained,
    )

    checkpoint = load_checkpoint(
        path=checkpoint_path,
        model=model,
        map_location=device,
    )

    model.to(device)
    model.eval()

    return model, checkpoint
