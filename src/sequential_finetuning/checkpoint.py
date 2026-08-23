from pathlib import Path

import torch


def save_checkpoint(
    path,
    model,
    optimizer=None,
    epoch=None,
    task=None,
    order=None,
    metrics=None,
    config=None,
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
    }

    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()

    torch.save(
        checkpoint,
        path,
    )


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
