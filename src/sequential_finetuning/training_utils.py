from typing import Tuple

import torch


def create_optimizer_and_scheduler(
    model,
    learning_rate: float,
    momentum: float,
    weight_decay: float,
    epochs: int,
):
    """
    Create a fresh SGD optimizer and cosine scheduler.

    A new optimizer/scheduler is intentionally created for
    every task in sequential fine-tuning.

    The model itself is NOT reinitialized.
    """

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
    )

    return optimizer, scheduler


def get_device() -> torch.device:
    """
    Return CUDA when available, otherwise CPU.
    """

    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def print_device_info(device: torch.device) -> None:
    """
    Print information about the active compute device.
    """

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

        print(f"CUDA: " f"{torch.version.cuda}")
