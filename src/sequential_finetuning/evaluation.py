from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    task: str,
) -> Dict[str, float]:
    """
    Evaluate a model on one task.
    """

    model.eval()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, targets in tqdm(
        dataloader,
        desc=f"Task {task}",
        leave=False,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        targets = targets.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Use the task-specific classification head
        # ----------------------------------------------------

        outputs = model(
            images,
            task,
        )

        loss = criterion(
            outputs,
            targets,
        )

        batch_size = images.size(0)

        total_loss += loss.item() * batch_size

        predictions = outputs.argmax(dim=1)

        correct += (predictions == targets).sum().item()

        total += batch_size

    if total == 0:
        raise RuntimeError("Evaluation dataset is empty.")

    return {
        "loss": total_loss / total,
        "accuracy": correct / total,
    }
