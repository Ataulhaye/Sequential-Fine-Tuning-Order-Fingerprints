import torch
import torch.nn as nn
from tqdm import tqdm


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    task,
):
    """
    Evaluate a model on one task.
    """

    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(
        dataloader,
        desc=f"Evaluate Task {task}",
        leave=False,
    ):

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        logits = model(
            images,
            task=task,
        )

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += batch_size

    avg_loss = total_loss / total

    accuracy = correct / total

    return avg_loss, accuracy
