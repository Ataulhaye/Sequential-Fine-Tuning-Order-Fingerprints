import torch.nn as nn
from tqdm import tqdm


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    task,
):
    """
    Train the model for one epoch on one task.

    Returns
    -------
    avg_loss : float
    accuracy : float
    """

    model.train()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        dataloader,
        desc=f"Task {task}",
        leave=True,
    )

    for images, labels in progress_bar:

        images = images.to(
            device,
            non_blocking=True,
        )

        labels = labels.to(
            device,
            non_blocking=True,
        )

        # ----------------------------------------------------
        # Forward
        # ----------------------------------------------------

        optimizer.zero_grad(set_to_none=True)

        logits = model(
            images,
            task=task,
        )

        loss = criterion(
            logits,
            labels,
        )

        # ----------------------------------------------------
        # Backward
        # ----------------------------------------------------

        loss.backward()

        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        batch_size = labels.size(0)

        total_loss += loss.item() * batch_size

        predictions = logits.argmax(dim=1)

        correct += (predictions == labels).sum().item()

        total += batch_size

        progress_bar.set_postfix(
            loss=f"{loss.item():.4f}",
            accuracy=f"{100.0 * correct / total:.2f}%",
        )

    avg_loss = total_loss / total

    accuracy = correct / total

    return avg_loss, accuracy
