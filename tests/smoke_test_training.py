import torch

from sequential_finetuning.dataset import (
    create_task_dataloader,
)
from sequential_finetuning.evaluation import (
    evaluate,
)
from sequential_finetuning.model import (
    ResNet18MultiTask,
)
from sequential_finetuning.train import (
    train_one_epoch,
)


def main():

    # ========================================================
    # Configuration
    # ========================================================

    task = "A"

    batch_size = 128

    num_workers = 2

    learning_rate = 0.001

    # ========================================================
    # Device
    # ========================================================

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("ResNet18 GPU Smoke Test")
    print("=" * 70)

    print(f"Device: {device}")

    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

        print(f"CUDA version: {torch.version.cuda}")

    # ========================================================
    # Data
    # ========================================================

    print()
    print("Loading Task A...")

    train_loader = create_task_dataloader(
        task_name=task,
        root="./data",
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = create_task_dataloader(
        task_name=task,
        root="./data",
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    print(f"Training samples: {len(train_loader.dataset)}")

    print(f"Test samples: {len(test_loader.dataset)}")

    # ========================================================
    # Model
    # ========================================================

    print()
    print("Creating ResNet18...")

    model = ResNet18MultiTask(
        num_classes_per_task=5,
        pretrained=False,
    )

    model = model.to(device)

    # ========================================================
    # Parameter count
    # ========================================================

    total_parameters = sum(parameter.numel() for parameter in model.parameters())

    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    print(f"Total parameters: {total_parameters:,}")

    print(f"Trainable parameters: " f"{trainable_parameters:,}")

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # ========================================================
    # Training
    # ========================================================

    print()
    print("Starting one training epoch...")

    train_loss, train_accuracy = train_one_epoch(
        model=model,
        dataloader=train_loader,
        optimizer=optimizer,
        device=device,
        task=task,
    )

    print()
    print("Training complete.")

    print(f"Train loss: {train_loss:.4f}")

    print(f"Train accuracy: " f"{100.0 * train_accuracy:.2f}%")

    # ========================================================
    # Evaluation
    # ========================================================

    print()
    print("Evaluating Task A...")

    test_loss, test_accuracy = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        task=task,
    )

    print()
    print("=" * 70)
    print("RESULT")
    print("=" * 70)

    print(f"Test loss: {test_loss:.4f}")

    print(f"Test accuracy: " f"{100.0 * test_accuracy:.2f}%")

    print("=" * 70)


if __name__ == "__main__":
    main()
