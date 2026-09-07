from pathlib import Path

import torch

from sequential_finetuning.checkpoint import (
    save_checkpoint,
)
from sequential_finetuning.config import (
    load_config,
)
from sequential_finetuning.dataset import (
    create_task_dataloader,
)
from sequential_finetuning.evaluation import (
    evaluate,
)
from sequential_finetuning.model import (
    ResNet18MultiTask,
)
from sequential_finetuning.seed import (
    set_seed,
)
from sequential_finetuning.train import (
    train_one_epoch,
)
from sequential_finetuning.training_utils import (
    create_optimizer_and_scheduler,
)


def train_single_task(
    task,
    config,
    device,
):
    """
    Train one independent model on one task.
    """

    print()
    print("=" * 70)
    print(f"SINGLE-TASK TRAINING: {task}")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(config["seed"])

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    dataset_root = config["dataset"]["root"]

    batch_size = config["training"]["batch_size"]

    num_workers = config["training"]["num_workers"]

    epochs = config["training"]["epochs_per_task"]

    learning_rate = config["training"]["learning_rate"]

    momentum = config["training"]["momentum"]

    weight_decay = config["training"]["weight_decay"]

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------

    train_loader = create_task_dataloader(
        task_name=task,
        root=dataset_root,
        train=True,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    test_loader = create_task_dataloader(
        task_name=task,
        root=dataset_root,
        train=False,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = ResNet18MultiTask(
        num_classes_per_task=5,
        pretrained=config["model"]["pretrained"],
    )

    model = model.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer, scheduler = create_optimizer_and_scheduler(
        model=model,
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        epochs=epochs,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    for epoch in range(epochs):

        print()
        print(f"Task {task} " f"Epoch {epoch + 1}/{epochs}")

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            task=task,
        )

        print(f"Train loss: " f"{train_loss:.4f}")

        print(f"Train accuracy: " f"{100.0 * train_accuracy:.2f}%")

        scheduler.step()

        print(f"Learning rate: " f"{scheduler.get_last_lr()[0]:.6f}")

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    test_results = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        task=task,
    )
    test_loss = test_results["loss"]

    test_accuracy = test_results["accuracy"]
    print()
    print(f"Final Task {task} test loss: " f"{test_loss:.4f}")

    print(f"Final Task {task} test accuracy: " f"{100.0 * test_accuracy:.2f}%")

    # --------------------------------------------------------
    # Save checkpoint
    # --------------------------------------------------------

    checkpoint_dir = Path(config["paths"]["single_checkpoints"])
    # add date and time to the checkpoint filename to avoid overwriting previous checkpoints

    # timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    checkpoint_path = checkpoint_dir / f"task_{task}.pt"

    save_checkpoint(
        path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epochs,
        task=task,
        order=[task],
        metrics={
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        config=config,
    )

    print()
    print(f"Checkpoint saved to:")

    print(checkpoint_path)

    return {
        "task": task,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "checkpoint": str(checkpoint_path),
    }


def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config("configs/experiment.yaml")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")

    if device.type == "cuda":
        print(f"GPU: " f"{torch.cuda.get_device_name(0)}")

    # --------------------------------------------------------
    # Train A, B, C
    # --------------------------------------------------------

    results = []

    for task in ["A", "B", "C"]:

        result = train_single_task(
            task=task,
            config=config,
            device=device,
        )

        results.append(result)

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SINGLE-TASK TRAINING SUMMARY")
    print("=" * 70)

    for result in results:

        print(f"Task {result['task']}: " f"{100.0 * result['test_accuracy']:.2f}%")


if __name__ == "__main__":
    main()
