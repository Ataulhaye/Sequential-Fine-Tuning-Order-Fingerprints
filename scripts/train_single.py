import time
from pathlib import Path

from sequential_finetuning.checkpoint import (
    create_model_from_initialization,
    ensure_initialization_checkpoint,
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
from sequential_finetuning.seed import (
    set_seed,
)
from sequential_finetuning.train import (
    train_one_epoch,
)
from sequential_finetuning.training_utils import (
    create_optimizer_and_scheduler,
    get_device,
)


def train_single_task(
    task,
    config,
    device,
):
    """
    Train one independent model on one task.
    """
    task_start_time = time.perf_counter()
    print()
    print("=" * 70)
    print(f"SINGLE-TASK TRAINING: {task}")
    print("=" * 70)

    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(config["seed"])

    initialization_path = ensure_initialization_checkpoint(
        path=config["paths"]["initialization_checkpoint"],
        config=config,
        device=device,
    )

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

    model = create_model_from_initialization(
        initialization_path=initialization_path,
        device=device,
        num_classes_per_task=5,
        pretrained=config["model"]["pretrained"],
    )

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
        parent_checkpoint=str(initialization_path),
        metrics={
            "test_loss": test_loss,
            "test_accuracy": test_accuracy,
        },
        config=config,
    )

    print()
    print(f"Checkpoint saved to:")

    print(checkpoint_path)

    task_duration = time.perf_counter() - task_start_time

    print()
    print(
        f"Task {task} total time: {task_duration:.2f} seconds "
        f"({task_duration / 60:.2f} minutes)"
    )

    return {
        "task": task,
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "checkpoint": str(checkpoint_path),
        "duration_seconds": task_duration,
    }


def main():

    # --------------------------------------------------------
    # Configuration
    # --------------------------------------------------------

    config = load_config("configs/experiment.yaml")

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    # --------------------------------------------------------
    # Train A, B, C
    # --------------------------------------------------------

    results = []

    order = config["experiment"]["sequential_orders"][0]

    total_start_time = time.perf_counter()

    for task in order:

        result = train_single_task(
            task=task,
            config=config,
            device=device,
        )

        results.append(result)

    total_duration = time.perf_counter() - total_start_time

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("SINGLE-TASK TRAINING SUMMARY")
    print("=" * 70)

    for result in results:

        print(
            f"Task {result['task']}: "
            f"{100.0 * result['test_accuracy']:.2f}% "
            f"(Time: {result['duration_seconds'] / 60:.2f} min)"
        )
    print()
    print("-" * 70)
    print(
        f"Total training time: "
        f"{total_duration:.2f} seconds "
        f"({total_duration / 60:.2f} minutes)"
    )

    avg_task_time = total_duration / len(results)

    print(
        f"Average task time: "
        f"{avg_task_time:.2f} seconds "
        f"({avg_task_time / 60:.2f} minutes)"
    )


if __name__ == "__main__":
    main()
