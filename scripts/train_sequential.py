import time
from pathlib import Path

from sequential_finetuning.checkpoint import (
    create_model_from_initialization,
    ensure_initialization_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from sequential_finetuning.config import (
    load_config,
)
from sequential_finetuning.dataset import (
    create_task_dataloader,
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


def train_task_stage(
    model,
    task,
    config,
    device,
    order,
    stage_index,
):
    """
    Train the existing model on one task.

    IMPORTANT:
    The model is NOT reinitialized.

    A new optimizer and scheduler are created for
    this task stage.
    """
    stage_start_time = time.perf_counter()
    print()
    print("=" * 70)
    print(f"SEQUENTIAL STAGE " f"{stage_index}: TASK {task}")
    print(f"Order: {' → '.join(order)}")
    print("=" * 70)

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

    stage_duration = time.perf_counter() - stage_start_time

    print()
    print(
        f"Task {task} stage time: "
        f"{stage_duration:.2f} seconds "
        f"({stage_duration / 60:.2f} minutes)"
    )

    return {
        "task": task,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "duration_seconds": stage_duration,
    }


def train_sequential_order(
    order,
    config,
    device,
):
    """
    Train one complete sequential task order.

    Example:

        A → B → C

    The same model is continuously fine-tuned.
    """
    order_start_time = time.perf_counter()
    print()
    print("#" * 70)
    print(f"SEQUENTIAL EXPERIMENT: " f"{' → '.join(order)}")
    print("#" * 70)

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
    # Create ONE model
    # --------------------------------------------------------

    model = create_model_from_initialization(
        initialization_path=initialization_path,
        device=device,
        num_classes_per_task=5,
        pretrained=config["model"]["pretrained"],
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    order_name = "_".join(order)

    checkpoint_dir = Path(config["paths"]["sequential_checkpoints"]) / order_name

    checkpoint_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Train each task sequentially
    # --------------------------------------------------------

    stage_results = []
    parent_checkpoint = initialization_path

    for stage_index, task in enumerate(
        order,
        start=1,
    ):

        load_checkpoint(
            path=parent_checkpoint,
            model=model,
            map_location=device,
        )

        result = train_task_stage(
            model=model,
            task=task,
            config=config,
            device=device,
            order=order,
            stage_index=stage_index,
        )

        stage_results.append(result)

        # ----------------------------------------------------
        # Save checkpoint after this task
        # ----------------------------------------------------

        checkpoint_name = f"{'_'.join(order[:stage_index])}.pt"

        checkpoint_path = checkpoint_dir / checkpoint_name

        save_checkpoint(
            path=checkpoint_path,
            model=model,
            epoch=config["training"]["epochs_per_task"],
            task=task,
            order=order[:stage_index],
            parent_checkpoint=str(parent_checkpoint),
            metrics=result,
            config=config,
        )

        parent_checkpoint = checkpoint_path

        print()
        print(f"Checkpoint saved:")

        print(checkpoint_path)
    order_duration = time.perf_counter() - order_start_time

    print()
    print("=" * 70)
    print(
        f"Order {' → '.join(order)} completed in "
        f"{order_duration:.2f} seconds "
        f"({order_duration / 60:.2f} minutes)"
    )
    print("=" * 70)

    return {
        "order": order,
        "stages": stage_results,
        "final_checkpoint": str(checkpoint_dir / f"{'_'.join(order)}.pt"),
        "duration_seconds": order_duration,
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
    # Read sequential orders from configuration
    # --------------------------------------------------------

    sequential_orders = config["experiment"]["sequential_orders"]

    if not sequential_orders:
        raise ValueError("No sequential orders found in " "configs/experiment.yaml")

    print()
    print("#" * 70)
    print("SEQUENTIAL EXPERIMENT MATRIX")
    print("#" * 70)

    print(f"Number of orders: " f"{len(sequential_orders)}")

    print()

    matrix_start_time = time.perf_counter()

    for index, order in enumerate(
        sequential_orders,
        start=1,
    ):
        print(f"{index}. " f"{' → '.join(order)}")

    # --------------------------------------------------------
    # Run all configured orders
    # --------------------------------------------------------

    all_results = []

    for experiment_index, order in enumerate(
        sequential_orders,
        start=1,
    ):

        print()
        print()
        print("#" * 70)
        print(
            f"SEQUENTIAL EXPERIMENT " f"{experiment_index}/" f"{len(sequential_orders)}"
        )
        print(f"Order: {' → '.join(order)}")
        print("#" * 70)

        result = train_sequential_order(
            order=order,
            config=config,
            device=device,
        )

        all_results.append(result)

        print()
        print("-" * 70)
        print(
            f"Experiment " f"{experiment_index}/" f"{len(sequential_orders)} complete"
        )
        print(f"Order: {' → '.join(result['order'])}")
        print(f"Duration: " f"{result['duration_seconds'] / 60:.2f} minutes")
        print(f"Final checkpoint: " f"{result['final_checkpoint']}")
        print("-" * 70)
    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    matrix_duration = time.perf_counter() - matrix_start_time
    print()
    print()
    print("#" * 70)
    print("ALL SEQUENTIAL EXPERIMENTS COMPLETE")
    print("#" * 70)

    for result in all_results:

        print(f"{' → '.join(result['order'])}")

        print(f"  Duration: " f"{result['duration_seconds'] / 60:.2f} min")

        for stage in result["stages"]:
            print(
                f"    Task {stage['task']}: "
                f"{stage['duration_seconds'] / 60:.2f} min"
            )

        print(f"  Final checkpoint: " f"{result['final_checkpoint']}")

    print()
    print("=" * 70)
    print(
        f"Total matrix time: "
        f"{matrix_duration:.2f} seconds "
        f"({matrix_duration / 3600:.2f} hours)"
    )

    avg_order_time = matrix_duration / len(all_results)

    print(
        f"Average order time: "
        f"{avg_order_time:.2f} seconds "
        f"({avg_order_time / 60:.2f} minutes)"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
