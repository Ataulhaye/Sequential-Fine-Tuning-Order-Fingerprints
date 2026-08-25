from collections import Counter

from sequential_finetuning.config import load_config
from sequential_finetuning.dataset import create_task_dataset


def inspect_task(task_name: str, tasks_config: dict):
    print("=" * 70)
    print(f"TASK {task_name}")
    print("=" * 70)

    print("Classes:")

    for index, class_name in enumerate(tasks_config[task_name]):
        print(f"  {index}: {class_name}")

    train_dataset = create_task_dataset(
        task_name=task_name,
        tasks_config=tasks_config,
        root="./data",
        train=True,
    )

    test_dataset = create_task_dataset(
        task_name=task_name,
        tasks_config=tasks_config,
        root="./data",
        train=False,
    )

    train_labels = [train_dataset[index][1] for index in range(len(train_dataset))]

    test_labels = [test_dataset[index][1] for index in range(len(test_dataset))]

    print()
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples:     {len(test_dataset)}")

    print()
    print("Training distribution:")
    print(Counter(train_labels))

    print()
    print("Test distribution:")
    print(Counter(test_labels))

    print()


def main():
    print("CIFAR-100 Sequential Fine-Tuning Project")
    print()

    # Load tasks configuration once at entry point
    tasks_config = load_config("configs/experiment.yaml")

    for task_name in tasks_config:
        inspect_task(task_name, tasks_config)


if __name__ == "__main__":
    main()
