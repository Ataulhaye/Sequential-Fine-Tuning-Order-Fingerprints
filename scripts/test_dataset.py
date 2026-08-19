from collections import Counter

from sequential_finetuning.dataset import TASKS, create_task_dataset


def inspect_task(task_name: str):
    print("=" * 70)
    print(f"TASK {task_name}")
    print("=" * 70)

    print("Classes:")

    for index, class_name in enumerate(TASKS[task_name]):
        print(f"  {index}: {class_name}")

    train_dataset = create_task_dataset(
        task_name=task_name,
        root="./data",
        train=True,
    )

    test_dataset = create_task_dataset(
        task_name=task_name,
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

    for task_name in TASKS:
        inspect_task(task_name)


if __name__ == "__main__":
    main()
