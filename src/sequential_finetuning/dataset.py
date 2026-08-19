from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

# ============================================================
# Project task definitions
# ============================================================

TASKS: Dict[str, List[str]] = {
    "A": [
        "beaver",
        "dolphin",
        "otter",
        "seal",
        "whale",
    ],
    "B": [
        "clock",
        "keyboard",
        "lamp",
        "telephone",
        "television",
    ],
    "C": [
        "bicycle",
        "bus",
        "motorcycle",
        "pickup_truck",
        "train",
    ],
}


# ============================================================
# CIFAR-100 normalization
# ============================================================

CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD = (0.2675, 0.2565, 0.2761)


# ============================================================
# Transforms
# ============================================================


def get_train_transform():
    """
    Data augmentation used during training.

    These augmentations will be used consistently across
    all experiments.
    """
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR100_MEAN,
                CIFAR100_STD,
            ),
        ]
    )


def get_test_transform():
    """
    Deterministic preprocessing for evaluation.
    """
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(
                CIFAR100_MEAN,
                CIFAR100_STD,
            ),
        ]
    )


# ============================================================
# CIFAR-100 task dataset
# ============================================================


class CIFAR100Task(Dataset):
    """
    A CIFAR-100 subset containing exactly the classes
    belonging to one project task.

    Original CIFAR-100 labels are remapped to:

        0, 1, 2, 3, 4

    according to the order specified in TASKS.
    """

    def __init__(
        self,
        root: str,
        task_classes: List[str],
        train: bool = True,
        download: bool = True,
    ):
        self.root = Path(root)
        self.task_classes = task_classes
        self.train = train

        transform = get_train_transform() if train else get_test_transform()

        self.base_dataset = datasets.CIFAR100(
            root=self.root,
            train=train,
            transform=transform,
            download=download,
        )

        # ----------------------------------------------------
        # Map CIFAR-100 class names to original class indices
        # ----------------------------------------------------

        class_to_idx = {
            class_name: idx for idx, class_name in enumerate(self.base_dataset.classes)
        }

        # ----------------------------------------------------
        # Check that requested classes exist
        # ----------------------------------------------------

        missing_classes = [
            class_name for class_name in task_classes if class_name not in class_to_idx
        ]

        if missing_classes:
            raise ValueError(f"Unknown CIFAR-100 classes: " f"{missing_classes}")

        # ----------------------------------------------------
        # Original CIFAR-100 labels for this task
        # ----------------------------------------------------

        selected_original_labels = {
            class_to_idx[class_name] for class_name in task_classes
        }

        # ----------------------------------------------------
        # Keep only examples belonging to this task
        # ----------------------------------------------------

        self.indices = [
            idx
            for idx, label in enumerate(self.base_dataset.targets)
            if label in selected_original_labels
        ]

        # ----------------------------------------------------
        # Map original labels → task-local labels
        # ----------------------------------------------------

        self.label_mapping = {
            class_to_idx[class_name]: task_idx
            for task_idx, class_name in enumerate(task_classes)
        }

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        original_index = self.indices[index]

        image, original_label = self.base_dataset[original_index]

        task_label = self.label_mapping[original_label]

        return image, task_label


# ============================================================
# Convenience functions
# ============================================================


def create_task_dataset(
    task_name: str,
    root: str = "./data",
    train: bool = True,
):
    """
    Create the dataset corresponding to task A, B, or C.
    """

    if task_name not in TASKS:
        raise ValueError(
            f"Unknown task '{task_name}'. " f"Available tasks: {list(TASKS.keys())}"
        )

    return CIFAR100Task(
        root=root,
        task_classes=TASKS[task_name],
        train=train,
    )


def create_task_dataloader(
    task_name: str,
    root: str = "./data",
    train: bool = True,
    batch_size: int = 128,
    num_workers: int = 2,
):
    """
    Create a DataLoader for one project task.
    """

    dataset = create_task_dataset(
        task_name=task_name,
        root=root,
        train=train,
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return loader
