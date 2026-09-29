#!/usr/bin/env python3
"""
save_cifar_examples.py

Extracts representative samples for each class in tasks A, B, and C
from CIFAR-100, saving:
1. Individual clearly labeled PNGs organized into task folders.
2. A single high-resolution grid graphic (collage) formatted for slides.
"""

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from torchvision.datasets import CIFAR100
import yaml

# Task definitions matching configs/experiment.yaml
TASKS = {
    "A": ["beaver", "dolphin", "otter", "seal", "whale"],
    "B": ["clock", "keyboard", "lamp", "telephone", "television"],
    "C": ["bicycle", "bus", "motorcycle", "pickup_truck", "train"],
}

# Number of sample images to save per class
SAMPLES_PER_CLASS = 3
OUTPUT_DIR = Path("figures/dataset_examples")


def get_dataset_root():
    """Attempt to read dataset root from configs/experiment.yaml, fallback to ./data."""
    cfg_path = Path("configs/experiment.yaml")
    if cfg_path.exists():
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
            return Path(cfg.get("dataset", {}).get("root", "./data"))
    return Path("./data")


def main():
    root = get_dataset_root()
    print(f"Loading CIFAR-100 dataset from: {root.resolve()}")

    # Download or load CIFAR-100 test split
    dataset = CIFAR100(root=str(root), train=False, download=True)
    class_to_idx = dataset.class_to_idx
    classes = dataset.classes

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Dictionary to store collected images for collage: {task: {class: [images]}}
    collected_samples = {task: {cls_name: [] for cls_name in class_list} for task, class_list in TASKS.items()}

    # Scan dataset to collect balanced samples
    for img, target in dataset:
        target_name = classes[target]
        for task_name, task_classes in TASKS.items():
            if target_name in task_classes:
                if len(collected_samples[task_name][target_name]) < SAMPLES_PER_CLASS:
                    collected_samples[task_name][target_name].append(img)
                break

        # Check if all classes have enough samples
        done = all(
            len(collected_samples[t][c]) == SAMPLES_PER_CLASS
            for t in TASKS
            for c in TASKS[t]
        )
        if done:
            break

    # 1. Save individual labeled images
    print("\nSaving individual images...")
    for task_name, class_dict in collected_samples.items():
        task_dir = OUTPUT_DIR / f"Task_{task_name}"
        task_dir.mkdir(parents=True, exist_ok=True)

        for class_name, img_list in class_dict.items():
            for idx, img in enumerate(img_list, start=1):
                # Informative filename format: Task_A_beaver_sample1.png
                filename = f"Task_{task_name}_{class_name}_sample{idx}.png"
                img.save(task_dir / filename)

    print(f"Individual images saved in {OUTPUT_DIR}/Task_A, Task_B, Task_C")

    # 2. Generate presentation summary grid figure
    print("\nGenerating presentation summary grid...")
    num_tasks = len(TASKS)
    classes_per_task = 5
    num_cols = classes_per_task * SAMPLES_PER_CLASS  # 5 * 3 = 15 columns
    num_rows = num_tasks  # 3 rows (Task A, Task B, Task C)

    fig, axes = plt.subplots(
        nrows=num_rows,
        ncols=num_cols,
        figsize=(18, 4.5),
        gridspec_kw={"wspace": 0.1, "hspace": 0.3},
    )

    task_subtitles = {
        "A": "Task A: Aquatic Mammals",
        "B": "Task B: Small Electronic Appliances",
        "C": "Task C: Vehicles 1",
    }

    for row_idx, (task_name, class_dict) in enumerate(collected_samples.items()):
        col_idx = 0
        for class_name, img_list in class_dict.items():
            for sample_idx, img in enumerate(img_list):
                ax = axes[row_idx, col_idx]
                ax.imshow(img)
                ax.set_xticks([])
                ax.set_yticks([])

                # Label the class name above the middle sample of each class
                if row_idx == 0 and sample_idx == 1:
                    ax.set_title(class_name.replace("_", "\n"), fontsize=9, fontweight="bold", pad=4)
                elif sample_idx == 1:
                    ax.set_title(class_name.replace("_", "\n"), fontsize=9, pad=4)

                # Add task row labels on the leftmost image
                if col_idx == 0:
                    ax.set_ylabel(
                        task_subtitles[task_name],
                        fontsize=11,
                        fontweight="bold",
                        labelpad=10,
                    )

                col_idx += 1

    fig.suptitle(
        "CIFAR-100 Superclass Tasks (3 Tasks, 5 Classes Each, 15 Total Classes)",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    collage_path = OUTPUT_DIR / "cifar100_tasks_overview_grid.png"
    plt.savefig(collage_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Presentation grid saved to: {collage_path.resolve()}")


if __name__ == "__main__":
    main()