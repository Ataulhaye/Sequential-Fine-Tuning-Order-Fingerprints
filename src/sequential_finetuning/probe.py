import json
import random
from pathlib import Path
from typing import Dict, List

from torchvision import datasets


def create_fixed_probe(
    root: str,
    classes: List[str],
    samples_per_class: int,
    seed: int,
    output_path: str,
) -> Dict:
    """
    Create a deterministic representation probe from the CIFAR-100
    test set.

    The same probe definition can then be reused for every model.
    """

    dataset = datasets.CIFAR100(
        root=root,
        train=False,
        transform=None,
        download=True,
    )

    class_to_idx = {class_name: idx for idx, class_name in enumerate(dataset.classes)}

    rng = random.Random(seed)

    probe_indices = {}
    all_indices = []

    for class_name in classes:

        if class_name not in class_to_idx:
            raise ValueError(f"Unknown CIFAR-100 class: {class_name}")

        original_label = class_to_idx[class_name]

        class_indices = [
            index
            for index, label in enumerate(dataset.targets)
            if label == original_label
        ]

        if samples_per_class > len(class_indices):
            raise ValueError(
                f"Requested {samples_per_class} samples for "
                f"{class_name}, but only {len(class_indices)} exist."
            )

        selected = sorted(
            rng.sample(
                class_indices,
                samples_per_class,
            )
        )

        probe_indices[class_name] = selected
        all_indices.extend(selected)

    probe = {
        "seed": seed,
        "samples_per_class": samples_per_class,
        "num_classes": len(classes),
        "num_samples": len(all_indices),
        "classes": list(classes),
        "indices": probe_indices,
        "all_indices": sorted(all_indices),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            probe,
            file,
            indent=2,
        )

    return probe


def load_probe(
    path: str,
) -> Dict:
    """
    Load an existing fixed probe definition.
    """

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def get_probe_indices(
    probe: Dict,
) -> List[int]:
    """
    Return the complete ordered list of probe indices.
    """

    return probe["all_indices"]
