from sequential_finetuning import dataset as dataset_module
from sequential_finetuning.combined_analysis import combine_all_results


class FakeCIFAR100:
    classes = ["project_a", "project_b", "outside"]
    targets = [0, 2, 1, 2, 0, 1]

    def __init__(self, root, train, download, transform):
        self.transform = transform

    def __getitem__(self, index):
        return index, self.targets[index]


def test_all_test_loader_selects_all_project_images_in_dataset_order(monkeypatch):
    monkeypatch.setattr(dataset_module, "CIFAR100", FakeCIFAR100)

    loader, indices = dataset_module.create_all_test_loader(
        dataset_root="unused",
        classes=["project_a", "project_b"],
        batch_size=2,
        num_workers=0,
    )

    # Class "outside" (label 1) must be excluded; no subsampling occurs.
    assert indices == [0, 2, 4, 5]
    assert list(loader.dataset.indices) == indices
    assert loader.sampler.__class__.__name__ == "SequentialSampler"


def test_project_classes_are_unique_and_preserve_task_order():
    classes = dataset_module.get_project_classes({"A": ["a", "b"], "B": ["b", "c"]})

    assert classes == ["a", "b", "c"]


def test_combined_results_preserve_representation_evaluation_metadata():
    representation = {
        "evaluation_set": {
            "mode": "all_test",
            "num_samples": 4,
            "num_classes": 2,
            "classes": ["project_a", "project_b"],
        },
        "orders": [
            {
                "order": ["A", "B"],
                "comparisons": {"A": {"cka": 0.5, "feature_drift": 0.4}},
                "prediction": {"cka": "A", "feature_drift": "A"},
                "correct": {"cka": True, "feature_drift": True},
                "checkpoint": "checkpoint.pt",
            }
        ],
    }
    sequential = {"orders": [{"order": ["A", "B"], "forgetting": {"A": 0.0, "B": 0.0}}]}
    weights = {
        "orders": [
            {
                "order": ["A", "B"],
                "full_model": {
                    "distances": {"A": 1.0, "B": 2.0},
                    "predicted_last_task": "A",
                    "correct": True,
                },
                "backbone": {
                    "distances": {"A": 1.0, "B": 2.0},
                    "predicted_last_task": "A",
                    "correct": True,
                },
                "checkpoint": "checkpoint.pt",
            }
        ]
    }

    combined = combine_all_results(
        orders=[["A", "B"]],
        sequential_results=sequential,
        weight_results=weights,
        representation_results=representation,
    )

    assert combined["orders"][0]["representation"]["evaluation_set"] == "all_test"
