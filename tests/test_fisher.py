import torch
from torch.utils.data import DataLoader, TensorDataset

from sequential_finetuning.analysis.fisher import (
    compute_fisher_trace,
    evaluate_fisher_predictions,
    guess_order_from_fisher,
    select_sample_indices,
)
from sequential_finetuning.model import ResNet18MultiTask
from sequential_finetuning.seed import set_seed


def create_loader(num_samples: int = 4) -> DataLoader:
    set_seed(0)
    images = torch.randn(num_samples, 3, 32, 32)
    labels = torch.randint(0, 5, (num_samples,))
    return DataLoader(TensorDataset(images, labels), batch_size=2, shuffle=False)


def create_model() -> ResNet18MultiTask:
    set_seed(1)
    return ResNet18MultiTask(num_classes_per_task=5, pretrained=False)


def test_sample_selection_is_deterministic():
    first = select_sample_indices(dataset_size=2500, num_samples=1000, seed=42)
    second = select_sample_indices(dataset_size=2500, num_samples=1000, seed=42)

    assert first == second
    assert len(first) == 1000
    assert len(set(first)) == 1000


def test_fisher_trace_is_reproducible_and_positive():
    model = create_model()
    loader = create_loader()
    device = torch.device("cpu")

    first = compute_fisher_trace(model, loader, "A", device, seed=7)
    second = compute_fisher_trace(model, loader, "A", device, seed=7)

    assert first["num_samples"] == 4
    assert first["fisher_total"] > 0.0
    assert first["fisher_total"] == second["fisher_total"]
    assert first["fisher_backbone"] > 0.0
    assert first["fisher_head"] > 0.0


def test_fisher_estimation_does_not_update_weights():
    model = create_model()
    loader = create_loader()

    before = {key: value.clone() for key, value in model.state_dict().items()}

    compute_fisher_trace(model, loader, "B", torch.device("cpu"), seed=3)

    after = model.state_dict()

    assert all(torch.equal(before[key], after[key]) for key in before)


def test_unused_heads_do_not_contribute_to_the_trace():
    model = create_model()
    loader = create_loader()

    result = compute_fisher_trace(model, loader, "A", torch.device("cpu"), seed=5)

    assert abs(
        result["fisher_total"] - (result["fisher_backbone"] + result["fisher_head"])
    ) < 1e-9 * max(result["fisher_total"], 1.0)


def test_both_reading_directions_are_reported():
    scores = {"A": 1.0, "B": 3.0, "C": 2.0}

    assert guess_order_from_fisher(scores, "high_is_recent") == ["A", "C", "B"]
    assert guess_order_from_fisher(scores, "low_is_recent") == ["B", "C", "A"]

    predictions = evaluate_fisher_predictions(scores, ["A", "C", "B"])

    assert predictions["high_is_recent"]["order_correct"]
    assert predictions["high_is_recent"]["last_task_correct"]
    assert not predictions["low_is_recent"]["order_correct"]
