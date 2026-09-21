from pathlib import Path

import torch

from sequential_finetuning.checkpoint import (
    compare_state_dicts,
    create_model_from_initialization,
    ensure_initialization_checkpoint,
    load_checkpoint,
    load_initialization_checkpoint,
    save_checkpoint,
    save_initialization_checkpoint,
)
from sequential_finetuning.model import ResNet18MultiTask
from sequential_finetuning.seed import set_seed


def test_canonical_initialization_is_loaded_exactly(tmp_path):
    set_seed(42)
    initialization_path = tmp_path / "initialization" / "base_model.pt"
    config = {"model": {"pretrained": False}}

    ensure_initialization_checkpoint(initialization_path, config, torch.device("cpu"))
    first = create_model_from_initialization(
        initialization_path, torch.device("cpu"), pretrained=False
    )
    second = create_model_from_initialization(
        initialization_path, torch.device("cpu"), pretrained=False
    )

    checkpoint = load_initialization_checkpoint(initialization_path)
    assert compare_state_dicts(checkpoint["model_state_dict"], first.state_dict())[
        "identical"
    ]
    assert compare_state_dicts(first.state_dict(), second.state_dict())["identical"]


def test_checkpoint_lineage_metadata_and_parent_state(tmp_path):
    set_seed(42)
    model = ResNet18MultiTask(num_classes_per_task=5, pretrained=False)
    base_path = tmp_path / "base_model.pt"
    child_path = tmp_path / "A.pt"
    grandchild_path = tmp_path / "A_B.pt"
    config = {"model": {"pretrained": False}}

    save_initialization_checkpoint(base_path, model, config)
    save_checkpoint(
        child_path,
        model,
        task="A",
        order=["A"],
        config=config,
        parent_checkpoint=str(base_path),
    )
    load_checkpoint(child_path, model)
    save_checkpoint(
        grandchild_path,
        model,
        task="B",
        order=["A", "B"],
        config=config,
        parent_checkpoint=str(child_path),
    )

    child = torch.load(child_path, weights_only=False)
    grandchild = torch.load(grandchild_path, weights_only=False)
    assert child["initialization_id"] == "base_model"
    assert child["parent_checkpoint"] == str(base_path)
    assert grandchild["parent_checkpoint"] == str(child_path)
    assert compare_state_dicts(
        child["model_state_dict"], grandchild["model_state_dict"]
    )["identical"]
    assert Path(child["parent_checkpoint"]).exists()
