import torch

from sequential_finetuning.analysis.weight_matching import (
    align_state_dict,
    apply_permutation,
    parameter_keys,
    per_layer_task_vector_cosine_similarity,
    resnet18_multitask_permutation_spec,
    state_dict_l2_distance,
    task_vector_cosine_similarity,
    validate_permutation_spec,
    verify_functional_equivalence,
)
from sequential_finetuning.model import ResNet18MultiTask
from sequential_finetuning.seed import set_seed

TASKS = ("A", "B", "C")


def create_model(seed: int) -> ResNet18MultiTask:
    set_seed(seed)
    model = ResNet18MultiTask(num_classes_per_task=5, pretrained=False)

    # Non-trivial batch-norm statistics make the invariance check meaningful.
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            module.running_mean.normal_()
            module.running_var.uniform_(0.5, 1.5)

    return model.eval()


def random_permutation(spec, state_dict, seed: int):
    generator = torch.Generator()
    generator.manual_seed(seed)

    return {
        permutation_name: torch.randperm(
            state_dict[axes[0][0]].shape[axes[0][1]],
            generator=generator,
        )
        for permutation_name, axes in spec.perm_to_axes.items()
    }


def test_permutation_spec_covers_every_checkpoint_parameter():
    model = create_model(seed=0)
    spec = resnet18_multitask_permutation_spec(TASKS)

    validate_permutation_spec(spec, model.state_dict())


def test_permuting_neurons_does_not_change_outputs():
    model = create_model(seed=1)
    spec = resnet18_multitask_permutation_spec(TASKS)

    permutation = random_permutation(spec, model.state_dict(), seed=7)

    permuted_model = ResNet18MultiTask(num_classes_per_task=5, pretrained=False)
    permuted_model.load_state_dict(
        apply_permutation(spec, permutation, model.state_dict())
    )

    images = torch.randn(8, 3, 32, 32)

    differences = verify_functional_equivalence(
        original_model=model,
        permuted_model=permuted_model,
        images=images,
        tasks=TASKS,
    )

    assert differences["max"] < 1e-4


def test_weight_matching_recovers_a_known_permutation():
    model = create_model(seed=2)
    spec = resnet18_multitask_permutation_spec(TASKS)

    reference_state = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }

    permuted_state = apply_permutation(
        spec,
        random_permutation(spec, reference_state, seed=11),
        reference_state,
    )

    keys = parameter_keys(model)

    distance_before = state_dict_l2_distance(permuted_state, reference_state, keys)

    aligned_state, _, info = align_state_dict(
        reference_state=reference_state,
        target_state=permuted_state,
        spec=spec,
        max_iterations=50,
    )

    distance_after = state_dict_l2_distance(aligned_state, reference_state, keys)

    assert distance_before > 0.0
    assert distance_after < 1e-5
    assert info["converged"]


def test_weight_matching_reduces_distance_between_independent_models():
    reference = create_model(seed=3)
    target = create_model(seed=4)

    spec = resnet18_multitask_permutation_spec(TASKS)
    keys = parameter_keys(reference)

    aligned_state, _, _ = align_state_dict(
        reference_state=reference.state_dict(),
        target_state=target.state_dict(),
        spec=spec,
        max_iterations=20,
    )

    distance_before = state_dict_l2_distance(
        target.state_dict(), reference.state_dict(), keys
    )
    distance_after = state_dict_l2_distance(aligned_state, reference.state_dict(), keys)

    assert distance_after <= distance_before


def test_task_vector_cosine_similarity_is_scale_invariant():
    initial = create_model(seed=5).state_dict()

    keys = ["heads.A.weight"]

    reference = {key: initial[key] + torch.ones_like(initial[key]) for key in keys}

    final_small = {
        key: initial[key] + 0.1 * torch.ones_like(initial[key]) for key in keys
    }
    final_large = {
        key: initial[key] + 10.0 * torch.ones_like(initial[key]) for key in keys
    }

    small_similarity = task_vector_cosine_similarity(
        final_state=final_small,
        reference_state=reference,
        initial_state=initial,
        keys=keys,
    )
    large_similarity = task_vector_cosine_similarity(
        final_state=final_large,
        reference_state=reference,
        initial_state=initial,
        keys=keys,
    )

    assert abs(small_similarity - 1.0) < 1e-5
    assert abs(large_similarity - 1.0) < 1e-5


def test_per_layer_cosine_similarity_reports_every_layer():
    initial = create_model(seed=6)
    keys = parameter_keys(initial)

    initial_state = {
        key: value.detach().clone().float()
        for key, value in initial.state_dict().items()
    }

    reference_state = {
        key: initial_state[key] + 0.01 * torch.randn_like(initial_state[key])
        for key in keys
    }
    final_state = {
        key: initial_state[key] + 0.02 * torch.randn_like(initial_state[key])
        for key in keys
    }

    similarities = per_layer_task_vector_cosine_similarity(
        final_state=final_state,
        reference_state=reference_state,
        initial_state=initial_state,
        keys=keys,
    )

    assert len(similarities) == len({key.rsplit(".", 1)[0] for key in keys})
    assert all(-1.0001 <= value <= 1.0001 for value in similarities.values())
