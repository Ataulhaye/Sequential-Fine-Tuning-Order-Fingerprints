"""
Weight Matching (Git Re-Basin)

Two networks can implement the same function with their neurons in a
different order, so a plain L2 distance between two checkpoints is
misleading. Weight matching removes this symmetry: the neurons (channels) of
one checkpoint are reordered so that they line up with the neurons of a
reference checkpoint, without changing the function the network computes.

The implementation follows the coordinate-descent weight-matching algorithm
of Ainsworth, Hayase and Srinivasa, "Git Re-Basin: Merging Models modulo
Permutation Symmetries" (ICLR 2023), adapted to the ResNet18 backbone with
task-specific heads used in this project.

Residual connections constrain which channels may be permuted: every tensor
that is added into the same residual stream must share one permutation.
"""

from collections import defaultdict
from typing import Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment

# Batch-norm statistics are permuted like the corresponding channels but are
# excluded from the matching cost, as in the reference implementation.
STATISTICS_SUFFIXES = (
    ".running_mean",
    ".running_var",
    ".num_batches_tracked",
)


class PermutationSpec(NamedTuple):
    """Description of which parameter axes are tied to which permutation."""

    perm_to_axes: Dict[str, List[Tuple[str, int]]]
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]]


def permutation_spec_from_axes_to_perm(
    axes_to_perm: Dict[str, Tuple[Optional[str], ...]],
) -> PermutationSpec:
    """Build the inverse index (permutation -> parameter axes)."""

    perm_to_axes: Dict[str, List[Tuple[str, int]]] = defaultdict(list)

    for parameter_name, axis_permutations in axes_to_perm.items():
        for axis, permutation_name in enumerate(axis_permutations):
            if permutation_name is not None:
                perm_to_axes[permutation_name].append((parameter_name, axis))

    return PermutationSpec(
        perm_to_axes=dict(perm_to_axes),
        axes_to_perm=axes_to_perm,
    )


def resnet18_multitask_permutation_spec(
    task_names: Sequence[str] = ("A", "B", "C"),
) -> PermutationSpec:
    """
    Permutation specification for :class:`ResNet18MultiTask`.

    Permutation groups:

    * ``P_stem``   output of ``conv1``/``bn1``; it is also the residual
      stream of ``layer1``, whose blocks use identity shortcuts.
    * ``P_layerX`` residual stream of ``layer2``/``layer3``/``layer4``; the
      downsample convolution writes into the same stream.
    * ``P_layerX_Y_inner`` output of the first convolution inside block ``Y``,
      which is free because it is consumed only by the second convolution.

    The 5 output units of each task head are class labels and are therefore
    never permuted.
    """

    def convolution(name, permutation_in, permutation_out):
        return {f"{name}.weight": (permutation_out, permutation_in, None, None)}

    def normalization(name, permutation):
        return {
            f"{name}.weight": (permutation,),
            f"{name}.bias": (permutation,),
            f"{name}.running_mean": (permutation,),
            f"{name}.running_var": (permutation,),
            f"{name}.num_batches_tracked": (),
        }

    def dense(name, permutation_in, permutation_out):
        return {
            f"{name}.weight": (permutation_out, permutation_in),
            f"{name}.bias": (permutation_out,),
        }

    axes_to_perm: Dict[str, Tuple[Optional[str], ...]] = {}

    axes_to_perm.update(convolution("backbone.conv1", None, "P_stem"))
    axes_to_perm.update(normalization("backbone.bn1", "P_stem"))

    layers = (
        ("layer1", False),
        ("layer2", True),
        ("layer3", True),
        ("layer4", True),
    )

    input_permutation = "P_stem"

    for layer_name, has_downsample in layers:

        # Without a downsample convolution the block adds into the incoming
        # residual stream, so that stream's permutation is reused.
        residual_permutation = (
            f"P_{layer_name}" if has_downsample else input_permutation
        )

        for block_index in range(2):

            prefix = f"backbone.{layer_name}.{block_index}"

            block_input_permutation = (
                input_permutation if block_index == 0 else residual_permutation
            )

            inner_permutation = f"P_{layer_name}_{block_index}_inner"

            axes_to_perm.update(
                convolution(
                    f"{prefix}.conv1",
                    block_input_permutation,
                    inner_permutation,
                )
            )
            axes_to_perm.update(normalization(f"{prefix}.bn1", inner_permutation))

            axes_to_perm.update(
                convolution(
                    f"{prefix}.conv2",
                    inner_permutation,
                    residual_permutation,
                )
            )
            axes_to_perm.update(normalization(f"{prefix}.bn2", residual_permutation))

            if has_downsample and block_index == 0:
                axes_to_perm.update(
                    convolution(
                        f"{prefix}.downsample.0",
                        block_input_permutation,
                        residual_permutation,
                    )
                )
                axes_to_perm.update(
                    normalization(f"{prefix}.downsample.1", residual_permutation)
                )

        input_permutation = residual_permutation

    for task_name in task_names:
        axes_to_perm.update(dense(f"heads.{task_name}", input_permutation, None))

    return permutation_spec_from_axes_to_perm(axes_to_perm)


def validate_permutation_spec(
    spec: PermutationSpec,
    state_dict: Dict[str, torch.Tensor],
) -> None:
    """Fail loudly when the specification does not match the checkpoint."""

    specification_keys = set(spec.axes_to_perm)
    state_dict_keys = set(state_dict)

    missing = sorted(specification_keys - state_dict_keys)
    unexpected = sorted(state_dict_keys - specification_keys)

    if missing:
        raise KeyError(f"Permutation spec references unknown parameters: {missing}")

    if unexpected:
        raise KeyError(
            f"Checkpoint parameters missing from permutation spec: {unexpected}"
        )

    for parameter_name, axis_permutations in spec.axes_to_perm.items():
        tensor = state_dict[parameter_name]
        if tensor.dim() != len(axis_permutations):
            raise ValueError(
                f"Parameter '{parameter_name}' has {tensor.dim()} dimensions, "
                f"but the spec describes {len(axis_permutations)}."
            )

    for permutation_name, axes in spec.perm_to_axes.items():
        sizes = {state_dict[name].shape[axis] for name, axis in axes}
        if len(sizes) != 1:
            raise ValueError(
                f"Permutation '{permutation_name}' is tied to axes of "
                f"different sizes: {sorted(sizes)}"
            )


def apply_permutation_to_parameter(
    spec: PermutationSpec,
    permutation: Dict[str, torch.Tensor],
    parameter_name: str,
    parameter: torch.Tensor,
    except_axis: Optional[int] = None,
) -> torch.Tensor:
    """Permute every tied axis of one parameter."""

    result = parameter

    for axis, permutation_name in enumerate(spec.axes_to_perm[parameter_name]):

        if permutation_name is None or axis == except_axis:
            continue

        result = torch.index_select(
            result,
            axis,
            permutation[permutation_name].to(result.device),
        )

    return result


def apply_permutation(
    spec: PermutationSpec,
    permutation: Dict[str, torch.Tensor],
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Apply a permutation to a complete state dictionary."""

    return {
        parameter_name: apply_permutation_to_parameter(
            spec=spec,
            permutation=permutation,
            parameter_name=parameter_name,
            parameter=parameter,
        )
        for parameter_name, parameter in state_dict.items()
    }


def weight_matching(
    spec: PermutationSpec,
    reference_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    max_iterations: int = 100,
    seed: int = 42,
    tolerance: float = 1e-12,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """
    Find the permutation that aligns ``target_state`` to ``reference_state``.

    Coordinate descent over the permutation groups: while all other groups are
    held fixed, the optimal permutation of one group is a linear assignment
    problem solved exactly by the Hungarian algorithm.

    Returns the permutation and convergence information.
    """

    validate_permutation_spec(spec, reference_state)
    validate_permutation_spec(spec, target_state)

    permutation_sizes = {
        permutation_name: reference_state[axes[0][0]].shape[axes[0][1]]
        for permutation_name, axes in spec.perm_to_axes.items()
    }

    permutation = {
        permutation_name: torch.arange(size)
        for permutation_name, size in permutation_sizes.items()
    }

    permutation_names = sorted(permutation_sizes)

    generator = torch.Generator()
    generator.manual_seed(seed)

    improvements: List[float] = []
    iterations_used = 0
    converged = False

    for iteration in range(max_iterations):

        iterations_used = iteration + 1
        total_improvement = 0.0

        visit_order = torch.randperm(
            len(permutation_names),
            generator=generator,
        ).tolist()

        for position in visit_order:

            permutation_name = permutation_names[position]
            size = permutation_sizes[permutation_name]

            similarity = torch.zeros(
                (size, size),
                dtype=torch.float64,
            )

            for parameter_name, axis in spec.perm_to_axes[permutation_name]:

                if parameter_name.endswith(STATISTICS_SUFFIXES):
                    continue

                weights_reference = (
                    reference_state[parameter_name].detach().cpu().double()
                )

                weights_target = apply_permutation_to_parameter(
                    spec=spec,
                    permutation=permutation,
                    parameter_name=parameter_name,
                    parameter=target_state[parameter_name].detach().cpu().double(),
                    except_axis=axis,
                )

                weights_reference = torch.movedim(weights_reference, axis, 0).reshape(
                    size, -1
                )
                weights_target = torch.movedim(weights_target, axis, 0).reshape(
                    size, -1
                )

                similarity += weights_reference @ weights_target.T

            rows, columns = linear_sum_assignment(
                similarity.numpy(),
                maximize=True,
            )

            row_index = torch.arange(size)

            old_objective = float(
                similarity[row_index, permutation[permutation_name]].sum().item()
            )

            new_permutation = torch.as_tensor(columns, dtype=torch.long)

            new_objective = float(similarity[row_index, new_permutation].sum().item())

            total_improvement += new_objective - old_objective

            permutation[permutation_name] = new_permutation

        improvements.append(total_improvement)

        if verbose:
            print(f"  iteration {iterations_used}: improvement {total_improvement:.6e}")

        if total_improvement <= tolerance:
            converged = True
            break

    info = {
        "iterations": iterations_used,
        "converged": converged,
        "final_improvement": improvements[-1] if improvements else 0.0,
        "num_permutation_groups": len(permutation_names),
        "is_identity": all(
            bool(torch.equal(values, torch.arange(values.numel())))
            for values in permutation.values()
        ),
    }

    return permutation, info


def align_state_dict(
    reference_state: Dict[str, torch.Tensor],
    target_state: Dict[str, torch.Tensor],
    spec: Optional[PermutationSpec] = None,
    max_iterations: int = 100,
    seed: int = 42,
    verbose: bool = False,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], Dict[str, object]]:
    """Align ``target_state`` to ``reference_state`` and return the new state."""

    if spec is None:
        spec = resnet18_multitask_permutation_spec()

    permutation, info = weight_matching(
        spec=spec,
        reference_state=reference_state,
        target_state=target_state,
        max_iterations=max_iterations,
        seed=seed,
        verbose=verbose,
    )

    aligned_state = apply_permutation(
        spec=spec,
        permutation=permutation,
        state_dict=target_state,
    )

    return aligned_state, permutation, info


# ============================================================
# Functional equivalence check
# ============================================================


@torch.no_grad()
def verify_functional_equivalence(
    original_model: nn.Module,
    permuted_model: nn.Module,
    images: torch.Tensor,
    tasks: Iterable[str],
) -> Dict[str, float]:
    """
    Check that permuting the neurons did not change the network's outputs.

    Both models are evaluated on the same images for every task head; the
    maximum absolute logit difference must stay at floating-point noise level.
    """

    original_model.eval()
    permuted_model.eval()

    differences = {}

    for task in tasks:

        original_logits = original_model(images, task=task)
        permuted_logits = permuted_model(images, task=task)

        differences[task] = float(
            (original_logits - permuted_logits).abs().max().item()
        )

    differences["max"] = max(differences.values())

    return differences


# ============================================================
# Distances and directions in parameter space
# ============================================================


def flatten_state_dict(
    state_dict: Dict[str, torch.Tensor],
    keys: Sequence[str],
) -> torch.Tensor:
    """Flatten the selected entries of a state dictionary into one vector."""

    if not keys:
        raise ValueError("No parameter keys selected.")

    return torch.cat(
        [state_dict[key].detach().cpu().float().reshape(-1) for key in keys]
    )


def parameter_keys(
    model: nn.Module,
    backbone_only: bool = False,
) -> List[str]:
    """Return trainable parameter names, optionally restricted to the backbone."""

    names = [name for name, _ in model.named_parameters()]

    if backbone_only:
        names = [name for name in names if name.startswith("backbone.")]

    return names


def state_dict_l2_distance(
    first: Dict[str, torch.Tensor],
    second: Dict[str, torch.Tensor],
    keys: Sequence[str],
) -> float:
    """Euclidean distance between two state dictionaries over selected keys."""

    difference = flatten_state_dict(first, keys) - flatten_state_dict(second, keys)

    return float(torch.linalg.vector_norm(difference, ord=2).item())


def task_vector_cosine_similarity(
    final_state: Dict[str, torch.Tensor],
    reference_state: Dict[str, torch.Tensor],
    initial_state: Dict[str, torch.Tensor],
    keys: Sequence[str],
) -> float:
    """
    Cosine similarity between the two update directions

        (theta_final - theta_init)  and  (theta_reference - theta_init).

    Unlike L2, this ignores how far training moved the weights in total and
    compares only the direction of movement.
    """

    similarity = optional_task_vector_cosine_similarity(
        final_state=final_state,
        reference_state=reference_state,
        initial_state=initial_state,
        keys=keys,
    )

    if similarity is None:
        raise ValueError("A task vector has zero norm; cosine is undefined.")

    return similarity


def optional_task_vector_cosine_similarity(
    final_state: Dict[str, torch.Tensor],
    reference_state: Dict[str, torch.Tensor],
    initial_state: Dict[str, torch.Tensor],
    keys: Sequence[str],
) -> Optional[float]:
    """
    Cosine similarity of the two update directions, or ``None`` when one of
    them is exactly zero.

    A zero direction occurs for the heads of tasks a reference model never
    trained on: those parameters never received a gradient and still hold
    their initialization values.
    """

    initial_vector = flatten_state_dict(initial_state, keys)

    final_direction = flatten_state_dict(final_state, keys) - initial_vector
    reference_direction = flatten_state_dict(reference_state, keys) - initial_vector

    final_norm = torch.linalg.vector_norm(final_direction, ord=2)
    reference_norm = torch.linalg.vector_norm(reference_direction, ord=2)

    if final_norm.item() == 0.0 or reference_norm.item() == 0.0:
        return None

    similarity = torch.dot(final_direction, reference_direction) / (
        final_norm * reference_norm
    )

    return float(similarity.item())


def layer_groups(keys: Sequence[str]) -> Dict[str, List[str]]:
    """Group parameter names by their owning module (weight/bias together)."""

    groups: Dict[str, List[str]] = defaultdict(list)

    for key in keys:
        module_name = key.rsplit(".", 1)[0]
        groups[module_name].append(key)

    return dict(groups)


def per_layer_task_vector_cosine_similarity(
    final_state: Dict[str, torch.Tensor],
    reference_state: Dict[str, torch.Tensor],
    initial_state: Dict[str, torch.Tensor],
    keys: Sequence[str],
) -> Dict[str, Optional[float]]:
    """
    Cosine similarity of the update directions, computed per layer.

    Layers whose direction is undefined (untrained heads) report ``None``.
    """

    return {
        module_name: optional_task_vector_cosine_similarity(
            final_state=final_state,
            reference_state=reference_state,
            initial_state=initial_state,
            keys=module_keys,
        )
        for module_name, module_keys in layer_groups(keys).items()
    }


def predict_minimum(values: Dict[str, float]) -> str:
    """Task with the smallest value."""

    if not values:
        raise ValueError("Cannot predict from an empty dictionary.")

    return min(values, key=values.get)


def predict_maximum(values: Dict[str, float]) -> str:
    """Task with the largest value."""

    if not values:
        raise ValueError("Cannot predict from an empty dictionary.")

    return max(values, key=values.get)
