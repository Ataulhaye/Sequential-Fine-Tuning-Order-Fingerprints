from typing import Dict

import torch
import torch.nn as nn


def get_parameter_vector(model: nn.Module) -> torch.Tensor:
    """
    Flatten all model parameters into one CPU vector.

    This includes the complete model:
        backbone + task-specific heads.
    """

    parameters = []

    for parameter in model.parameters():
        parameters.append(parameter.detach().cpu().float().reshape(-1))

    if not parameters:
        raise ValueError("Model contains no parameters.")

    return torch.cat(parameters)


def get_backbone_parameter_vector(
    model: nn.Module,
) -> torch.Tensor:
    """
    Flatten only the shared backbone parameters.

    Task-specific classifier heads are excluded.

    This is particularly useful for studying representation
    similarity because the ResNet18 backbone contains the
    shared learned representation.
    """

    if not hasattr(model, "backbone"):
        raise AttributeError("Model does not contain a 'backbone' attribute.")

    parameters = []

    for parameter in model.backbone.parameters():
        parameters.append(parameter.detach().cpu().float().reshape(-1))

    if not parameters:
        raise ValueError("Model backbone contains no parameters.")

    return torch.cat(parameters)


def calculate_l2_distance(
    model_a: nn.Module,
    model_b: nn.Module,
) -> float:
    """
    Calculate Euclidean L2 distance between all model parameters.

        D = ||theta_a - theta_b||_2
    """

    vector_a = get_parameter_vector(model_a)
    vector_b = get_parameter_vector(model_b)

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Models have different numbers of parameters: "
            f"{vector_a.numel()} vs {vector_b.numel()}"
        )

    distance = torch.linalg.vector_norm(
        vector_a - vector_b,
        ord=2,
    )

    return float(distance.item())


def calculate_backbone_l2_distance(
    model_a: nn.Module,
    model_b: nn.Module,
) -> float:
    """
    Calculate Euclidean L2 distance using only the shared
    ResNet18 backbone parameters.

        D_backbone =
            ||theta_backbone_a - theta_backbone_b||_2

    Task-specific classification heads are ignored.
    """

    vector_a = get_backbone_parameter_vector(model_a)
    vector_b = get_backbone_parameter_vector(model_b)

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Backbones have different numbers of parameters: "
            f"{vector_a.numel()} vs {vector_b.numel()}"
        )

    distance = torch.linalg.vector_norm(
        vector_a - vector_b,
        ord=2,
    )

    return float(distance.item())


def calculate_relative_l2_distance(
    model_a: nn.Module,
    model_b: nn.Module,
) -> float:
    """
    Calculate full-model L2 distance normalized by the
    reference model norm.
    """

    vector_a = get_parameter_vector(model_a)
    vector_b = get_parameter_vector(model_b)

    if vector_a.shape != vector_b.shape:
        raise ValueError(
            "Models have different numbers of parameters: "
            f"{vector_a.numel()} vs {vector_b.numel()}"
        )

    difference = torch.linalg.vector_norm(
        vector_a - vector_b,
        ord=2,
    )

    reference_norm = torch.linalg.vector_norm(
        vector_b,
        ord=2,
    )

    if reference_norm.item() == 0.0:
        raise ValueError("Reference model has zero parameter norm.")

    return float((difference / reference_norm).item())


def calculate_all_weight_distances(
    sequential_model: nn.Module,
    single_task_models: Dict[str, nn.Module],
) -> Dict[str, float]:
    """
    Calculate full-model L2 distance from a sequential model
    to every single-task reference model.
    """

    distances = {}

    for task, single_model in single_task_models.items():
        distances[task] = calculate_l2_distance(
            sequential_model,
            single_model,
        )

    return distances


def calculate_all_backbone_distances(
    sequential_model: nn.Module,
    single_task_models: Dict[str, nn.Module],
) -> Dict[str, float]:
    """
    Calculate backbone-only L2 distance from a sequential model
    to every single-task reference model.
    """

    distances = {}

    for task, single_model in single_task_models.items():
        distances[task] = calculate_backbone_l2_distance(
            sequential_model,
            single_model,
        )

    return distances


def predict_closest_task(
    distances: Dict[str, float],
) -> str:
    """
    Predict the task corresponding to the smallest distance.
    """

    if not distances:
        raise ValueError("Cannot predict from an empty distance dictionary.")

    return min(
        distances,
        key=distances.get,
    )
