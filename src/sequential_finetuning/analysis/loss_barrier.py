"""
Loss-Barrier Analysis

Measures the "smoothness" of the optimization landscape between two models
by interpolating in weight space and evaluating loss at intermediate points.

A low barrier indicates similar loss landscapes, suggesting the models
found similar solutions. A high barrier suggests different local minima.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def get_model_weights_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """
    Extract all model weights as a dictionary.
    """
    return {name: param.clone().detach() for name, param in model.named_parameters()}


def interpolate_models(
    weights_A: Dict[str, torch.Tensor],
    weights_B: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """
    Linear interpolation between two sets of model weights.

    weights_interp = (1 - alpha) * weights_A + alpha * weights_B

    Parameters
    ----------
    weights_A : Dict
        Source model weights
    weights_B : Dict
        Target model weights
    alpha : float
        Interpolation coefficient [0, 1]
        alpha=0 → weights_A
        alpha=1 → weights_B

    Returns
    -------
    Dict
        Interpolated weights
    """

    interpolated = {}

    for name in weights_A.keys():
        if name in weights_B:
            interpolated[name] = (1 - alpha) * weights_A[name] + alpha * weights_B[name]
        else:
            # If weight not in B, use A
            interpolated[name] = weights_A[name].clone()

    return interpolated


def set_model_weights(model: nn.Module, weights: Dict[str, torch.Tensor]):
    """
    Set model weights from a dictionary.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in weights:
                param.copy_(weights[name])


def evaluate_loss_at_interpolation(
    model: nn.Module,
    weights_A: Dict[str, torch.Tensor],
    weights_B: Dict[str, torch.Tensor],
    dataloader,
    device: torch.device,
    task: str,
    num_points: int = 11,
) -> Tuple[List[float], List[float]]:
    """
    Evaluate model loss at multiple interpolation points between two weight sets.

    Parameters
    ----------
    model : nn.Module
        The model to evaluate
    weights_A : Dict
        Weights of model A (reference)
    weights_B : Dict
        Weights of model B (sequential)
    dataloader : DataLoader
        Data to evaluate loss on
    device : torch.device
        Device to run on
    task : str
        Task identifier for forward pass
    num_points : int
        Number of interpolation points (default 11 gives alpha=[0, 0.1, ..., 1.0])

    Returns
    -------
    alphas : List[float]
        Interpolation coefficients
    losses : List[float]
        Loss values at each alpha
    """

    model.eval()
    criterion = nn.CrossEntropyLoss()

    alphas = []
    losses = []

    with torch.no_grad():
        for i in range(num_points):
            alpha = i / (num_points - 1)

            # Interpolate weights
            interp_weights = interpolate_models(weights_A, weights_B, alpha)

            # Set model to interpolated weights
            set_model_weights(model, interp_weights)

            # Evaluate loss
            total_loss = 0.0
            num_samples = 0

            for images, labels in dataloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                logits = model(images, task=task)
                loss = criterion(logits, labels)

                batch_size = labels.size(0)
                total_loss += loss.item() * batch_size
                num_samples += batch_size

            avg_loss = total_loss / num_samples if num_samples > 0 else 0.0

            alphas.append(alpha)
            losses.append(avg_loss)

    # Restore model to weights_B
    set_model_weights(model, weights_B)

    return alphas, losses


def compute_barrier_height(alphas: List[float], losses: List[float]) -> float:
    """
    Compute barrier height as the maximum loss - minimum loss.

    A high barrier suggests different optimization landscapes.
    A low barrier suggests similar landscapes.

    Returns
    -------
    float
        Barrier height (difference between max and min loss)
    """
    if not losses:
        return 0.0

    return max(losses) - min(losses)


def compute_barrier_area(alphas: List[float], losses: List[float]) -> float:
    """
    Compute approximate area under the loss curve using trapezoidal integration.

    Higher area can indicate more complex landscape.

    Returns
    -------
    float
        Approximate area under the curve
    """
    if len(alphas) < 2:
        return 0.0

    area = 0.0
    for i in range(len(alphas) - 1):
        # Trapezoidal rule
        delta_alpha = alphas[i + 1] - alphas[i]
        area += delta_alpha * (losses[i] + losses[i + 1]) / 2.0

    return area
