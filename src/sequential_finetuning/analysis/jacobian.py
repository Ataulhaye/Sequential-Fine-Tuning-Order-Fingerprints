"""
Jacobian Sensitivity Analysis

Computes input-output Jacobian sensitivity to understand how sensitive
model predictions are to input perturbations on task probe images.

This helps identify which input features most strongly influence task predictions
and reveals task-specific sensitivity patterns.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn


def compute_jacobian_batch(
    model: nn.Module,
    images: torch.Tensor,
    task: str,
    device: torch.device,
    compute_class_jacobian: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute input-output Jacobian for a batch of images.

    Jacobian[i, j, k, l] = d(output[i, j]) / d(input[i, k, l])
    where i = batch index, j = class index, k,l = spatial indices

    For computational efficiency, we compute Jacobian for predicted class only.

    Parameters
    ----------
    model : nn.Module
        Model with forward(images, task) method
    images : torch.Tensor
        Input images [N, 3, 32, 32]
    task : str
        Task identifier
    device : torch.device
        Device to run on
    compute_class_jacobian : bool
        If True, compute Jacobian w.r.t. predicted class only (faster)
        If False, compute full Jacobian for all classes

    Returns
    -------
    jacobians : torch.Tensor
        Jacobians [N, 3, 32, 32] (per-sample) or [N, num_classes, 3, 32, 32] (full)
    predicted_classes : torch.Tensor
        Predicted class for each sample [N]
    """

    N = images.shape[0]
    images_shape = images.shape
    jacobians = []
    predicted_classes = []

    model.eval()

    for i in range(N):
        # Single image
        img = images[i : i + 1].clone().detach().requires_grad_(True)

        # Forward pass
        logits = model(img, task=task)

        if compute_class_jacobian:
            # Only compute Jacobian for predicted class
            pred_class = logits.argmax(dim=1).item()
            pred_logit = logits[0, pred_class]

            # Backward pass
            grad = torch.autograd.grad(
                pred_logit,
                img,
                create_graph=False,
                retain_graph=False,
            )[0]

            jacobians.append(grad.squeeze(0))  # [3, 32, 32]
            predicted_classes.append(pred_class)
        else:
            # Compute full Jacobian (all classes)
            num_classes = logits.shape[1]
            full_jacobian = []

            for class_idx in range(num_classes):
                logit = logits[0, class_idx]

                grad = torch.autograd.grad(
                    logit,
                    img,
                    create_graph=False,
                    retain_graph=(class_idx < num_classes - 1),
                )[0]

                full_jacobian.append(grad.squeeze(0))  # [3, 32, 32]

            jacobians.append(torch.stack(full_jacobian))  # [num_classes, 3, 32, 32]
            predicted_classes.append(logits.argmax(dim=1).item())

    if compute_class_jacobian:
        jacobians_tensor = torch.stack(jacobians)  # [N, 3, 32, 32]
    else:
        jacobians_tensor = torch.stack(jacobians)  # [N, num_classes, 3, 32, 32]

    return jacobians_tensor, torch.tensor(predicted_classes, device=device)


def compute_jacobian_sensitivity_metrics(
    jacobians: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute various sensitivity metrics from Jacobian tensors.

    Parameters
    ----------
    jacobians : torch.Tensor
        Jacobians from compute_jacobian_batch [N, 3, 32, 32] or [N, num_classes, 3, 32, 32]

    Returns
    -------
    Dict with metrics:
        - mean_sensitivity: Average Jacobian norm across batch
        - max_sensitivity: Maximum Jacobian norm in batch
        - min_sensitivity: Minimum Jacobian norm in batch
        - std_sensitivity: Std of Jacobian norms across batch
        - mean_gradient_magnitude: Average L2 norm of gradients
    """

    if jacobians.dim() == 4:
        # [N, 3, 32, 32] - per-class Jacobians
        jac_norms = torch.norm(jacobians.reshape(jacobians.shape[0], -1), dim=1)
    elif jacobians.dim() == 5:
        # [N, num_classes, 3, 32, 32] - full Jacobians
        jac_norms = torch.norm(
            jacobians.reshape(jacobians.shape[0], jacobians.shape[1], -1), dim=2
        )
        # Average over classes
        jac_norms = jac_norms.mean(dim=1)
    else:
        raise ValueError(f"Unexpected Jacobian shape: {jacobians.shape}")

    return {
        "mean_sensitivity": float(jac_norms.mean().item()),
        "max_sensitivity": float(jac_norms.max().item()),
        "min_sensitivity": float(jac_norms.min().item()),
        "std_sensitivity": float(jac_norms.std().item()),
        "median_sensitivity": float(jac_norms.median().item()),
    }


def compute_spatial_sensitivity(jacobians: torch.Tensor) -> torch.Tensor:
    """
    Compute spatial sensitivity map by averaging Jacobian across batch and color channels.

    Shows which spatial regions most influence predictions.

    Parameters
    ----------
    jacobians : torch.Tensor
        [N, 3, 32, 32] or [N, num_classes, 3, 32, 32]

    Returns
    -------
    torch.Tensor
        Spatial sensitivity map [32, 32]
    """

    if jacobians.dim() == 4:
        # [N, 3, 32, 32]
        jac_spatial = jacobians.abs().mean(dim=0).mean(dim=0)  # [32, 32]
    elif jacobians.dim() == 5:
        # [N, num_classes, 3, 32, 32]
        jac_spatial = jacobians.abs().mean(dim=0).mean(dim=0).mean(dim=0)  # [32, 32]
    else:
        raise ValueError(f"Unexpected Jacobian shape: {jacobians.shape}")

    return jac_spatial


def compute_channel_sensitivity(jacobians: torch.Tensor) -> torch.Tensor:
    """
    Compute per-channel sensitivity by averaging Jacobian across batch and spatial.

    Shows which color channels most influence predictions.

    Parameters
    ----------
    jacobians : torch.Tensor
        [N, 3, 32, 32] or [N, num_classes, 3, 32, 32]

    Returns
    -------
    torch.Tensor
        Channel sensitivity [3]
    """

    if jacobians.dim() == 4:
        # [N, 3, 32, 32]
        jac_channel = jacobians.abs().mean(dim=0).mean(dim=(1, 2))  # [3]
    elif jacobians.dim() == 5:
        # [N, num_classes, 3, 32, 32]
        jac_channel = jacobians.abs().mean(dim=0).mean(dim=0).mean(dim=(1, 2))  # [3]
    else:
        raise ValueError(f"Unexpected Jacobian shape: {jacobians.shape}")

    return jac_channel


def task_jacobian_comparison(
    jacobians_dict: Dict[str, torch.Tensor],
) -> Dict[str, Dict[str, float]]:
    """
    Compare Jacobian sensitivities across tasks.

    Parameters
    ----------
    jacobians_dict : Dict[str, torch.Tensor]
        Dictionary mapping task names to their Jacobian tensors

    Returns
    -------
    Dict
        Sensitivity metrics per task
    """

    results = {}

    for task_name, jacobians in jacobians_dict.items():
        metrics = compute_jacobian_sensitivity_metrics(jacobians)
        results[task_name] = metrics

    return results
