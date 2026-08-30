from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader


def center_features(features: torch.Tensor) -> torch.Tensor:
    """Center feature matrix by subtracting the mean of each feature dimension."""
    return features - features.mean(dim=0, keepdim=True)


def linear_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
) -> float:
    """Compute linear CKA between two representation matrices [N, D]."""
    if features_x.ndim != 2 or features_y.ndim != 2:
        raise ValueError("Features must be 2-dimensional tensors with shape [N, D].")

    if features_x.shape[0] != features_y.shape[0]:
        raise ValueError(
            "Both representations must contain the same number of samples."
        )

    x = center_features(features_x)
    y = center_features(features_y)

    # Cross-covariance / Gram interaction
    xy = x.T @ y
    numerator = torch.sum(xy**2)

    xx = x.T @ x
    yy = y.T @ y

    denominator = torch.sqrt(torch.sum(xx**2) * torch.sum(yy**2))

    if denominator <= 0:
        return 0.0

    cka = numerator / denominator
    return float(cka.item())


def feature_drift(
    reference_features: torch.Tensor,
    comparison_features: torch.Tensor,
) -> float:
    """Compute mean L2 distance between normalized feature vectors [N, D]."""
    if reference_features.ndim != 2 or comparison_features.ndim != 2:
        raise ValueError("Features must be 2-dimensional tensors with shape [N, D].")

    if reference_features.shape != comparison_features.shape:
        raise ValueError("Representation matrices must have identical shapes.")

    reference = torch.nn.functional.normalize(reference_features, p=2, dim=1)
    comparison = torch.nn.functional.normalize(comparison_features, p=2, dim=1)

    distances = torch.norm(reference - comparison, p=2, dim=1)
    return float(distances.mean().item())


def compare_representations(
    reference_features: torch.Tensor,
    comparison_features: torch.Tensor,
) -> Dict[str, float]:
    """Compute both CKA and feature drift, returning a dict for downstream callers."""
    cka = linear_cka(reference_features, comparison_features)
    drift = feature_drift(reference_features, comparison_features)

    return {"cka": cka, "feature_drift": drift}


@torch.no_grad()
def extract_features(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Extract backbone representations for a model over a dataset probe.
    Centralized here so test and analysis scripts produce identical features.
    """
    model.eval()
    model.to(device)

    features = []
    for images, _ in dataloader:
        images = images.to(device)
        batch_features = model.extract_features(images)
        features.append(batch_features.detach().cpu())

    return torch.cat(features, dim=0)
