"""
Fisher Information Analysis

Estimates the trace of the Fisher information matrix of a checkpoint with
respect to the data distribution of every task.

The estimator is the "true" (model-sampled) Fisher, not the empirical one:
for every sample the label is drawn from the model's own predictive
distribution instead of using the dataset ground-truth label.

    F_T = (1 / N) * sum_n sum_p ( d/dtheta_p  -log p(y_n_hat | x_n) )^2

    with  y_n_hat ~ p(. | x_n)  and  x_n  a sample of task T.

Sorting F_A, F_B and F_C yields a guessed training order. Because it is not
known a priori whether a recently learned task produces a high or a low
Fisher trace, both reading directions are reported.
"""

from typing import Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torch.utils.data import DataLoader

HIGH_IS_RECENT = "high_is_recent"
LOW_IS_RECENT = "low_is_recent"

HYPOTHESES = (HIGH_IS_RECENT, LOW_IS_RECENT)


def select_sample_indices(
    dataset_size: int,
    num_samples: int,
    seed: int,
) -> List[int]:
    """
    Deterministically select a subset of dataset indices.

    The same indices are reused for every checkpoint so that the Fisher
    traces of different models are computed on identical images.
    """

    if dataset_size <= 0:
        raise ValueError("Dataset is empty.")

    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    permutation = torch.randperm(
        dataset_size,
        generator=generator,
    )

    return permutation[: min(num_samples, dataset_size)].tolist()


def compute_fisher_trace(
    model: nn.Module,
    dataloader: DataLoader,
    task: str,
    device: torch.device,
    seed: int = 42,
) -> Dict[str, float]:
    """
    Estimate the Fisher information trace of one model on one task.

    For every sample individually:
        1. forward pass through the checkpoint,
        2. sample a label from the predicted class probabilities,
        3. compute the loss for that sampled label and back-propagate,
        4. square and accumulate every gradient entry.

    The gradients are never applied: no optimizer step is taken and the
    gradients are cleared after every sample.

    Returns the mean squared gradient sum over all parameters
    (``fisher_total``) and the same quantity restricted to the shared
    backbone and to the task head.
    """

    model.eval()

    generator = torch.Generator()
    generator.manual_seed(seed)

    head_prefix = f"heads.{task}."

    totals = {
        "fisher_total": torch.zeros((), device=device, dtype=torch.float64),
        "fisher_backbone": torch.zeros((), device=device, dtype=torch.float64),
        "fisher_head": torch.zeros((), device=device, dtype=torch.float64),
    }

    num_samples = 0

    for images, _ in dataloader:

        images = images.to(device)

        for index in range(images.shape[0]):

            single_image = images[index : index + 1]

            logits = model(single_image, task=task)

            probabilities = torch.softmax(
                logits.detach(),
                dim=1,
            ).cpu()

            # The ground-truth label is intentionally ignored: the label is
            # drawn from the model's own predictive distribution.
            sampled_label = torch.multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
            ).view(1)

            loss = functional.cross_entropy(
                logits,
                sampled_label.to(device),
            )

            model.zero_grad(set_to_none=True)

            loss.backward()

            for name, parameter in model.named_parameters():

                if parameter.grad is None:
                    continue

                squared_sum = parameter.grad.detach().double().pow(2).sum()

                totals["fisher_total"] += squared_sum

                if name.startswith("backbone."):
                    totals["fisher_backbone"] += squared_sum
                elif name.startswith(head_prefix):
                    totals["fisher_head"] += squared_sum

            num_samples += 1

    model.zero_grad(set_to_none=True)

    if num_samples == 0:
        raise ValueError("No samples were provided for Fisher estimation.")

    result = {key: float((value / num_samples).item()) for key, value in totals.items()}

    result["num_samples"] = num_samples

    return result


def guess_order_from_fisher(
    scores: Dict[str, float],
    hypothesis: str,
) -> List[str]:
    """
    Sort the tasks into a guessed training order.

    ``high_is_recent``: the largest Fisher trace belongs to the task that was
    learned last, so the guessed order is ascending.

    ``low_is_recent``: the smallest Fisher trace belongs to the task that was
    learned last, so the guessed order is descending.
    """

    if hypothesis not in HYPOTHESES:
        raise ValueError(
            f"Unknown hypothesis '{hypothesis}'. " f"Available: {list(HYPOTHESES)}"
        )

    descending = hypothesis == LOW_IS_RECENT

    return sorted(
        scores,
        key=lambda task: scores[task],
        reverse=descending,
    )


def evaluate_fisher_predictions(
    scores: Dict[str, float],
    actual_order: Sequence[str],
) -> Dict[str, Dict]:
    """
    Evaluate both reading directions of the Fisher ranking.

    For each hypothesis the guessed full order and the guessed last task are
    compared against the ground truth.
    """

    actual_order = list(actual_order)

    predictions = {}

    for hypothesis in HYPOTHESES:

        guessed_order = guess_order_from_fisher(
            scores=scores,
            hypothesis=hypothesis,
        )

        predictions[hypothesis] = {
            "predicted_order": guessed_order,
            "predicted_last_task": guessed_order[-1],
            "order_correct": guessed_order == actual_order,
            "last_task_correct": guessed_order[-1] == actual_order[-1],
        }

    return predictions
