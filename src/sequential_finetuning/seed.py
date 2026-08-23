import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible experiments.

    Parameters
    ----------
    seed:
        Global random seed.
    """

    # Python
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Python hash randomization
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
