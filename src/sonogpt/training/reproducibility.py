"""Random-state controls for repeatable CPU and single-GPU experiments."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_reproducible_seed(seed: int, *, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
