"""The single place where random seeds are set."""

import random

import numpy as np
import torch


def set_seed(seed: int) -> np.random.Generator:
    """Seed `random` and `torch` (all devices), and return a numpy Generator for that seed.

    numpy has no global seed here: pass the returned Generator to code that needs randomness.
    """
    random.seed(seed)
    torch.manual_seed(seed)  # also seeds cuda and mps
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return np.random.default_rng(seed)
