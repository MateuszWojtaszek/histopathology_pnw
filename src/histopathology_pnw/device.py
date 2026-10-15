"""The single place where the compute device is chosen."""

import torch


def get_device() -> torch.device:
    """Return the best available device: cuda, then mps, then cpu."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
