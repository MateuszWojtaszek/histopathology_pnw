import torch

from histopathology_pnw.device import get_device


def test_get_device_returns_usable_device():
    device = get_device()
    assert isinstance(device, torch.device)
    assert torch.ones(2, device=device).sum().item() == 2
