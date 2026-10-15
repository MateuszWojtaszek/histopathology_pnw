import random

import torch

from histopathology_pnw.seed import set_seed


def _draw(seed: int):
    rng = set_seed(seed)
    return random.random(), rng.random(), torch.rand(1).item()


def test_same_seed_gives_same_draws():
    assert _draw(0) == _draw(0)


def test_different_seed_gives_different_draws():
    assert _draw(0) != _draw(1)
