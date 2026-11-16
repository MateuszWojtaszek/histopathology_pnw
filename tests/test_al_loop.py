"""The AL loop end to end with the model replaced by fakes: no weights, no training."""

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from histopathology_pnw.al import random_loop, selectors
from histopathology_pnw.al.random_loop import RandomALConfig, run_random_al
from histopathology_pnw.datasets.monuseg import Sample

HARD = {"img1", "img4", "img6"}  # the fake student finds nothing on these


def _mask() -> np.ndarray:
    mask = np.zeros((16, 16), dtype=np.uint16)
    mask[2:7, 2:7] = 1
    mask[9:14, 9:14] = 2
    return mask


def _image(image_id: str) -> np.ndarray:
    # channel 0 carries the oracle mask, channel 1 says whether the fake student finds it
    image = np.zeros((16, 16, 3), dtype=np.uint16)
    image[..., 0] = _mask()
    image[..., 1] = 0 if image_id in HARD else 1
    return image


def _fake_predict(model, images, **kwargs):
    return [im[..., 0] * im[..., 1] for im in images]


@pytest.fixture
def fake_models(monkeypatch):
    def train_seg(net, save_path, model_name, **kwargs):
        path = Path(save_path) / "models" / model_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"weights")
        return path, [0.5], None

    def fake_init(seed, path):
        path.write_bytes(b"init")

    student = SimpleNamespace(net=None)
    monkeypatch.setattr(random_loop, "build_student_init", fake_init)
    monkeypatch.setattr(random_loop, "load_student", lambda path, device: student)
    monkeypatch.setattr(random_loop, "train_seg", train_seg)
    monkeypatch.setattr(random_loop, "oracle_masks", lambda samples: [_mask() for _ in samples])
    monkeypatch.setattr(random_loop, "read_image", lambda path: _image(Path(path).stem))
    monkeypatch.setattr(random_loop, "read_mask", lambda sample: _mask())
    monkeypatch.setattr(random_loop, "predict", _fake_predict)
    monkeypatch.setattr(selectors, "predict", _fake_predict)


def _samples(n: int, prefix: str = "img") -> list[Sample]:
    return [
        Sample(f"{prefix}{i}", "train", Path(f"{prefix}{i}.tif"), Path(f"{prefix}{i}.xml"))
        for i in range(n)
    ]


def _cfg(selector: str, max_labeled: int) -> RandomALConfig:
    return RandomALConfig(
        seed=0, selector=selector, max_labeled=max_labeled, gradient_checkpointing=False
    )


def _labeled(results: pd.DataFrame) -> list[list[str]]:
    return [json.loads(x) for x in results["labeled_ids"]]


def test_random_and_peek_share_the_start_and_then_differ(fake_models, tmp_path):
    pool, val = _samples(8), _samples(2, "val")
    order = [pool[i].image_id for i in np.random.default_rng(0).permutation(len(pool))]

    rand = run_random_al(_cfg("random", 6), pool, val, tmp_path / "random")
    assert _labeled(rand) == [order[:2], order[:4], order[:6]]

    peek = run_random_al(_cfg("oracle_peek", 6), pool, val, tmp_path / "peek")
    sets = _labeled(peek)
    assert sets[0] == order[:2]  # the same start as random
    hard_left = [i for i in order if i in HARD and i not in order[:2]]
    assert sets[1] == order[:2] + hard_left[:2]  # the worst images, ties by permutation
    assert [len(s) for s in sets] == [2, 4, 6]

    table = pd.read_csv(tmp_path / "peek" / "round_02" / "selection.csv")
    assert len(table) == 6  # every unlabelled image, not only the chosen ones
    assert set(table.loc[table["selected"], "image_id"]) == set(hard_left[:2])
    assert set(table.loc[table["pq_oracle"] == 0, "image_id"]) == set(hard_left)
    assert json.loads(peek["selected_next"].iloc[-1]) == []


def test_extending_a_finished_run_continues_from_its_last_round(fake_models, tmp_path):
    pool, val = _samples(8), _samples(2, "val")
    whole = run_random_al(_cfg("oracle_peek", 6), pool, val, tmp_path / "whole")
    run_random_al(_cfg("oracle_peek", 4), pool, val, tmp_path / "parts")
    parts = run_random_al(_cfg("oracle_peek", 6), pool, val, tmp_path / "parts")
    assert _labeled(parts) == _labeled(whole)
