import numpy as np
import pytest

from histopathology_pnw.metrics import evaluate, evaluate_all, panoptic_quality, summarize


def _two_squares() -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:8] = 1
    mask[10:16, 10:16] = 2
    return mask


def test_identical_masks_are_perfect():
    gt = _two_squares()
    result = panoptic_quality(gt, gt)
    assert (result["pq"], result["tp"], result["fp"], result["fn"]) == (1.0, 2, 0, 0)


def test_label_permutation_does_not_matter():
    gt = _two_squares()
    pred = np.where(gt == 1, 2, np.where(gt == 2, 1, 0)).astype(np.uint16)
    assert panoptic_quality(gt, pred)["pq"] == 1.0


def test_missing_and_extra_instances():
    gt = _two_squares()
    pred = np.zeros_like(gt)
    pred[2:8, 2:8] = 1  # matches gt 1
    pred[17:19, 0:2] = 2  # false positive; gt 2 is missed
    result = panoptic_quality(gt, pred)
    assert (result["tp"], result["fp"], result["fn"]) == (1, 1, 1)
    assert result["dq"] == pytest.approx(1 / (1 + 0.5 + 0.5))
    assert result["pq"] == pytest.approx(0.5)


def test_low_overlap_is_not_a_match():
    gt = np.zeros((10, 10), dtype=np.uint16)
    gt[0:4, 0:4] = 1
    pred = np.zeros_like(gt)
    pred[2:6, 2:6] = 1  # IoU = 4 / 28
    result = panoptic_quality(gt, pred)
    assert (result["tp"], result["fp"], result["fn"], result["pq"]) == (0, 1, 1, 0.0)


def test_empty_prediction():
    result = evaluate(_two_squares(), np.zeros((20, 20), dtype=np.uint16))
    assert result["pq"] == 0.0
    assert result["fn"] == 2


def test_summarize_pools_counts():
    gt = _two_squares()
    table = evaluate_all([gt, gt], [gt, np.zeros_like(gt)], ["a", "b"])
    summary = summarize(table)
    assert summary["pq"] == pytest.approx(0.5)  # mean of 1 and 0
    assert summary["pq_pooled"] == pytest.approx(2 / 3)  # tp=2, fn=2: dq = 2/3, sq = 1
