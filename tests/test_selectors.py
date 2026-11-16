import numpy as np
import pandas as pd
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk

from histopathology_pnw.al.selectors import (
    combine_tta,
    peek_table,
    pick,
    random_scores,
    tta_image_row,
    tta_kd_scores,
)


def _two_squares() -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:8] = 1
    mask[10:16, 10:16] = 2
    return mask


def test_random_selector_follows_the_permutation():
    rank = {"c": 0, "a": 1, "d": 2, "b": 3}
    chosen, table = pick(random_scores(["a", "b", "d"], rank), 2, rank)
    assert chosen == ["a", "d"]
    assert table.set_index("image_id")["selected"].to_dict() == {"a": True, "b": False, "d": True}


def test_pick_takes_highest_scores_and_breaks_ties_by_permutation():
    rank = {"a": 2, "b": 0, "c": 1, "d": 3}
    scores = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "score": [0.5, 0.5, 0.1, 0.9]})
    assert pick(scores, 3, rank)[0] == ["d", "b", "a"]
    assert pick(scores, 0, rank)[0] == []


def test_peek_ranks_the_worst_segmented_image_first():
    oracle = _two_squares()
    half = np.where(oracle == 1, 1, 0).astype(np.uint16)  # one of two nuclei found
    table = peek_table(
        ["perfect", "empty", "half"],
        [oracle, np.zeros_like(oracle), half],
        [oracle, oracle, oracle],
    )
    by_id = table.set_index("image_id")
    assert by_id["pq_oracle"].to_dict() == {"perfect": 1.0, "empty": 0.0, "half": 2 / 3}
    assert (by_id.loc["half", "tp"], by_id.loc["half", "fn"], by_id.loc["half", "fp"]) == (1, 1, 0)
    rank = {"perfect": 0, "empty": 1, "half": 2}
    assert pick(table, 2, rank)[0] == ["empty", "half"]


def test_combine_tta_ranks_by_contour_and_detection_instability():
    table = pd.DataFrame(
        {
            "image_id": ["stable", "wobbly", "missing"],
            "K_mean": [0.05, 0.30, 0.10],
            "low_D": [0.00, 0.10, 0.40],
            "E_extra": [0.00, 0.00, 0.10],
        }
    )
    out = combine_tta(table).set_index("image_id")
    assert out["unstable"].to_dict() == pytest.approx(
        {"stable": 0.0, "wobbly": 0.1, "missing": 0.5}
    )
    # ranks (of 3): stable 1/3 and 1/3; wobbly 1 and 2/3; missing 2/3 and 1
    assert out["score"].to_dict() == pytest.approx(
        {"stable": 1 / 3, "wobbly": 5 / 6, "missing": 5 / 6}
    )
    rank = {"stable": 0, "missing": 1, "wobbly": 2}
    assert pick(out.reset_index(), 2, rank)[0] == ["missing", "wobbly"]  # tie: seed order


def test_image_without_nuclei_is_the_most_unstable():
    row = tta_image_row(pd.DataFrame({"K": [], "D": []}), {"E_extra": 0.0})
    assert (row["n_pred"], row["K_mean"], row["low_D"]) == (0, 1.0, 1.0)


class _Model:
    """Bright blobs; with `closing` it joins horizontal neighbours, so it is not rotation-stable."""

    def __init__(self, closing: bool = False):
        self.closing = closing

    def eval(self, img, batch_size=8, **kwargs):
        fg = img[..., 0] > 128
        if self.closing:
            fg = ndi.binary_closing(fg, structure=np.ones((1, 9)))
        lab, _ = ndi.label(fg)
        return lab, [None, None, np.where(fg, 4.0, -4.0).astype(np.float32)], None


def _blobs(centres) -> np.ndarray:
    img = np.zeros((120, 120, 3), np.uint8)
    for y, x in centres:
        img[disk((y, x), 7)] = 255
    return img


def test_tta_selector_prefers_the_image_segmented_differently_under_rotation():
    images = {
        "apart": _blobs([(30, 30), (90, 90), (30, 90)]),  # the same in every version
        "touching": _blobs([(30, 30), (30, 48), (90, 90)]),  # 4 px gap: closed only horizontally
    }
    oracle = {i: ndi.label(im[..., 0] > 128)[0] for i, im in images.items()}
    table = tta_kd_scores(_Model(closing=True), list(images), images, oracle, eval_kwargs={})
    by_id = table.set_index("image_id")
    assert by_id.loc["apart", "K_mean"] < 1e-9 and by_id.loc["apart", "unstable"] == 0
    assert by_id.loc["touching", "unstable"] > 0
    assert by_id["score"].idxmax() == "touching"
    assert by_id.loc["apart", "pq_oracle_offline"] == 1.0  # stored for analysis, not used
    # without labels the score is the same: the selector does not need them
    blind = tta_kd_scores(_Model(closing=True), list(images), images, eval_kwargs={})
    assert blind["score"].tolist() == table["score"].tolist()
    assert "pq_oracle_offline" not in blind.columns
