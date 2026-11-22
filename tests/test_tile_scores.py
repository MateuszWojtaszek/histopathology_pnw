import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk

from histopathology_pnw.metrics import panoptic_quality
from histopathology_pnw.uncertainty.tile_scores import score_round


class ThresholdModel:
    """Threshold + connected components: the same nuclei in every orientation (no instability)."""

    def eval(self, img, batch_size=8, **kw):
        fg = img[..., 0] > 128
        lab, _ = ndi.label(fg)
        return lab, [None, None, np.where(fg, 4.0, -4.0).astype(np.float32)], None


def scene(centers, shape=(200, 200), r=6):
    img = np.zeros((*shape, 3), np.uint8)
    lab = np.zeros(shape, np.int32)
    for i, c in enumerate(centers, 1):
        rr = disk(c, r, shape=shape)
        img[rr] = 255
        lab[rr] = i
    return img, lab


@pytest.fixture
def data():
    img_a, lab_a = scene([(20, 20), (30, 150), (150, 40), (160, 160)])
    img_b, _ = scene([])  # no nuclei at all
    # oracle of image a: one nucleus missed by the student, one more found
    _, oracle_a = scene([(20, 20), (30, 150), (150, 40), (100, 100)])
    images = {"a": img_a, "b": img_b}
    oracle = {"a": oracle_a, "b": np.zeros((200, 200), np.int32)}
    return images, oracle, lab_a


def test_score_round_tables(data):
    images, oracle, _ = data
    tiles, imgs, nuclei = score_round(
        ThresholdModel(), ["a", "b"], images, oracle, {"a": "pool", "b": "val"}, tile=100
    )
    assert len(tiles) == 8 and set(tiles["part"]) == {"pool", "val"}
    assert imgs.set_index("image_id").loc["a", "n_nuclei"] == 4
    # equivariant model: every nucleus is found in every version with the same contour
    assert (nuclei["D"] == 1).all() and np.allclose(nuclei["K"], 0)
    assert nuclei["tile"].tolist() == [0, 1, 2, 3]
    assert nuclei["iou_oracle"].tolist() == pytest.approx([1, 1, 1, 0])


def test_tile_counts_add_up_to_image_pq(data):
    images, oracle, lab_a = data
    tiles, imgs, _ = score_round(ThresholdModel(), ["a"], images, oracle, {"a": "pool"}, tile=100)
    pq = panoptic_quality(oracle["a"], lab_a)
    assert (tiles["tp"].sum(), tiles["fp"].sum(), tiles["fn"].sum()) == (
        pq["tp"],
        pq["fp"],
        pq["fn"],
    )
    assert imgs["pq"].iloc[0] == pytest.approx(pq["pq"])
    t = tiles.set_index("tile")
    assert t.loc[3, "fp"] == 1 and t.loc[3, "fn"] == 1  # (160, 160) vs (100, 100) both in tile 3


def test_empty_image_gets_highest_instability(data):
    images, oracle, _ = data
    _, imgs, nuclei = score_round(ThresholdModel(), ["b"], images, oracle, {"b": "val"}, tile=100)
    row = imgs.iloc[0]
    assert row["n_nuclei"] == 0 and len(nuclei) == 0
    assert row["K_mean"] == 1.0 and row["low_D"] == 1.0
