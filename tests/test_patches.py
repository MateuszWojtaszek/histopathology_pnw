import numpy as np
import pandas as pd
import pytest
from skimage.draw import disk

from histopathology_pnw.uncertainty.patches import (
    TileGrid,
    centroids,
    extra_labels,
    extra_per_tile,
    image_from_tiles,
    tile_quality,
    tile_uncertainty,
)
from histopathology_pnw.uncertainty.tta import TTAPrediction


def blobs(shape, centers, r=6):
    m = np.zeros(shape, np.int32)
    for i, (y, x) in enumerate(centers, 1):
        m[disk((y, x), r, shape=shape)] = i
    return m


def pq_global(gt, pred):
    """Reference PQ written independently of the tile code."""
    n_gt, n_pred = int(gt.max()), int(pred.max())
    tp, s = 0, 0.0
    for i in range(1, n_gt + 1):
        for j in range(1, n_pred + 1):
            a, b = gt == i, pred == j
            inter = (a & b).sum()
            if inter == 0:
                continue
            iou = inter / (a | b).sum()
            if iou > 0.5:
                tp, s = tp + 1, s + iou
    fp, fn = n_pred - tp, n_gt - tp
    return tp, fp, fn, (s / tp) * tp / (tp + fp / 2 + fn / 2)


# ---- grid ---------------------------------------------------------------------------------------
def test_grid_shape_and_assignment():
    g = TileGrid(1000, 1000, 250)
    assert (g.n_rows, g.n_cols, g.n_tiles) == (4, 4, 16)
    assert list(g.tile_of([0, 249.9, 250, 999], [0, 0, 260, 999])) == [0, 0, 5, 15]
    assert g.tile_of([np.nan], [10])[0] == -1


def test_grid_uneven_last_tile():
    g = TileGrid(1001, 600, 250)
    f = g.frame()
    assert (g.n_rows, g.n_cols) == (5, 3)
    assert f["h"].iloc[-1] == 1 and f["w"].iloc[-1] == 100
    assert f["h"].mul(f["w"]).sum() == 1001 * 600


def test_grid_overlapping_tiles():
    g = TileGrid(1000, 1000, 256, stride=248)
    f = g.frame()
    assert (g.n_rows, g.n_cols) == (4, 4)
    assert list(f["y0"].iloc[::4]) == [0, 248, 496, 744]
    assert (f["h"] == 256).all() and (f["w"] == 256).all()  # last tile ends at the border
    # ownership: bands of 248 px, the last one 256 px; the 8 px overlap goes to the next tile
    assert list(g.tile_of([247.9, 248, 743.9, 744, 999], [0] * 5)) == [0, 4, 8, 12, 12]
    img = np.arange(1000 * 1000).reshape(1000, 1000)
    assert img[g.box(5)].shape == (256, 256) and img[g.box(5)][0, 0] == img[248, 248]


def test_centroids():
    m = blobs((100, 100), [(20, 30), (70, 80)])
    cy, cx = centroids(m)
    np.testing.assert_allclose(cy, [20, 70], atol=1e-9)
    np.testing.assert_allclose(cx, [30, 80], atol=1e-9)


# ---- tile quality --------------------------------------------------------------------------------
@pytest.fixture
def pair():
    shape = (200, 200)
    oracle = blobs(shape, [(20, 20), (20, 120), (120, 20), (120, 120), (180, 180)])
    # matched with a 1 px shift, one nucleus missed (180, 180), one false positive (60, 170)
    student = blobs(shape, [(21, 20), (20, 121), (121, 21), (119, 120), (60, 170)])
    return oracle, student


def test_tile_counts_sum_to_image(pair):
    oracle, student = pair
    tq = tile_quality(student, oracle, TileGrid(200, 200, 100))
    tp, fp, fn, pq = pq_global(oracle, student)
    assert (tq["tp"].sum(), tq["fp"].sum(), tq["fn"].sum()) == (tp, fp, fn)
    pooled = tq["sum_iou"].sum() / (tp + fp / 2 + fn / 2)
    assert pooled == pytest.approx(pq)


def test_single_tile_equals_image_pq(pair):
    oracle, student = pair
    tq = tile_quality(student, oracle, TileGrid(200, 200, 500))
    assert len(tq) == 1
    assert tq["pq"].iloc[0] == pytest.approx(pq_global(oracle, student)[3])


def test_tile_quality_places_errors(pair):
    oracle, student = pair
    tq = tile_quality(student, oracle, TileGrid(200, 200, 100)).set_index("tile")
    assert tq.loc[1, "fp"] == 1  # false positive at (60, 170)
    assert tq.loc[3, "fn"] == 1  # missed nucleus at (180, 180)
    assert tq.loc[0, "fp"] == tq.loc[0, "fn"] == 0 and tq.loc[0, "pq"] > 0.6


def test_empty_tile_is_nan():
    shape = (200, 200)
    m = blobs(shape, [(20, 20)])
    tq = tile_quality(m, m, TileGrid(200, 200, 100)).set_index("tile")
    assert tq.loc[0, "pq"] == pytest.approx(1.0)
    assert np.isnan(tq.loc[3, "pq"])


# ---- extra nuclei and tile scores ----------------------------------------------------------------
def make_pred():
    shape = (200, 200)
    ref = blobs(shape, [(20, 20), (20, 120), (120, 20)])
    same = ref.copy()
    with_extra = blobs(shape, [(20, 20), (20, 120), (120, 20), (150, 150)])
    zeros = [np.zeros(shape, np.float32)] * 3
    return TTAPrediction(["identity", "a", "b"], [ref, same, with_extra], zeros)


def test_extra_labels_and_tiles():
    pred = make_pred()
    assert list(extra_labels(pred.masks[0], pred.masks[2])) == [4]
    e = extra_per_tile(pred, TileGrid(200, 200, 100))
    assert e[3] == pytest.approx(0.5)  # one extra nucleus in one of two versions
    assert e[[0, 1, 2]].sum() == 0


def test_tile_uncertainty_aggregates():
    pred = make_pred()
    nuclei = pd.DataFrame(
        {
            "label": [1, 2, 3],
            "D": [1.0, 0.5, 1.0],
            "K": [0.1, 0.3, 0.2],
            "U_MI": [0.01, 0.02, 0.03],
            "Top": [0.0, 0.5, 0.0],
        }
    )
    t = tile_uncertainty(nuclei, pred, TileGrid(200, 200, 100)).set_index("tile")
    assert list(t["n_nuclei"]) == [1, 1, 1, 0]
    assert t.loc[1, "K_mean"] == pytest.approx(0.3)
    assert t.loc[1, "frac_D_lt1"] == pytest.approx(1.0)
    assert np.isnan(t.loc[3, "K_mean"])
    assert t.loc[3, "E_extra"] == pytest.approx(0.5)  # no reference nuclei: divided by 1
    assert t.loc[3, "unstable"] == pytest.approx(0.5)
    assert t.loc[0, "unstable"] == pytest.approx(0.0)


def test_image_from_tiles():
    t = pd.DataFrame({"n_nuclei": [10, 10, 10, 2], "s": [0.1, 0.4, 0.3, 0.9]})
    assert image_from_tiles(t, "s", "max") == pytest.approx(0.9)
    assert image_from_tiles(t, "s", "max", min_nuclei=5) == pytest.approx(0.4)
    assert image_from_tiles(t, "s", "topk", k=2, min_nuclei=5) == pytest.approx(0.35)
    assert image_from_tiles(t, "s", "mean", min_nuclei=5) == pytest.approx(0.8 / 3)
    assert np.isnan(image_from_tiles(t, "s", "max", min_nuclei=50))
