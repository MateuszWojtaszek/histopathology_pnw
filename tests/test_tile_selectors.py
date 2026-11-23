import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.draw import disk

from histopathology_pnw.al.tile_random_loop import split_key, tile_key
from histopathology_pnw.al.tile_selectors import (
    combine_tta,
    oracle_peek_tile_scores,
    pick_tiles,
    tta_kd_tile_scores,
)


class ThresholdModel:
    """Threshold + connected components: the same nuclei in every orientation."""

    def eval(self, img, batch_size=8, **kw):
        fg = img[..., 0] > 128
        lab, _ = ndi.label(fg)
        return lab, [None, None, np.where(fg, 4.0, -4.0).astype(np.float32)], None


def scene(centers, shape=(1000, 1000), r=8):
    img = np.zeros((*shape, 3), np.uint8)
    lab = np.zeros(shape, np.int32)
    for i, c in enumerate(centers, 1):
        rr = disk(c, r, shape=shape)
        img[rr] = 255
        lab[rr] = i
    return img, lab


def grid_centers(y0, x0, n=6, step=30):
    return [(y0 + 20 + step * (i // 3), x0 + 20 + step * (i % 3)) for i in range(n)]


def test_oracle_peek_ranks_the_tile_the_student_gets_wrong():
    # 6 nuclei in tile 0 and in tile 5; the oracle has 3 more nuclei in tile 5 (student misses them)
    img, _ = scene(grid_centers(0, 0) + grid_centers(248, 248))
    _, oracle = scene(
        grid_centers(0, 0) + grid_centers(248, 248) + [(400, 300), (420, 340), (440, 380)]
    )
    keys = [tile_key("a", t) for t in range(16)]
    s = oracle_peek_tile_scores(
        ThresholdModel(), keys, {"a": img}, {"a": oracle}, split_key, tile_key
    ).set_index("tile_id")
    assert s.loc["a#t00", "score"] == 0 and s.loc["a#t05", "score"] > 0.15  # 6 TP, 3 FN: 1 - 6/7.5
    assert s.loc["a#t00", "n_nuclei"] == 6 and s.loc["a#t15", "score"] == 0  # empty tile
    rank = {k: i for i, k in enumerate(keys)}
    chosen, table = pick_tiles(s.reset_index(), 2, rank)
    assert chosen == ["a#t05", "a#t00"] and table["selected"].sum() == 2


def test_tta_kd_scores_only_rank_tiles_with_enough_nuclei():
    img, _ = scene([*grid_centers(0, 0), (300, 300)])  # tile 0: 6 nuclei, tile 5: 1 nucleus
    keys = [tile_key("a", t) for t in range(16)]
    s = tta_kd_tile_scores(ThresholdModel(), keys, {"a": img}, split_key, tile_key).set_index(
        "tile_id"
    )
    assert s.loc["a#t00", "n_nuclei"] == 6 and s.loc["a#t05", "n_nuclei"] == 1
    assert np.isfinite(s.loc["a#t00", "rank_K"]) and np.isnan(s.loc["a#t05", "rank_K"])
    assert s.loc["a#t05", "score"] == 0


def test_combine_tta_and_pick_fill_with_permutation_order():
    t = pd.DataFrame(
        {
            "tile_id": ["x", "y", "z", "w"],
            "n_nuclei": [10, 10, 2, 0],
            "K_mean": [0.1, 0.3, 0.9, np.nan],
            "unstable": [0.2, 0.4, 0.9, 0.0],
        }
    )
    s = combine_tta(t).set_index("tile_id")
    assert s.loc["y", "score"] == 1.0 and s.loc["x", "score"] == 0.5 and s.loc["z", "score"] == 0
    chosen, _ = pick_tiles(s.reset_index(), 3, {"x": 3, "y": 2, "z": 1, "w": 0})
    assert chosen == ["y", "x", "w"]  # ineligible tiles last, in permutation order
