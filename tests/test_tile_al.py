import numpy as np
import pytest
from skimage.draw import disk

from histopathology_pnw.al.random_loop import round_sizes
from histopathology_pnw.al.test_eval import round_dir
from histopathology_pnw.al.tile_random_loop import (
    TileALConfig,
    crop_flows,
    cut_pool,
    draw_order,
    loss_ignoring_unannotated,
    margin_target,
    n_labels,
    split_key,
    tile_key,
)
from histopathology_pnw.metrics import panoptic_quality
from histopathology_pnw.tile_metrics import crop_mask, evaluate_tiles, summarize_tiles
from histopathology_pnw.uncertainty.patches import TileGrid


def nuclei(centers, shape=(1000, 1000), r=8):
    m = np.zeros(shape, np.uint16)
    for i, c in enumerate(centers, 1):
        m[disk(c, r, shape=shape)] = i
    return m


def test_tile_keys_roundtrip():
    k = tile_key("TCGA-18-5592-01Z-00-DX1", 7)
    assert k == "TCGA-18-5592-01Z-00-DX1#t07" and split_key(k) == ("TCGA-18-5592-01Z-00-DX1", 7)


def test_default_rounds_match_image_al_budget():
    cfg = TileALConfig()
    sizes = round_sizes(cfg, 30 * 16)
    assert sizes[0] == 32 and sizes[-1] == 480 and len(sizes) == 15  # image AL: 2, 4, ..., 30


def test_cut_pool_covers_images_and_relabels():
    img = np.random.default_rng(0).integers(0, 255, (1000, 1000, 3), np.uint8)
    # a nucleus on the border of tiles 0 and 1 (x = 250): cut into two fragments
    mask = nuclei([(100, 250), (600, 600)])
    keys, timgs, tmasks = cut_pool({"a": img}, {"a": mask}, 256, 248)
    assert len(keys) == 16 and keys[0] == "a#t00"
    assert all(t.shape == (256, 256, 3) for t in timgs.values())
    assert np.array_equal(timgs["a#t05"], img[248:504, 248:504])
    assert tmasks["a#t00"].max() == 1 and tmasks["a#t01"].max() == 1  # fragment kept on both
    assert tmasks["a#t15"].max() == 0
    assert tmasks["a#t10"].max() == 1 and tmasks["a#t10"].dtype == np.uint16


def test_draw_order_is_a_fixed_permutation():
    keys = [tile_key("a", t) for t in range(16)]
    a = draw_order(keys, 0)
    assert a == draw_order(keys, 0) and sorted(a) == sorted(keys) and a != draw_order(keys, 1)


def test_crop_mask_relabels():
    m = nuclei([(20, 20), (500, 500), (30, 60)])
    c = crop_mask(m, (slice(0, 100), slice(0, 100)))
    assert sorted(np.unique(c)) == [0, 1, 2]


def test_evaluate_tiles():
    gt = nuclei([(100, 100), (100, 400), (600, 600), (900, 900)])
    pred = nuclei([(100, 100), (100, 400), (600, 600), (300, 900)])  # 1 FN, 1 FP
    grid = TileGrid(1000, 1000, 256, stride=248)
    t = evaluate_tiles(gt, pred, grid).set_index("tile")
    assert len(t) == 16
    assert t.loc[0, "pq"] > 0.99 and t.loc[0, "own_pq"] > 0.99
    assert np.isnan(t.loc[2, "pq"]) and np.isnan(t.loc[2, "own_pq"])  # empty on both sides
    assert t.loc[15, "pq"] == 0 and t.loc[15, "own_fn"] == 1  # GT nucleus missed
    # owned counts sum to the image counts
    img = panoptic_quality(gt, pred)
    assert (t["own_tp"].sum(), t["own_fp"].sum(), t["own_fn"].sum()) == (
        img["tp"],
        img["fp"],
        img["fn"],
    )
    s = summarize_tiles(t.reset_index())
    assert s["n_tiles"] == 5 and 0 < s["tile_pq"] < 1


def test_round_dir_finds_both_namings(tmp_path):
    for name in ("round_04", "round_032"):
        (tmp_path / name / "models").mkdir(parents=True)
        (tmp_path / name / "models" / "student").write_bytes(b"x")
    assert round_dir(tmp_path, 4).name == "round_04"
    assert round_dir(tmp_path, 32).name == "round_032"
    with pytest.raises(FileNotFoundError):
        round_dir(tmp_path, 64)


def test_crop_flows_keeps_whole_image_flow():
    # target [labels, foreground, flow Y, flow X]; nucleus centred at x = 250 is cut by tile 0
    m = nuclei([(100, 250)], shape=(300, 300), r=10)
    flows = np.zeros((4, 300, 300), np.float32)
    flows[0], flows[1] = m, m > 0
    flows[3] = np.where(m > 0, 1.0, 0.0)  # all flow X pointing right, as towards a centre at 250
    t = crop_flows(flows, (slice(0, 256), slice(0, 256)))
    assert t.shape == (4, 256, 256) and t.dtype == np.float32
    assert n_labels(t) == 1 and set(np.unique(t[0])) == {0, 1}
    assert (t[3][t[0] > 0] == 1).all()  # flows are not recomputed for the fragment


def test_margin_target_marks_context_unannotated():
    m = nuclei([(100, 250), (400, 400)], shape=(500, 500), r=10)
    flows = np.stack([m, m > 0, np.ones_like(m), np.ones_like(m)]).astype(np.float32)
    wide, t = margin_target(flows, (slice(0, 256), slice(0, 256)), 64)
    assert wide == (slice(0, 320), slice(0, 320)) and t.shape == (4, 320, 320)
    assert (t[1, 256:] == -1).all() and (t[1, :, 256:] == -1).all() and (t[2:, 256:] == 0).all()
    assert n_labels(t) == 1 and (t[0, :, 256:] == 0).all()  # labels only inside the tile
    assert t[1, 100, 250] == 1 and t[3, 100, 250] == 1  # the cut nucleus keeps its flow


def test_loss_ignoring_unannotated_matches_cellpose_without_margin():
    import torch
    from cellpose import train as cp_train

    g = torch.Generator().manual_seed(0)
    y = torch.randn(2, 3, 16, 16, generator=g)
    lbl = torch.randn(2, 3, 16, 16, generator=g)
    lbl[:, 0] = (lbl[:, 0] > 0).float()
    original = cp_train._loss_fn_seg(lbl, y, "cpu")
    with loss_ignoring_unannotated():
        same = cp_train._loss_fn_seg(lbl, y, "cpu")
        lbl2 = lbl.clone()
        lbl2[:, :, :, 8:] = 0
        lbl2[:, 0, :, 8:] = -1  # right half not annotated
        y2 = y.clone()
        y2[:, :, :, 8:] = 100.0  # any prediction there
        masked = cp_train._loss_fn_seg(lbl2, y2, "cpu")
        half = cp_train._loss_fn_seg(lbl[..., :8], y[..., :8], "cpu")
    assert torch.allclose(original, same, atol=1e-5)
    assert torch.allclose(masked, half, atol=1e-5)
    assert cp_train._loss_fn_seg is not None and cp_train._loss_fn_seg(lbl, y, "cpu") == original
