import numpy as np
import pytest
from scipy import ndimage as ndi
from skimage.draw import disk

from histopathology_pnw.uncertainty.tta import (
    D4,
    D4_SMALL,
    TTAPrediction,
    match_version,
    nucleus_uncertainty,
    predict_tta,
)


# ---- transforms ----
@pytest.mark.parametrize("s", D4, ids=lambda s: s.name)
def test_inverse_restores_non_square_and_channels(s):
    x = np.random.default_rng(0).random((7, 11, 3))
    y = s.forward(x)
    assert np.array_equal(s.inverse(y), x)
    assert np.array_equal(s.inverse(s.forward(x[..., 0])), x[..., 0])


def test_d4_named_elements():
    x = np.arange(12).reshape(3, 4)
    by = {s.name: s for s in D4}
    assert np.array_equal(by["flip_lr"].forward(x), np.fliplr(x))
    assert np.array_equal(by["flip_ud"].forward(x), np.flipud(x))
    assert np.array_equal(by["rot180"].forward(x), np.rot90(x, 2))
    outs = {by[n].forward(x).tobytes() + bytes(by[n].forward(x).shape) for n in by}
    assert len(outs) == 8  # all distinct
    assert [s.name for s in D4_SMALL] == ["identity", "flip_lr", "flip_ud", "rot180"]


# ---- matching ----
def _scene():
    ref = np.zeros((100, 100), np.int32)
    ref[disk((20, 20), 8)] = 1  # A
    ref[disk((50, 40), 8)] = 2  # B
    ref[disk((50, 58), 8)] = 3  # C (touches B region)
    return ref


def test_match_example_merge_and_extra():
    ref = _scene()
    oth = np.zeros_like(ref)
    oth[disk((20, 21), 8)] = 1  # A' slightly shifted
    oth[(ref == 2) | (ref == 3)] = 2  # B and C merged
    oth[disk((85, 85), 6)] = 3  # D: extra nucleus
    r = match_version(ref, oth)
    # IoU(B, B∪C) = |B| / (|B| + |C|) = 0.5 for equal disks, so B and C have no match (> 0.5 needed)
    assert r["matched"].tolist() == [True, False, False]
    assert r["best_iou"][1] == pytest.approx(0.5, abs=0.01)
    assert r["merge"].tolist() == [False, True, True]
    assert r["split"].tolist() == [False, False, False]
    assert r["n_extra"] == 1


def test_match_split():
    ref = np.zeros((60, 60), np.int32)
    ref[20:40, 10:50] = 1
    oth = np.zeros_like(ref)
    oth[20:40, 10:30] = 1
    oth[20:40, 30:50] = 2
    r = match_version(ref, oth)
    assert r["split"][0] and not r["merge"][0] and not r["matched"][0]


# ---- fake Cellpose models ----
class EquivariantModel:
    """Threshold + connected components: gives the same nuclei in every orientation."""

    def eval(self, img, batch_size=8, **kw):
        fg = img[..., 0] > 128
        lab, _ = ndi.label(fg)
        cp = np.where(fg, 4.0, -4.0).astype(np.float32)
        return lab, [None, None, cp], None


class AnisotropicModel(EquivariantModel):
    """Closes gaps along image rows only, so neighbours side by side merge in some orientations."""

    def eval(self, img, batch_size=8, **kw):
        fg = ndi.binary_closing(img[..., 0] > 128, structure=np.ones((1, 9)))
        lab, _ = ndi.label(fg)
        cp = np.where(fg, 4.0, -4.0).astype(np.float32) + (img[..., 1] / 255.0 - 0.5)
        return lab, [None, None, cp], None


def _image():
    img = np.zeros((120, 120, 3), np.uint8)
    for y, x in [(30, 30), (30, 48), (80, 80), (90, 30)]:  # first two: 4 px apart horizontally
        img[disk((y, x), 7)] = 255
    img[..., 1] = np.random.default_rng(1).integers(0, 255, (120, 120))
    return img


def test_equivariant_model_is_perfectly_stable():
    pred = predict_tta(EquivariantModel(), _image(), D4, eval_kwargs={})
    df, st = nucleus_uncertainty(pred)
    assert len(df) == 4 and st["n_versions"] == 7
    assert (df["D"] == 1).all() and (df["K"] < 1e-9).all() and (df["Top"] == 0).all()
    assert (df["U_TTA"] < 1e-12).all()
    assert st["E_extra"] == 0


def test_anisotropic_model_shows_topology_instability():
    pred = predict_tta(AnisotropicModel(), _image(), D4, eval_kwargs={})
    df, _ = nucleus_uncertainty(pred)
    # the original merges the two horizontal neighbours into one nucleus; rotated versions split it
    big = df.sort_values("area").iloc[-1]
    assert big["Top"] > 0.3 and big["D"] < 1
    lone = df.sort_values("area").iloc[0]
    assert lone["Top"] == 0
    assert (df["U_TTA"] > 0).any()


def test_prediction_roundtrip(tmp_path):
    pred = predict_tta(EquivariantModel(), _image(), D4_SMALL, eval_kwargs={})
    pred.save(tmp_path / "p.npz")
    back = TTAPrediction.load(tmp_path / "p.npz")
    assert back.names == pred.names and np.array_equal(back.masks[2], pred.masks[2])


def test_first_symmetry_must_be_identity():
    with pytest.raises(ValueError):
        predict_tta(EquivariantModel(), _image(), D4[1:], eval_kwargs={})
