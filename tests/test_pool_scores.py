import numpy as np
from scipy import ndimage as ndi
from skimage.draw import disk

from histopathology_pnw.uncertainty.pool_scores import score_images


class ThresholdModel:
    """Segments bright blobs; equivariant under rotations and mirrors, so TTA is stable."""

    def eval(self, img, batch_size=8, **kwargs):
        fg = img[..., 0] > 128
        lab, _ = ndi.label(fg)
        return lab, [None, None, np.where(fg, 4.0, -4.0).astype(np.float32)], None


def _image(centres) -> np.ndarray:
    img = np.zeros((100, 100, 3), np.uint8)
    for y, x in centres:
        img[disk((y, x), 7)] = 255
    return img


def test_one_row_per_nucleus_and_per_image_with_offline_oracle_columns():
    images = {"a": _image([(25, 25), (70, 70)]), "b": _image([(50, 50)])}
    full = ndi.label(images["a"][..., 0] > 128)[0]
    oracle = {"a": full, "b": np.zeros((100, 100), np.int32)}  # the oracle sees nothing on b
    nuclei, per_image = score_images(
        ThresholdModel(), ["a", "b"], images, oracle, {"a": "pool", "b": "val"}, eval_kwargs={}
    )
    assert nuclei.groupby("image_id").size().to_dict() == {"a": 2, "b": 1}
    assert (nuclei["D"] == 1).all() and (nuclei["Top"] == 0).all()
    by_id = nuclei.groupby("image_id")["iou_oracle"].min()
    assert by_id["a"] == 1.0 and by_id["b"] == 0.0
    img = per_image.set_index("image_id")
    assert img["pq_oracle"].to_dict() == {"a": 1.0, "b": 0.0}
    assert img["part"].to_dict() == {"a": "pool", "b": "val"}
    assert (img["n_nuclei"].to_dict(), img["n_oracle"].to_dict()) == (
        {"a": 2, "b": 1},
        {"a": 2, "b": 0},
    )
