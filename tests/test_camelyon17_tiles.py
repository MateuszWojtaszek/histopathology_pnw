import numpy as np
import pandas as pd
import pytest
from skimage.io import imsave

from histopathology_pnw.al.camelyon17_students import draw_order
from histopathology_pnw.config import Settings
from histopathology_pnw.datasets import camelyon17_tiles as c17


class CountingModel:
    """Fake Cellpose model: one square nucleus per tile, counts its calls."""

    calls = 0

    def eval(self, img, batch_size=8, **kw):
        CountingModel.calls += 1
        m = np.zeros(img.shape[:2], np.uint16)
        m[2:6, 2:6] = 1
        return m, None, None


@pytest.fixture
def settings(tmp_path):
    s = Settings(data_dir=tmp_path / "data", outputs_dir=tmp_path / "outputs")
    root = c17.tiles_root(s)
    (root / "tiles").mkdir(parents=True)
    rows = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        tid = f"patient_00{i}_node_0_x0_y0"
        imsave(
            root / "tiles" / f"{tid}.png",
            np.full((8, 8, 3), 40 * i, np.uint8),
            check_contrast=False,
        )
        rows.append({"tile_id": tid, "split": split, "group": "guz"})
    pd.DataFrame(rows).to_csv(root / "tiles_manifest.csv", index=False)
    return s


def test_tile_ids_by_split_and_read(settings):
    assert len(c17.tile_ids("train", settings)) == 2
    assert c17.tile_ids("val", settings) == ["patient_002_node_0_x0_y0"]
    img = c17.read_tile("patient_001_node_0_x0_y0", settings)
    assert img.shape == (8, 8, 3) and img.dtype == np.uint8 and img[0, 0, 0] == 40
    with pytest.raises(ValueError):
        c17.tile_ids("pool", settings)


def test_oracle_masks_are_cached(settings):
    ids = c17.tile_ids("train", settings)
    CountingModel.calls = 0
    first = c17.oracle_tile_masks(ids, model=CountingModel(), settings=settings)
    again = c17.oracle_tile_masks(ids, model=CountingModel(), settings=settings)
    assert CountingModel.calls == 2  # second call reads the cache
    assert all(m.max() == 1 for m in first + again)
    assert sorted(
        p.name for p in (settings.outputs_dir / "oracle_masks" / "camelyon17").iterdir()
    ) == [f"{i}.tif" for i in sorted(ids)]


def test_draw_order_is_a_fixed_permutation():
    ids = [f"t{i}" for i in range(20)]
    a, b = draw_order(ids, 0), draw_order(ids, 0)
    assert a == b and sorted(a) == sorted(ids) and a != draw_order(ids, 1)
