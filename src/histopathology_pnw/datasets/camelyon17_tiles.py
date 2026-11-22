"""CAMELYON17 tiles for nucleus segmentation (read only; the tiles are made on the `camelyon17` branch).

Tiles are 1000 x 1000 px RGB crops at level 0 (as MoNuSeg), cut from the slides with lesion
annotations: per slide up to 4 "guz" (>= 50% tumour) and 4 "normalna" (0% tumour) tiles with
dense tissue. `tiles_manifest.csv` holds one row per tile with its slide, patient, centre and
the split (train / val / test, by patient). CAMELYON17 has no nucleus masks: the oracle
(`cpsam`) is the annotator and the reference, as in the AL simulation on MoNuSeg.

The test tiles are never used for training, model selection or AL selection.
Data: `uv run dvc pull data/processed/camelyon17.dvc`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.io import imread

from histopathology_pnw.config import Settings
from histopathology_pnw.inference import predict


def tiles_root(settings: Settings | None = None) -> Path:
    return (settings or Settings()).data_dir / "processed" / "camelyon17"


def manifest(settings: Settings | None = None) -> pd.DataFrame:
    path = tiles_root(settings) / "tiles_manifest.csv"
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} not found; run `uv run dvc pull data/processed/camelyon17.dvc`"
        )
    return pd.read_csv(path)


def tile_ids(split: str, settings: Settings | None = None) -> list[str]:
    """Tile ids of one split ("train", "val" or "test"), in manifest order."""
    if split not in ("train", "val", "test"):
        raise ValueError(f"unknown split {split!r}")
    m = manifest(settings)
    return m.loc[m["split"] == split, "tile_id"].tolist()


def read_tile(tile_id: str, settings: Settings | None = None) -> np.ndarray:
    """H x W x 3 uint8 RGB tile."""
    image = imread(tiles_root(settings) / "tiles" / f"{tile_id}.png")
    if image.ndim == 3 and image.shape[2] == 4:  # drop an alpha channel if a viewer added one
        image = image[..., :3]
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{tile_id}: expected HxWx3 uint8, got {image.shape} {image.dtype}")
    return image


def oracle_tile_masks(
    ids: list[str], cache_dir: Path | None = None, model=None, settings: Settings | None = None
) -> list[np.ndarray]:
    """Oracle masks of tiles, computed once and cached as `<cache_dir>/<tile_id>.tif`."""
    settings = settings or Settings()
    cache_dir = cache_dir or settings.outputs_dir / "oracle_masks" / "camelyon17"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{t}.tif" for t in ids]
    todo = [i for i, p in enumerate(paths) if not p.is_file()]
    if todo:
        if model is None:
            from histopathology_pnw.models.oracle import load_oracle

            model = load_oracle()
        for start in range(0, len(todo), 16):  # small chunks: a killed job keeps what it finished
            chunk = todo[start : start + 16]
            preds = predict(model, [read_tile(ids[i], settings) for i in chunk], progress=False)
            for i, mask in zip(chunk, preds, strict=True):
                tmp = paths[i].with_suffix(".partial.tif")
                tifffile.imwrite(tmp, mask, compression="zlib")
                tmp.replace(paths[i])
    return [tifffile.imread(p) for p in paths]
