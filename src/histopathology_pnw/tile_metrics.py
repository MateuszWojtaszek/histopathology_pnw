"""Segmentation metrics per tile of an image (the image is predicted whole, only the scoring is split).

Two views of the same tiles, both returned by `evaluate_tiles`:
- crop (`pq`, `dq` = F1@0.5, `aji`, ...): both masks are cut to the tile box and scored with
  `metrics.evaluate` as if the tile were an image. Nuclei cut by the border become fragments on
  both sides, as they would for an annotator who sees only the tile. A tile with no nucleus on
  either side gets NaN (not the 1.0 of an empty image), so it does not inflate the mean.
- owned (`own_pq`, `own_sq`, `own_dq`): matches are found on the whole image and every nucleus
  goes to the tile that owns its centroid (`patches.tile_quality`); no cropping, the tile counts
  sum to the image counts (pooled over tiles = the image PQ).
"""

import warnings

import numpy as np
import pandas as pd
from skimage.segmentation import relabel_sequential

from histopathology_pnw.metrics import evaluate
from histopathology_pnw.uncertainty.patches import TileGrid, tile_quality

CROP_METRICS = ("pq", "sq", "dq", "pq@0.75", "aji", "ap@0.5")


def crop_mask(mask: np.ndarray, box: tuple[slice, slice]) -> np.ndarray:
    """Instance mask cut to a box and relabelled 1..N (fragments of cut nuclei are kept)."""
    return relabel_sequential(np.ascontiguousarray(mask[box]))[0].astype(np.uint16)


def evaluate_tiles(gt: np.ndarray, pred: np.ndarray, grid: TileGrid) -> pd.DataFrame:
    """One row per tile: crop metrics and owned PQ/SQ/DQ (see the module docstring)."""
    rows = []
    for t in range(grid.n_tiles):
        box = grid.box(t)
        g, p = crop_mask(gt, box), crop_mask(pred, box)
        if g.max() == 0 and p.max() == 0:
            m = {k: np.nan for k in CROP_METRICS} | {"n_gt": 0, "n_pred": 0}
        else:
            with warnings.catch_warnings():  # AJI of a tile with no GT nucleus: 0/0 warnings
                warnings.simplefilter("ignore", RuntimeWarning)
                m = evaluate(g, p)
        rows.append({"tile": t, **{k: m[k] for k in (*CROP_METRICS, "n_gt", "n_pred")}})
    own = tile_quality(pred, gt, grid)[["tile", "tp", "fp", "fn", "sum_iou", "pq", "sq", "dq"]]
    own = own.rename(columns={c: f"own_{c}" for c in own.columns if c != "tile"})
    return pd.DataFrame(rows).merge(own, on="tile")


def summarize_tiles(per_tile: pd.DataFrame) -> dict:
    """Mean over tiles (NaN tiles skipped) of the crop metrics and of the owned PQ."""
    out = {f"tile_{k}": float(per_tile[k].mean()) for k in CROP_METRICS}
    out["tile_own_pq"] = float(per_tile["own_pq"].mean())
    out["n_tiles"] = int(per_tile[list(CROP_METRICS)].notna().any(axis=1).sum())
    return out
