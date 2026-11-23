"""Selectors for AL on tiles: which unlabelled tiles are queried next.

The image-level selectors of the AL loop (`random`, `oracle_peek`, `tta_kd`; branch
`oracle-peek`, `al/selectors.py`) carried over to tiles. The student always segments the whole
pool image (full context, no artificial borders); only the scores are split into tiles of the
same grid as the pool (256 px every 248 px). A nucleus belongs to the tile that owns its
centroid (`patches.TileGrid.tile_of`).

- "random": the next tiles of the seed's permutation (nested labelled sets).
- "oracle_peek": a DIAGNOSTIC, not a method: score = 1 - PQ of the student against the oracle
  per tile (matches found on the whole image, `patches.tile_quality`). It reads the pool labels,
  so it bounds what any "query the hardest tiles" selector can gain over random.
- "tta_kd": no labels. Per tile, from the D4 TTA of the whole image: `K_mean` (mean contour
  instability of its nuclei) and `unstable` = share of nuclei not found in every version +
  extra nuclei per nucleus (`patches.tile_uncertainty`). Score = mean of the two percentile
  ranks among the candidate tiles.

A tile with fewer than `min_nuclei` nuclei of the student (background, stroma, a cut corner) has
too few nuclei for a stable tile score (1 - PQ of a tile with one nucleus is 0 or 1). Such tiles
are candidates only after all the others, in the order of the seed's permutation. At image level
this rule was not needed (every MoNuSeg image has hundreds of nuclei).
"""

from collections import defaultdict

import numpy as np
import pandas as pd

from histopathology_pnw.inference import predict
from histopathology_pnw.uncertainty import tta
from histopathology_pnw.uncertainty.patches import (
    TileGrid,
    centroids,
    tile_quality,
    tile_uncertainty,
)

SELECTORS = ("random", "oracle_peek", "tta_kd")
MIN_NUCLEI = 5


def _by_image(keys: list[str], split_key) -> dict[str, list[int]]:
    tiles = defaultdict(list)
    for key in keys:
        image_id, t = split_key(key)
        tiles[image_id].append(t)
    return tiles


def _student_counts(mask: np.ndarray, grid: TileGrid) -> np.ndarray:
    cy, cx = centroids(mask)
    owner = grid.tile_of(cy, cx)
    return np.bincount(owner[owner >= 0], minlength=grid.n_tiles)


def oracle_peek_tile_scores(
    student, keys, images, oracle, split_key, tile_key, size=256, stride=248
) -> pd.DataFrame:
    """1 - PQ (student vs oracle) of every unlabelled tile; one prediction per pool image."""
    rows = []
    for image_id, tiles in _by_image(keys, split_key).items():
        pred = predict(student, [images[image_id]], progress=False)[0]
        grid = TileGrid(*pred.shape, size=size, stride=stride)
        q = tile_quality(pred, oracle[image_id], grid).set_index("tile")
        n = _student_counts(pred, grid)
        for t in tiles:
            rows.append(
                {
                    "tile_id": tile_key(image_id, t),
                    "n_nuclei": int(n[t]),
                    "n_oracle": int(q.loc[t, "n_oracle"]),
                    "pq_oracle": q.loc[t, "pq"],
                    "score": 1.0 - q.loc[t, "pq"] if np.isfinite(q.loc[t, "pq"]) else 0.0,
                }
            )
    return pd.DataFrame(rows)


def tta_kd_tile_scores(
    student, keys, images, split_key, tile_key, oracle=None, size=256, stride=248
) -> pd.DataFrame:
    """Mean rank of contour and detection instability under D4 TTA, per unlabelled tile.

    `oracle` is NOT used for the score; when given, the student's PQ against it per tile is
    stored as `pq_oracle_offline` for analysis after the run.
    """
    rows = []
    for image_id, tiles in _by_image(keys, split_key).items():
        pred = tta.predict_tta(student, images[image_id])
        nuclei, _ = tta.nucleus_uncertainty(pred)
        grid = TileGrid(*pred.masks[0].shape, size=size, stride=stride)
        u = tile_uncertainty(nuclei, pred, grid).set_index("tile")
        q = None
        if oracle is not None:
            q = tile_quality(pred.masks[0], oracle[image_id], grid).set_index("tile")
        for t in tiles:
            row = {
                "tile_id": tile_key(image_id, t),
                "n_nuclei": int(u.loc[t, "n_nuclei"]),
                "K_mean": u.loc[t, "K_mean"],
                "frac_D_lt1": u.loc[t, "frac_D_lt1"],
                "E_extra": u.loc[t, "E_extra"],
                "unstable": u.loc[t, "unstable"],
            }
            if q is not None:
                row["pq_oracle_offline"] = q.loc[t, "pq"]
            rows.append(row)
    return combine_tta(pd.DataFrame(rows))


def combine_tta(table: pd.DataFrame, min_nuclei: int = MIN_NUCLEI) -> pd.DataFrame:
    """Score = mean percentile rank of K_mean and unstable among tiles with >= min_nuclei."""
    ok = table["n_nuclei"] >= min_nuclei
    ranks = table.loc[ok, ["K_mean", "unstable"]].rank(pct=True)
    table = table.assign(rank_K=ranks["K_mean"], rank_unstable=ranks["unstable"])
    return table.assign(score=table[["rank_K", "rank_unstable"]].mean(axis=1).fillna(0.0))


def pick_tiles(
    scores: pd.DataFrame, k: int, rank: dict[str, int], min_nuclei: int = MIN_NUCLEI
) -> tuple[list[str], pd.DataFrame]:
    """The `k` best tiles: candidates with >= min_nuclei by score (ties: earlier in the seed's
    permutation), then the remaining tiles in permutation order. Returns the chosen ids (best
    first) and `scores` with `eligible`, `order_rank` and `selected` columns."""
    table = scores.assign(
        eligible=scores["n_nuclei"] >= min_nuclei,
        order_rank=[rank[t] for t in scores["tile_id"]],
    )
    best = table.sort_values(["eligible", "score", "order_rank"], ascending=[False, False, True])
    chosen = best["tile_id"].head(k).tolist()
    return chosen, table.assign(selected=table["tile_id"].isin(chosen))
