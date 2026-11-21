"""Tile-level (patch-level) view of the TTA uncertainty: scores and difficulty per tile, not per image.

An image (MoNuSeg: 1000 x 1000) holds hundreds of nuclei, so a mean over the whole image hides a
small hard region. Here the image is cut into a grid of square tiles (default 250 px, 4 x 4) and
every quantity is computed per tile. TTA itself still runs on the whole image (full context, no
artificial borders); only the results are split.

Assignment rule (the same everywhere): an object belongs to the tile that contains its centroid,
so a nucleus cut by a tile border is counted once and never cropped.

Per tile, from the student's predictions only (usable by a selector):
- n_nuclei          number of reference nuclei (original prediction),
- K_mean, U_MI_mean, Top_mean, D_mean   mean of the per-nucleus measures (`tta.nucleus_uncertainty`),
- frac_D_lt1        share of nuclei not found in every version (D < 1),
- E_extra           mean over versions of (#extra nuclei with centroid in the tile) / max(n_nuclei, 1),
                    extra = a nucleus of a version with < alpha of its pixels on reference nuclei,
- unstable          frac_D_lt1 + E_extra (the image-level score of the `tta_kd` selector, per tile).

Per tile, against the oracle (offline evaluation only, never for selection):
- PQ of the student versus the oracle with matches (IoU > 0.5) found on the whole image; a matched
  pair and an unmatched student nucleus (FP) go to the tile of the student nucleus, an unmatched
  oracle nucleus (FN) to the tile of the oracle nucleus. Tile counts sum to the image counts, so
  pooling tiles gives back the image PQ. A tile with no nucleus on either side has PQ = NaN.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from histopathology_pnw.uncertainty.tta import TTAPrediction, overlap


# ---- grid ------------------------------------------------------------------------------------
@dataclass(frozen=True)
class TileGrid:
    """Non-overlapping square tiles covering an image; the last row/column may be smaller."""

    height: int
    width: int
    size: int = 250

    @property
    def n_rows(self) -> int:
        return -(-self.height // self.size)

    @property
    def n_cols(self) -> int:
        return -(-self.width // self.size)

    @property
    def n_tiles(self) -> int:
        return self.n_rows * self.n_cols

    def tile_of(self, cy: np.ndarray, cx: np.ndarray) -> np.ndarray:
        """Tile id (row-major) of every point; -1 for NaN coordinates."""
        cy, cx = np.asarray(cy, float), np.asarray(cx, float)
        ok = np.isfinite(cy) & np.isfinite(cx)
        r = np.clip(np.floor(np.where(ok, cy, 0) / self.size), 0, self.n_rows - 1).astype(int)
        c = np.clip(np.floor(np.where(ok, cx, 0) / self.size), 0, self.n_cols - 1).astype(int)
        return np.where(ok, r * self.n_cols + c, -1)

    def frame(self) -> pd.DataFrame:
        """One row per tile: tile id, row, col and its pixel box."""
        rows = []
        for t in range(self.n_tiles):
            r, c = divmod(t, self.n_cols)
            y0, x0 = r * self.size, c * self.size
            rows.append(
                {
                    "tile": t,
                    "row": r,
                    "col": c,
                    "y0": y0,
                    "x0": x0,
                    "h": min(self.size, self.height - y0),
                    "w": min(self.size, self.width - x0),
                }
            )
        return pd.DataFrame(rows)


def centroids(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Centroid (cy, cx) of labels 1..max; NaN for a label with no pixels."""
    n = int(mask.max())
    lab = mask.ravel().astype(np.int64)
    yy, xx = np.indices(mask.shape)
    area = np.bincount(lab, minlength=n + 1)[1:].astype(float)
    sy = np.bincount(lab, weights=yy.ravel(), minlength=n + 1)[1:]
    sx = np.bincount(lab, weights=xx.ravel(), minlength=n + 1)[1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(area > 0, sy / area, np.nan), np.where(area > 0, sx / area, np.nan)


# ---- extra nuclei, located --------------------------------------------------------------------
def extra_labels(ref: np.ndarray, other: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """Labels of `other` with < alpha of their pixels on reference nuclei (same rule as `tta`)."""
    inter = overlap(ref, other)
    area_oth = inter.sum(axis=0)[1:].astype(float)
    on_ref = inter[1:, 1:].sum(axis=0) / np.maximum(area_oth, 1)
    keep = (on_ref < alpha) & (area_oth > 0)
    return np.flatnonzero(keep) + 1


def extra_per_tile(pred: TTAPrediction, grid: TileGrid, alpha: float = 0.2) -> np.ndarray:
    """Mean over versions t = 1..T of the number of extra nuclei whose centroid is in each tile."""
    ref = pred.masks[0]
    counts = np.zeros((len(pred.masks) - 1, grid.n_tiles))
    for t, other in enumerate(pred.masks[1:]):
        labs = extra_labels(ref, other, alpha)
        if labs.size == 0:
            continue
        cy, cx = centroids(other)
        tiles = grid.tile_of(cy[labs - 1], cx[labs - 1])
        tiles = tiles[tiles >= 0]
        counts[t] = np.bincount(tiles, minlength=grid.n_tiles)
    return counts.mean(axis=0) if len(counts) else np.zeros(grid.n_tiles)


# ---- student-only tile scores ----------------------------------------------------------------
def tile_uncertainty(
    nuclei: pd.DataFrame, pred: TTAPrediction, grid: TileGrid, alpha: float = 0.2
) -> pd.DataFrame:
    """Tile scores from the per-nucleus table of `tta.nucleus_uncertainty` (needs column `label`)."""
    cy, cx = centroids(pred.masks[0])
    lab = nuclei["label"].to_numpy()
    nuc = nuclei.assign(
        tile=grid.tile_of(cy[lab - 1], cx[lab - 1]), D_lt1=(nuclei["D"] < 1).astype(float)
    )
    agg = (
        nuc[nuc["tile"] >= 0]
        .groupby("tile")
        .agg(
            n_nuclei=("label", "size"),
            K_mean=("K", "mean"),
            U_MI_mean=("U_MI", "mean"),
            Top_mean=("Top", "mean"),
            D_mean=("D", "mean"),
            frac_D_lt1=("D_lt1", "mean"),
        )
    )
    out = grid.frame().merge(agg, left_on="tile", right_index=True, how="left")
    out["n_nuclei"] = out["n_nuclei"].fillna(0).astype(int)
    out["n_extra"] = extra_per_tile(pred, grid, alpha)
    out["E_extra"] = out["n_extra"] / np.maximum(out["n_nuclei"], 1)
    out["unstable"] = out["frac_D_lt1"].fillna(0) + out["E_extra"]
    return out


# ---- offline difficulty against the oracle ---------------------------------------------------
def tile_quality(student: np.ndarray, oracle: np.ndarray, grid: TileGrid) -> pd.DataFrame:
    """Per-tile TP/FP/FN, PQ, SQ, DQ of `student` against `oracle` (matching on the whole image)."""
    inter = overlap(oracle, student).astype(float)
    a_o, a_s = inter.sum(axis=1)[1:], inter.sum(axis=0)[1:]
    inter = inter[1:, 1:]
    union = a_o[:, None] + a_s[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    oi, si = np.nonzero(iou > 0.5)  # unique pairs for IoU > 0.5

    scy, scx = centroids(student)
    ocy, ocx = centroids(oracle)
    s_tile, o_tile = grid.tile_of(scy, scx), grid.tile_of(ocy, ocx)
    s_matched = np.zeros(len(a_s), bool)
    o_matched = np.zeros(len(a_o), bool)
    s_matched[si], o_matched[oi] = True, True

    n = grid.n_tiles

    def count(tiles, weights=None):
        ok = tiles >= 0
        w = None if weights is None else np.asarray(weights)[ok]
        return np.bincount(tiles[ok], weights=w, minlength=n)

    present_s, present_o = a_s > 0, a_o > 0
    tp = count(s_tile[si])
    sum_iou = count(s_tile[si], iou[oi, si])
    fp = count(s_tile[~s_matched & present_s])
    fn = count(o_tile[~o_matched & present_o])
    denom = tp + 0.5 * fp + 0.5 * fn
    with np.errstate(invalid="ignore", divide="ignore"):
        sq = np.where(tp > 0, sum_iou / np.maximum(tp, 1), 0.0)
        dq = np.where(denom > 0, tp / denom, np.nan)
    pq = np.where(denom > 0, sq * dq, np.nan)
    return pd.DataFrame(
        {
            "tile": np.arange(n),
            "tp": tp.astype(int),
            "fp": fp.astype(int),
            "fn": fn.astype(int),
            "sum_iou": sum_iou,
            "n_oracle": count(o_tile[present_o]).astype(int),
            "pq": pq,
            "sq": np.where(denom > 0, sq, np.nan),
            "dq": dq,
            "u": 1 - pq,
        }
    )


# ---- from tiles back to an image score -------------------------------------------------------
def image_from_tiles(
    tiles: pd.DataFrame, col: str, how: str = "max", k: int = 2, min_nuclei: int = 0
) -> float:
    """Image score from tile scores: 'mean', 'max' or 'topk' (mean of the k highest tiles)."""
    v = tiles.loc[tiles["n_nuclei"] >= min_nuclei, col].dropna().sort_values(ascending=False)
    if v.empty:
        return float("nan")
    if how == "mean":
        return float(v.mean())
    if how == "max":
        return float(v.iloc[0])
    if how == "topk":
        return float(v.iloc[:k].mean())
    raise ValueError(f"unknown aggregation {how!r}")


def pct_rank(x: pd.Series) -> pd.Series:
    """Percentile rank in (0, 1]; ties get the average rank, NaN stays NaN."""
    return x.rank(pct=True, method="average")
