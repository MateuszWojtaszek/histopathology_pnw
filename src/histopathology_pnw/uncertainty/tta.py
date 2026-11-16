"""Per-nucleus model uncertainty from test-time augmentation (TTA) with the 8 symmetries of a square.

The student segments the image and its 7 rotated/mirrored versions (group D4); each prediction is
mapped back to the original orientation, which is exact for D4 (no interpolation). Nuclei of the
original prediction are the reference; each reference nucleus is compared with its counterparts:

Notation: P_i = pixels of reference nucleus i (i = 1..N), Q_tj = pixels of nucleus j in version t
(t = 1..T, already mapped back), |X| = number of pixels.
- IoU(P_i, Q_tj) = |P_i ∩ Q_tj| / |P_i ∪ Q_tj|,  b_t(i) = max_j IoU(P_i, Q_tj),
  m_t(i) = 1 if b_t(i) > 0.5 else 0 (a match is unique for IoU > 0.5),
- detection        D(i)   = mean_t m_t(i)                         (1 = found in every version),
- contour          K(i)   = 1 - sum_t m_t b_t / sum_t m_t         (K = 1 if never matched),
- topology         Top(i) = mean_t [split_t(i) or merge_t(i)], with coverage c(i, j) = |P_i ∩ Q_tj| / |P_i|:
    split_t(i): at least two nuclei of version t cover >= alpha of P_i each,
    merge_t(i): the nucleus of version t that covers most of P_i also covers >= alpha of another
                reference nucleus,
- pixel uncertainty from the cellprob maps of all T+1 predictions, p_t = sigmoid(cellprob_t), in a
  band B_i of +-ring pixels around the boundary of P_i:
    U_TTA(i) = mean_{x in B_i} Var_t p_t(x)                       (spread between versions),
    U_H(i)   = mean_{x in B_i} H(p_0(x))                          (entropy of the original prediction),
    U_MI(i)  = mean_{x in B_i} [H(mean_t p_t(x)) - mean_t H(p_t(x))] (disagreement part of the entropy),
  with H(p) = -p log2 p - (1-p) log2 (1-p).
Image level: extra nuclei E = mean_t (#nuclei of version t with < alpha of their pixels on any
reference nucleus) / N, i.e. nuclei the original prediction probably missed.
"""

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation
from skimage.morphology import disk
from skimage.segmentation import expand_labels, find_boundaries, relabel_sequential


# ---- the 8 symmetries of a square (D4), exact and invertible ----------------------------------
@dataclass(frozen=True)
class Symmetry:
    """x -> rot90(fliplr(x) if flip else x, k) on the first two axes (rows, columns)."""

    name: str
    flip: bool
    k: int

    def forward(self, x: np.ndarray) -> np.ndarray:
        y = x[:, ::-1] if self.flip else x
        return np.ascontiguousarray(np.rot90(y, self.k, axes=(0, 1)))

    def inverse(self, y: np.ndarray) -> np.ndarray:
        x = np.rot90(y, -self.k, axes=(0, 1))
        return np.ascontiguousarray(x[:, ::-1] if self.flip else x)


D4 = (
    Symmetry("identity", False, 0),
    Symmetry("rot90", False, 1),
    Symmetry("rot180", False, 2),
    Symmetry("rot270", False, 3),
    Symmetry("flip_lr", True, 0),
    Symmetry("flip_lr_rot90", True, 1),  # mirror in a diagonal
    Symmetry("flip_ud", True, 2),
    Symmetry("flip_lr_rot270", True, 3),  # mirror in the other diagonal
)
D4_SMALL = (D4[0], D4[4], D4[6], D4[2])  # identity, mirror left-right, mirror up-down, rot180


# ---- predictions -----------------------------------------------------------------------------
@dataclass
class TTAPrediction:
    """Masks and cellprob logits of every version, all in the orientation of the original image."""

    names: list[str]
    masks: list[np.ndarray]  # int32, labels 1..N
    cellprob: list[np.ndarray]  # float32 logits

    def save(self, path) -> None:
        np.savez_compressed(path, names=np.array(self.names), masks=np.stack(self.masks),
                            cellprob=np.stack(self.cellprob))

    @classmethod
    def load(cls, path) -> "TTAPrediction":
        d = np.load(path)
        return cls(list(d["names"]), list(d["masks"]), list(d["cellprob"]))


def predict_tta(model, image: np.ndarray, symmetries: Sequence[Symmetry] = D4, batch_size: int = 8,
                eval_kwargs: dict | None = None) -> TTAPrediction:
    """Run `model.eval` (a CellposeModel) on every version of `image` (H x W x C) and map back.

    The first symmetry must be the identity: its prediction is the reference.
    """
    if symmetries[0].flip or symmetries[0].k % 4:
        raise ValueError("the first symmetry must be the identity (the reference prediction)")
    if eval_kwargs is None:
        from histopathology_pnw.inference import EVAL_KWARGS

        eval_kwargs = EVAL_KWARGS
    names, masks, probs = [], [], []
    for s in symmetries:
        m, flows, _ = model.eval(s.forward(image), batch_size=batch_size, **eval_kwargs)
        cp = np.asarray(flows[2], dtype=np.float32)  # cellprob logits (cellpose 4: flows[2])
        m = np.asarray(m)
        if cp.shape != m.shape:
            raise ValueError(f"cellprob {cp.shape} != mask {m.shape}; check `flows` of this cellpose version")
        names.append(s.name)
        masks.append(s.inverse(m).astype(np.int32))
        probs.append(s.inverse(cp))
    masks[0] = relabel_sequential(masks[0])[0].astype(np.int32)
    return TTAPrediction(names, masks, probs)


# ---- matching ---------------------------------------------------------------------------------
def overlap(ref: np.ndarray, other: np.ndarray) -> np.ndarray:
    """Pixel counts |P_i ∩ Q_j|, shape (n_ref + 1, n_other + 1); row/column 0 = background."""
    n, m = int(ref.max()), int(other.max())
    pair = ref.astype(np.int64).ravel() * (m + 1) + other.astype(np.int64).ravel()
    return np.bincount(pair, minlength=(n + 1) * (m + 1)).reshape(n + 1, m + 1)


def match_version(ref: np.ndarray, other: np.ndarray, alpha: float = 0.2) -> dict:
    """Compare every reference nucleus with one version. Arrays have one entry per reference nucleus.

    Returns best_iou (b_t), matched (m_t), split, merge, and n_extra (nuclei of `other` that lie
    outside every reference nucleus: < alpha of their pixels on reference nuclei).
    """
    inter = overlap(ref, other)
    area_ref = inter.sum(axis=1)[1:].astype(float)  # |P_i|
    area_oth = inter.sum(axis=0)[1:].astype(float)  # |Q_j|
    inter = inter[1:, 1:].astype(float)
    n, m = inter.shape
    if n == 0:
        return {"best_iou": np.zeros(0), "matched": np.zeros(0, bool), "split": np.zeros(0, bool),
                "merge": np.zeros(0, bool), "n_extra": int(m)}
    if m == 0:
        z = np.zeros(n)
        return {"best_iou": z, "matched": z.astype(bool), "split": z.astype(bool),
                "merge": z.astype(bool), "n_extra": 0}
    union = area_ref[:, None] + area_oth[None, :] - inter
    iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    best = iou.max(axis=1)
    cov = inter / np.maximum(area_ref[:, None], 1)  # share of P_i covered by Q_j
    split = (cov >= alpha).sum(axis=1) >= 2
    j_main = inter.argmax(axis=1)  # version nucleus covering most of P_i
    covers_ref = (cov >= alpha).sum(axis=0)  # how many reference nuclei each Q_j covers
    merge = (inter.max(axis=1) > 0) & (covers_ref[j_main] >= 2)
    on_ref = inter.sum(axis=0) / np.maximum(area_oth, 1)  # share of Q_j lying on reference nuclei
    return {"best_iou": best, "matched": best > 0.5, "split": split, "merge": merge,
            "n_extra": int((on_ref < alpha).sum())}


# ---- pixel uncertainty helpers ----------------------------------------------------------------
def binary_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def boundary_band(mask: np.ndarray, ring: int = 2) -> np.ndarray:
    """Label image of the band of +-ring pixels around each nucleus boundary (0 elsewhere)."""
    inner = binary_dilation(find_boundaries(mask, mode="inner"),
                            structure=disk(max(ring - 1, 0)).astype(bool)) & (mask > 0)
    outer = expand_labels(mask, ring)
    outer[mask > 0] = 0
    return np.where(inner, mask, outer)


def per_label_mean(labels: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    sel = labels > 0
    s = np.bincount(labels[sel], weights=values[sel], minlength=n + 1)[1:]
    c = np.bincount(labels[sel], minlength=n + 1)[1:]
    return np.divide(s, c, out=np.full(n, np.nan), where=c > 0)


# ---- per-nucleus uncertainty ------------------------------------------------------------------
def nucleus_uncertainty(pred: TTAPrediction, alpha: float = 0.2, ring: int = 2) -> tuple[pd.DataFrame, dict]:
    """One row per reference nucleus with D, K, Top (and its parts), U_TTA, U_H, U_MI; plus image stats."""
    ref = pred.masks[0]
    n = int(ref.max())
    versions = [match_version(ref, m, alpha) for m in pred.masks[1:]]
    T = len(versions)
    if T == 0:
        raise ValueError("need at least one transformed version besides the reference")
    best = np.stack([v["best_iou"] for v in versions]) if n else np.zeros((T, 0))
    matched = np.stack([v["matched"] for v in versions]) if n else np.zeros((T, 0), bool)
    split = np.stack([v["split"] for v in versions]) if n else np.zeros((T, 0), bool)
    merge = np.stack([v["merge"] for v in versions]) if n else np.zeros((T, 0), bool)

    n_match = matched.sum(axis=0)
    K = np.where(n_match > 0, 1 - (best * matched).sum(axis=0) / np.maximum(n_match, 1), 1.0)

    p = 1 / (1 + np.exp(-np.stack(pred.cellprob).astype(np.float64)))  # (T+1, H, W)
    var = p.var(axis=0)
    h_mean = binary_entropy(p.mean(axis=0))
    mean_h = binary_entropy(p).mean(axis=0)
    band = boundary_band(ref, ring)

    labels = np.arange(1, n + 1)
    area = np.bincount(ref.ravel(), minlength=n + 1)[1:]
    df = pd.DataFrame({
        "label": labels,
        "area": area,
        "D": matched.mean(axis=0),
        "K": K,
        "Top": (split | merge).mean(axis=0),
        "split_rate": split.mean(axis=0),
        "merge_rate": merge.mean(axis=0),
        "mean_best_iou": best.mean(axis=0),
        "U_TTA": per_label_mean(band, var, n),
        "U_H": per_label_mean(band, binary_entropy(p[0]), n),
        "U_MI": per_label_mean(band, h_mean - mean_h, n),
    })
    stats = {
        "n_nuclei": n,
        "n_versions": T,
        "versions": pred.names[1:],
        "E_extra": float(np.mean([v["n_extra"] for v in versions]) / max(n, 1)),
        "n_extra_per_version": [v["n_extra"] for v in versions],
    }
    return df, stats


def tta_uncertainty(model, image: np.ndarray, symmetries: Sequence[Symmetry] = D4, alpha: float = 0.2,
                    ring: int = 2, batch_size: int = 8, eval_kwargs: dict | None = None):
    """Predict all versions and compute per-nucleus uncertainty. Returns (df, stats, pred)."""
    t0 = time.time()
    pred = predict_tta(model, image, symmetries, batch_size, eval_kwargs)
    df, stats = nucleus_uncertainty(pred, alpha, ring)
    stats["seconds"] = round(time.time() - t0, 2)
    return df, stats, pred
