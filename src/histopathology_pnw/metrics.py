"""Instance segmentation metrics, one evaluation function for every model.

Panoptic Quality (Kirillov et al., CVPR 2019) for a single class:
- a predicted and a ground-truth instance match when IoU > threshold; for threshold >= 0.5 every
  instance has at most one match, so no assignment step is needed,
- SQ = mean IoU of matched pairs, DQ = TP / (TP + FP/2 + FN/2) (this is F1 at that IoU),
- PQ = SQ * DQ. The standard PQ uses threshold 0.5, i.e. PQ = PQ@0.5.

Also reported: AJI (Kumar et al., 2017) and AP@0.5 = TP / (TP + FP + FN) as in Cellpose,
both computed with `cellpose.metrics`.
"""

import numpy as np
import pandas as pd
from cellpose.metrics import aggregated_jaccard_index, average_precision


def iou_matrix(gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """IoU between every ground-truth (rows) and predicted (columns) instance, background dropped."""
    n_gt, n_pred = int(gt.max()), int(pred.max())
    pair = gt.astype(np.int64).ravel() * (n_pred + 1) + pred.astype(np.int64).ravel()
    inter = np.bincount(pair, minlength=(n_gt + 1) * (n_pred + 1)).reshape(n_gt + 1, n_pred + 1)
    area_gt = inter.sum(axis=1, keepdims=True)
    area_pred = inter.sum(axis=0, keepdims=True)
    union = area_gt + area_pred - inter
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, inter / union, 0.0)
    return iou[1:, 1:]


def panoptic_quality(gt: np.ndarray, pred: np.ndarray, iou_threshold: float = 0.5) -> dict:
    """PQ, SQ, DQ and the counts needed to pool them over a dataset."""
    if iou_threshold < 0.5:
        raise ValueError("matching is unique only for iou_threshold >= 0.5")
    iou = iou_matrix(gt, pred)
    n_gt, n_pred = iou.shape
    matched = iou[iou > iou_threshold]
    tp = int(matched.size)
    fp, fn = n_pred - tp, n_gt - tp
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "sum_iou": float(matched.sum()),
        **_pq(tp, fp, fn, float(matched.sum())),
    }


def _pq(tp: int, fp: int, fn: int, sum_iou: float) -> dict:
    denom = tp + 0.5 * fp + 0.5 * fn
    if denom == 0:  # no instances on either side: perfect agreement
        return {"pq": 1.0, "sq": 1.0, "dq": 1.0}
    sq = sum_iou / tp if tp else 0.0
    dq = tp / denom
    return {"pq": sq * dq, "sq": sq, "dq": dq}


def evaluate(gt: np.ndarray, pred: np.ndarray) -> dict:
    """All metrics for one image."""
    pq50 = panoptic_quality(gt, pred, 0.5)
    pq75 = panoptic_quality(gt, pred, 0.75)
    ap, _, _, _ = average_precision([gt], [pred], threshold=[0.5])
    return {
        "pq": pq50["pq"],
        "sq": pq50["sq"],
        "dq": pq50["dq"],  # = F1@0.5
        "pq@0.75": pq75["pq"],
        "aji": float(aggregated_jaccard_index([gt], [pred])[0]),
        "ap@0.5": float(ap[0, 0]),
        "n_gt": int(gt.max()),
        "n_pred": int(pred.max()),
        "tp": pq50["tp"],
        "fp": pq50["fp"],
        "fn": pq50["fn"],
        "sum_iou": pq50["sum_iou"],
    }


def evaluate_all(gts: list[np.ndarray], preds: list[np.ndarray], ids: list[str]) -> pd.DataFrame:
    """One row per image."""
    rows = [{"image_id": i, **evaluate(g, p)} for i, g, p in zip(ids, gts, preds, strict=True)]
    return pd.DataFrame(rows)


def summarize(per_image: pd.DataFrame) -> dict:
    """Mean over images (the usual way MoNuSeg results are reported) plus pooled PQ from summed
    TP/FP/FN over the whole set (`pq_pooled`), which weights images by their number of nuclei."""
    means = per_image[["pq", "sq", "dq", "pq@0.75", "aji", "ap@0.5"]].mean().to_dict()
    pooled = _pq(
        int(per_image["tp"].sum()),
        int(per_image["fp"].sum()),
        int(per_image["fn"].sum()),
        float(per_image["sum_iou"].sum()),
    )
    return {**means, "pq_pooled": pooled["pq"], "n_images": len(per_image)}
