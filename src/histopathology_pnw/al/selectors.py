"""Selectors: which unlabelled images are queried next in the AL loop.

A selector scores every unlabelled image (higher = queried earlier) and returns one row per image;
`pick` then takes the `k` best. The whole table is saved each round, not only the chosen images,
so the scores can be analysed afterwards.

- "random": the score is the position in the seed's permutation of the pool, so the labelled sets
  are the nested sets `order[:n]` of the random baseline.
- "oracle_peek": a DIAGNOSTIC, not a method. It reads the oracle masks of the unlabelled pool,
  which a real selector never sees, and queries the images on which the current student agrees
  least with the oracle (lowest PQ). It bounds what any "query the hardest images" selector can
  gain over random.
- "tta_kd": model uncertainty from test-time augmentation (`uncertainty/tta.py`), no labels. Per
  image: `K_mean` (mean contour instability of its nuclei) and `unstable` (share of nuclei not
  found in every version, plus extra nuclei that appear only in other versions). The score is
  the mean of the two percentile ranks within the unlabelled pool.
"""

import numpy as np
import pandas as pd
from cellpose.models import CellposeModel

from histopathology_pnw.inference import predict
from histopathology_pnw.metrics import panoptic_quality
from histopathology_pnw.uncertainty import tta

SELECTORS = ("random", "oracle_peek", "tta_kd")


def random_scores(unlabeled: list[str], rank: dict[str, int]) -> pd.DataFrame:
    """Earlier in the seed's permutation = higher score."""
    return pd.DataFrame({"image_id": unlabeled, "score": [-float(rank[i]) for i in unlabeled]})


def peek_table(
    unlabeled: list[str], preds: list[np.ndarray], oracle: list[np.ndarray]
) -> pd.DataFrame:
    """Score = 1 - PQ(student mask vs oracle mask) per image, with the counts behind it."""
    rows = []
    for image_id, pred, ref in zip(unlabeled, preds, oracle, strict=True):
        pq = panoptic_quality(ref, pred)
        rows.append(
            {
                "image_id": image_id,
                "score": 1.0 - pq["pq"],
                "pq_oracle": pq["pq"],
                "sq_oracle": pq["sq"],
                "dq_oracle": pq["dq"],
                "n_oracle": int(ref.max()),
                "n_pred": int(pred.max()),
                "tp": pq["tp"],
                "fp": pq["fp"],
                "fn": pq["fn"],
            }
        )
    return pd.DataFrame(rows)


def oracle_peek_scores(
    student: CellposeModel,
    unlabeled: list[str],
    images: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Segment the unlabelled images with the current student and compare with the oracle."""
    preds = predict(student, [images[i] for i in unlabeled], progress=False)
    return peek_table(unlabeled, preds, [labels[i] for i in unlabeled])


def tta_image_row(nuclei: pd.DataFrame, stats: dict) -> dict:
    """Image-level TTA measures from the per-nucleus table of `tta.nucleus_uncertainty`.

    An image on which the student finds no nucleus gets the highest instability: nothing was
    segmented, so nothing is known to be stable.
    """
    if len(nuclei) == 0:
        return {"n_pred": 0, "K_mean": 1.0, "low_D": 1.0, "E_extra": stats["E_extra"]}
    return {
        "n_pred": len(nuclei),
        "K_mean": float(nuclei["K"].mean()),
        "low_D": float((nuclei["D"] < 1).mean()),
        "E_extra": stats["E_extra"],
    }


def combine_tta(table: pd.DataFrame) -> pd.DataFrame:
    """Add `unstable` = low_D + E_extra and `score` = mean percentile rank of K_mean and unstable."""
    table = table.assign(unstable=table["low_D"] + table["E_extra"])
    ranks = table[["K_mean", "unstable"]].rank(pct=True)
    return table.assign(
        rank_K=ranks["K_mean"], rank_unstable=ranks["unstable"], score=ranks.mean(axis=1)
    )


def tta_kd_scores(
    student: CellposeModel,
    unlabeled: list[str],
    images: dict[str, np.ndarray],
    labels: dict[str, np.ndarray] | None = None,
    eval_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Score = mean rank of contour instability and detection instability under D4 TTA.

    `labels` (oracle masks) are NOT used for the score. When given, the student's PQ against
    them is stored as `pq_oracle_offline`, for analysis after the run only.
    """
    rows = []
    for image_id in unlabeled:
        pred = tta.predict_tta(student, images[image_id], eval_kwargs=eval_kwargs)
        nuclei, stats = tta.nucleus_uncertainty(pred)
        row = {"image_id": image_id, **tta_image_row(nuclei, stats)}
        if labels is not None:
            row["pq_oracle_offline"] = panoptic_quality(labels[image_id], pred.masks[0])["pq"]
        rows.append(row)
    return combine_tta(pd.DataFrame(rows))


def pick(scores: pd.DataFrame, k: int, rank: dict[str, int]) -> tuple[list[str], pd.DataFrame]:
    """The `k` highest scores; ties go to the image earlier in the seed's permutation.

    Returns the chosen ids (best first) and `scores` with `order_rank` and `selected` columns.
    """
    table = scores.assign(order_rank=[rank[i] for i in scores["image_id"]])
    best = table.sort_values(["score", "order_rank"], ascending=[False, True])
    chosen = best["image_id"].head(k).tolist()
    return chosen, table.assign(selected=table["image_id"].isin(chosen))
