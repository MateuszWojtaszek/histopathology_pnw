"""TTA uncertainty of saved AL students on the images they had not been trained on.

For one AL run (selector, mode, seed) and every round whose checkpoint is kept, the student of
that round segments each unlabelled pool image and each validation image in the 8 versions of
`uncertainty/tta.py`. Written next to the round's `selection.csv`:
- `tta_nuclei.csv`: one row per nucleus of the student's prediction with the TTA measures,
- `tta_images.csv`: one row per image (number of nuclei, extra nuclei, seconds).

`iou_oracle` (best IoU of the nucleus with an oracle nucleus) and `pq_oracle` are added for
OFFLINE analysis only: they say whether the measures point at badly segmented nuclei and images.
A selector must not read them. Rounds that already have both files are skipped (resume).

    uv run python -m histopathology_pnw.uncertainty.pool_scores --selector oracle_peek --seed 0
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from histopathology_pnw.config import Settings
from histopathology_pnw.datasets import monuseg
from histopathology_pnw.device import get_device
from histopathology_pnw.metrics import iou_matrix, panoptic_quality
from histopathology_pnw.models.oracle import oracle_masks
from histopathology_pnw.models.student import load_student
from histopathology_pnw.uncertainty import tta

SYMMETRIES = {"D4": tta.D4, "D4_SMALL": tta.D4_SMALL}


def score_images(
    model,
    ids: list[str],
    images: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    part: dict[str, str],
    symmetries=tta.D4,
    eval_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """TTA measures for `ids`: (one row per nucleus, one row per image)."""
    nuclei, per_image = [], []
    for image_id in ids:
        t0 = time.time()
        pred = tta.predict_tta(model, images[image_id], symmetries, eval_kwargs=eval_kwargs)
        df, stats = tta.nucleus_uncertainty(pred)
        ref = pred.masks[0]
        iou = iou_matrix(oracle[image_id], ref)  # oracle nuclei x student nuclei
        df["iou_oracle"] = iou.max(axis=0) if iou.shape[0] else 0.0
        nuclei.append(df.assign(image_id=image_id, part=part[image_id]))
        pq = panoptic_quality(oracle[image_id], ref)
        per_image.append(
            {
                "image_id": image_id,
                "part": part[image_id],
                "n_nuclei": stats["n_nuclei"],
                "n_oracle": int(oracle[image_id].max()),
                "E_extra": stats["E_extra"],
                "pq_oracle": pq["pq"],
                "seconds": round(time.time() - t0, 2),
            }
        )
    return pd.concat(nuclei, ignore_index=True), pd.DataFrame(per_image)


def main() -> None:
    parser = argparse.ArgumentParser(description="TTA uncertainty of saved AL students.")
    parser.add_argument("--selector", default="oracle_peek")
    parser.add_argument("--mode", default="full")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, nargs="*", help="default: every kept checkpoint")
    parser.add_argument("--symmetries", choices=tuple(SYMMETRIES), default="D4")
    parser.add_argument("--max-images", type=int, help="first N images only (quick check)")
    args = parser.parse_args()

    run_dir = Settings().outputs_dir / f"al_{args.selector}" / args.mode / f"seed{args.seed}"
    results = pd.read_csv(run_dir / "results.csv")
    split = monuseg.load_train_val_split()
    by_id = monuseg.samples_by_id("train")
    everything = split["train"] + split["val"]
    images = {i: monuseg.read_image(by_id[i].image_path) for i in everything}
    oracle = dict(zip(everything, oracle_masks([by_id[i] for i in everything]), strict=True))
    device = get_device()
    print(f"device: {device} | run: {run_dir} | versions: {args.symmetries}")

    for row in results.itertuples():
        n = int(row.n_labeled)
        round_dir = run_dir / f"round_{n:02d}"
        checkpoint = round_dir / "models" / "student"
        suffix = "" if args.max_images is None else f"_first{args.max_images}"
        out_nuclei = round_dir / f"tta_nuclei{suffix}.csv"
        out_images = round_dir / f"tta_images{suffix}.csv"
        if args.rounds and n not in args.rounds:
            continue
        if not checkpoint.is_file():
            print(f"round n_labeled={n}: no checkpoint, skipping")
            continue
        if out_nuclei.is_file() and out_images.is_file():
            print(f"round n_labeled={n}: already done, skipping")
            continue
        labeled = set(json.loads(row.labeled_ids))
        part = {i: "pool" for i in split["train"] if i not in labeled}
        part |= {i: "val" for i in split["val"]}
        ids = list(part)[: args.max_images]

        t0 = time.time()
        student = load_student(checkpoint, device)
        nuclei, per_image = score_images(
            student, ids, images, oracle, part, SYMMETRIES[args.symmetries]
        )
        _write(nuclei.assign(n_labeled=n, seed=args.seed), out_nuclei)
        _write(per_image.assign(n_labeled=n, seed=args.seed), out_images)
        print(
            f"round n_labeled={n}: {len(ids)} images, {len(nuclei)} nuclei, "
            f"{per_image['seconds'].mean():.1f} s per image ({(time.time() - t0) / 60:.1f} min)"
        )
        del student


def _write(df: pd.DataFrame, path: Path) -> None:
    partial = path.with_suffix(".partial")  # a killed job never leaves a half-written table
    df.to_csv(partial, index=False)
    partial.replace(path)


if __name__ == "__main__":
    main()
