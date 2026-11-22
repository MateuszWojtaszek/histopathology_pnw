"""Tile-level TTA uncertainty of the saved students of an AL run (inference only).

For every seed and every round that still has its checkpoint, the student of that round
segments each unlabelled pool image and each validation image in the D4 versions of
`uncertainty/tta.py`. The results are split into tiles (`uncertainty/patches.py`) and written,
one folder per round, to `<out-dir>/seed<k>/round_XX/`:
- `tiles.csv`: one row per tile with the student-only scores and, for OFFLINE analysis only,
  the tile PQ of the student against the oracle (`pq`, `u` = 1 - PQ, tp/fp/fn),
- `images.csv`: one row per image with the image-level scores (mean over nuclei, as the
  `tta_kd` selector) and the image PQ against the oracle,
- `nuclei.csv`: one row per nucleus with the TTA measures, its tile and its best IoU with an
  oracle nucleus (offline).

The oracle columns say whether the scores point at badly segmented tiles; a selector must not
read them. The checkpoints are only read, so `--runs-dir` may belong to someone else; outputs go
to `--out-dir`. Rounds that already have all three files are skipped (resume).

MoNuSeg images and CAMELYON17 tiles (`--dataset camelyon17`, 1000 px like MoNuSeg) are handled
the same way; the CAMELYON17 students come from `al/camelyon17_students.py`.

    uv run --no-sync python -m histopathology_pnw.uncertainty.tile_scores --seeds 0 --rounds 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from histopathology_pnw.config import Settings
from histopathology_pnw.datasets import camelyon17_tiles as c17
from histopathology_pnw.datasets import monuseg
from histopathology_pnw.device import get_device
from histopathology_pnw.metrics import iou_matrix, panoptic_quality
from histopathology_pnw.models.oracle import oracle_masks
from histopathology_pnw.models.student import load_student
from histopathology_pnw.uncertainty import patches, tta

SYMMETRIES = {"D4": tta.D4, "D4_SMALL": tta.D4_SMALL}
FILES = ("tiles.csv", "images.csv", "nuclei.csv")


def score_image(
    pred: tta.TTAPrediction,
    oracle: np.ndarray,
    tile: int = 250,
    alpha: float = 0.2,
    ring: int = 2,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """Tile table, image row and nucleus table of one image from its TTA prediction."""
    ref = pred.masks[0]
    nuclei, stats = tta.nucleus_uncertainty(pred, alpha, ring)
    grid = patches.TileGrid(*ref.shape[:2], tile)
    tiles = patches.tile_uncertainty(nuclei, pred, grid, alpha).merge(
        patches.tile_quality(ref, oracle, grid), on="tile"
    )
    cy, cx = patches.centroids(ref)
    labels = nuclei["label"].to_numpy()
    iou = iou_matrix(oracle, ref)  # oracle nuclei x student nuclei
    nuclei = nuclei.assign(
        tile=grid.tile_of(cy[labels - 1], cx[labels - 1]),
        iou_oracle=iou.max(axis=0) if iou.shape[0] else 0.0,
    )
    pq = panoptic_quality(oracle, ref)
    n = len(nuclei)
    # an image without nuclei gets the highest instability, as in the `tta_kd` selector
    low_d = float((nuclei["D"] < 1).mean()) if n else 1.0
    image = {
        "n_nuclei": n,
        "n_oracle": int(oracle.max()),
        "K_mean": float(nuclei["K"].mean()) if n else 1.0,
        "U_MI_mean": float(nuclei["U_MI"].mean()) if n else np.nan,
        "Top_mean": float(nuclei["Top"].mean()) if n else np.nan,
        "low_D": low_d,
        "E_extra": stats["E_extra"],
        "unstable": low_d + stats["E_extra"],
        "pq": pq["pq"],
        "u": 1.0 - pq["pq"],
    }
    return tiles, image, nuclei


def score_round(
    model,
    ids: list[str],
    images: dict[str, np.ndarray],
    oracle: dict[str, np.ndarray],
    part: dict[str, str],
    symmetries=tta.D4,
    tile: int = 250,
    alpha: float = 0.2,
    ring: int = 2,
    eval_kwargs: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """TTA + tile split for `ids`: (tiles, images, nuclei), each with `image_id` and `part`."""
    tiles, rows, nuclei = [], [], []
    for image_id in ids:
        t0 = time.time()
        pred = tta.predict_tta(model, images[image_id], symmetries, eval_kwargs=eval_kwargs)
        t, row, nuc = score_image(pred, oracle[image_id], tile, alpha, ring)
        key = {"image_id": image_id, "part": part[image_id]}
        tiles.append(t.assign(**key))
        nuclei.append(nuc.assign(**key))
        rows.append({**key, **row, "seconds": round(time.time() - t0, 2)})
    return (
        pd.concat(tiles, ignore_index=True),
        pd.DataFrame(rows),
        pd.concat(nuclei, ignore_index=True),
    )


def load_dataset(name: str) -> tuple[dict, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """({"train": ids, "val": ids}, images, oracle masks) of the training and validation images.

    The test images are never loaded.
    """
    if name == "monuseg":
        split = monuseg.load_train_val_split()
        by_id = monuseg.samples_by_id("train")
        ids = split["train"] + split["val"]
        images = {i: monuseg.read_image(by_id[i].image_path) for i in ids}
        masks = oracle_masks([by_id[i] for i in ids])
    elif name == "camelyon17":
        split = {"train": c17.tile_ids("train"), "val": c17.tile_ids("val")}
        ids = split["train"] + split["val"]
        images = {i: c17.read_tile(i) for i in ids}
        masks = c17.oracle_tile_masks(ids)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    return split, images, dict(zip(ids, masks, strict=True))


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Tile-level TTA uncertainty of saved AL students.")
    parser.add_argument(
        "--runs-dir",
        type=Path,
        help="folder with seed<k>/results.csv and seed<k>/round_XX/models/student (read only); "
        "default: <outputs>/al_random/full (MoNuSeg) or <outputs>/al_random_c17 (CAMELYON17)",
    )
    parser.add_argument(
        "--out-dir", type=Path, help="default: <outputs>/tile_scores/<runs folder name>"
    )
    parser.add_argument("--dataset", choices=("monuseg", "camelyon17"), default="monuseg")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--rounds", type=int, nargs="*", help="n_labeled; default: every checkpoint"
    )
    parser.add_argument("--symmetries", choices=tuple(SYMMETRIES), default="D4")
    parser.add_argument("--tile", type=int, default=250)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--ring", type=int, default=2)
    parser.add_argument("--max-images", type=int, help="first N images only (quick check)")
    args = parser.parse_args()

    runs = {"monuseg": "al_random/full", "camelyon17": "al_random_c17"}[args.dataset]
    args.runs_dir = args.runs_dir or settings.outputs_dir / runs
    args.out_dir = args.out_dir or settings.outputs_dir / "tile_scores" / runs.replace("/", "_")
    split, images, oracle = load_dataset(args.dataset)
    device = get_device()
    print(f"device: {device} | runs: {args.runs_dir} | out: {args.out_dir} | {args.symmetries}")

    for seed in args.seeds:
        results = pd.read_csv(args.runs_dir / f"seed{seed}" / "results.csv")
        for row in results.itertuples():
            n = int(row.n_labeled)
            if args.rounds and n not in args.rounds:
                continue
            checkpoint = args.runs_dir / f"seed{seed}" / f"round_{n:02d}" / "models" / "student"
            suffix = "" if args.max_images is None else f"_first{args.max_images}"
            out = args.out_dir / f"seed{seed}" / f"round_{n:02d}{suffix}"
            if not checkpoint.is_file():
                print(f"seed {seed} round {n}: no checkpoint, skipping")
                continue
            if all((out / f).is_file() for f in FILES):
                print(f"seed {seed} round {n}: already done, skipping")
                continue
            labeled = set(json.loads(row.labeled_ids))
            part = {i: "pool" for i in split["train"] if i not in labeled}
            part |= {i: "val" for i in split["val"]}
            ids = list(part)[: args.max_images]

            t0 = time.time()
            student = load_student(checkpoint, device)
            tables = score_round(
                student,
                ids,
                images,
                oracle,
                part,
                SYMMETRIES[args.symmetries],
                args.tile,
                args.alpha,
                args.ring,
            )
            info = {"seed": seed, "n_labeled": n}
            for name, table in zip(FILES, tables, strict=True):
                _write(table.assign(**info), out / name)
            (out / "info.json").write_text(
                json.dumps({**info, **vars(args)}, default=str, indent=1)
            )
            print(
                f"seed {seed} round {n}: {len(ids)} images, {len(tables[0])} tiles, "
                f"{tables[1]['seconds'].mean():.1f} s per image ({(time.time() - t0) / 60:.1f} min)"
            )
            del student


def _write(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(".partial")  # a killed job never leaves a half-written table
    df.to_csv(partial, index=False)
    partial.replace(path)


if __name__ == "__main__":
    main()
