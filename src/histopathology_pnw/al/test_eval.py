"""Test-set evaluation of saved AL students, for the final learning curves only.

The MoNuSeg test set is used here and nowhere else: never for training, model selection,
hyperparameters or AL selection. Every student of a run (image AL or tile AL) segments the whole
test images and is scored twice against the ground truth:
(a) per image, exactly as before (`metrics.evaluate_all` / `summarize`),
(b) per 256 px tile of the test images, same grid as the tile AL (`tile_metrics`): mean over
    tiles of the crop metrics and of the owned PQ.
The oracle is scored the same way once (`oracle.csv`), as the ceiling.

Reads `<runs-dir>/seed<k>/results.csv` and `round_XX[X]/models/student` (read only); writes
`<out-dir>/seed<k>/test_metrics.csv` (one row per round) and, per round, the per-image and
per-tile tables. Finished rounds are skipped when rerun.

    uv run --no-sync python -m histopathology_pnw.al.test_eval --runs-dir outputs/al_random_tiles/full --seeds 0 1 2
    uv run --no-sync python -m histopathology_pnw.al.test_eval --runs-dir outputs/al_random/full --seeds 0 1 2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from histopathology_pnw.config import Settings
from histopathology_pnw.datasets import monuseg
from histopathology_pnw.device import get_device
from histopathology_pnw.inference import predict
from histopathology_pnw.metrics import evaluate_all, summarize
from histopathology_pnw.models.oracle import oracle_masks
from histopathology_pnw.models.student import load_student
from histopathology_pnw.tile_metrics import evaluate_tiles, summarize_tiles
from histopathology_pnw.uncertainty.patches import TileGrid


def score(
    gts: list[np.ndarray], preds: list[np.ndarray], ids: list[str], tile: int, stride: int
) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    """Summary row, per-image table and per-tile table of predictions against GT."""
    per_image = evaluate_all(gts, preds, ids)
    per_tile = pd.concat(
        [
            evaluate_tiles(g, p, TileGrid(*g.shape, size=tile, stride=stride)).assign(image_id=i)
            for g, p, i in zip(gts, preds, ids, strict=True)
        ],
        ignore_index=True,
    )
    return {**summarize(per_image), **summarize_tiles(per_tile)}, per_image, per_tile


def round_dir(seed_dir: Path, n: int) -> Path:
    """Image AL names rounds round_XX, tile AL round_XXX."""
    for name in (f"round_{n:03d}", f"round_{n:02d}"):
        if (seed_dir / name / "models" / "student").is_file():
            return seed_dir / name
    raise FileNotFoundError(f"no checkpoint for round {n} in {seed_dir}")


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Test-set curves of saved AL students.")
    parser.add_argument("--runs-dir", type=Path, required=True, help="folder with seed<k>/")
    parser.add_argument("--out-dir", type=Path, help="default: <outputs>/test_eval/<runs-dir name>")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--rounds", type=int, nargs="+", help="n_labeled values; default: all")
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--stride", type=int, default=248)
    args = parser.parse_args()
    runs = args.runs_dir.resolve()
    try:
        name = str(runs.relative_to(settings.outputs_dir.resolve())).replace("\\", "_")
    except ValueError:
        name = runs.name
    out_dir = args.out_dir or settings.outputs_dir / "test_eval" / name.replace("/", "_")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    print(f"device: {device} | runs: {runs} | out: {out_dir}")

    test = monuseg.list_split("test")
    ids = [s.image_id for s in test]
    images = [monuseg.read_image(s.image_path) for s in test]
    gts = [monuseg.read_mask(s) for s in test]

    oracle_path = out_dir / "oracle.csv"
    if not oracle_path.is_file():
        row, _, _ = score(gts, oracle_masks(test), ids, args.tile, args.stride)
        pd.DataFrame([row]).to_csv(oracle_path, index=False)

    for seed in args.seeds:
        seed_dir = runs / f"seed{seed}"
        results = pd.read_csv(seed_dir / "results.csv")
        rounds = [n for n in results["n_labeled"] if args.rounds is None or n in args.rounds]
        out_seed = out_dir / f"seed{seed}"
        out_seed.mkdir(parents=True, exist_ok=True)
        table_path = out_seed / "test_metrics.csv"
        table = pd.read_csv(table_path) if table_path.is_file() else pd.DataFrame()
        done = set(table["n_labeled"]) if len(table) else set()
        for n in rounds:
            if n in done:
                continue
            rdir = round_dir(seed_dir, n)
            student = load_student(rdir / "models" / "student", device)
            row, per_image, per_tile = score(
                gts, predict(student, images, progress=False), ids, args.tile, args.stride
            )
            del student
            per_image.to_csv(out_seed / f"{rdir.name}_test_images.csv", index=False)
            per_tile.to_csv(out_seed / f"{rdir.name}_test_tiles.csv", index=False)
            table = pd.concat([table, pd.DataFrame([{"n_labeled": n, **row}])], ignore_index=True)
            table.sort_values("n_labeled").to_csv(table_path, index=False)
            print(
                f"seed {seed} n_labeled={n}: test PQ {row['pq']:.4f}, "
                f"tile PQ {row['tile_pq']:.4f}, AJI {row['aji']:.4f}"
            )


if __name__ == "__main__":
    main()
