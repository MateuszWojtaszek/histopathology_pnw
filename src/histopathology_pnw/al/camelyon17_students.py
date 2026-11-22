"""A few random-AL students on CAMELYON17 tiles, for the TTA uncertainty study (not a full AL run).

Same recipe as the MoNuSeg random AL (`al/random_loop.py`): the training tiles are drawn in one
random order fixed by the seed, so labelled sets are nested; every student starts from the SAM
initialisation (cold start) and is trained on the oracle masks of the first `n` tiles with the
same `train_seg` settings. Only the rounds asked for are trained (e.g. 4 and 10 tiles), to keep
the GPU cost small. Students are evaluated on the validation tiles against the oracle masks
(there is no nucleus ground truth on CAMELYON17); test tiles are never touched.

Layout of `--out-dir` (as the MoNuSeg runs, so `uncertainty/tile_scores.py` can read it):
`results.csv` (one row per round, with `labeled_ids`) and `round_XX/models/student`.

    uv run --no-sync python -m histopathology_pnw.al.camelyon17_students --seed 0 --rounds 4 10
"""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cellpose.train import train_seg

from histopathology_pnw.al.random_loop import RandomALConfig, _free_memory
from histopathology_pnw.config import Settings
from histopathology_pnw.datasets import camelyon17_tiles as c17
from histopathology_pnw.device import get_device
from histopathology_pnw.inference import predict
from histopathology_pnw.metrics import evaluate_all, summarize
from histopathology_pnw.models.student import (
    build_student_init,
    enable_gradient_checkpointing,
    load_student,
)
from histopathology_pnw.seed import set_seed


def draw_order(ids: list[str], seed: int) -> list[str]:
    """The seed's permutation of the training tiles; round n labels the first n."""
    rng = np.random.default_rng(seed)
    return [ids[i] for i in rng.permutation(len(ids))]


def main() -> None:
    settings = Settings()
    parser = argparse.ArgumentParser(description="Random-AL students on CAMELYON17 tiles.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, nargs="+", default=[4, 10], help="labelled tiles")
    parser.add_argument("--out-dir", type=Path, help="default: <outputs>/al_random_c17/seed<k>")
    parser.add_argument("--gradient-checkpointing", action="store_true", help="for < 40 GB GPUs")
    parser.add_argument("--no-tf32", action="store_true")
    args = parser.parse_args()

    cfg = RandomALConfig(seed=args.seed, gradient_checkpointing=args.gradient_checkpointing)
    out = args.out_dir or settings.outputs_dir / "al_random_c17" / f"seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(
        json.dumps({**asdict(cfg), **vars(args)}, default=str, indent=1)
    )
    torch.backends.cuda.matmul.allow_tf32 = not args.no_tf32
    torch.backends.cudnn.allow_tf32 = not args.no_tf32
    device = get_device()

    train_ids, val_ids = c17.tile_ids("train"), c17.tile_ids("val")
    order = draw_order(train_ids, args.seed)
    needed = order[: max(args.rounds)]
    t0 = time.time()
    labels = dict(zip(needed, c17.oracle_tile_masks(needed), strict=True))
    val_oracle = c17.oracle_tile_masks(val_ids)
    print(
        f"oracle masks: {len(needed)} train + {len(val_ids)} val tiles ({time.time() - t0:.0f} s)"
    )
    images = {t: c17.read_tile(t) for t in needed}
    val_images = [c17.read_tile(t) for t in val_ids]

    init_path = out / "student_init.pth"
    if not init_path.is_file():
        build_student_init(args.seed, init_path)

    results_path = out / "results.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    done = set(results["n_labeled"]) if len(results) else set()
    for n in sorted(args.rounds):
        if n in done:
            print(f"round n_labeled={n}: already done, skipping")
            continue
        labeled = order[:n]
        round_dir = out / f"round_{n:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed * 1000 + n)

        t0 = time.time()
        student = load_student(init_path, device)
        if cfg.gradient_checkpointing:
            enable_gradient_checkpointing(student.net)
        checkpoint, losses, _ = train_seg(
            student.net,
            train_data=[images[t] for t in labeled],
            train_labels=[labels[t] for t in labeled],
            channel_axis=-1,
            n_epochs=cfg.n_epochs,
            nimg_per_epoch=cfg.nimg_per_epoch,
            batch_size=cfg.batch_size,
            learning_rate=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            min_train_masks=1,
            save_path=round_dir,
            model_name="student",
        )
        train_time = time.time() - t0

        preds = predict(student, val_images, progress=False)
        per_tile = evaluate_all(val_oracle, preds, val_ids)
        per_tile.to_csv(round_dir / "val_metrics_oracle.csv", index=False)
        summary = summarize(per_tile)
        row = {
            "n_labeled": n,
            "labeled_ids": json.dumps(labeled),
            **{f"val_oracle_{k}": v for k, v in summary.items()},
            "final_train_loss": float(losses[-1]),
            "train_time_s": round(train_time, 1),
            "checkpoint": str(checkpoint),
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        print(
            f"round n_labeled={n}: val PQ vs oracle {summary['pq']:.4f} ({train_time / 60:.1f} min)"
        )
        del student
        _free_memory()


if __name__ == "__main__":
    main()
