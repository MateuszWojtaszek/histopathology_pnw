"""Active learning loop with random image selection (task I3).

Simulation: the oracle annotates the images that are queried, so the student is trained on
oracle masks; MoNuSeg ground truth is used only to evaluate. Each round:
1. the labelled set grows by `step` images drawn at random (the draw order is one permutation
   fixed by the seed, so labelled sets are nested),
2. a fresh student is built from the SAM initialisation (cold start, the same for every
   selector) and trained on the labelled images,
3. the student is evaluated on the validation images against ground truth and against the
   oracle masks (student-oracle agreement: the only measure available on datasets without GT,
   so it is the one comparable across datasets).
The loop stops when the student reaches the oracle's PQ on validation or the pool runs out.
Each finished round is appended to `results.csv`; rerunning skips finished rounds (resume).
"""

import gc
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cellpose.train import train_seg

from histopathology_pnw.datasets.monuseg import Sample, read_image, read_mask
from histopathology_pnw.device import get_device
from histopathology_pnw.inference import predict
from histopathology_pnw.metrics import evaluate_all, summarize
from histopathology_pnw.models.oracle import oracle_masks
from histopathology_pnw.models.student import (
    build_student_init,
    enable_gradient_checkpointing,
    load_student,
)
from histopathology_pnw.seed import set_seed


@dataclass
class RandomALConfig:
    seed: int = 0
    initial: int = 2  # labelled images in the first round
    step: int = 2  # images added per round
    max_labeled: int | None = None  # None: the whole pool
    n_epochs: int = 200
    # random 256 px crops per epoch; fixed, so every round gets the same number of steps
    nimg_per_epoch: int = 64
    batch_size: int = 4
    # recompute encoder activations in backward: needed to fit batch 4 in 16 GB VRAM (float32)
    gradient_checkpointing: bool = True
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    stop_at_oracle: bool = True


def round_sizes(cfg: RandomALConfig, pool_size: int) -> list[int]:
    last = min(cfg.max_labeled or pool_size, pool_size)
    sizes = list(range(cfg.initial, last + 1, cfg.step))
    if not sizes or sizes[-1] != last:
        sizes.append(last)
    return sizes


def _free_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif torch.backends.mps.is_available():
        torch.mps.empty_cache()


def run_random_al(
    cfg: RandomALConfig,
    pool: list[Sample],
    val: list[Sample],
    out_dir: Path,
    oracle_val_pq: float | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run (or resume) one random-selection AL experiment; return one row per round."""
    device = device or get_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    results_path = out_dir / "results.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    done = set(results["n_labeled"]) if len(results) else set()

    rng = np.random.default_rng(cfg.seed)
    order = [pool[i] for i in rng.permutation(len(pool))]

    init_path = out_dir / "student_init.pth"
    if not init_path.is_file():
        build_student_init(cfg.seed, init_path)

    # oracle "annotations" for the whole pool, computed once and cached
    labels = dict(zip([s.image_id for s in pool], oracle_masks(pool), strict=True))
    images = {s.image_id: read_image(s.image_path) for s in pool}
    val_images = [read_image(s.image_path) for s in val]
    val_gt = [read_mask(s) for s in val]
    # oracle masks for val (cached): reference for the student-oracle agreement
    val_oracle = oracle_masks(val)
    val_ids = [s.image_id for s in val]

    reached = cfg.stop_at_oracle and oracle_val_pq is not None
    if reached and len(results) and results["val_pq"].max() >= oracle_val_pq:
        print("student already reached the oracle in a previous run; nothing to do")
        return results

    for n in round_sizes(cfg, len(pool)):
        if n in done:
            print(f"round n_labeled={n}: already done, skipping")
            continue
        labeled = [s.image_id for s in order[:n]]
        round_dir = out_dir / f"round_{n:02d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed * 1000 + n)

        t0 = time.time()
        student = load_student(init_path, device)
        if cfg.gradient_checkpointing:
            enable_gradient_checkpointing(student.net)
        checkpoint, train_losses, _ = train_seg(
            student.net,
            train_data=[images[i] for i in labeled],
            train_labels=[labels[i] for i in labeled],
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

        preds = predict(student, val_images)
        per_image = evaluate_all(val_gt, preds, val_ids)
        per_image.to_csv(round_dir / "val_metrics.csv", index=False)
        summary = summarize(per_image)
        per_image_oracle = evaluate_all(val_oracle, preds, val_ids)
        per_image_oracle.to_csv(round_dir / "val_metrics_oracle.csv", index=False)
        summary_oracle = summarize(per_image_oracle)

        row = {
            "n_labeled": n,
            "labeled_ids": json.dumps(labeled),
            **{f"val_{k}": v for k, v in summary.items()},
            **{f"val_oracle_{k}": v for k, v in summary_oracle.items()},
            "final_train_loss": float(train_losses[-1]),
            "train_time_s": round(train_time, 1),
            "checkpoint": str(checkpoint),
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        print(
            f"round n_labeled={n}: val PQ={summary['pq']:.4f}, "
            f"vs oracle PQ={summary_oracle['pq']:.4f} ({train_time / 60:.1f} min)"
        )

        del student
        _free_memory()
        if cfg.stop_at_oracle and oracle_val_pq is not None and summary["pq"] >= oracle_val_pq:
            print(f"student reached the oracle (PQ {oracle_val_pq:.4f}); stopping")
            break

    return results
