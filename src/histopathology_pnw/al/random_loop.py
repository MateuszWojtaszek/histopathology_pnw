"""Active learning loop: random image selection (task I3) and other selectors.

Simulation: the oracle annotates the images that are queried, so the student is trained on
oracle masks; MoNuSeg ground truth is used only to evaluate. Each round:
1. a fresh student is built from the SAM initialisation (cold start, the same for every
   selector) and trained on the labelled images,
2. the student is evaluated on the validation images against ground truth and against the
   oracle masks (student-oracle agreement: the only measure available on datasets without GT,
   so it is the one comparable across datasets),
3. the selector (`al/selectors.py`) scores the unlabelled images and the `step` best join the
   labelled set of the next round.
Every selector starts from the same first `initial` images of the seed's permutation of the
pool. With the random selector the labelled sets are the nested sets `order[:n]`.
The loop stops when the student reaches the oracle's PQ on validation or the pool runs out.
Each finished round is appended to `results.csv`; rerunning skips finished rounds (resume).

Command line (one seed; on WCSS `slurm/monuseg_random_al.sbatch` runs one seed per array task):
    uv run python -m histopathology_pnw.al.random_loop --mode full --seed 0
    uv run python -m histopathology_pnw.al.random_loop --mode full --seed 0 \
        --selector oracle_peek --max-labeled 12 --keep-checkpoints all
(`--selector tta_kd` for the TTA uncertainty selector.)
"""

import argparse
import gc
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, replace
from importlib.metadata import version
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from cellpose.train import train_seg

from histopathology_pnw.al.selectors import (
    SELECTORS,
    oracle_peek_scores,
    pick,
    random_scores,
    tta_kd_scores,
)
from histopathology_pnw.config import ROOT, Settings
from histopathology_pnw.datasets import monuseg
from histopathology_pnw.datasets.monuseg import Sample, read_image, read_mask
from histopathology_pnw.device import get_device
from histopathology_pnw.inference import predict
from histopathology_pnw.metrics import evaluate_all, summarize
from histopathology_pnw.models import sha256
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
    # "all": keep every round's checkpoint (~1.2 GB each); "best": only the best round on val
    keep_checkpoints: Literal["all", "best"] = "all"
    # how the next images are chosen, see al/selectors.py ("oracle_peek" is a diagnostic)
    selector: Literal["random", "oracle_peek", "tta_kd"] = "random"

    def __post_init__(self) -> None:
        if self.selector not in SELECTORS:
            raise ValueError(f"selector must be one of {SELECTORS}, not {self.selector!r}")
        if self.keep_checkpoints not in ("all", "best"):
            raise ValueError(
                f"keep_checkpoints must be 'all' or 'best', not {self.keep_checkpoints!r}"
            )


def preset(mode: str, seed: int) -> RandomALConfig:
    """Settings of notebook 03. "smoke" only checks that everything runs (PQ stays 0: in the
    single epoch the learning rate is still warming up); "full" has the hyperparameters of the
    6-seed baseline of the I1/I3 report. Numerical settings (TF32) are chosen by the caller."""
    if mode == "smoke":
        return RandomALConfig(
            seed=seed,
            max_labeled=4,
            n_epochs=1,
            nimg_per_epoch=2,
            batch_size=1,
            stop_at_oracle=False,
        )
    if mode == "full":
        return RandomALConfig(seed=seed)
    raise ValueError(f"unknown mode {mode!r}")


def round_sizes(cfg: RandomALConfig, pool_size: int) -> list[int]:
    last = min(cfg.max_labeled or pool_size, pool_size)
    sizes = list(range(cfg.initial, last + 1, cfg.step))
    if not sizes or sizes[-1] != last:
        sizes.append(last)
    return sizes


def _round_dir(out_dir: Path, n_labeled: int) -> Path:
    return out_dir / f"round_{n_labeled:02d}"


def _checkpoint(out_dir: Path, n_labeled: int) -> Path:
    # where train_seg saves (save_path=round dir, model_name="student"); built from out_dir, so
    # it stays valid when the outputs folder moves (the absolute path in results.csv does not)
    return _round_dir(out_dir, n_labeled) / "models" / "student"


def prune_checkpoints(results: pd.DataFrame, out_dir: Path) -> None:
    """Delete the checkpoints of all rounds except the best on val PQ (ties: the earlier round).

    Only `round_XX/models` under `out_dir` is removed; metrics stay. The best round is the one
    evaluated on test, so its checkpoint is the only one needed afterwards.
    """
    best = int(results.loc[results["val_pq"].idxmax(), "n_labeled"])
    for n in results["n_labeled"]:
        models = _round_dir(out_dir, int(n)) / "models"
        if int(n) != best and models.is_dir():
            shutil.rmtree(models)


def resume_state(results: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Labelled ids of the last finished round and the ids it selected for the next one.

    The selection is empty when the last round was the final one of that run, and for results
    written before selectors existed (no `selected_next` column); it is then computed again.
    """
    last = results.iloc[-1]
    labeled = json.loads(last["labeled_ids"])
    if "selected_next" not in results.columns:
        return labeled, []
    return labeled, json.loads(last["selected_next"])


def select_next(
    cfg: RandomALConfig,
    student,
    labeled: list[str],
    k: int,
    rank: dict[str, int],
    images: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
) -> tuple[list[str], pd.DataFrame]:
    """Score every unlabelled image with the configured selector and choose `k` of them."""
    unlabeled = sorted(set(rank) - set(labeled), key=rank.__getitem__)
    if cfg.selector == "oracle_peek":
        scores = oracle_peek_scores(student, unlabeled, images, labels)
    elif cfg.selector == "tta_kd":
        scores = tta_kd_scores(student, unlabeled, images, labels)
    else:
        scores = random_scores(unlabeled, rank)
    return pick(scores, k, rank)


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
    """Run (or resume) one AL experiment with `cfg.selector`; return one row per round."""
    device = device or get_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    results_path = out_dir / "results.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    done = set(results["n_labeled"]) if len(results) else set()

    rng = np.random.default_rng(cfg.seed)
    order = [pool[i] for i in rng.permutation(len(pool))]
    rank = {s.image_id: r for r, s in enumerate(order)}

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

    sizes = round_sizes(cfg, len(pool))
    labeled: list[str] = []
    selected = [s.image_id for s in order[: cfg.initial]]  # joins the labelled set next round
    if len(results):
        labeled, selected = resume_state(results)
    for idx, n in enumerate(sizes):
        if n in done:
            print(f"round n_labeled={n}: already done, skipping")
            continue
        if not selected:  # resumed after a run's final round, which selected nothing
            previous = None
            if cfg.selector != "random":  # the selector needs the student of that round
                previous = load_student(_checkpoint(out_dir, len(labeled)), device)
            selected, _ = select_next(
                cfg, previous, labeled, n - len(labeled), rank, images, labels
            )
            del previous
        labeled = labeled + selected
        if len(labeled) != n or len(set(labeled)) != n:
            raise RuntimeError(f"round n_labeled={n}: labelled set has {len(labeled)} images")
        round_dir = _round_dir(out_dir, n)
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

        # scores of all unlabelled images (also after the last round: they describe the pool)
        k = sizes[idx + 1] - n if idx + 1 < len(sizes) else 0
        selected, selection = select_next(cfg, student, labeled, k, rank, images, labels)
        selection.to_csv(round_dir / "selection.csv", index=False)

        row = {
            "n_labeled": n,
            "labeled_ids": json.dumps(labeled),
            "selected_next": json.dumps(selected),
            **{f"val_{k}": v for k, v in summary.items()},
            **{f"val_oracle_{k}": v for k, v in summary_oracle.items()},
            "final_train_loss": float(train_losses[-1]),
            "train_time_s": round(train_time, 1),
            "checkpoint": str(checkpoint),
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        if cfg.keep_checkpoints == "best":
            prune_checkpoints(results, out_dir)
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


def baseline_init_path(mode: str, seed: int) -> Path:
    """`student_init.pth` of the random run with the same mode and seed."""
    return Settings().outputs_dir / "al_random" / mode / f"seed{seed}" / "student_init.pth"


def oracle_pq(samples: list[Sample]) -> float:
    """The oracle's PQ against ground truth on `samples` (on val: the stopping threshold)."""
    gt = [read_mask(s) for s in samples]
    return summarize(evaluate_all(gt, oracle_masks(samples), [s.image_id for s in samples]))["pq"]


def evaluate_on_test(
    results: pd.DataFrame, out_dir: Path, device: torch.device | None = None
) -> pd.DataFrame:
    """One evaluation on the MoNuSeg test set (14 images) of the round that is best on val.

    The test set only reports the final result; it never influences a choice. Rows: student and
    oracle against ground truth, and the student against the oracle masks.
    """
    best = results.loc[results["val_pq"].idxmax()]
    test = monuseg.list_split("test")
    ids = [s.image_id for s in test]
    gt = [read_mask(s) for s in test]
    oracle = oracle_masks(test)
    student = load_student(_checkpoint(out_dir, int(best["n_labeled"])), device or get_device())
    preds = predict(student, [read_image(s.image_path) for s in test])
    final = pd.DataFrame(
        {
            "student vs GT": summarize(evaluate_all(gt, preds, ids)),
            "oracle vs GT": summarize(evaluate_all(gt, oracle, ids)),
            "student vs oracle": summarize(evaluate_all(oracle, preds, ids)),
        }
    ).T
    final.insert(0, "n_labeled", int(best["n_labeled"]))
    final.insert(1, "val_pq", float(best["val_pq"]))
    return final


def _run_info(args: argparse.Namespace, device: torch.device, init_path: Path) -> dict:
    """What is needed to reproduce a run: code version, environment, arguments."""

    def git(*cmd: str) -> str:
        out = subprocess.run(["git", *cmd], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip()

    return {
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "host": platform.node(),
        "device": str(device),
        "torch": torch.__version__,
        "cellpose": version("cellpose"),
        "tf32": torch.get_float32_matmul_precision() != "highest",
        "student_init_sha256": sha256(init_path) if init_path.is_file() else None,
        "args": vars(args),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="AL on MoNuSeg with one selector, one seed.")
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="faster, needs more GPU memory (batch 4 fits in 80 GB, not in 16 GB)",
    )
    parser.add_argument(
        "--no-tf32", action="store_true", help="exact float32 matmuls (TF32 is on by default)"
    )
    parser.add_argument("--keep-checkpoints", choices=("all", "best"), default="best")
    parser.add_argument("--selector", choices=SELECTORS, default="random")
    parser.add_argument(
        "--max-labeled", type=int, default=None, help="last round (default: the mode's preset)"
    )
    args = parser.parse_args()

    # Cellpose training logs (loss per epoch) to stdout, i.e. the Slurm log of this job;
    # cellpose.io.logger_setup() would share and truncate ~/.cellpose/run.log across jobs
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s: %(message)s", stream=sys.stdout
    )
    if not args.no_tf32:
        torch.set_float32_matmul_precision("high")  # TF32 where the GPU has it
    device = get_device()
    cfg = replace(
        preset(args.mode, args.seed),
        gradient_checkpointing=not args.no_gradient_checkpointing,
        keep_checkpoints=args.keep_checkpoints,
        selector=args.selector,
    )
    if args.max_labeled is not None:
        cfg = replace(cfg, max_labeled=args.max_labeled)

    split = monuseg.load_train_val_split()
    by_id = monuseg.samples_by_id("train")
    pool = [by_id[i] for i in split["train"]]
    val = [by_id[i] for i in split["val"]]
    if args.mode == "smoke":
        val = val[:2]

    test = monuseg.list_split("test")

    out_dir = Settings().outputs_dir / f"al_{cfg.selector}" / args.mode / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # paired comparison: start from the very file the random run of this seed started from
    baseline_init = baseline_init_path(args.mode, args.seed)
    init_path = out_dir / "student_init.pth"
    if cfg.selector != "random" and not init_path.is_file() and baseline_init.is_file():
        partial = init_path.with_suffix(".partial")  # never leave a half-copied init behind
        shutil.copyfile(baseline_init, partial)
        partial.replace(init_path)
        print(f"student init copied from {baseline_init}")
    # one line per (re)start; appending keeps earlier lines intact if a job dies while writing
    with (out_dir / "run_info.jsonl").open("a") as f:
        f.write(json.dumps(_run_info(args, device, init_path)) + "\n")
    print(f"device: {device} | pool: {len(pool)} | val: {len(val)} | results: {out_dir}")

    # fill the oracle cache for every image this run needs with a single oracle load
    # (the smoke run fills it for the full array too: same pool, full val, test)
    oracle_masks(pool + [by_id[i] for i in split["val"]] + test)

    threshold = oracle_pq(val)
    print(f"oracle PQ on val: {threshold:.4f}")
    results = run_random_al(cfg, pool, val, out_dir, oracle_val_pq=threshold, device=device)
    if cfg.keep_checkpoints == "best":
        prune_checkpoints(results, out_dir)  # also after a resumed run that trained nothing

    if cfg.selector != "random":
        # the test set reports final results only: selectors under development (and the
        # oracle_peek diagnostic, which reads pool labels) are compared on validation
        print(f"{cfg.selector}: test set not evaluated")
        return

    final = evaluate_on_test(results, out_dir, device)
    final.to_csv(out_dir / "final_test.csv")
    print(final.round(4).to_string())


if __name__ == "__main__":
    main()
