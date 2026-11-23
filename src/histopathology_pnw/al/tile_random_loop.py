"""Active learning on tiles instead of whole images: random selection (the tile baseline) and the
tile versions of the image selectors (`--selector oracle_peek | tta_kd`, `al/tile_selectors.py`).

Same simulation and training recipe as `random_loop.py`, but the unit of annotation is a tile:
1. pool: the training images (the frozen train/val split, never the test set) cut into 256 x 256
   tiles on a 4 x 4 grid started every 248 px (1000 = 3 * 248 + 256, 8 px overlap), 16 tiles per
   image, every pixel covered,
2. annotation: the oracle predicts the whole image (full context, cached as before) and the mask
   of a queried tile is cut out of it; nuclei cut by the border stay as fragments,
   (`--full-image-flows`: the Cellpose training target - labels and flows - is computed on the
   whole-image mask and then cut, so a cut nucleus keeps the flow towards its real centre),
3. the labelled set grows by `step` tiles drawn at random (one permutation fixed by the seed,
   nested sets); 16 tiles = 1 image in pixels, so the defaults (32 + 32 per round) give round k
   the pixel budget of round k of the image AL (2 + 2 images per round),
4. every round a fresh student is built from the SAM initialisation (cold start) and trained on
   the labelled tiles only, with the same `train_seg` settings and the same number of steps,
5. evaluation on validation as in `random_loop.py` (whole images against GT and against the
   oracle), plus the per-tile metrics of the validation images (`tile_metrics`). The test set
   is evaluated separately and only for reporting (`al/test_eval.py`).

Each finished round is appended to `results.csv`; rerunning skips finished rounds (resume).

    uv run --no-sync python -m histopathology_pnw.al.tile_random_loop --seeds 0 1 2
"""

import argparse
import json
import shutil
import time
from contextlib import contextmanager, nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from cellpose.train import train_seg

from histopathology_pnw.al.random_loop import RandomALConfig, _free_memory, round_sizes
from histopathology_pnw.al.tile_selectors import (
    SELECTORS,
    oracle_peek_tile_scores,
    pick_tiles,
    tta_kd_tile_scores,
)
from histopathology_pnw.config import Settings
from histopathology_pnw.datasets import monuseg
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
from histopathology_pnw.tile_metrics import crop_mask, evaluate_tiles, summarize_tiles
from histopathology_pnw.uncertainty.patches import TileGrid

KEY_SEP = "#t"  # tile key: "<image_id>#t<tile:02d>"


@dataclass
class TileALConfig(RandomALConfig):
    initial: int = 32  # labelled tiles in the first round (= 2 images in pixels)
    step: int = 32  # tiles added per round
    tile: int = 256
    stride: int = 248
    # True: Cellpose flows computed on the whole-image oracle mask, then cut (a nucleus cut by the
    # border keeps the flow towards its real centre); False: flows computed on the cut tile mask
    full_image_flows: bool = False
    # > 0 (needs full_image_flows): the student also sees this many px of tissue around the tile,
    # marked as not annotated (foreground = -1) and left out of the loss; the budget is unchanged
    margin: int = 0
    rounds: list[int] | None = None  # train only these n_labeled (default: round_sizes)
    # how the next tiles are chosen (al/tile_selectors.py; "oracle_peek" is a diagnostic)
    tile_selector: str = "random"

    def __post_init__(self) -> None:
        if self.tile_selector not in SELECTORS:
            raise ValueError(f"tile_selector must be one of {SELECTORS}")


def tile_key(image_id: str, tile: int) -> str:
    return f"{image_id}{KEY_SEP}{tile:02d}"


def split_key(key: str) -> tuple[str, int]:
    image_id, tile = key.rsplit(KEY_SEP, 1)
    return image_id, int(tile)


def full_image_flows(mask: np.ndarray, device: torch.device | None = None) -> np.ndarray:
    """Cellpose training target of a whole image: [labels, foreground, flow Y, flow X]."""
    from cellpose.dynamics import labels_to_flows

    return labels_to_flows([mask.astype(np.int32)], device=device)[0]


def crop_flows(flows: np.ndarray, box: tuple[slice, slice]) -> np.ndarray:
    """Cut a [4, H, W] training target to a tile; labels relabelled 1..N, flows unchanged."""
    out = np.ascontiguousarray(flows[:, box[0], box[1]]).astype(np.float32)
    out[0] = crop_mask(out[0].astype(np.int32), (slice(None), slice(None)))
    return out


def margin_target(
    flows: np.ndarray, box: tuple[slice, slice], margin: int
) -> tuple[tuple[slice, slice], np.ndarray]:
    """Box widened by `margin` (clipped to the image) and its target with the margin unannotated."""
    h, w = flows.shape[-2:]
    ys, xs = box
    wide = (
        slice(max(ys.start - margin, 0), min(ys.stop + margin, h)),
        slice(max(xs.start - margin, 0), min(xs.stop + margin, w)),
    )
    out = np.ascontiguousarray(flows[:, wide[0], wide[1]]).astype(np.float32)
    inside = np.zeros(out.shape[1:], bool)
    inside[
        ys.start - wide[0].start : ys.stop - wide[0].start,
        xs.start - wide[1].start : xs.stop - wide[1].start,
    ] = True
    out[0][~inside] = 0
    out[0] = crop_mask(out[0].astype(np.int32), (slice(None), slice(None)))
    out[1][~inside] = -1  # not annotated: ignored by the loss
    out[2:, ~inside] = 0
    return wide, out


@contextmanager
def loss_ignoring_unannotated():
    """Cellpose segmentation loss without the pixels whose foreground target is -1.

    Same as `cellpose.train._loss_fn_seg` (MSE of 5 x flows / 2 + BCE of the cell probability,
    both averaged over pixels) but averaged over annotated pixels only. Patched for the duration
    of the `with` block; identical to the original when no pixel is -1.
    """
    import cellpose.train as cp_train

    original = cp_train._loss_fn_seg

    def loss_fn(lbl, y, device):
        valid = (lbl[:, -3] > -0.5).to(y.dtype)
        n = valid.sum().clamp(min=1)
        mse = (((y[:, -3:-1] - 5.0 * lbl[:, -2:]) ** 2) * valid[:, None]).sum() / (2 * n)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            y[:, -1], (lbl[:, -3] > 0.5).to(y.dtype), reduction="none"
        )
        return mse / 2.0 + (bce * valid).sum() / n

    cp_train._loss_fn_seg = loss_fn
    try:
        yield
    finally:
        cp_train._loss_fn_seg = original


def n_labels(target: np.ndarray) -> int:
    """Number of nuclei in a tile target (a label mask or a [4, H, W] flow target)."""
    return int(target[0].max() if target.ndim == 3 else target.max())


def cut_pool(
    images: dict[str, np.ndarray],
    masks: dict[str, np.ndarray],
    tile: int,
    stride: int,
    flows: dict[str, np.ndarray] | None = None,
    margin: int = 0,
) -> tuple[list[str], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Tile keys (image order, then row-major), tile images and tile targets.

    Targets are the cut masks (relabelled), or the cut whole-image flows when `flows` is given.
    """
    keys, tile_images, tile_masks = [], {}, {}
    for image_id, image in images.items():
        grid = TileGrid(*image.shape[:2], size=tile, stride=stride)
        for t in range(grid.n_tiles):
            box = grid.box(t)
            key = tile_key(image_id, t)
            keys.append(key)
            if margin:
                if flows is None:
                    raise ValueError("margin needs the whole-image flows")
                wide, tile_masks[key] = margin_target(flows[image_id], box, margin)
                tile_images[key] = np.ascontiguousarray(image[wide])
                continue
            tile_images[key] = np.ascontiguousarray(image[box])
            if flows is None:
                tile_masks[key] = crop_mask(masks[image_id], box)
            else:
                tile_masks[key] = crop_flows(flows[image_id], box)
    return keys, tile_images, tile_masks


def draw_order(keys: list[str], seed: int) -> list[str]:
    """The seed's permutation of the pool tiles; round n labels the first n."""
    rng = np.random.default_rng(seed)
    return [keys[i] for i in rng.permutation(len(keys))]


def evaluate_val_tiles(
    gts: list[np.ndarray], preds: list[np.ndarray], ids: list[str], cfg: TileALConfig
) -> pd.DataFrame:
    frames = []
    for gt, pred, image_id in zip(gts, preds, ids, strict=True):
        grid = TileGrid(*gt.shape, size=cfg.tile, stride=cfg.stride)
        frames.append(evaluate_tiles(gt, pred, grid).assign(image_id=image_id))
    return pd.concat(frames, ignore_index=True)


def run_tile_random_al(
    cfg: TileALConfig,
    pool: list[Sample],
    val: list[Sample],
    out_dir: Path,
    oracle_val_pq: float | None = None,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """Run (or resume) one random tile-selection AL experiment; return one row per round."""
    device = device or get_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
    results_path = out_dir / "results.csv"
    results = pd.read_csv(results_path) if results_path.is_file() else pd.DataFrame()
    done = set(results["n_labeled"]) if len(results) else set()

    # oracle "annotations": whole pool images, cached, then cut into tiles
    pool_ids = [s.image_id for s in pool]
    masks = dict(zip(pool_ids, oracle_masks(pool), strict=True))
    images = {s.image_id: read_image(s.image_path) for s in pool}
    flows = None
    if cfg.full_image_flows:
        flows = {i: full_image_flows(m, device) for i, m in masks.items()}
    keys, tile_images, tile_masks = cut_pool(images, masks, cfg.tile, cfg.stride, flows, cfg.margin)
    del flows
    if cfg.tile_selector == "random":
        del images, masks  # the other selectors segment the whole pool images
    order = draw_order(keys, cfg.seed)
    rank = {k: r for r, k in enumerate(order)}

    def select_next(student, labeled: list[str], k: int) -> tuple[list[str], pd.DataFrame | None]:
        done_tiles = set(labeled)
        unlabeled = [t for t in order if t not in done_tiles]
        if cfg.tile_selector == "random" or k == 0:
            return unlabeled[:k], None
        if cfg.tile_selector == "oracle_peek":
            scores = oracle_peek_tile_scores(
                student, unlabeled, images, masks, split_key, tile_key, cfg.tile, cfg.stride
            )
        else:
            scores = tta_kd_tile_scores(
                student, unlabeled, images, split_key, tile_key, masks, cfg.tile, cfg.stride
            )
        return pick_tiles(scores, k, rank)

    val_images = [read_image(s.image_path) for s in val]
    val_gt = [read_mask(s) for s in val]
    val_oracle = oracle_masks(val)
    val_ids = [s.image_id for s in val]

    init_path = out_dir / "student_init.pth"
    if not init_path.is_file():
        build_student_init(cfg.seed, init_path)

    reached = cfg.stop_at_oracle and oracle_val_pq is not None and len(results) > 0
    if reached and results["val_pq"].max() >= oracle_val_pq:
        print("student already reached the oracle in a previous run; nothing to do")
        return results

    sizes = cfg.rounds or round_sizes(cfg, len(keys))
    # every selector starts from the first `initial` tiles of the seed's permutation; random
    # keeps the nested sets order[:n], the others add what the previous student chose
    labeled: list[str] = []
    selected = order[: sizes[0]]
    if cfg.tile_selector != "random" and len(results):
        last = results.iloc[-1]
        labeled, selected = json.loads(last["labeled_ids"]), json.loads(last["selected_next"])
    for idx, n in enumerate(sizes):
        if n in done:
            print(f"round n_labeled={n}: already done, skipping")
            continue
        labeled = order[:n] if cfg.tile_selector == "random" else labeled + selected
        if len(labeled) != n or len(set(labeled)) != n:
            raise RuntimeError(f"round n_labeled={n}: labelled set has {len(labeled)} tiles")
        round_dir = out_dir / f"round_{n:03d}"
        round_dir.mkdir(parents=True, exist_ok=True)
        set_seed(cfg.seed * 1000 + n)

        t0 = time.time()
        student = load_student(init_path, device)
        if cfg.gradient_checkpointing:
            enable_gradient_checkpointing(student.net)
        with loss_ignoring_unannotated() if cfg.margin else nullcontext():
            checkpoint, train_losses, _ = train_seg(
                student.net,
                train_data=[tile_images[k] for k in labeled],
                train_labels=[tile_masks[k] for k in labeled],
                channel_axis=-1,
                n_epochs=cfg.n_epochs,
                nimg_per_epoch=cfg.nimg_per_epoch,
                batch_size=cfg.batch_size,
                learning_rate=cfg.learning_rate,
                weight_decay=cfg.weight_decay,
                min_train_masks=1,  # keep tiles with few nuclei (the default 5 drops them silently)
                save_path=round_dir,
                model_name="student",
            )
        train_time = time.time() - t0

        preds = predict(student, val_images, progress=False)
        per_image = evaluate_all(val_gt, preds, val_ids)
        per_image.to_csv(round_dir / "val_metrics.csv", index=False)
        summary = summarize(per_image)
        per_image_oracle = evaluate_all(val_oracle, preds, val_ids)
        per_image_oracle.to_csv(round_dir / "val_metrics_oracle.csv", index=False)
        summary_oracle = summarize(per_image_oracle)
        per_tile = evaluate_val_tiles(val_gt, preds, val_ids, cfg)
        per_tile.to_csv(round_dir / "val_metrics_tiles.csv", index=False)

        t1 = time.time()
        k_next = sizes[idx + 1] - n if idx + 1 < len(sizes) else 0
        selected, selection = select_next(student, labeled, k_next)
        if selection is not None:
            selection.to_csv(round_dir / "selection.csv", index=False)
        select_time = time.time() - t1

        row = {
            "n_labeled": n,
            "n_images_equiv": n / 16,
            "n_source_images": len({split_key(k)[0] for k in labeled}),
            "n_labeled_nuclei": sum(n_labels(tile_masks[k]) for k in labeled),
            "labeled_ids": json.dumps(labeled),
            "selected_next": json.dumps(selected),
            **{f"val_{k}": v for k, v in summary.items()},
            **{f"val_{k}": v for k, v in summarize_tiles(per_tile).items()},
            **{f"val_oracle_{k}": v for k, v in summary_oracle.items()},
            "final_train_loss": float(train_losses[-1]),
            "train_time_s": round(train_time, 1),
            "select_time_s": round(select_time, 1),
            "checkpoint": str(checkpoint),
        }
        results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
        results.to_csv(results_path, index=False)
        print(
            f"round n_labeled={n} tiles ({row['n_source_images']} images): "
            f"val PQ={summary['pq']:.4f}, tile PQ={row['val_tile_pq']:.4f}, "
            f"vs oracle PQ={summary_oracle['pq']:.4f} ({train_time / 60:.1f} min)"
        )

        del student
        _free_memory()
        if cfg.stop_at_oracle and oracle_val_pq is not None and summary["pq"] >= oracle_val_pq:
            print(f"student reached the oracle (PQ {oracle_val_pq:.4f}); stopping")
            break

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Random tile-selection AL on MoNuSeg.")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--out-dir", type=Path, help="default: <outputs>/al_random_tiles/{full,smoke}"
    )
    parser.add_argument("--max-labeled", type=int, help="last round (tiles); default: whole pool")
    parser.add_argument(
        "--no-gradient-checkpointing", action="store_true", help="for >= 40 GB GPUs"
    )
    parser.add_argument("--no-stop-at-oracle", action="store_true")
    parser.add_argument("--rounds", type=int, nargs="+", help="only these n_labeled (tiles)")
    parser.add_argument(
        "--full-image-flows", action="store_true", help="flows from the whole-image mask, then cut"
    )
    parser.add_argument(
        "--margin", type=int, default=0, help="px of unannotated context (needs --full-image-flows)"
    )
    parser.add_argument("--selector", choices=SELECTORS, default="random")
    parser.add_argument(
        "--init-from", type=Path, help="copy student_init.pth from this run (paired comparison)"
    )
    parser.add_argument("--smoke", action="store_true", help="2 tiny rounds, 1 epoch (code check)")
    args = parser.parse_args()

    settings = Settings()
    mode = "smoke" if args.smoke else "full"
    out_root = args.out_dir or settings.outputs_dir / "al_random_tiles" / mode
    split = monuseg.load_train_val_split()
    by_id = monuseg.samples_by_id("train")
    pool = [by_id[i] for i in split["train"]]
    val = [by_id[i] for i in split["val"]]
    oracle_val_pq = summarize(
        evaluate_all([read_mask(s) for s in val], oracle_masks(val), [s.image_id for s in val])
    )["pq"]
    device = get_device()
    print(f"device: {device} | pool {len(pool)} images | val {len(val)} | out {out_root}")
    print(f"oracle val PQ: {oracle_val_pq:.4f}")

    for seed in args.seeds:
        cfg = TileALConfig(
            seed=seed,
            max_labeled=args.max_labeled,
            gradient_checkpointing=not args.no_gradient_checkpointing,
            stop_at_oracle=not args.no_stop_at_oracle,
            full_image_flows=args.full_image_flows,
            margin=args.margin,
            rounds=args.rounds,
            tile_selector=args.selector,
        )
        if args.smoke:
            cfg.initial, cfg.step, cfg.max_labeled = 4, 4, 8
            cfg.n_epochs, cfg.nimg_per_epoch, cfg.batch_size = 1, 4, 2
        out_dir = out_root / f"seed{seed}"
        init = out_dir / "student_init.pth"
        if args.init_from and not init.is_file():
            source = args.init_from / f"seed{seed}" / "student_init.pth"
            out_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, init.with_suffix(".partial"))
            init.with_suffix(".partial").replace(init)
            print(f"student init copied from {source}")
        run_tile_random_al(cfg, pool, val, out_dir, oracle_val_pq, device)


if __name__ == "__main__":
    main()
