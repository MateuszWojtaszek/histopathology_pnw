# Losowe AL na kaflach CAMELYON17, segmentacja jąder (student Cellpose-SAM, "adnotator" = wyrocznia cpsam).
# Umieść w: src/histopathology_pnw/al/camelyon17_random_loop.py
#
# CAMELYON17 nie ma masek jąder, więc:
#  - uczymy studenta na maskach wyroczni (jak w MoNuSeg),
#  - jakość studenta mierzymy ZGODNOŚCIĄ Z WYROCZNIĄ na kaflach val/test (PQ względem masek wyroczni),
#  - poprzeczką jest samozgodność wyroczni (PQ wyroczni na kaflu i na jego odbiciu), bo szum
#    wyroczni to naturalny sufit zgodności.
# To NIE jest dokładność względem patologa. Prawdziwe GT daje tylko test na MoNuSeg (patrz notebook 03).
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from histopathology_pnw.datasets import camelyon17 as c17
from histopathology_pnw.inference import predict
from histopathology_pnw.metrics import evaluate_all, summarize
from histopathology_pnw.models.student import build_student_init, load_student


def _outputs() -> Path:
    from histopathology_pnw.config import Settings

    return Settings().outputs_dir


def oracle_tile_masks(oracle, tile_ids: list[str], batch: int = 8) -> list[np.ndarray]:
    # maski wyroczni dla kafli, liczone raz (cache: outputs/oracle_masks/camelyon17/<tile_id>.npy)
    cache = _outputs() / "oracle_masks" / "camelyon17"
    cache.mkdir(parents=True, exist_ok=True)
    missing = [t for t in tile_ids if not (cache / f"{t}.npy").exists()]
    for i in range(0, len(missing), batch):
        chunk = missing[i : i + batch]
        preds = predict(oracle, [c17.read_tile(t) for t in chunk])
        for t, m in zip(chunk, preds, strict=True):
            np.save(cache / f"{t}.npy", np.asarray(m).astype(np.uint16))
    return [np.load(cache / f"{t}.npy") for t in tile_ids]


def oracle_self_consistency(oracle, tile_ids: list[str], ref_masks: list[np.ndarray]):
    # PQ między maską wyroczni a maską z odbitego (lewo-prawo) kafla, odbitą z powrotem.
    # Szacuje szum samej wyroczni: student nie powinien być "zgodniejszy" niż ona sama.
    flipped = [np.ascontiguousarray(c17.read_tile(t)[:, ::-1]) for t in tile_ids]
    back = [np.ascontiguousarray(np.asarray(m)[:, ::-1]) for m in predict(oracle, flipped)]
    table = evaluate_all(ref_masks, back, tile_ids)
    return summarize(table), table


def enable_gradient_checkpointing(net) -> int:
    # Jak w pętli MoNuSeg: bloki encodera liczone ponownie w backwardzie (mniej VRAM, ~30% wolniej).
    # Zakłada, że net.encoder.blocks to lista bloków ViT; jeśli nie, nic nie robi i zwraca 0.
    from torch.utils.checkpoint import checkpoint

    blocks = getattr(getattr(net, "encoder", None), "blocks", None)
    if blocks is None:
        return 0

    def wrap(block):
        orig = block.forward

        def fwd(*a, **k):
            if torch.is_grad_enabled() and block.training:
                return checkpoint(orig, *a, use_reentrant=False, **k)
            return orig(*a, **k)

        block.forward = fwd

    for b in blocks:
        wrap(b)
    return len(blocks)


@dataclass
class TileALConfig:
    seed: int = 0
    initial: int = 4
    step: int = 4
    max_labeled: int | None = None
    n_epochs: int = 200
    nimg_per_epoch: int = 64
    batch_size: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.1
    gradient_checkpointing: bool = True
    stop_at_target: bool = True


def round_sizes(cfg: TileALConfig, n_pool: int) -> list[int]:
    top = min(cfg.max_labeled or n_pool, n_pool)
    sizes = list(range(min(cfg.initial, top), top + 1, cfg.step))
    if not sizes or sizes[-1] != top:
        sizes.append(top)
    return sizes


def train_student(init, imgs, masks, out_dir: Path, cfg: TileALConfig, device, seed: int):
    from cellpose import train as cp_train

    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = load_student(init, device)
    n_ckpt = enable_gradient_checkpointing(model.net) if cfg.gradient_checkpointing else 0
    t0 = time.time()
    path, train_losses, _ = cp_train.train_seg(
        model.net, train_data=imgs, train_labels=masks,
        batch_size=cfg.batch_size, n_epochs=cfg.n_epochs, nimg_per_epoch=cfg.nimg_per_epoch,
        learning_rate=cfg.learning_rate, weight_decay=cfg.weight_decay,
        save_path=str(out_dir), model_name="student",
    )
    return Path(path), dict(final_train_loss=float(np.asarray(train_losses)[-1]),
                            train_time_s=time.time() - t0, n_checkpointed_blocks=n_ckpt)


def _status(out_dir: Path, **kw) -> None:
    # status.json czyta camelyon17_monitor.py (postęp, ETA)
    p = out_dir / "status.json"
    cur = json.loads(p.read_text()) if p.exists() else {}
    cur.update(kw, updated_at=time.strftime("%Y-%m-%d %H:%M:%S"))
    p.write_text(json.dumps(cur, indent=2, default=float))


def run_random_al(cfg: TileALConfig, pool: list[str], val: list[str], oracle_masks: dict[str, np.ndarray],
                  out_dir: Path, target_pq: float, device) -> pd.DataFrame:
    # pool, val: tile_id; oracle_masks: tile_id -> maska wyroczni (adnotacje + "GT" do ewaluacji)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    order = [pool[i] for i in rng.permutation(len(pool))]
    (out_dir / "order.json").write_text(json.dumps(order, indent=2))

    sizes = round_sizes(cfg, len(pool))
    _status(out_dir, seed=cfg.seed, sizes=sizes, n_rounds=len(sizes), n_epochs=cfg.n_epochs,
            nimg_per_epoch=cfg.nimg_per_epoch, batch_size=cfg.batch_size, target_pq=target_pq,
            device=str(device), phase="start", round=0, n_labeled=None)

    init = build_student_init(seed=cfg.seed, out_path=out_dir / "student_init.pth")
    val_imgs = [c17.read_tile(t) for t in val]
    val_gt = [oracle_masks[t] for t in val]

    rows = []
    for idx, n in enumerate(sizes, 1):
        rd = out_dir / f"round_{n:03d}"
        res_path = rd / "result.json"
        if res_path.exists():
            row = json.loads(res_path.read_text())
        else:
            rd.mkdir(parents=True, exist_ok=True)
            labeled = order[:n]
            t_round = time.time()
            _status(out_dir, phase="training", round=idx, n_labeled=n,
                    round_started=time.strftime("%Y-%m-%d %H:%M:%S"))
            ckpt, info = train_student(
                init, [c17.read_tile(t) for t in labeled], [oracle_masks[t] for t in labeled],
                rd, cfg, device, seed=cfg.seed * 1000 + n)
            _status(out_dir, phase="evaluating")
            t_eval = time.time()
            student = load_student(ckpt, device)
            table = evaluate_all(val_gt, predict(student, val_imgs), val)
            row = dict(n_labeled=n, labeled=labeled, checkpoint=str(ckpt), **info,
                       eval_time_s=time.time() - t_eval, round_time_s=time.time() - t_round,
                       **{f"val_{k}": v for k, v in summarize(table).items()})
            res_path.write_text(json.dumps(row, indent=2, default=float))
        rows.append(row)
        print(f"n={n}: val PQ (względem wyroczni) = {row['val_pq']:.4f} (poprzeczka {target_pq:.4f})", flush=True)
        if cfg.stop_at_target and row["val_pq"] >= target_pq:
            break
    _status(out_dir, phase="finished")
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "results.csv", index=False)
    return res