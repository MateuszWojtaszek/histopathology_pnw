# Wspólny kod dla notebooków 02 i 03 (CAMELYON17).
# Umieść w: src/histopathology_pnw/camelyon17_al.py
#
# Zadanie: klasyfikacja patchy 256x256 px (poziom 0) jako guz / tkanka normalna.
# Jednostką puli AL jest slajd (WSI) z adnotacją XML; "adnotatorem" są ręczne poligony
# z CAMELYON17 (maski z notebooka 01), więc nie ma tu osobnej wyroczni-labelera.
from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# korzeń repo: src/histopathology_pnw/camelyon17_al.py -> parents[2]
ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("C17_ROOT", ROOT / "data" / "raw" / "camelyon17"))
IMAGES = DATA_ROOT / "images"
PROCESSED = Path(os.environ.get("C17_PROCESSED", ROOT / "data" / "processed" / "camelyon17"))
SPLIT_PATH = ROOT / "configs" / "camelyon17_split.json"
try:  # ścieżka wyników jak w pozostałych notebookach (PNW_* / .env)
    from histopathology_pnw.config import Settings

    OUTPUTS = Path(os.environ.get("C17_OUTPUTS") or Settings().outputs_dir)
except Exception:
    OUTPUTS = Path(os.environ.get("C17_OUTPUTS", ROOT / "outputs"))

PATCH = 256          # rozmiar patcha na poziomie 0 [px]
INPUT_SIZE = 224     # rozmiar wejścia ViT (patch jest skalowany 256 -> 224)
ARCH = "vit_small_patch16_224.dino"
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
SLIDE_RE = re.compile(r"patient_(\d+)_node_(\d+)")


# ----------------------------------------------------------------------------- dane
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@lru_cache(maxsize=1)
def slide_paths() -> dict[str, Path]:
    return {p.stem: p for p in sorted(IMAGES.rglob("*.tif")) if SLIDE_RE.search(p.stem)}


def slide_meta(slide_id: str) -> tuple[int, int]:
    # (patient, center); 20 pacjentów na ośrodek
    patient = int(SLIDE_RE.search(slide_id).group(1))
    return patient, patient // 20


def ensure_test_split(path: Path = SPLIT_PATH, seed: int = 1, frac: float = 0.2) -> dict:
    # Notebook 01 zamraża train/val. Tu (raz) wydzielamy test z puli train, po pacjencie
    # i osobno w każdym ośrodku. Jeśli klucz "test" już istnieje, nic nie zmieniamy.
    split = json.loads(path.read_text())
    if "test" in split:
        return split
    rng = np.random.default_rng(seed)
    by_center: dict[int, dict[int, list[str]]] = {}
    for sid in split["train"]:
        patient, center = slide_meta(sid)
        by_center.setdefault(center, {}).setdefault(patient, []).append(sid)
    test: list[str] = []
    for _center, pats in sorted(by_center.items()):
        keys = sorted(pats)
        if len(keys) < 2:
            continue
        rng.shuffle(keys)
        for k in keys[: max(1, round(frac * len(keys)))]:
            test += pats[k]
    split["train"] = sorted(set(split["train"]) - set(test))
    split["test"] = sorted(test)
    split["test_seed"] = seed
    path.write_text(json.dumps(split, indent=2))
    return split


@lru_cache(maxsize=None)
def _load_masks(slide_id: str):
    z = np.load(PROCESSED / f"{slide_id}.npz")
    return z["tumor"], z["tissue"], float(z["downsample"]), int(z["level"])


def grid_shape(slide_id: str) -> tuple[int, int]:
    tumor, _, ds, _ = _load_masks(slide_id)
    k = max(int(round(PATCH / ds)), 1)
    return tumor.shape[0] // k, tumor.shape[1] // k


@lru_cache(maxsize=None)
def _slide_patches_cached(slide_id: str) -> pd.DataFrame:
    tumor, tissue, ds, _ = _load_masks(slide_id)
    k = max(int(round(PATCH / ds)), 1)
    h, w = (tumor.shape[0] // k) * k, (tumor.shape[1] // k) * k
    t = tumor[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    s = tissue[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    r, c = np.nonzero(s >= 0.5)
    tf = t[r, c]
    df = pd.DataFrame(
        dict(slide_id=slide_id, row=r, col=c,
             x0=(c * k * ds).astype(int), y0=(r * k * ds).astype(int),
             tumor_frac=tf, tissue_frac=s[r, c])
    )
    df["label"] = np.where(tf >= 0.5, 1, np.where(tf == 0, 0, -1))  # -1 = graniczny
    return df


def slide_patches(slide_id: str, keep_border: bool = False) -> pd.DataFrame:
    df = _slide_patches_cached(slide_id)
    return (df if keep_border else df[df["label"] >= 0]).reset_index(drop=True)


def patches_for(slide_ids) -> pd.DataFrame:
    return pd.concat([slide_patches(s) for s in slide_ids], ignore_index=True)


def usable_slides(slide_ids, min_tumor: int = 20, min_normal: int = 20) -> list[str]:
    # slajdy, z których da się wylosować obie klasy
    out = []
    for s in slide_ids:
        lab = slide_patches(s)["label"]
        if (lab == 1).sum() >= min_tumor and (lab == 0).sum() >= min_normal:
            out.append(s)
    return out


def sample_eval_table(slide_ids, per_class: int = 1000, seed: int = 0) -> pd.DataFrame:
    # stały, zrównoważony podzbiór patchy do ewaluacji (do `per_class` na klasę i slajd).
    # Uwaga: prewalencja guza jest tu sztuczna, dlatego raportujemy AUROC / bal_acc / F1.
    rng = np.random.default_rng(seed)
    parts = []
    for s in slide_ids:
        df = slide_patches(s)
        for lab in (0, 1):
            g = df[df["label"] == lab]
            if len(g) > per_class:
                g = g.iloc[np.sort(rng.choice(len(g), per_class, replace=False))]
            parts.append(g)
    return pd.concat(parts, ignore_index=True)


def read_thumbnail(slide_id: str) -> np.ndarray:
    import openslide

    _, _, _, level = _load_masks(slide_id)
    with openslide.OpenSlide(str(slide_paths()[slide_id])) as sl:
        w, h = sl.level_dimensions[level]
        return np.asarray(sl.read_region((0, 0), level, (w, h)).convert("RGB"))


def to_grid(tab: pd.DataFrame, col: str, shape: tuple[int, int]) -> np.ndarray:
    grid = np.full(shape, np.nan, dtype=np.float32)
    grid[tab["row"].to_numpy(), tab["col"].to_numpy()] = tab[col].to_numpy()
    return grid


class PatchDataset(Dataset):
    def __init__(self, table: pd.DataFrame, train: bool = False):
        from torchvision import transforms as T

        t = table.reset_index(drop=True)
        self.sid = t["slide_id"].to_numpy()
        self.x0 = t["x0"].to_numpy()
        self.y0 = t["y0"].to_numpy()
        self.label = t["label"].to_numpy()
        self.train = train
        self._open: dict = {}
        self.aug = T.Compose([
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomApply([T.RandomRotation((90, 90))], p=0.5),
            T.ColorJitter(0.25, 0.25, 0.25, 0.04),
        ])
        self.tf = T.Compose([T.Resize(INPUT_SIZE), T.ToTensor(), T.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.sid)

    def _slide(self, sid):
        import openslide

        if sid not in self._open:  # uchwyt otwierany leniwie, osobno w każdym workerze
            self._open[sid] = openslide.OpenSlide(str(slide_paths()[sid]))
        return self._open[sid]

    def __getitem__(self, i):
        img = self._slide(self.sid[i]).read_region(
            (int(self.x0[i]), int(self.y0[i])), 0, (PATCH, PATCH)).convert("RGB")
        if self.train:
            img = self.aug(img)
        return self.tf(img), float(self.label[i]), i


# --------------------------------------------------------------------------- model
def build_model(seed: int = 0, pretrained: bool = True):
    # encoder ViT-S/16 (DINO, ImageNet) + losowa głowica 1 logit (seed)
    import timm

    torch.manual_seed(seed)
    return timm.create_model(ARCH, pretrained=pretrained, num_classes=1)


def save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(path: Path, device) -> torch.nn.Module:
    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return model.to(device).eval()


def sampling_weights(table: pd.DataFrame) -> np.ndarray:
    # 50% klasa guz / 50% normalna; wewnątrz klasy każdy slajd ma równą wagę
    if set(table["label"].unique()) != {0, 1}:
        raise ValueError("w zbiorze treningowym brakuje jednej z klas")
    cnt = table.groupby(["slide_id", "label"])["x0"].transform("size")
    n_slides = table.groupby("label")["slide_id"].transform("nunique")
    return (0.5 / (n_slides * cnt)).to_numpy()


def _lr_factor(step: int, total: int, warmup: float = 0.1) -> float:
    w = max(int(total * warmup), 1)
    if step < w:
        return (step + 1) / w
    return 0.5 * (1 + float(np.cos(np.pi * (step - w) / max(total - w, 1))))


def train_model(model, slide_ids, *, steps: int, batch_size: int = 64, lr: float = 5e-5,
                weight_decay: float = 0.05, seed: int = 0, device=None, num_workers: int = 8,
                log_every: int = 100) -> dict:
    from torch.utils.data import DataLoader, WeightedRandomSampler

    device = device or get_device()
    table = patches_for(slide_ids)
    gen = torch.Generator().manual_seed(seed)
    sampler = WeightedRandomSampler(
        torch.as_tensor(sampling_weights(table), dtype=torch.double),
        num_samples=steps * batch_size, replacement=True, generator=gen)
    dl = DataLoader(PatchDataset(table, train=True), batch_size=batch_size, sampler=sampler,
                    num_workers=num_workers, drop_last=True, pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    amp = device.type == "cuda"
    model.to(device).train()
    losses, t0 = [], time.time()
    for step, (x, y, _) in enumerate(dl):
        for pg in opt.param_groups:
            pg["lr"] = lr * _lr_factor(step, steps)
        x, y = x.to(device, non_blocking=True), y.float().to(device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logit = model(x).squeeze(1)
        loss = loss_fn(logit.float(), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(float(loss))
        if log_every and (step + 1) % log_every == 0:
            print(f"krok {step + 1}/{steps}  loss {np.mean(losses[-log_every:]):.4f}")
    model.eval()
    return dict(n_steps=len(losses), final_train_loss=float(np.mean(losses[-50:])),
                train_time_s=time.time() - t0, n_train_patches=len(table))


# ------------------------------------------------------------------------ ewaluacja
@torch.inference_mode()
def add_predictions(model, table: pd.DataFrame, device=None, batch_size: int = 256,
                    num_workers: int = 8) -> pd.DataFrame:
    from torch.utils.data import DataLoader

    device = device or get_device()
    model.to(device).eval()
    dl = DataLoader(PatchDataset(table, train=False), batch_size=batch_size, shuffle=False,
                    num_workers=num_workers, pin_memory=device.type == "cuda")
    amp = device.type == "cuda"
    out = []
    for x, _, _ in dl:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=amp):
            logit = model(x.to(device, non_blocking=True)).squeeze(1)
        out.append(torch.sigmoid(logit.float()).cpu().numpy())
    res = table.reset_index(drop=True).copy()
    res["p"] = np.concatenate(out) if out else np.array([], dtype=np.float32)
    return res


def patch_metrics(y: np.ndarray, p: np.ndarray, thr: float = 0.5) -> dict:
    from sklearn.metrics import (average_precision_score, balanced_accuracy_score, f1_score,
                                 roc_auc_score)

    y = np.asarray(y).astype(int)
    pred = np.asarray(p) >= thr
    both = 0 < y.sum() < len(y)
    tp, fn = int((pred & (y == 1)).sum()), int((~pred & (y == 1)).sum())
    tn, fp = int((~pred & (y == 0)).sum()), int((pred & (y == 0)).sum())
    return dict(
        n=len(y), n_tumor=int(y.sum()),
        auroc=float(roc_auc_score(y, p)) if both else float("nan"),
        auprc=float(average_precision_score(y, p)) if both else float("nan"),
        f1=float(f1_score(y, pred, zero_division=0)),
        bal_acc=float(balanced_accuracy_score(y, pred)) if both else float("nan"),
        sens=tp / max(tp + fn, 1), spec=tn / max(tn + fp, 1),
    )


def score_table(tab: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    # tab: kolumny slide_id, label, p. Zwraca (metryki zbiorcze, tabela per slajd)
    pooled = patch_metrics(tab["label"].to_numpy(), tab["p"].to_numpy())
    per = []
    for sid, g in tab.groupby("slide_id"):
        m = patch_metrics(g["label"].to_numpy(), g["p"].to_numpy())
        per.append(dict(slide_id=sid, center=slide_meta(sid)[1], **m))
    per = pd.DataFrame(per)
    pooled["slide_mean_auroc"] = (
        float(np.nanmean(per["auroc"])) if per["auroc"].notna().any() else float("nan"))
    pooled["n_slides"] = len(per)
    return pooled, per


# ------------------------------------------------------------------ pętla AL (losowa)
@dataclass
class RandomALConfig:
    seed: int = 0
    initial: int = 2
    step: int = 2
    max_labeled: int | None = None
    steps: int = 1500
    batch_size: int = 64
    lr: float = 5e-5
    num_workers: int = 8
    per_class_eval: int = 1000
    stop_at_oracle: bool = True
    metric: str = "auroc"
    tol: float = 0.0


def round_sizes(cfg: RandomALConfig, n_pool: int) -> list[int]:
    top = min(cfg.max_labeled or n_pool, n_pool)
    sizes = list(range(min(cfg.initial, top), top + 1, cfg.step))
    if not sizes or sizes[-1] != top:
        sizes.append(top)
    return sizes


def run_random_al(cfg: RandomALConfig, pool: list[str], val: list[str], out_dir: Path,
                  oracle_val_metric: float, device=None) -> pd.DataFrame:
    device = device or get_device()
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    order = [pool[i] for i in rng.permutation(len(pool))]
    (out_dir / "order.json").write_text(json.dumps(order, indent=2))

    init_path = out_dir / "student_init.pt"
    if not init_path.exists():
        save_model(build_model(seed=cfg.seed), init_path)
    init_state = torch.load(init_path, map_location="cpu", weights_only=True)
    val_table = sample_eval_table(val, cfg.per_class_eval, seed=0)  # ten sam w każdej rundzie

    rows = []
    for n in round_sizes(cfg, len(pool)):
        rd = out_dir / f"round_{n:03d}"
        res_path = rd / "result.json"
        if res_path.exists():
            row = json.loads(res_path.read_text())
        else:
            rd.mkdir(parents=True, exist_ok=True)
            labeled = order[:n]
            model = build_model(pretrained=False)
            model.load_state_dict(init_state)  # cold start z inicjalizacji SAM/DINO
            info = train_model(model, labeled, steps=cfg.steps, batch_size=cfg.batch_size,
                               lr=cfg.lr, seed=cfg.seed * 1000 + n, device=device,
                               num_workers=cfg.num_workers)
            ckpt = rd / "model.pt"
            save_model(model, ckpt)
            tab = add_predictions(model, val_table, device, num_workers=cfg.num_workers)
            pooled, _ = score_table(tab)
            lab = patches_for(labeled)
            row = dict(n_labeled=n, labeled=labeled, n_labeled_patches=len(lab),
                       n_tumor_patches=int((lab["label"] == 1).sum()),
                       checkpoint=str(ckpt), **info,
                       **{f"val_{k}": v for k, v in pooled.items()})
            res_path.write_text(json.dumps(row, indent=2))
        rows.append(row)
        print(f"n={n}: val_{cfg.metric} = {row['val_' + cfg.metric]:.4f} "
              f"(wyrocznia {oracle_val_metric:.4f})")
        if cfg.stop_at_oracle and row["val_" + cfg.metric] >= oracle_val_metric - cfg.tol:
            break
    res = pd.DataFrame(rows)
    res.to_csv(out_dir / "results.csv", index=False)
    return res
