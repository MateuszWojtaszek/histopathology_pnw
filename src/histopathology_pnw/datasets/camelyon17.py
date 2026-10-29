# Kafle CAMELYON17 do segmentacji jąder.
# Umieść w: src/histopathology_pnw/datasets/camelyon17.py
#
# CAMELYON17 nie ma masek jąder. Tniemy slajdy (te z XML, z notebooka 01) na kafle 1000x1000 px
# (jak w MoNuSeg, poziom 0), a "adnotatorem" jąder jest wyrocznia cpsam (patrz al/camelyon17_random_loop.py).
# Maski guza z XML służą tylko do stratyfikacji kafli (guz / normalna) i analizy.
# Do DVC trafiają kafle (data/processed/camelyon17/tiles) i tiles_manifest.csv, nie surowe slajdy.
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

# src/histopathology_pnw/datasets/camelyon17.py -> parents[3] = korzeń repo
ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = Path(os.environ.get("C17_ROOT", ROOT / "data" / "raw" / "camelyon17"))
IMAGES = DATA_ROOT / "images"
PROCESSED = Path(os.environ.get("C17_PROCESSED", ROOT / "data" / "processed" / "camelyon17"))
TILES_DIR = PROCESSED / "tiles"
MANIFEST = PROCESSED / "tiles_manifest.csv"
PARAMS = PROCESSED / "tiles_params.json"
SPLIT_PATH = ROOT / "configs" / "camelyon17_split.json"

TILE = 1000  # jak w MoNuSeg [px, poziom 0]
SLIDE_RE = re.compile(r"patient_(\d+)_node_(\d+)")


@lru_cache(maxsize=1)
def slide_paths() -> dict[str, Path]:
    return {p.stem: p for p in sorted(IMAGES.rglob("*.tif")) if SLIDE_RE.search(p.stem)}


def slide_meta(slide_id: str) -> tuple[int, int]:
    # (patient, center); 20 pacjentów na ośrodek
    patient = int(SLIDE_RE.search(slide_id).group(1))
    return patient, patient // 20


def ensure_test_split(path: Path = SPLIT_PATH, seed: int = 1, frac: float = 0.2) -> dict:
    # Notebook 01 zamraża train/val po pacjencie. Tu (raz) wydzielamy test z puli train,
    # po pacjencie i osobno w każdym ośrodku. Jeśli klucz "test" już istnieje, nic nie zmieniamy.
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
    z = np.load(PROCESSED / f"{slide_id}.npz")  # maski z notebooka 01
    return z["tumor"], z["tissue"], float(z["downsample"])


def tile_grid(tumor: np.ndarray, tissue: np.ndarray, ds: float, tile: int = TILE) -> pd.DataFrame:
    # siatka nienakładających się okien ~tile px (poziom 0) liczona na maskach niskiej rozdzielczości
    k = max(int(round(tile / ds)), 1)
    h, w = (tumor.shape[0] // k) * k, (tumor.shape[1] // k) * k
    t = tumor[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    s = tissue[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    r, c = np.meshgrid(np.arange(h // k), np.arange(w // k), indexing="ij")
    r, c = r.ravel(), c.ravel()
    return pd.DataFrame(dict(
        row=r, col=c, x0=(c * k * ds).astype(int), y0=(r * k * ds).astype(int),
        tissue_frac=s.ravel(), tumor_frac=t.ravel()))


def extract_tiles(per_group: int = 4, min_tissue: float = 0.8, seed: int = 0,
                  slides: list[str] | None = None, force: bool = False) -> pd.DataFrame:
    # Wycina do `per_group` kafli "guz" (>=50% guza) i `per_group` kafli "normalna" (0% guza)
    # z każdego slajdu, tylko z gęstą tkanką (>= min_tissue). Wynik: PNG + tiles_manifest.csv.
    import openslide
    from PIL import Image

    params = dict(per_group=per_group, min_tissue=min_tissue, seed=seed, tile=TILE,
                  slides=sorted(slides) if slides else None)
    if MANIFEST.exists() and not force:
        same = PARAMS.exists() and json.loads(PARAMS.read_text()) == params
        if same or not slide_paths():  # bez surowych slajdów używamy gotowych kafli
            return pd.read_csv(MANIFEST)

    split = ensure_test_split()
    role = {s: name for name in ("train", "val", "test") for s in split[name]}
    TILES_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for sid in sorted(role):
        if slides is not None and sid not in slides:
            continue
        if not (PROCESSED / f"{sid}.npz").exists() or sid not in slide_paths():
            continue
        tumor, tissue, ds = _load_masks(sid)
        grid = tile_grid(tumor, tissue, ds)
        grid = grid[grid["tissue_frac"] >= min_tissue]
        patient, center = slide_meta(sid)
        rng = np.random.default_rng(seed * 100003 + patient * 10 + int(SLIDE_RE.search(sid).group(2)))
        with openslide.OpenSlide(str(slide_paths()[sid])) as sl:
            W, H = sl.dimensions
            ok = (grid["x0"] + TILE <= W) & (grid["y0"] + TILE <= H)
            for group, cond in (("guz", grid["tumor_frac"] >= 0.5), ("normalna", grid["tumor_frac"] == 0)):
                cand = grid[ok & cond]
                if len(cand) > per_group:
                    cand = cand.iloc[np.sort(rng.choice(len(cand), per_group, replace=False))]
                for _, c in cand.iterrows():
                    x0, y0 = int(c["x0"]), int(c["y0"])
                    tile_id = f"{sid}_x{x0}_y{y0}"
                    path = TILES_DIR / f"{tile_id}.png"
                    if not path.exists():
                        img = sl.read_region((x0, y0), 0, (TILE, TILE)).convert("RGB")
                        Image.fromarray(np.asarray(img)).save(path)
                    rows.append(dict(tile_id=tile_id, slide_id=sid, patient=patient, center=center,
                                     split=role[sid], group=group, x0=x0, y0=y0,
                                     tissue_frac=float(c["tissue_frac"]), tumor_frac=float(c["tumor_frac"])))
    manifest = pd.DataFrame(rows)
    manifest.to_csv(MANIFEST, index=False)
    PARAMS.write_text(json.dumps(params, indent=2))
    return manifest


def list_tiles(split: str | None = None, group: str | None = None) -> pd.DataFrame:
    m = pd.read_csv(MANIFEST)
    if split is not None:
        m = m[m["split"] == split]
    if group is not None:
        m = m[m["group"] == group]
    return m.reset_index(drop=True)


def tile_path(tile_id: str) -> Path:
    return TILES_DIR / f"{tile_id}.png"


def read_tile(tile_id: str) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(tile_path(tile_id)).convert("RGB"))


def main() -> None:
    # uv run python -m histopathology_pnw.datasets.camelyon17 --per-group 4   (stage dvc)
    import argparse

    ap = argparse.ArgumentParser(description="Wycinanie kafli CAMELYON17")
    ap.add_argument("--per-group", type=int, default=4)
    ap.add_argument("--min-tissue", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    m = extract_tiles(a.per_group, a.min_tissue, a.seed, force=a.force)
    print(m.groupby(["split", "group"]).size())


if __name__ == "__main__":
    main()
