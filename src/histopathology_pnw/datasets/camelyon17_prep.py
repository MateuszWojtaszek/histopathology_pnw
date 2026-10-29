# Przetwarzanie danych CAMELYON17 bez notebooka: maski guza/tkanki, podział po pacjencie, kafle.
# Umieść w: src/histopathology_pnw/datasets/camelyon17_prep.py
#
#   export C17_ROOT=/lustre/pd01/hpc-martinta-1783643755/alehen0076/camelyon17/raw
#   uv run python -m histopathology_pnw.datasets.camelyon17_prep --per-group 4
#
# To samo, co robi notebook 01 (maski + podział) i datasets.camelyon17 (kafle), w jednym poleceniu.
from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

try:  # libopenslide z pakietu openslide-bin, gdy nie ma jej w systemie
    import openslide_bin  # noqa: F401
except ImportError:
    pass
import openslide
from skimage.color import rgb2hsv
from skimage.draw import polygon as draw_polygon
from skimage.filters import threshold_otsu
from skimage.morphology import remove_small_holes, remove_small_objects

from histopathology_pnw.datasets import camelyon17 as c17

ANNOT = c17.DATA_ROOT / "lesion_annotations"
MASK_LEVEL = 5  # poziom piramidy dla masek (downsample ~32x)


def annotated_slides() -> list[str]:
    paths = c17.slide_paths()
    return sorted(s for s in paths if (ANNOT / f"{s}.xml").exists())


def read_polygons(xml_path):
    # lista (grupa, typ, tablica (N, 2) w pikselach poziomu 0)
    out = []
    for a in ET.parse(xml_path).getroot().iter("Annotation"):
        cs = sorted(a.iter("Coordinate"), key=lambda c: float(c.attrib.get("Order", 0)))
        xy = np.array([[float(c.attrib["X"]), float(c.attrib["Y"])] for c in cs]).reshape(-1, 2)
        out.append((a.attrib.get("PartOfGroup", "None").lower(), a.attrib.get("Type"), xy))
    return out


def polygons_to_masks(polys, shape, downsample, level0_size):
    tumor = np.zeros(shape, bool)
    normal = np.zeros(shape, bool)
    stats = dict(n_polygons=len(polys), n_too_few_vertices=0, n_metastases=0, n_normal=0,
                 n_out_of_bounds=0, n_empty=0)
    W, H = level0_size
    for group, _typ, xy in polys:
        if len(xy) < 3:
            stats["n_too_few_vertices"] += 1
            continue
        if (xy[:, 0] < 0).any() or (xy[:, 1] < 0).any() or (xy[:, 0] > W).any() or (xy[:, 1] > H).any():
            stats["n_out_of_bounds"] += 1
        rr, cc = draw_polygon(xy[:, 1] / downsample, xy[:, 0] / downsample, shape)
        if len(rr) == 0:
            stats["n_empty"] += 1
            continue
        if group == "normal":
            normal[rr, cc] = True
            stats["n_normal"] += 1
        else:
            tumor[rr, cc] = True
            stats["n_metastases"] += 1
    return tumor & ~normal, stats


def tissue_mask(rgb):
    sat = rgb2hsv(rgb)[..., 1]
    m = sat > threshold_otsu(sat)
    m = remove_small_objects(m, 64)
    return remove_small_holes(m, 64)


def read_level(slide, level):
    w, h = slide.level_dimensions[level]
    return np.asarray(slide.read_region((0, 0), level, (w, h)).convert("RGB"))


def build_masks(slide_ids: list[str], skip_existing: bool = False) -> pd.DataFrame:
    c17.PROCESSED.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, sid in enumerate(slide_ids, 1):
        patient, center = c17.slide_meta(sid)
        node = int(c17.SLIDE_RE.search(sid).group(2))
        out = c17.PROCESSED / f"{sid}.npz"
        if skip_existing and out.exists():
            z = np.load(out)
            rows.append(dict(slide_id=sid, patient=patient, node=node, center=center,
                             tissue_px=int(z["tissue"].sum()), tumor_px=int(z["tumor"].sum())))
            print(f"[{i}/{len(slide_ids)}] {sid}: pominięto (maska już jest)")
            continue
        with openslide.OpenSlide(str(c17.slide_paths()[sid])) as sl:
            level = min(MASK_LEVEL, sl.level_count - 1)
            ds = float(sl.level_downsamples[level])
            thumb = read_level(sl, level)
            polys = read_polygons(ANNOT / f"{sid}.xml")
            tumor, stats = polygons_to_masks(polys, thumb.shape[:2], ds, sl.dimensions)
            tissue = tissue_mask(thumb)
        np.savez_compressed(out, tumor=tumor, tissue=tissue, downsample=ds, level=level)
        mean_rgb = thumb[tissue].mean(axis=0) if tissue.any() else np.full(3, np.nan)
        rows.append(dict(
            slide_id=sid, patient=patient, node=node, center=center, mask_level=level, downsample=ds,
            **stats, tissue_px=int(tissue.sum()), tumor_px=int(tumor.sum()),
            tumor_outside_tissue_px=int((tumor & ~tissue).sum()),
            tumor_frac_of_tissue=float(tumor.sum() / max(tissue.sum(), 1)),
            mean_r=mean_rgb[0], mean_g=mean_rgb[1], mean_b=mean_rgb[2]))
        print(f"[{i}/{len(slide_ids)}] {sid}: guz {stats['n_metastases']} poligonów, "
              f"tkanka {int(tissue.sum())} px (poziom {level})")
    manifest = pd.DataFrame(rows)
    manifest.to_csv(c17.PROCESSED / "manifest.csv", index=False)
    return manifest


def make_split(manifest: pd.DataFrame, seed: int = 0, val_frac: float = 0.2, test_frac: float = 0.2) -> dict:
    # podział po pacjencie: <15 pacjentów globalnie, inaczej osobno w każdym ośrodku
    rng = np.random.default_rng(seed)
    val_p, test_p = set(), set()
    patients = sorted(manifest["patient"].unique())
    if len(patients) < 15:
        rng.shuffle(patients)
        if len(patients) >= 3:
            n_val = max(1, round(val_frac * len(patients)))
            n_test = max(1, round(test_frac * len(patients)))
            val_p = set(patients[:n_val])
            test_p = set(patients[n_val : n_val + n_test])
        elif len(patients) == 2:
            val_p = set(patients[:1])
    else:
        for _c, g in manifest.groupby("center"):
            pats = sorted(g["patient"].unique())
            rng.shuffle(pats)
            if len(pats) >= 3:
                n_val = max(1, round(val_frac * len(pats)))
                n_test = max(1, round(test_frac * len(pats)))
                val_p |= set(pats[:n_val])
                test_p |= set(pats[n_val : n_val + n_test])
            elif len(pats) == 2:
                val_p |= set(pats[:1])
    is_val = manifest["patient"].isin(val_p)
    is_test = manifest["patient"].isin(test_p)
    return dict(seed=seed, group="patient",
                train=sorted(manifest.loc[~is_val & ~is_test, "slide_id"]),
                val=sorted(manifest.loc[is_val, "slide_id"]),
                test=sorted(manifest.loc[is_test, "slide_id"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="Maski, podział i kafle CAMELYON17 (bez notebooka)")
    ap.add_argument("--per-group", type=int, default=4, help="kafli 'guz' i 'normalna' na slajd")
    ap.add_argument("--min-tissue", type=float, default=0.8)
    ap.add_argument("--skip-existing", action="store_true", help="nie licz ponownie istniejących masek")
    ap.add_argument("--keep-split", action="store_true", help="nie zmieniaj podziału, nawet gdy zbiór slajdów się zmienił")
    ap.add_argument("--force-tiles", action="store_true", help="wytnij kafle od nowa")
    ap.add_argument("--no-tiles", action="store_true", help="tylko maski i podział")
    a = ap.parse_args()

    slides = annotated_slides()
    print(f"dane: {c17.DATA_ROOT}\nslajdy z adnotacją XML: {len(slides)}")
    if not slides:
        raise SystemExit(f"Brak slajdów z XML. Sprawdź C17_ROOT i foldery images/ oraz lesion_annotations/ w {c17.DATA_ROOT}")

    manifest = build_masks(slides, skip_existing=a.skip_existing)

    split_changed = False
    if c17.SPLIT_PATH.exists():
        old = json.loads(c17.SPLIT_PATH.read_text())
        old_ids = set(old["train"]) | set(old["val"]) | set(old.get("test", []))
        if old_ids != set(slides) and not a.keep_split:
            print("zbiór slajdów się zmienił: tworzę nowy podział")
            split_changed = True
    else:
        split_changed = True
    if split_changed:
        split = make_split(manifest)
        c17.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        c17.SPLIT_PATH.write_text(json.dumps(split, indent=2))
    else:
        split = json.loads(c17.SPLIT_PATH.read_text())
    print("podział (slajdy):", {k: len(split.get(k, [])) for k in ("train", "val", "test")})

    if not a.no_tiles:
        tiles = c17.extract_tiles(per_group=a.per_group, min_tissue=a.min_tissue,
                                  force=a.force_tiles or split_changed)
        print(f"kafle: {len(tiles)}")
        print(tiles.groupby(["split", "group"]).size().unstack(fill_value=0))
    print("gotowe. Maski i kafle:", c17.PROCESSED)


if __name__ == "__main__":
    main()
