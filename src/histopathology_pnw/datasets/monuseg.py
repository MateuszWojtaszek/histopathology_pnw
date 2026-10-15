"""MoNuSeg: reading XML polygon annotations and turning them into instance masks.

Layout of the raw data (`data/raw/monuseg`, tracked by DVC):
- train: `MoNuSeg 2018 Training Data/Tissue Images/<id>.tif` + `.../Annotations/<id>.xml` (37 images)
- test:  `MoNuSegTestData/<id>.tif` + `MoNuSegTestData/<id>.xml` (14 images)

Masks are written to `data/processed/monuseg/<split>/<id>_masks.tif` (uint16, 0 = background,
1..N = nuclei). Run: `uv run python -m histopathology_pnw.datasets.monuseg` (or `uv run dvc repro`).
"""

import json
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from skimage.draw import polygon as draw_polygon
from skimage.segmentation import relabel_sequential

from histopathology_pnw.config import ROOT, Settings

SPLITS = ("train", "test")
SPLIT_FILE = ROOT / "configs" / "monuseg_split.json"
ORGANS_FILE = ROOT / "configs" / "monuseg_train_organs.csv"


@dataclass(frozen=True)
class Sample:
    image_id: str
    split: str  # official split: "train" or "test"
    image_path: Path
    xml_path: Path


@dataclass(frozen=True)
class MaskStats:
    n_polygons: int  # regions in the XML
    n_too_few_vertices: int  # polygons with < 3 vertices, skipped
    n_empty: int  # polygons that cover no pixel inside the image, skipped
    n_overlap_pixels: int  # pixels claimed by more than one polygon (the later polygon wins)
    n_lost: int  # instances completely covered by later polygons
    n_instances: int  # instances in the final mask


def raw_dir(settings: Settings | None = None) -> Path:
    return (settings or Settings()).data_dir / "raw" / "monuseg"


def processed_dir(settings: Settings | None = None) -> Path:
    return (settings or Settings()).data_dir / "processed" / "monuseg"


def list_split(split: str, root: Path | None = None) -> list[Sample]:
    """All samples of an official split, sorted by image id."""
    root = root or raw_dir()
    if split == "train":
        image_dir = root / "MoNuSeg 2018 Training Data" / "Tissue Images"
        xml_dir = root / "MoNuSeg 2018 Training Data" / "Annotations"
    elif split == "test":
        image_dir = xml_dir = root / "MoNuSegTestData"
    else:
        raise ValueError(f"unknown split {split!r}, expected one of {SPLITS}")

    samples = [
        Sample(p.stem, split, p, xml_dir / f"{p.stem}.xml") for p in sorted(image_dir.glob("*.tif"))
    ]
    if not samples:
        raise FileNotFoundError(f"no images in {image_dir}; run `uv run dvc pull`")
    missing = [s.xml_path for s in samples if not s.xml_path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing annotations: {missing}")
    return samples


def read_image(path: Path) -> np.ndarray:
    """H x W x 3 uint8 RGB image."""
    image = tifffile.imread(path)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError(f"{path}: expected HxWx3 uint8, got {image.shape} {image.dtype}")
    return image


def read_polygons(xml_path: Path) -> list[np.ndarray]:
    """One (N, 2) float array of (x, y) vertices per annotated nucleus, in file order."""
    root = ET.parse(xml_path).getroot()
    polygons = []
    for region in root.iter("Region"):
        vertices = region.find("Vertices")
        points = [] if vertices is None else vertices.findall("Vertex")
        polygons.append(
            np.array([[float(v.get("X")), float(v.get("Y"))] for v in points], dtype=np.float64)
        )
    return polygons


def polygons_to_instance_mask(
    polygons: list[np.ndarray], shape: tuple[int, int]
) -> tuple[np.ndarray, MaskStats]:
    """Rasterise polygons into a uint16 instance mask.

    Vertex (x, y) is column x, row y; vertices outside the image are clipped. Where polygons
    overlap, the later one in the file wins. Labels are consecutive 1..N.
    """
    mask = np.zeros(shape, dtype=np.int32)
    too_few = empty = overlap = 0
    label = 0
    for poly in polygons:
        if len(poly) < 3:
            too_few += 1
            continue
        rr, cc = draw_polygon(poly[:, 1], poly[:, 0], shape=shape)
        if rr.size == 0:
            empty += 1
            continue
        label += 1
        overlap += int(np.count_nonzero(mask[rr, cc]))
        mask[rr, cc] = label

    mask, _, _ = relabel_sequential(mask)
    n_instances = int(mask.max())
    stats = MaskStats(
        n_polygons=len(polygons),
        n_too_few_vertices=too_few,
        n_empty=empty,
        n_overlap_pixels=overlap,
        n_lost=label - n_instances,
        n_instances=n_instances,
    )
    return mask.astype(np.uint16), stats


def instance_mask(sample: Sample) -> tuple[np.ndarray, MaskStats]:
    """Ground-truth instance mask built directly from the XML (no cache)."""
    shape = tifffile.TiffFile(sample.image_path).pages[0].shape[:2]
    return polygons_to_instance_mask(read_polygons(sample.xml_path), shape)


def mask_path(sample: Sample, out_dir: Path | None = None) -> Path:
    return (out_dir or processed_dir()) / sample.split / f"{sample.image_id}_masks.tif"


def read_mask(sample: Sample, out_dir: Path | None = None) -> np.ndarray:
    """Preprocessed ground-truth mask; run `preprocess()` (or `dvc repro`) first."""
    path = mask_path(sample, out_dir)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `uv run dvc repro`")
    return tifffile.imread(path)


def preprocess(out_dir: Path | None = None) -> pd.DataFrame:
    """Write masks for both official splits and a manifest; return the manifest."""
    out_dir = out_dir or processed_dir()
    rows = []
    for split in SPLITS:
        (out_dir / split).mkdir(parents=True, exist_ok=True)
        for sample in list_split(split):
            mask, stats = instance_mask(sample)
            tifffile.imwrite(mask_path(sample, out_dir), mask, compression="zlib")
            rows.append({"image_id": sample.image_id, "split": split, **asdict(stats)})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(out_dir / "manifest.csv", index=False)
    return manifest


def organs() -> pd.DataFrame:
    """Organ per training image, from the PDF shipped with MoNuSeg (24 of 37 images).

    The PDF says "Liver" for the lung images; the `organ` column is corrected from the disease.
    """
    return pd.read_csv(ORGANS_FILE)


def make_train_val_split(seed: int = 0, n_val: int = 7) -> dict:
    """Random split of the official training images by image. Used once; the result is frozen
    in `configs/monuseg_split.json`. The official test split is never touched."""
    ids = [s.image_id for s in list_split("train")]
    rng = np.random.default_rng(seed)
    val = sorted(rng.choice(ids, size=n_val, replace=False).tolist())
    return {"seed": seed, "train": sorted(set(ids) - set(val)), "val": val}


def load_train_val_split() -> dict[str, list[str]]:
    return json.loads(SPLIT_FILE.read_text())


def samples_by_id(split: str) -> dict[str, Sample]:
    return {s.image_id: s for s in list_split(split)}


def main() -> None:
    manifest = preprocess()
    print(manifest.groupby("split")[["n_instances", "n_lost", "n_overlap_pixels"]].sum())
    print(f"masks written to {processed_dir()}")


if __name__ == "__main__":
    main()
