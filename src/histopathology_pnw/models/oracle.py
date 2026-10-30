"""Oracle: frozen Cellpose-SAM `cpsam` (April 2025 release), inference only.

Weights: `models/oracle/cpsam` (DVC), from huggingface.co/mouseland/cellpose-sam at commit
7c61431b5fbb078f3296754bd15d9f51b320f837. Never load it by name and never use `cpsam_v2`.
"""

import os
import tempfile
from pathlib import Path

import numpy as np
import tifffile
import torch
from cellpose.models import CellposeModel

from histopathology_pnw.config import Settings
from histopathology_pnw.datasets.monuseg import Sample, read_image
from histopathology_pnw.device import get_device
from histopathology_pnw.inference import predict
from histopathology_pnw.models import load_cellpose, verify_file

ORACLE_SHA256 = "e1440429eb384f95afe32bcba6510f90d518eaedc917ede549bed6804004abe2"


def oracle_path(settings: Settings | None = None) -> Path:
    return (settings or Settings()).models_dir / "oracle" / "cpsam"


def load_oracle(device: torch.device | None = None) -> CellposeModel:
    path = oracle_path()
    verify_file(path, ORACLE_SHA256)
    return load_cellpose(path, device or get_device())


def oracle_masks(
    samples: list[Sample], cache_dir: Path | None = None, model: CellposeModel | None = None
) -> list[np.ndarray]:
    """Oracle predictions for `samples`, computed once and cached as `<cache_dir>/<id>.tif`.

    These are the "annotations" the oracle gives in the simulated AL loop.
    """
    cache_dir = cache_dir or Settings().outputs_dir / "oracle_masks" / "monuseg"
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{s.image_id}.tif" for s in samples]
    todo = [i for i, p in enumerate(paths) if not p.is_file()]
    if todo:
        model = model or load_oracle()
        preds = predict(model, [read_image(samples[i].image_path) for i in todo])
        for i, mask in zip(todo, preds, strict=True):
            # write a uniquely named file, then rename: parallel jobs (Slurm array, possibly on
            # different nodes of a shared filesystem) never read or share a half-written file
            fd, tmp = tempfile.mkstemp(dir=cache_dir, prefix=f".{paths[i].stem}.", suffix=".tif")
            os.close(fd)
            tifffile.imwrite(tmp, mask, compression="zlib")
            Path(tmp).replace(paths[i])
    return [tifffile.imread(p) for p in paths]
