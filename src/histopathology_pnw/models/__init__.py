"""Model loading: the frozen oracle and the student (shared helpers below)."""

import hashlib
from pathlib import Path

import torch
from cellpose.models import CellposeModel


def sha256(path: Path, chunk_size: int = 1 << 24) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def verify_file(path: Path, expected_sha256: str) -> None:
    """Fail loudly if a weights file is missing or not byte-identical to the pinned one."""
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found; run `uv run dvc pull`")
    actual = sha256(path)
    if actual != expected_sha256:
        raise ValueError(f"{path}: SHA-256 {actual} != expected {expected_sha256}")


def load_cellpose(path: Path, device: torch.device) -> CellposeModel:
    """Load a Cellpose-SAM checkpoint by path. The path must exist: Cellpose silently falls back
    to `cpsam_v2` for a missing path. bfloat16 only on CUDA, float32 elsewhere."""
    if not path.is_file():
        raise FileNotFoundError(path)
    return CellposeModel(
        pretrained_model=str(path), device=device, use_bfloat16=device.type == "cuda"
    )
