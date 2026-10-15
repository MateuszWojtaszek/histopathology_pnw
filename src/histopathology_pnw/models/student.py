"""Student: Cellpose-SAM architecture whose encoder starts from the original SAM ViT-L weights.

Never initialised from a Cellpose checkpoint. `cellpose.vit.CPSAM.load_pretrained()` cannot be
used: it replaces the adapted encoder with an unmodified SAM encoder (16 px patches, 64x64
position embeddings), which does not match the rest of CPSAM. Instead the SAM encoder weights
are adapted the same way `CPSAM.__init__` adapts its (random) encoder and loaded with
`strict=True`, so any missing or unexpected key is an error.
"""

from pathlib import Path

import torch
from cellpose.models import CellposeModel
from cellpose.vit import CPSAM
from torch.utils.checkpoint import checkpoint

from histopathology_pnw.config import Settings
from histopathology_pnw.device import get_device
from histopathology_pnw.models import load_cellpose, verify_file

# https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth, 1 249 524 607 bytes
SAM_VIT_L_SHA256 = "3adcc4315b642a4d2101128f611684e8734c41232a17c648ed1693702a49a622"


def sam_checkpoint_path(settings: Settings | None = None) -> Path:
    return (settings or Settings()).models_dir / "student_init" / "sam_vit_l_0b3195.pth"


def build_student_init(seed: int, out_path: Path) -> Path:
    """Write a Cellpose-SAM checkpoint: SAM ViT-L encoder + randomly initialised readout.

    The readout (`out`) is random, seeded with `seed`. The file is a normal Cellpose 4 checkpoint,
    so it loads with `load_student`. Returns `out_path`.
    """
    sam_path = sam_checkpoint_path()
    verify_file(sam_path, SAM_VIT_L_SHA256)

    torch.manual_seed(seed)
    net = CPSAM(dtype=torch.float32)

    sam = torch.load(sam_path, map_location="cpu", weights_only=True)
    prefix = "image_encoder."
    encoder = {k.removeprefix(prefix): v for k, v in sam.items() if k.startswith(prefix)}

    # same adaptation as CPSAM.__init__: 8 px patches and position embeddings for 256 px input
    patch = encoder["patch_embed.proj.weight"]
    stride = patch.shape[-1] // net.ps
    encoder["patch_embed.proj.weight"] = patch[:, :, ::stride, ::stride].contiguous()
    pos = encoder["pos_embed"]
    ds = pos.shape[1] // net.encoder.pos_embed.shape[1]
    encoder["pos_embed"] = pos[:, ::ds, ::ds].contiguous()

    net.encoder.load_state_dict(encoder, strict=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_model(out_path)
    return out_path


def load_student(path: Path, device: torch.device | None = None) -> CellposeModel:
    return load_cellpose(path, device or get_device())


def enable_gradient_checkpointing(net: CPSAM) -> None:
    """Recompute encoder block activations in the backward pass instead of storing them.

    Cellpose trains in float32 and every ViT-L block keeps several 1024x1024 attention maps per
    head for backward, so a batch of 4 crops does not fit in 16 GB. With checkpointing only the
    block inputs are kept (~30% slower). Only the instances' `forward` is replaced, so parameter
    names and saved checkpoints are unchanged; without gradients (inference) nothing changes.
    """
    for block in net.encoder.blocks:
        plain_forward = block.forward

        def forward(x: torch.Tensor, _plain=plain_forward) -> torch.Tensor:
            if torch.is_grad_enabled():
                return checkpoint(_plain, x, use_reentrant=False)
            return _plain(x)

        block.forward = forward
