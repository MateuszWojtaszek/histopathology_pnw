"""Running a Cellpose model over images, with the same settings for every model."""

import numpy as np
from cellpose.models import CellposeModel
from tqdm.auto import tqdm

# one set of eval settings for oracle and student alike (Cellpose defaults, no rescaling)
EVAL_KWARGS = {
    "channel_axis": -1,
    "diameter": None,
    "flow_threshold": 0.4,
    "cellprob_threshold": 0.0,
}


def predict(
    model: CellposeModel, images: list[np.ndarray], batch_size: int = 8, progress: bool = True
) -> list[np.ndarray]:
    """Instance masks (uint16) for HxWx3 images."""
    masks = []
    for image in tqdm(images, desc="predict", disable=not progress):
        mask, _flows, _styles = model.eval(image, batch_size=batch_size, **EVAL_KWARGS)
        masks.append(np.asarray(mask).astype(np.uint16))
    return masks
