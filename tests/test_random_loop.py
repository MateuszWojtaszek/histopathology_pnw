import json

import numpy as np
import pandas as pd
import pytest

from histopathology_pnw.al.random_loop import (
    RandomALConfig,
    preset,
    prune_checkpoints,
    resume_state,
    round_sizes,
    select_next,
)


def _rounds(out_dir, val_pq: dict[int, float]) -> pd.DataFrame:
    """Fake finished rounds: a checkpoint and a metrics file per round."""
    for n in val_pq:
        models = out_dir / f"round_{n:02d}" / "models"
        models.mkdir(parents=True)
        (models / "student").write_bytes(b"weights")
        (out_dir / f"round_{n:02d}" / "val_metrics.csv").write_text("pq\n0.5\n")
    return pd.DataFrame({"n_labeled": list(val_pq), "val_pq": list(val_pq.values())})


def _kept(out_dir) -> list[str]:
    return sorted(p.parent.name for p in out_dir.glob("round_*/models"))


def test_prune_keeps_only_the_best_round(tmp_path):
    prune_checkpoints(_rounds(tmp_path, {2: 0.50, 4: 0.60, 6: 0.55}), tmp_path)
    assert _kept(tmp_path) == ["round_04"]
    assert len(list(tmp_path.glob("round_*/val_metrics.csv"))) == 3


def test_prune_tie_keeps_the_earlier_round(tmp_path):
    prune_checkpoints(_rounds(tmp_path, {2: 0.60, 4: 0.60}), tmp_path)
    assert _kept(tmp_path) == ["round_02"]


def test_full_preset_is_the_reported_baseline():
    cfg = preset("full", seed=3)
    assert cfg.seed == 3
    assert (cfg.n_epochs, cfg.nimg_per_epoch, cfg.batch_size) == (200, 64, 4)
    assert (cfg.learning_rate, cfg.weight_decay) == (5e-5, 0.1)
    assert round_sizes(cfg, 30) == list(range(2, 31, 2))


def test_random_selector_reproduces_the_nested_sets_of_the_baseline():
    # the loop's bookkeeping (start from order[:initial], then add what the selector picks)
    # must give exactly order[:n], the labelled sets of the baseline runs
    cfg = preset("full", seed=1)
    ids = [f"img{i:02d}" for i in range(30)]
    order = [ids[i] for i in np.random.default_rng(cfg.seed).permutation(len(ids))]
    rank = {image_id: r for r, image_id in enumerate(order)}
    sizes = round_sizes(cfg, len(ids))
    labeled, selected = [], order[: cfg.initial]
    for idx, n in enumerate(sizes):
        labeled = labeled + selected
        assert labeled == order[:n]
        k = sizes[idx + 1] - n if idx + 1 < len(sizes) else 0
        selected, table = select_next(cfg, None, labeled, k, rank, {}, {})
        assert len(table) == len(ids) - n
    assert selected == []


def test_resume_state_reads_the_last_round():
    results = pd.DataFrame(
        {
            "n_labeled": [2, 4],
            "labeled_ids": [json.dumps(["a", "b"]), json.dumps(["a", "b", "c", "d"])],
            "selected_next": [json.dumps(["c", "d"]), json.dumps(["e", "f"])],
        }
    )
    assert resume_state(results) == (["a", "b", "c", "d"], ["e", "f"])
    # results written before selectors existed: nothing selected yet
    assert resume_state(results.drop(columns="selected_next")) == (["a", "b", "c", "d"], [])


def test_unknown_selector_is_rejected():
    with pytest.raises(ValueError, match="selector"):
        RandomALConfig(selector="fuzzy")
