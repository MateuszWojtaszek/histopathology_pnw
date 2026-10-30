import pandas as pd

from histopathology_pnw.al.random_loop import preset, prune_checkpoints, round_sizes


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
