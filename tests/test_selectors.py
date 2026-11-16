import numpy as np
import pandas as pd

from histopathology_pnw.al.selectors import peek_table, pick, random_scores


def _two_squares() -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint16)
    mask[2:8, 2:8] = 1
    mask[10:16, 10:16] = 2
    return mask


def test_random_selector_follows_the_permutation():
    rank = {"c": 0, "a": 1, "d": 2, "b": 3}
    chosen, table = pick(random_scores(["a", "b", "d"], rank), 2, rank)
    assert chosen == ["a", "d"]
    assert table.set_index("image_id")["selected"].to_dict() == {"a": True, "b": False, "d": True}


def test_pick_takes_highest_scores_and_breaks_ties_by_permutation():
    rank = {"a": 2, "b": 0, "c": 1, "d": 3}
    scores = pd.DataFrame({"image_id": ["a", "b", "c", "d"], "score": [0.5, 0.5, 0.1, 0.9]})
    assert pick(scores, 3, rank)[0] == ["d", "b", "a"]
    assert pick(scores, 0, rank)[0] == []


def test_peek_ranks_the_worst_segmented_image_first():
    oracle = _two_squares()
    half = np.where(oracle == 1, 1, 0).astype(np.uint16)  # one of two nuclei found
    table = peek_table(
        ["perfect", "empty", "half"],
        [oracle, np.zeros_like(oracle), half],
        [oracle, oracle, oracle],
    )
    by_id = table.set_index("image_id")
    assert by_id["pq_oracle"].to_dict() == {"perfect": 1.0, "empty": 0.0, "half": 2 / 3}
    assert (by_id.loc["half", "tp"], by_id.loc["half", "fn"], by_id.loc["half", "fp"]) == (1, 1, 0)
    rank = {"perfect": 0, "empty": 1, "half": 2}
    assert pick(table, 2, rank)[0] == ["empty", "half"]
