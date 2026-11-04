import numpy as np
import pytest
import mir_eval
from gmv.evaluation.metrics import onset_score
from gmv.evaluation.corpus import generate


def test_match_is_one_to_one_with_duplicates():
    result = onset_score([1, 2], [1, 1.01, 2.02, 4])
    assert (result["tp"], result["fp"], result["fn"], result["duplicates"]) == (2, 2, 0, 1)
    f, p, r = mir_eval.onset.f_measure(np.array([1.0, 2]), np.array([1.0, 1.01, 2.02, 4]))
    assert result["f1"] == pytest.approx(f) and result["precision"] == p and result["recall"] == r


def test_empty_and_strict_tolerance():
    assert onset_score([], [])["correct_empty"]
    assert onset_score([], [1])["fp"] == 1
    assert onset_score([1], [])["fn"] == 1
    assert onset_score([1], [1.04], 0.025)["tp"] == 0
    assert onset_score([1], [1.04], 0.05)["tp"] == 1


def test_corpus_reproducible_and_split_patterns_differ():
    a, b = next(generate()), next(generate())
    holdout = next(generate("holdout"))
    assert a.manifest()["sha256_float32_le"] == b.manifest()["sha256_float32_le"]
    assert a.onsets != holdout.onsets
    simultaneous = next(f for f in generate() if f.category == "simultaneous")
    assert len(simultaneous.onsets) * 3 == len(simultaneous.attacks)
