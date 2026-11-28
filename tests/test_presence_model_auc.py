"""A5.2 Step 4, S4.5: ROC-AUC and PR-AUC (protocol §11; Amendment 3, A3.5 B20), on generated
scores only."""

import itertools
from fractions import Fraction

import numpy as np
import pytest

from gmv.character.research.presence_model import auc, scoring

CENTRES = np.arange(-50, 500) * 0.02 + 0.0232


def brute_roc(scores, labels):
    pos = [s for s, y in zip(scores, labels, strict=True) if y]
    neg = [s for s, y in zip(scores, labels, strict=True) if not y]
    wins = sum(Fraction(1) if p > n else Fraction(1, 2) if p == n else Fraction(0) for p, n in itertools.product(pos, neg))
    return wins / (len(pos) * len(neg))


def brute_ap(scores, labels):
    """AP by its definition: thresholds at each distinct score, highest first."""
    n1 = sum(labels)
    total, previous = Fraction(0), Fraction(0)
    for threshold in sorted(set(scores), reverse=True):
        tp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and y)
        fp = sum(1 for s, y in zip(scores, labels, strict=True) if s >= threshold and not y)
        recall = Fraction(tp, n1)
        total += (recall - previous) * Fraction(tp, tp + fp)
        previous = recall
    return total


@pytest.mark.parametrize("seed", range(6))
def test_roc_and_ap_equal_their_definitions_with_ties(seed):
    rng = np.random.default_rng(seed)
    scores = np.round(rng.random(60), 1)  # many ties
    labels = rng.random(60) < 0.4
    labels[:2] = [True, False]
    assert auc.roc_auc(scores, labels) == brute_roc(list(scores), list(labels))
    assert auc.average_precision(scores, labels) == brute_ap(list(scores), list(labels))


def test_known_values():
    assert auc.roc_auc([0.1, 0.2, 0.3, 0.4], [False, False, True, True]) == 1
    assert auc.roc_auc([0.5, 0.5], [True, False]) == Fraction(1, 2)  # a tie counts half
    assert auc.average_precision([0.5, 0.5, 0.5, 0.5], [True, False, False, False]) == Fraction(1, 4)  # one step
    assert auc.average_precision([0.9, 0.8, 0.7], [True, False, True]) == Fraction(1, 2) + Fraction(1, 2) * Fraction(2, 3)


def scored(group, scores, confirmed=((2.0, 6.0),), silent=None):
    return scoring.Scored(group, CENTRES, 10.0, tuple(confirmed), (),
                          np.zeros(len(CENTRES), bool) if silent is None else silent, np.asarray(scores, float))


def test_quiet_frames_are_scored_zero_and_collars_and_pre_roll_are_left_out():
    rng = np.random.default_rng(1)
    s = rng.random(len(CENTRES))
    silent = np.zeros(len(CENTRES), bool)
    silent[300:320] = True
    sc, lab = auc.frames(scored("g", s, silent=silent))
    assert (sc == 0).sum() == 20 and len(sc) == 500 - 2 * 30  # two interior boundaries, 30 frames each
    assert lab.sum() == 200 - 15 * 2  # half of each collar lies inside the voice


def test_eligibility_needs_two_seconds_of_each_class():
    rng = np.random.default_rng(2)
    values = auc.group_values([scored("both", rng.random(len(CENTRES))),
                               scored("short", rng.random(len(CENTRES)), confirmed=((2.0, 3.5),)),
                               scored("none", rng.random(len(CENTRES)), confirmed=())])
    assert sorted(values) == ["both"]


def test_the_macro_is_over_eligible_groups_per_seed_then_over_o():
    rng = np.random.default_rng(3)
    per_seed = {s: auc.group_values([scored(g, rng.random(len(CENTRES))) for g in ("a", "b", "c")]) for s in range(5)}
    m = auc.macro(per_seed)
    expected = np.mean([np.mean([float(per_seed[s][g].roc) for g in "abc"]) for s in range(5)])
    assert m.roc == pytest.approx(expected, rel=1e-12) and m.eligible_groups == 3
    assert m.mean_voiced_share == pytest.approx(float(per_seed[0]["a"].prevalence))
    assert set(auc.public(m)) == {"roc_auc_macro", "pr_auc_macro", "eligible_groups", "mean_voiced_share"}


def test_b0_and_b0t_share_one_value_because_no_threshold_enters():
    rng = np.random.default_rng(4)
    s = rng.random(len(CENTRES))
    b0 = auc.macro({0: auc.group_values([scored("a", s)])})
    b0t = auc.macro({0: auc.group_values([scored("a", s)])})
    assert b0 == b0t


def test_disagreeing_eligibility_or_no_eligible_group_is_refused():
    rng = np.random.default_rng(5)
    a = auc.group_values([scored("a", rng.random(len(CENTRES)))])
    with pytest.raises(ValueError):
        auc.macro({0: a, 1: {}})
    with pytest.raises(ValueError):
        auc.macro({0: {}})
