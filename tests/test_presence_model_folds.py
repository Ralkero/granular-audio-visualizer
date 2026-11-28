"""A5.2 Stage 1, protocol §16 "Folds": the class-aware inner folds of §9. Synthetic group
records only (``group_001`` …); never a real group key. These are test fixtures, not the
project's fold files."""

import json
import math

import numpy as np
import pytest

from gmv.character.research.presence_model import folds


def synthetic(seed, n=18, novoice_only=3, voice_only=1):
    """Generated (voiced_s, novoice_s) per synthetic group."""
    rng = np.random.default_rng(seed)
    groups = {}
    for i in range(n):
        key = f"group_{i + 1:03d}"
        if i < novoice_only:
            groups[key] = (0.0, float(rng.uniform(20, 180)))
        elif i < novoice_only + voice_only:
            groups[key] = (float(rng.uniform(20, 80)), 0.0)
        else:
            groups[key] = (float(rng.uniform(5, 110)), float(rng.uniform(10, 60)))
    return groups


def members(assignment, k=folds.FOLDS):
    return [[g for g, f in assignment.folds.items() if f == i] for i in range(k)]


@pytest.mark.parametrize("seed", range(8))
def test_assignments_satisfy_every_registered_check(seed):
    groups = synthetic(seed)
    result = folds.assign(groups)
    assert sorted(result.folds) == sorted(groups)  # every group exactly once
    assert set(result.folds.values()) <= set(range(folds.FOLDS))
    assert folds.check(groups, members(result)) == []
    total_v = sum(v for v, _ in groups.values())
    total_n = sum(n for _, n in groups.values())
    for fold in members(result):
        assert len(fold) <= folds.CAPACITY
        assert total_v - sum(groups[g][0] for g in fold) >= 0.65 * total_v - 1e-9
        assert total_n - sum(groups[g][1] for g in fold) >= 0.65 * total_n - 1e-9
    for kind in ("voice_only", "novoice_only"):
        n_type = sum(1 for v, n in groups.values() if folds.group_type(v, n) == kind)
        for fold in members(result):
            assert sum(1 for g in fold if folds.group_type(*groups[g]) == kind) <= math.ceil(n_type / folds.FOLDS)


def test_the_same_inputs_give_the_same_assignment_and_fingerprint():
    groups = synthetic(3)
    first, second = folds.assign(groups), folds.assign(dict(reversed(list(groups.items()))))
    assert first.folds == second.folds and first.attempt == second.attempt
    assert folds.fingerprint(first) == folds.fingerprint(second)
    assert folds.serialise(first) == folds.serialise(second)
    other = folds.assign(groups, seed="another-seed")
    assert folds.check(groups, members(other)) == []


def test_single_class_groups_are_spread_before_two_class_groups_fill_the_folds():
    groups = synthetic(5, novoice_only=4, voice_only=2)
    result = folds.assign(groups)
    for kind in ("voice_only", "novoice_only"):
        counts = [sum(1 for g in fold if folds.group_type(*groups[g]) == kind) for fold in members(result)]
        assert max(counts) <= 1


def test_the_outer_group_is_never_assigned():
    groups = synthetic(2, n=19)
    for outer in groups:
        result = folds.for_outer(groups, outer)
        assert outer not in result.folds and len(result.folds) == 18
    with pytest.raises(ValueError):
        folds.for_outer(groups, "group_999")


def test_retries_are_deterministic_and_recorded():
    """A stricter share (a test-only setting; the registered value is 0.65) makes the first
    placement fail, so the deterministic retry order is exercised."""
    groups = synthetic(4)
    first = folds.assign(groups, share=0.72)
    again = folds.assign(groups, share=0.72)
    assert first.attempt == again.attempt == 3
    assert first.folds == again.folds
    assert folds.check(groups, members(first), share=0.72) == []
    assert first.summary["attempt"] == 3


def test_an_impossible_assignment_stops_instead_of_relaxing():
    # One group holds most of the voiced time: whichever fold holds it, the training
    # partition without that fold keeps less than 65% of the voiced time.
    groups = synthetic(1)
    groups["group_001"] = (5000.0, 30.0)
    with pytest.raises(folds.FoldAssignmentError) as caught:
        folds.assign(groups)
    assert "100 attempts" in str(caught.value) and "group_" not in str(caught.value)
    with pytest.raises(folds.FoldAssignmentError):  # 21 groups exceed 5 folds of 4
        folds.assign(synthetic(0, n=21))
    with pytest.raises(folds.FoldAssignmentError):  # nothing voiced at all: cannot balance
        folds.assign({f"group_{i:03d}": (0.0, 30.0) for i in range(5)})


def test_public_summaries_hold_aggregates_but_no_group_keys():
    groups = synthetic(6)
    result = folds.assign(groups)
    text = json.dumps(result.summary)
    assert "group_" not in text
    assert sum(row["groups"] for row in result.summary["folds"]) == len(groups)
    assert sum(row["voiced_s"] for row in result.summary["folds"]) == pytest.approx(sum(v for v, _ in groups.values()), abs=1e-2)
    for row in result.summary["folds"]:
        assert 0.65 <= row["training_share_voiced"] <= 1 and 0.65 <= row["training_share_novoice"] <= 1


def test_failure_messages_hold_no_group_keys():
    groups = synthetic(4)
    failures = folds.check(groups, [list(groups)[:10], list(groups)[10:], [], [], []])
    assert failures and not any("group_" in f for f in failures)


def test_invalid_records_are_refused():
    with pytest.raises(ValueError):
        folds.assign({})
    with pytest.raises(ValueError):
        folds.assign({"group_001": (-1.0, 10.0), "group_002": (10.0, 10.0)})
    with pytest.raises(ValueError):
        folds.assign({"group_001": (float("nan"), 10.0)})
    with pytest.raises(ValueError):  # under 2 s of both classes: the protocol does not say where it goes
        folds.assign({"group_001": (1.0, 1.5), "group_002": (30.0, 30.0)})


def test_group_types_use_the_inclusive_two_second_minimum():
    assert folds.group_type(2.0, 2.0) == "both"
    assert folds.group_type(2.0, 1.99) == "voice_only"
    assert folds.group_type(0.0, 2.0) == "novoice_only"
