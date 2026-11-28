"""A5.2 Step 4, S4.3: the ``horizon`` ledger and B1's matched-start check (protocol Amendment 3,
A3.3; A3.10 "Seeds"). Generated frames only; model C's matched-start tests need PyTorch and are in
``test_presence_model_ctrain.py``."""

import copy
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from test_presence_model_b1train import data  # noqa: E402

from gmv.character.research.presence_model import b1train, matched, seedplan  # noqa: E402

SOURCE = "gmv-a5.2-gate:C:outer-0:conv:0"


@pytest.fixture(scope="module")
def production_plan():
    return seedplan.plan("C")


@pytest.fixture
def ledger(production_plan):
    return seedplan.RunLedger(production_plan)


def test_there_are_16_declared_relations_on_the_four_gate_sources():
    relations = seedplan.horizon_relations()
    assert len(relations) == 16
    assert relations["H-C-outer-0:conv:0-45"] == (SOURCE, 30, 45)
    assert relations["H-C-outer-0:conv:0-68"] == (SOURCE, 45, 68)
    assert relations["H-B1-inner-0:conv:1-135"] == ("gmv-a5.2-gate:B1:inner-0:conv:1", 90, 135)
    assert {b for b, _s, _l in relations.values()} == {f"gmv-a5.2-gate:{a}:{s}" for a in ("C", "B1")
                                                        for s in seedplan.HORIZON_SOURCES}


def test_a_declared_relation_runs_after_its_source_at_e_and_once(ledger):
    ledger.claim_production("gate-C-1", SOURCE, epochs=30)
    ledger.claim_horizon("H-C-outer-0:conv:0-45", SOURCE, 45)
    ledger.claim_horizon("H-C-outer-0:conv:0-68", SOURCE, 68)
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_horizon("H-C-outer-0:conv:0-45", SOURCE, 45)


def test_a_relation_before_its_source_is_refused(ledger):
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_horizon("H-C-outer-0:conv:0-45", SOURCE, 45)


def test_a_source_that_ran_at_another_length_is_refused(ledger):
    ledger.claim_production("gate-C-1", SOURCE, epochs=31)
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_horizon("H-C-outer-0:conv:0-45", SOURCE, 45)


def test_the_e_plus_plus_run_needs_the_e_plus_run_first(ledger):
    ledger.claim_production("gate-C-1", SOURCE, epochs=30)
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_horizon("H-C-outer-0:conv:0-68", SOURCE, 68)


@pytest.mark.parametrize("relation, text, epochs", (
    ("H-C-outer-0:conv:9-45", SOURCE, 45),                       # undeclared
    ("H-C-outer-0:conv:0-45", "gmv-a5.2-gate:C:outer-0:conv:1", 45),  # not the matching source
    ("H-C-outer-0:conv:0-45", SOURCE, 46),                       # not the relation's length
))
def test_mismatched_claims_are_refused(ledger, relation, text, epochs):
    ledger.claim_production("gate-C-1", SOURCE, epochs=30)
    ledger.claim_production("gate-C-2", "gmv-a5.2-gate:C:outer-0:conv:1", epochs=30)
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_horizon(relation, text, epochs)


def test_only_gate_identities_have_relations_and_other_reuse_stops(ledger, production_plan):
    registered = "gmv-a5.2:C:0:outer:0"
    ledger.claim_production("outer", registered, epochs=30)
    forged = {"H-X": (registered, 30, 45)}
    other = seedplan.RunLedger(production_plan, horizons=forged)
    other.claim_production("outer", registered, epochs=30)
    with pytest.raises(seedplan.ReplayError):
        other.claim_horizon("H-X", registered, 45)  # a registered identity can never be paired
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_production("again", registered, epochs=45)


# -- B1's matched-start check ----------------------------------------------------------------------


def b1_pair(short=4, long=6, identity="gmv-a5.2-gate:B1:outer-0:conv:0"):
    d = data()
    at = matched.b1_epoch(short)
    a = b1train.train(identity, d, epochs=short, batch=256, snapshot_after=(at,))
    b = b1train.train(identity, d, epochs=long, batch=256, snapshot_after=(at,))
    return a, b


def test_a_genuine_b1_pair_passes_and_parts_after_the_first_half():
    a, b = b1_pair()
    matched.check_b1(a, b)
    assert matched.b1_epoch(60) == 30 and matched.b1_epoch(90) == 45
    assert not np.array_equal(a.params["w1"], b.params["w1"])  # the schedules differ afterwards


def test_a_perturbed_initialisation_fails():
    a, b = b1_pair()
    b = copy.deepcopy(b)
    b.initial["w1"][0, 0] += 1e-3
    with pytest.raises(matched.MatchedStartError):
        matched.check_b1(a, b)


def test_a_perturbed_data_order_fails():
    a, b = b1_pair()
    b = copy.deepcopy(b)
    b.order_sha256[1] = "0" * 64
    with pytest.raises(matched.MatchedStartError):
        matched.check_b1(a, b)


def test_a_perturbed_b1_first_half_fails():
    """The longer run drops its rate too early: its first half no longer matches."""
    d = data()
    identity = "gmv-a5.2-gate:B1:outer-0:conv:0"
    a = b1train.train(identity, d, epochs=6, batch=256, snapshot_after=(3,))
    b = b1train.train(identity, d, epochs=4, batch=256, snapshot_after=(3,))  # drops after 2 epochs
    b.epochs, a.epochs = 8, 6  # pretend it was the longer recipe
    with pytest.raises(matched.MatchedStartError):
        matched.check_b1(a, b)


def test_a_missing_snapshot_or_different_identity_fails():
    a, b = b1_pair()
    b.snapshots.clear()
    with pytest.raises(matched.MatchedStartError):
        matched.check_b1(a, b)
    c, d = b1_pair(identity="gmv-a5.2-gate:B1:outer-0:conv:1")
    with pytest.raises(matched.MatchedStartError):
        matched.check_b1(a, d)
