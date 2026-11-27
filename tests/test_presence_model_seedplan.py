"""A5.2 Step 1, S1.1: the seed plan (enumeration, consumer streams, collisions, replays).

Synthetic only: the strings are built from public index values; no project data is read.
The expected enumeration hash is the provisional value from the readiness proposal; this
test checks the proposed serialisation against it and does not adjust anything to match.
"""

import hashlib

import pytest

from gmv.character.research.presence_model import seedplan, seeds

PROVISIONAL_SHA256 = "0436bb715365843deac45c27a01be640d441850889ef9df12df8f6a6c81cb8f5"


def test_every_family_has_the_permitted_values_and_counts():
    reg = seedplan.registered()
    assert sum(":outer:" not in s and ":final:" not in s for s in reg) == 380
    assert sum(":outer:" in s for s in reg) == 190
    assert sum(":final:" in s for s in reg) == 10
    assert len(seedplan.gate()) == 22
    sens = seedplan.sensitivity()
    assert sum(":outer:" not in s for s in sens) == 720 and sum(":outer:" in s for s in sens) == 360
    every = seedplan.production()
    assert len(every) == len(set(every)) == 1682
    assert {seedplan.family_of(s) for s in every} == {"registered", "gate", "sensitivity"}
    # outer indices: 0-18 registered, 0-17 sensitivity; inner 0-4
    assert {int(s.split(":")[2]) for s in reg if ":final:" not in s} == set(range(19))
    assert {int(s.split(":")[2]) for s in sens} == set(range(18))
    assert {int(s.split(":")[3]) for s in reg + sens if ":outer:" not in s and ":final:" not in s} == set(range(5))


def test_policy_d_keeps_the_leading_seed_indices():
    reg_d, sens_d = seedplan.registered("D"), seedplan.sensitivity("D")
    assert len(reg_d) == 310 and len(sens_d) == 576
    assert set(reg_d) <= set(seedplan.registered()) and set(sens_d) <= set(seedplan.sensitivity())
    assert {int(s.split(":")[-1]) for s in reg_d if ":outer:" in s or ":final:" in s} == {0, 1, 2}
    assert {int(s.split(":")[-1]) for s in reg_d if ":outer:" not in s and ":final:" not in s} == {0}


def test_integer_of_is_the_registered_formula():
    for text in seedplan.registered():
        _prefix, arm, outer, inner, index = text.split(":")
        assert seedplan.integer_of(text) == seeds.training_seed(arm, outer, inner, index)


def test_the_canonical_serialisation_reproduces_the_provisional_hash():
    lines = sorted(f"{s}\t{seedplan.integer_of(s)}" for s in seedplan.production())
    independent = hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()
    assert seedplan.enumeration_sha256(seedplan.production()) == independent
    assert independent == PROVISIONAL_SHA256, "the proposed serialisation does not reproduce the provisional hash"


@pytest.fixture(scope="module")
def production_plan():
    return seedplan.plan("C")


def test_the_production_plan_has_no_collision_at_any_consumer(production_plan):
    p = production_plan
    assert p.remapped == 0 and len(p.identities) == 1682
    assert all(b == f for b, f, _n in p.identities)
    assert len({n for _b, _f, n in p.identities}) == 1682
    c_count = sum(seedplan.arm_of(b) == "C" for b, _f, _n in p.identities)
    assert c_count == 841
    assert p.torch_values == 2 * c_count  # C: initialisation and dropout
    assert p.numpy_states == 5 * c_count + 2 * (1682 - c_count)  # C: order, 4 augmentation; B1: init, order
    assert p.sha256 == PROVISIONAL_SHA256
    assert seedplan.plan("C").consumers_sha256 == p.consumers_sha256  # deterministic


def test_policy_d_plan_is_consistent():
    p = seedplan.plan("D")
    assert len(p.identities) == 310 + 22 + 576 and p.remapped == 0


# -- forced collisions through the injectable seam (SC1-SC7) -------------------------------------

A, B = "gmv-a5.2:C:0:0:0", "gmv-a5.2:C:0:0:1"
G = "gmv-a5.2-gate:C:outer-0:conv:1"
S1, S2 = "gmv-a5.2-sensT:B1:0:0:0", "gmv-a5.2-sensT:B1:0:1:0"


def overriding(table):
    return lambda text: table.get(text, seedplan.integer_of(text))


def test_sc1_a_registered_integer_collision_stops_without_remapping():
    with pytest.raises(seedplan.SeedCollisionError):
        seedplan.plan("C", integer_of=overriding({B: seedplan.integer_of(A)}))


def test_sc2_a_registered_derived_stream_collision_stops():
    na, nb = seedplan.integer_of(A), seedplan.integer_of(B)

    def torch_seed_of(n, key):
        if (n, key) in ((na, (0,)), (nb, (3,))):
            return 12345
        return seedplan.torch_seed_of(n, key)

    with pytest.raises(seedplan.SeedCollisionError):
        seedplan.plan("C", torch_seed_of=torch_seed_of)


def test_sc3_a_gate_collision_with_a_registered_seed_takes_c1():
    p = seedplan.plan("C", integer_of=overriding({G: seedplan.integer_of(A)}))
    assert p.final(G) == G + ":c1" and p.final(A) == A and p.remapped == 1


def test_sc4_the_later_of_two_colliding_sensitivity_strings_is_remapped():
    assert S1 < S2
    p = seedplan.plan("C", integer_of=overriding({S2: seedplan.integer_of(S1)}))
    assert p.final(S1) == S1 and p.final(S2) == S2 + ":c1"


def test_sc5_when_c1_still_collides_c2_is_used():
    table = {G: seedplan.integer_of(A), G + ":c1": seedplan.integer_of(B)}
    p = seedplan.plan("C", integer_of=overriding(table))
    assert p.final(G) == G + ":c2"


def test_sc6_the_record_and_hash_follow_the_remapped_strings():
    table = {G: seedplan.integer_of(A), G + ":c1": seedplan.integer_of(B)}
    p = seedplan.plan("C", integer_of=overriding(table))
    finals = [f for _b, f, _n in p.identities]
    assert len(set(finals)) == len({n for _b, _f, n in p.identities}) == 1682
    lines = sorted(f"{f}\t{n}" for _b, f, n in p.identities)
    assert p.sha256 == hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest() != PROVISIONAL_SHA256
    assert G + ":c2" in finals and G not in finals


def test_sc7_remapping_is_deterministic():
    table = {G: seedplan.integer_of(A)}
    one = seedplan.plan("C", integer_of=overriding(table))
    two = seedplan.plan("C", integer_of=overriding(table))
    assert one == two


def test_duplicate_strings_are_refused():
    with pytest.raises(seedplan.SeedPlanError):
        seedplan.plan("C", strings=[A, A])


# -- replays (RP1-RP5) ---------------------------------------------------------------------------


@pytest.fixture
def ledger(production_plan):
    return seedplan.RunLedger(production_plan)


def test_rp1_a_declared_replay_is_accepted_after_its_source(ledger):
    source = seedplan.REPLAYS["G-C3"]
    ledger.claim_production("G-C1", source)
    ledger.claim_replay("G-C3", source)
    seedplan.verify_replay(b"\x01\x02", b"\x01\x02")


def test_rp2_an_undeclared_reuse_stops(ledger):
    ledger.claim_production("run-1", A)
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_production("run-2", A)


def test_rp3_an_unknown_replay_id_stops(ledger):
    ledger.claim_production("run-1", A)
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_replay("G-C9", A)


def test_rp4_a_replay_with_the_wrong_source_or_before_its_source_stops(ledger):
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_replay("G-C3", seedplan.REPLAYS["G-C3"])  # the source has not run
    ledger.claim_production("G-B1-1", seedplan.REPLAYS["G-B1-3"])
    with pytest.raises(seedplan.ReplayError):
        ledger.claim_replay("G-C3", seedplan.REPLAYS["G-B1-3"])


def test_rp5_a_replay_that_differs_stops():
    with pytest.raises(seedplan.ReplayError):
        seedplan.verify_replay(b"\x01\x02", b"\x01\x03")


def test_a_seed_outside_the_plan_is_refused(ledger):
    with pytest.raises(seedplan.SeedPlanError):
        ledger.claim_production("run-x", "gmv-a5.2:C:99:0:0")
