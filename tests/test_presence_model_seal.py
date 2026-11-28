"""A5.2 Step 4, S4.8: the prediction seal and its verifier (protocol Amendment 3, A3.5 N2;
A3.10 "N2"). Generated runs only: small generated model and prediction files."""

import builtins
import copy
import hashlib
import json
import os
import stat

import pytest

from gmv.character.research.presence_model import seal

EPOCHS = {"C": 30, "B1": 60}


def fold_index(outer=3, inner=2, sens_outer=2):
    return json.dumps({"main": {str(o): list(range(inner)) for o in range(outer)},
                       "sensT": {str(o): list(range(inner)) for o in range(sens_outer)},
                       "sensV": {str(o): list(range(inner)) for o in range(sens_outer)}}, sort_keys=True).encode()


def decisions(index, policy="D", sets=("sensT", "sensV")):
    return ({"fold_index_sha256": hashlib.sha256(index).hexdigest(), "sensitivity_sets": list(sets)},
            {"policy": policy, "epochs": dict(EPOCHS)})


def declaration(policy="D", reruns=("sensT",), commit="a" * 40):
    return seal.Declaration("run-1", "step 6", commit, policy, dict(EPOCHS), tuple(reruns))


def header(decl, index):
    return {"format": seal.FORMAT, "version": seal.VERSION, "run": decl.run, "step": decl.step, "commit": decl.commit,
            "integrity_sha256": "1" * 64, "fold_index_sha256": hashlib.sha256(index).hexdigest(),
            "seed_format": "a5.2-seed-format-v1", "seed_enumeration_sha256": "2" * 64, "policy": decl.policy,
            "epochs": dict(decl.epochs), "thresholds_sha256": "3" * 64}


def materialise(root, inventory):
    """Write generated model and prediction files and return the inventory with their hashes."""
    out = []
    for n, entry in enumerate(inventory):
        entry = dict(entry)
        for field, digest in (("model_file", "model_sha256"), ("predictions_file", "predictions_sha256")):
            if entry[field] is None:
                entry[digest] = None
                continue
            path = root / entry[field]
            path.parent.mkdir(parents=True, exist_ok=True)
            data = f"generated {field} {n}".encode()
            path.write_bytes(data)
            entry[digest] = hashlib.sha256(data).hexdigest()
        out.append(entry)
    return out


@pytest.fixture
def world(tmp_path):
    index = fold_index()
    a5, a6 = decisions(index)
    decl = declaration()
    expected = seal.expectation(index, a5, a6, decl)
    root = tmp_path / "run"
    root.mkdir()
    inventory = materialise(root, expected["inventory"])
    record = tmp_path / "record.json"
    seal.record_expectation(record, expected)
    return {"index": index, "a5": a5, "a6": a6, "decl": decl, "expected": expected, "root": root,
            "inventory": inventory, "record": record, "tmp": tmp_path}


def write(world, inventory=None, head=None, name="seal.json"):
    path = world["tmp"] / name
    seal.write(path, head or header(world["decl"], world["index"]), world["inventory"] if inventory is None else inventory)
    return path


def check(world, path):
    return seal.verify(path, world["root"], world["expected"], world["record"], world["decl"], world["a5"])


def test_a_matching_seal_verifies_and_is_read_only(world):
    path = write(world)
    verified = check(world, path)
    assert verified.entries == len(world["inventory"])
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    assert verified.seal_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        write(world)


def test_the_registered_design_gives_the_registered_model_counts():
    index = json.dumps({"main": {str(o): list(range(5)) for o in range(19)}}).encode()
    for policy, per_arm in (("C", 190 + 95 + 5), ("D", 95 + 57 + 3)):
        a5, a6 = decisions(index, policy, sets=())
        inv = seal.expectation(index, a5, a6, declaration(policy, reruns=()))["inventory"]
        assert len(inv) == 2 * per_arm
        assert {e["identity"] for e in inv if e["role"] == "final"} == {
            f"gmv-a5.2:{a}:final:final:{s}" for a in ("B1", "C") for s in seal.POLICIES[policy]["outer"]}


def test_sensitivity_reruns_have_no_final_models_and_their_own_namespace():
    index = fold_index()
    a5, a6 = decisions(index)
    inv = seal.expectation(index, a5, a6, declaration())["inventory"]
    sens = [e for e in inv if e["namespace"] == "sensT"]
    assert sens and not any(e["role"] == "final" for e in sens)
    assert all(e["identity"].startswith("gmv-a5.2-sensT:") for e in sens)


def test_the_expectation_is_identical_with_or_without_a_seal_and_never_opens_it(world, monkeypatch):
    before = seal.expectation_digest(world["expected"])
    path = write(world)
    opened = []
    real_open, real_os_open = builtins.open, os.open
    monkeypatch.setattr(builtins, "open", lambda p, *a, **k: (opened.append(str(p)), real_open(p, *a, **k))[1])
    monkeypatch.setattr(os, "open", lambda p, *a, **k: (opened.append(str(p)), real_os_open(p, *a, **k))[1])
    again = seal.expectation(world["index"], world["a5"], world["a6"], world["decl"])
    monkeypatch.undo()
    assert seal.expectation_digest(again) == before
    assert str(path) not in opened


def test_the_expectation_must_be_recorded_before_the_seal_is_checked(world):
    path = write(world)
    world["record"].unlink()
    world["record"].write_text(json.dumps({"expectation_sha256": "0" * 64}))
    with pytest.raises(seal.SealError, match="run record"):
        check(world, path)


def mutate(world, change):
    inventory = copy.deepcopy(world["inventory"])
    change(inventory)
    return inventory


def first(inv, **match):
    return next(e for e in inv if all(e[k] == v for k, v in match.items()))


REFUSALS = {
    "outside the list": lambda inv: first(inv, role="outer").update(arm="B0"),
    "seed outside": lambda inv: first(inv, role="outer").update(seed=4),
    "undeclared rerun": lambda inv: first(inv, namespace="sensT").update(namespace="sensV"),
    "outside Amendment 5": lambda inv: first(inv, namespace="sensT").update(namespace="sensX"),
    "not in the fold index": lambda inv: first(inv, role="inner").update(outer="7"),
    "relabelled": lambda inv: first(inv, role="outer", seed=1).update(identity=first(inv, role="outer", seed=0)["identity"]),
    "twice": lambda inv: inv.append(dict(inv[0])),
    "missing": lambda inv: inv.pop(),
}


@pytest.mark.parametrize("case", sorted(REFUSALS))
def test_each_refused_case_is_refused(world, case):
    path = write(world, mutate(world, REFUSALS[case]))
    with pytest.raises(seal.SealError, match=case):
        check(world, path)


def test_extra_files_on_disk_are_refused(world):
    path = write(world)
    (world["root"] / "models" / "stray.model").write_bytes(b"x")
    with pytest.raises(seal.SealError, match="files on disk"):
        check(world, path)


def test_a_weights_hash_mismatch_is_refused(world):
    path = write(world)
    target = world["root"] / first(world["inventory"], arm="C", role="final")["model_file"]
    target.write_bytes(b"changed weights")
    with pytest.raises(seal.SealError, match="does not match its hash"):
        check(world, path)


def test_a_header_that_differs_from_the_declaration_is_refused(world):
    head = header(world["decl"], world["index"])
    head["commit"] = "b" * 40
    with pytest.raises(seal.SealError, match="commit"):
        check(world, write(world, head=head))


def test_a_declaration_disagreeing_with_amendment_6_is_refused(world):
    with pytest.raises(seal.SealError, match="policy"):
        seal.expectation(world["index"], world["a5"], world["a6"], declaration(policy="C"))
    bad = seal.Declaration("run-1", "step 6", "a" * 40, "D", {"C": 45, "B1": 60}, ("sensT",))
    with pytest.raises(seal.SealError, match="epochs"):
        seal.expectation(world["index"], world["a5"], world["a6"], bad)
    with pytest.raises(seal.SealError, match="outside Amendment 5"):
        seal.expectation(world["index"], world["a5"], world["a6"], declaration(reruns=("sensX",)))


def test_a_fold_index_that_does_not_match_amendment_5_is_refused(world):
    with pytest.raises(seal.SealError, match="fold index"):
        seal.expectation(world["index"] + b" ", world["a5"], world["a6"], world["decl"])


def test_policy_c_against_d_and_a_policy_c_seal_under_a_d_declaration(world):
    a5, a6c = decisions(world["index"], "C")
    c = seal.expectation(world["index"], a5, a6c, declaration("C"))
    assert len(c["inventory"]) > len(world["expected"]["inventory"])
    seeds_c = {e["seed"] for e in c["inventory"] if e["role"] == "outer"}
    assert seeds_c == {0, 1, 2, 3, 4} and {e["seed"] for e in world["expected"]["inventory"]
                                            if e["role"] == "outer"} == {0, 1, 2}


def test_an_incomplete_header_cannot_be_written(world):
    head = header(world["decl"], world["index"])
    del head["thresholds_sha256"]
    with pytest.raises(seal.SealError):
        seal.write(world["tmp"] / "x.json", head, world["inventory"])
