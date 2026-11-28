"""A5.2 Step 2a, S2a.3 and S2a.4: the private review file, the neutral public wording, and the
hiding commitments (protocol Amendment 3, A3.6 and A3.10), on generated inputs only."""

import hashlib
import itertools
import json
import secrets
import stat
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step2_synth  # noqa: E402

from gmv.character.research.presence import identity  # noqa: E402
from gmv.character.research.presence_model import commitment, musdb, readers, review  # noqa: E402

CANARY = "CANARY" + secrets.token_hex(6)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(track_id, split="train", digest=None):
    return {"dataset": "musdb18hq", "dataset_track_id": track_id, "split": split, "sha256": digest or sha(track_id)}


REGISTRY = {"entries": {
    "alpha": {"partition": "development", "nodes": [f"name:{identity.normalise('Alpha ' + CANARY)}"]},
    "beta": {"partition": "development", "nodes": ["name:beta", f"rec:{sha('beta-own')}"]},
}}


# -- the review file ---------------------------------------------------------------------------------------


def test_the_review_file_is_private_complete_and_never_printed(tmp_path, capsys):
    records = [record(f"Alpha {CANARY} - Song (with Beta)"), record("Anyone - Copy", "validation", sha("beta-own"))]
    result = musdb.match(records, REGISTRY, ({}, []), {}, None)
    pairs = musdb.review_pairs(result)
    path = review.write_review_file(tmp_path / "review.json", pairs, REGISTRY)
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    body = json.loads(path.read_text())
    by_group = {(i["group"], i["musdb_artist"]): i for i in body["pairs"]}
    alpha = by_group[("alpha", f"Alpha {CANARY}")]
    assert alpha["credit_from"] == ["artist"] and alpha["decision_if_not_the_same_artist"]["reason"] == "not-the-same-artist"
    assert by_group[("beta", f"Alpha {CANARY}")]["credit_from"] == ["bracket"]
    assert "cannot be rejected" in by_group[("beta", "Anyone")]["decision_if_not_the_same_artist"]
    captured = capsys.readouterr()
    assert CANARY not in captured.out + captured.err
    with pytest.raises(FileExistsError):  # never overwritten
        review.write_review_file(tmp_path / "review.json", pairs, REGISTRY)


def test_a_copied_decision_round_trips_through_the_decisions_reader(tmp_path):
    track = f"Alpha {CANARY} - Song"
    result = musdb.match([record(track, "train")], REGISTRY, ({}, []), {}, None)
    path = review.write_review_file(tmp_path / "review.json", musdb.review_pairs(result), REGISTRY)
    decision = json.loads(path.read_text())["pairs"][0]["decision_if_not_the_same_artist"]
    decisions = tmp_path / "decisions.json"
    decisions.write_text(json.dumps({"format": readers.DECISIONS_FORMAT, "version": 1, "rejections": [decision]}))
    rejected = readers.read_decisions(decisions).value
    after = musdb.exposure(result, REGISTRY, {"alpha": "paired", "beta": "paired"}, rejections=rejected)
    assert after.g_t == frozenset()


def test_a_test_split_stop_gives_a_review_file_of_the_test_pairs_only(tmp_path):
    with pytest.raises(musdb.TestSplitOverlap) as caught:
        musdb.match([record(f"Alpha {CANARY} - Song", "test"), record("Beta - Other", "train")], REGISTRY, ({}, []), {}, None)
    path = review.write_review_file(tmp_path / "r.json", review.test_split_pairs(caught.value), REGISTRY, test_split_only=True)
    body = json.loads(path.read_text())
    assert [(i["group"], i["corpus_split"]) for i in body["pairs"]] == [("alpha", "test")]
    assert review.summary(review.test_split_pairs(caught.value)) == {
        "name_based_pairs": 1, "file_matches": 0, "pairs_by_split": {"test": 1}}


def test_the_only_public_step_2_sentence_is_the_halt_and_it_names_no_cause():
    lowered = review.HALT.lower()
    assert not any(word in lowered for word in ("match", "test", "musdb", "confirm", "reject", "alias", "review file"))
    assert review.HALT == "Step 2 stopped for owner review. No figures from this run are published."
    assert not hasattr(review, "RERUN")  # Amendment 3A: re-runs have no public wording


# -- commitments -------------------------------------------------------------------------------------------


def hashes(decisions_sha):
    return {"corpus_manifest": sha("corpus"), "registry": sha("registry"), "aliases": sha("aliases"),
            "titles": sha("titles"), "credit_readings": sha("readings"), "review_decisions": decisions_sha}


def candidate_decision_files(tmp_path, pairs):
    """Every possible review-decisions file for these pairs: each subset of them rejected."""
    for n in range(len(pairs) + 1):
        for subset in itertools.combinations(pairs, n):
            path = step2_synth.write_decisions(tmp_path / f"cand-{n}-{abs(hash(subset))}.json", list(subset))
            yield subset, hashlib.sha256(path.read_bytes()).hexdigest()


def plain_digest(h):
    """Step 1b's public input digest: SHA-256 of the sorted hashes (the positive control)."""
    return hashlib.sha256(json.dumps(sorted(h.items()), separators=(",", ":")).encode()).hexdigest()


def test_an_attacker_who_knows_every_other_input_cannot_find_the_decisions(tmp_path):
    pairs = [(f"Alpha {CANARY} - Song {i}", "alpha") for i in range(4)]
    actual = step2_synth.write_decisions(tmp_path / "actual.json", pairs[1:3])
    actual_sha = hashlib.sha256(actual.read_bytes()).hexdigest()
    public, _opening = commitment.commit(hashes(actual_sha))
    tried = 0
    for _subset, candidate_sha in candidate_decision_files(tmp_path, pairs):
        tried += 1
        assert plain_digest(hashes(candidate_sha)) != public  # no repeatable calculation reproduces it
    assert tried == 16


def test_the_plain_digest_leaks_the_decisions_positive_control(tmp_path):
    pairs = [(f"Alpha {CANARY} - Song {i}", "alpha") for i in range(4)]
    actual = step2_synth.write_decisions(tmp_path / "actual.json", list(pairs[1:3]))
    public = plain_digest(hashes(hashlib.sha256(actual.read_bytes()).hexdigest()))
    recovered = [s for s, c in candidate_decision_files(tmp_path, pairs) if plain_digest(hashes(c)) == public]
    assert recovered == [tuple(pairs[1:3])]  # the old digest gives the decisions away


def test_identical_inputs_give_different_public_values_and_the_opening_verifies():
    h = hashes(sha("decisions"))
    first, opening = commitment.commit(h)
    second, _ = commitment.commit(h)
    assert first != second
    assert commitment.verify(first, opening)
    record_ = opening.to_private_record()
    assert commitment.verify(first, commitment.Opening.from_private_record(record_))
    changed = dict(record_, hashes=dict(record_["hashes"], titles=sha("other")))
    assert not commitment.verify(first, commitment.Opening.from_private_record(changed))
    flipped = dict(record_, secret=("0" if record_["secret"][0] != "0" else "1") + record_["secret"][1:])
    assert not commitment.verify(first, commitment.Opening.from_private_record(flipped))


def test_the_secret_and_the_private_hashes_never_appear_in_public_or_printed_form():
    h = hashes(sha("decisions"))
    public, opening = commitment.commit(h)
    secret = opening.to_private_record()["secret"]
    assert secret not in public and not any(v in public for v in h.values())
    assert repr(opening) == str(opening) == "<commitment opening: withheld>"
    assert secret not in repr(opening)
    with pytest.raises(TypeError):
        json.dumps({"opening": opening})


@pytest.mark.parametrize("bad", ({}, {"x": "not-a-sha"}, {"": "0" * 64}, ["0" * 64]))
def test_malformed_hashes_are_refused(bad):
    with pytest.raises(commitment.CommitmentError):
        commitment.commit(bad)
