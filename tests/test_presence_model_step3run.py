"""A5.2 Step 2a, S2a.8: step 3's MUSDB part, the public count report, and test C4-T (protocol
Amendment 3, A3.6, A3.7 with the proposed 3A, and A3.10), on generated inputs only."""

import itertools
import json
import secrets

import pytest

from gmv.character.research.presence import identity
from gmv.character.research.presence_model import musdb, step3, step3run

CANARY = "CANARY" + secrets.token_hex(6)
RATES = {44100: "supported", 22050: "not_constructible"}


def sha(text):
    import hashlib

    return hashlib.sha256(text.encode()).hexdigest()


def setting(n_paired=8, n_validation=2, matched_train=(), matched_val=(), test_rejected=()):
    """A registry, development records and a deferred step 2 result. Group names carry canaries."""
    entries = {f"p{i}-{CANARY}": {"partition": "development",
                                  "nodes": [f"name:{identity.normalise(f'paired {i} {CANARY}')}"]} for i in range(n_paired)}
    entries[f"low-{CANARY}"] = {"partition": "development", "nodes": [f"name:{identity.normalise(f'low {CANARY}')}"]}
    for i in range(n_validation):
        entries[f"v{i}-{CANARY}"] = {"partition": "validation", "nodes": [f"name:{identity.normalise(f'val {i} {CANARY}')}"]}
    registry = {"entries": entries}
    development = [{"id": f"{k}-0", "group": k, "sample_rate": 22050 if k.startswith("low") else 44100}
                   for k, e in entries.items() if e["partition"] == "development"]
    records, kind = [], {"p": "paired", "low": "low", "v": "val"}
    for split, groups in (("train", matched_train), ("validation", matched_val)):
        for g in groups:
            prefix, _, rest = g.partition(":")
            label = f"{kind[prefix]} {rest} {CANARY}" if rest else f"{kind[prefix]} {CANARY}"
            records.append({"dataset": "musdb18hq", "dataset_track_id": f"{label} - Song {split}", "split": split,
                            "sha256": sha(f"{label}{split}")})
    for i, g in enumerate(test_rejected):
        prefix, _, rest = g.partition(":")
        label = f"{kind[prefix]} {rest} {CANARY}" if rest else f"{kind[prefix]} {CANARY}"
        records.append({"dataset": "musdb18hq", "dataset_track_id": f"{label} - Leak {i}", "split": "test",
                        "sha256": sha(f"{label}test{i}")})
    if not records:
        records.append({"dataset": "musdb18hq", "dataset_track_id": "Nobody - Tune", "split": "train", "sha256": sha("n")})
    test_pairs = {(r["dataset_track_id"], next(k for k, e in entries.items()
                                               if f"name:{identity.normalise(r['dataset_track_id'].split(' - ')[0])}" in e["nodes"]))
                  for r in records if r["split"] == "test"}
    result = musdb.match(records, registry, ({}, []), {}, None, {}, test_pairs)
    stats = {k: (300 + i, 400 + i, 320 + i, 410 + i) for i, k in enumerate(sorted(entries)) if k.startswith("p")}
    public2 = {"musdb_records": result.musdb_records, "unparsed_records": result.unparsed,
               "credits_read_by_reading": result.readings_used}
    return {"development_records": development, "registry": registry, "step2_result": result,
            "applied_test_rejections": frozenset(test_pairs), "decisions": frozenset(test_pairs),
            "rate_class": RATES.get, "stats": stats, "step2_public": public2, "commit": "0" * 40}


def test_the_public_report_holds_figures_through_the_check_and_no_group_identity():
    inputs = setting(matched_train=["p:0", "p:1", "p:2", "low"], matched_val=["p:0"])
    private, public = step3run.compute(**inputs)
    text = json.dumps(public)
    assert CANARY not in text and identity.normalise(CANARY) not in text
    assert public["item_d"] == {"train": 4, "validation": 1, "test": 0}
    assert public["g_t_size"] == "withheld"  # an exact 4 here could pin the low-rate and validation groups
    assert public["low_rate_cells"] == "withheld: one group"
    assert public["voiced_and_no_voice_durations"].startswith("withheld")
    assert len(private["g_t"]) == 3 and private["matched"]["train"] == sorted(
        [k for k in private["categories"] if k.startswith(("p0", "p1", "p2", "low"))])


def test_reruns_are_considered_from_the_item_d_figure_and_their_sets_stay_private():
    inputs = setting(n_paired=10, matched_train=["p:0", "p:1", "p:2"])  # 10 - 3 leaves 7, enough for F1
    private, public = step3run.compute(**inputs)
    training = public["sensitivity_reruns"]["training"]
    assert training["considered"] is True and training["data_checks"] == "passed"
    assert set(training) == {"considered", "data_checks", "folds_commitment"}  # no size, no set, no fold count
    assert private["reruns"]["training"]["removed"] == sorted(private["g_t"])
    assert public["sensitivity_reruns"]["validation"] == {"considered": False}


def test_train_and_validation_rejections_are_applied_at_step_3_and_test_ones_are_not_applied_twice():
    inputs = setting(matched_train=["p:0", "p:1"], test_rejected=["p:3"])
    kept = [t for t, s, k, _x in inputs["step2_result"].matches if s == "train" and k.startswith("p1")]
    inputs["decisions"] = inputs["decisions"] | {(kept[0], next(k for k in inputs["registry"]["entries"] if k.startswith("p1")))}
    private, public = step3run.compute(**inputs)
    assert public["item_d"]["train"] == 1 and len(private["g_t"]) == 1


def test_a_remaining_test_split_match_is_refused():
    inputs = setting(matched_train=["p:0"])
    inputs["step2_result"] = musdb.Result(
        counts={}, musdb_records={"train": 1, "validation": 0, "test": 1}, unparsed={s: 0 for s in musdb.SPLITS},
        exposed=None, private=musdb.Private(None, None, [("T - x", "test", next(iter(inputs["registry"]["entries"])),
                                                          ("exact",))], {}), readings_used={})
    with pytest.raises(step3.Step3Error):
        step3run.compute(**inputs)


# -- test C4-T, the test-split part ---------------------------------------------------------------------------


def c4t_worlds(sizes):
    """Each assessed group's MUSDB-test state: none, a candidate the owner rejected, or a
    confirmed candidate. Groups are exchangeable within a category, so a world is the count of
    each state per category."""
    def compositions(n, parts=3):
        for cut in itertools.combinations(range(n + parts - 1), parts - 1):
            b = (-1, *cut, n + parts - 1)
            yield tuple(b[i + 1] - b[i] - 1 for i in range(parts))

    return list(itertools.product(*(list(compositions(n)) for n in sizes)))


def observe(world, wording):
    confirmed = sum(c[2] for c in world)
    rejected = sum(c[1] for c in world)
    if confirmed:
        return ("stop",)  # the research-safety stop: the neutral sentence, nothing else
    if wording == "approved":
        return ("report", "re-run sentence" if rejected else "no re-run sentence")
    return ("report",)  # only the final outcome is public


def pinned(worlds, sizes):
    """Protected single statuses that have only one possible value: (category, before or after)."""
    out = set()
    for ci, n in enumerate(sizes):
        if not n:
            continue
        before = {v for w in worlds for v, ok in ((1, w[ci][1] + w[ci][2] >= 1), (0, w[ci][0] >= 1)) if ok}
        after = {v for w in worlds for v, ok in ((1, w[ci][2] >= 1), (0, w[ci][0] + w[ci][1] >= 1)) if ok}
        if len(before) < 2:
            out.add((ci, "before review"))
        if len(after) < 2:
            out.add((ci, "after review"))
    return out


@pytest.mark.parametrize("sizes", ((3, 1, 2), (4, 1, 2)))
def test_c4t_the_approved_re_run_sentence_reveals_that_no_group_had_a_test_candidate(sizes):
    """A finding for the owner: with the approved public re-run sentence, a report without it
    tells the reader that no assessed group had any MUSDB-test candidate before the review."""
    worlds = c4t_worlds(sizes)
    groups = {}
    for w in worlds:
        groups.setdefault(observe(w, "approved"), []).append(w)
    leak = pinned(groups[("report", "no re-run sentence")], sizes)
    assert {(ci, "before review") for ci in range(3)} <= leak
    # after review, every status is 0 in a published report: that is item D's test figure of 0, exempt


@pytest.mark.parametrize("sizes", ((3, 1, 2), (4, 1, 2)))
def test_c4t_publishing_only_the_final_outcome_keeps_every_before_review_status_uncertain(sizes):
    worlds = c4t_worlds(sizes)
    groups = {}
    for w in worlds:
        groups.setdefault(observe(w, "final outcome only"), []).append(w)
    for outcome, members in groups.items():
        leak = pinned(members, sizes)
        allowed = {(ci, "after review") for ci in range(3)} if outcome == ("report",) else set()  # item D: 0
        assert leak <= allowed, (outcome, leak)
