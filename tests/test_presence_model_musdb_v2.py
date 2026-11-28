"""A5.2 Step 2a, S2a.2: the matcher's Amendment 3 rules (A3.6), on generated inputs only.

* L3: ambiguous credits ("and"/"with" as words in any capitalisation, brackets, non-Latin
  letters) are listed, need a registered reading, and otherwise stop step 2; they never become
  silent non-matches.
* Credit sources for the owner's review: artist field, title bracket, piece, link.
* Option C: deciding the paired/low-rate split later (step 3) gives exactly the matches,
  counts, G_T and G_V of deciding it first.
* The owner's review: test-split rejections before the test-split check, train and validation
  rejections at step 3; a file match cannot be rejected.
"""

import hashlib
import random
import secrets

import pytest

from gmv.character.research.presence import identity
from gmv.character.research.presence_model import musdb

CANARY = "CANARY" + secrets.token_hex(6)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(track_id, split="train", digest=None):
    return {"dataset": "musdb18hq", "dataset_track_id": track_id, "split": split, "sha256": digest or sha(f"m:{track_id}")}


def entry(partition, names=(), recordings=()):
    return {"partition": partition,
            "nodes": [f"name:{identity.normalise(n)}" for n in names] + [f"rec:{r}" for r in recordings]}


def registry_of(**groups):
    """group key -> (partition, names, recordings)."""
    return {"entries": {k: entry(*v) for k, v in groups.items()}}


NO_ALIASES = ({}, [])


def run(records, registry, readings=None, categories="all-paired", **kwargs):
    if categories == "all-paired":
        categories = {k: "paired" for k, e in registry["entries"].items() if e["partition"] == "development"}
    return musdb.match(records, registry, NO_ALIASES, {}, categories, readings=readings, **kwargs)


def pairs(result):
    return {(t, k): types for t, _s, k, types in result.matches}


# -- L3: what is ambiguous ---------------------------------------------------------------------------------


@pytest.mark.parametrize("artist", (
    "Clara and Wooldog", "Clara And Wooldog", "Clara AND Wooldog", "Clara aNd Wooldog",
    "Clara with Wooldog", "Clara With Wooldog", "Clara WITH Wooldog",
    "Clara And Wooldog", "Clara  and   Wooldog", "Ｃｌａｒａ ＡＮＤ Ｗｏｏｌｄｏｇ",
    "And Then There Were None", "Rock-and-Roll Band", "The Easton Ellises (Baumi)", "Band [Live]",
    "Мумий Тролль", "Clara 和 Wooldog",
))
def test_ambiguous_credits_are_recognised_in_every_form(artist):
    assert musdb.is_ambiguous(artist)


@pytest.mark.parametrize("artist", ("Andy", "Withers", "Sandwith", "Brandon", "Alexander", "Sigur Rós", "Björk",
                                    "Jokers, Jacks & Kings", "Triviul feat. The Fiend", "A vs. B", "Mr. Bubble"))
def test_words_merely_containing_and_or_with_raise_no_false_alarm(artist):
    assert not musdb.is_ambiguous(artist)


def test_the_listing_phase_lists_ambiguous_credits_only_and_reads_only_musdb_records():
    records = [record("Clara And Wooldog - Air"), record("Plain Name - Song"), record("Clara And Wooldog - Stella"),
               {"dataset": "other", "dataset_track_id": "Mixed And Matched - X", "split": "train", "sha256": "x"}]
    assert musdb.list_ambiguous(records) == ["Clara And Wooldog"]


# -- L3: readings ------------------------------------------------------------------------------------------


WORLD = registry_of(wooldog=("development", ["Wooldog"]), clara=("development", ["Clara"]),
                    duo=("development", ["Clara and Wooldog"]))


@pytest.mark.parametrize("artist", ("Clara and Wooldog", "Clara And Wooldog", "CLARA AND WOOLDOG"))
def test_an_ambiguous_credit_without_a_reading_stops_step_2_with_counts_only(artist):
    with pytest.raises(musdb.UnparsedRecords) as caught:
        run([record(f"{artist} - Air {CANARY}")], WORLD)
    message = str(caught.value)
    assert "1 with an ambiguous credit without a reading" in message and CANARY not in message
    assert "Clara" not in message and "Wooldog" not in message


def test_the_default_reading_credits_the_whole_name_and_each_piece_and_flags_pieces():
    result = run([record("Clara And Wooldog - Air")], WORLD, readings={"Clara And Wooldog": "whole-and-pieces"})
    assert set(pairs(result)) == {("Clara And Wooldog - Air", k) for k in ("wooldog", "clara", "duo")}
    assert result.sources[("Clara And Wooldog - Air", "wooldog")] == ("piece",)
    assert result.sources[("Clara And Wooldog - Air", "duo")] == ("artist",)
    assert result.readings_used == {"train": 1, "validation": 0, "test": 0}


def test_a_single_name_reading_credits_only_the_whole_name():
    result = run([record("Clara And Wooldog - Air")], WORLD, readings={"Clara And Wooldog": "single-name"})
    assert set(pairs(result)) == {("Clara And Wooldog - Air", "duo")}


def test_each_reading_gives_exactly_its_credits():
    keys = musdb.exact_keys("The Easton Ellises (Baumi)", "Song", {}, [], reading="whole-and-pieces")
    assert keys == {"easton ellises baumi", "easton ellises", "baumi"}
    assert musdb.exact_keys("The Easton Ellises (Baumi)", "Song", {}, [], reading="single-name") == {"easton ellises baumi"}


def test_a_stale_reading_is_refused():
    with pytest.raises(musdb.MusdbInputError) as caught:
        run([record("Plain - Song")], WORLD, readings={f"Gone And {CANARY}": "single-name"})
    assert "not on the listing" in str(caught.value) and CANARY not in str(caught.value)


def test_no_ambiguous_credit_ends_with_neither_a_match_nor_a_refusal():
    """On random credits: an ambiguous credit with a reading is matched by its reading; one
    without a reading stops; there is no third, silent outcome."""
    rng = random.Random(7)
    words = ["and", "And", "AND", "with", "With", "&", "feat.", "x"]
    for _ in range(200):
        left, right = f"left{rng.randrange(1000)}", f"right{rng.randrange(1000)}"
        artist = f"{left} {rng.choice(words)} {right}"
        reg = registry_of(r=("development", [right]))
        rec = [record(f"{artist} - Song")]
        if musdb.is_ambiguous(artist):
            with pytest.raises(musdb.UnparsedRecords):
                run(rec, reg)
            assert ("%s - Song" % artist, "r") in pairs(run(rec, reg, readings={artist: "whole-and-pieces"}))
        else:  # a registered separator: the exact tier already splits it, in any capitalisation
            assert (f"{artist} - Song", "r") in pairs(run(rec, reg))


@pytest.mark.parametrize("form", ("{a} feat. {b}", "{a} FEAT. {b}", "{a} Feat. {b}", "{a} vs. {b}", "{a} VS. {b}",
                                  "{a} & {b}", "{a} x {b}", "{a} X {b}", "{a}, {b}", "{a} / {b}"))
def test_supported_separators_split_identically_in_every_capitalisation(form):
    assert musdb.exact_keys(form.format(a="Left", b="Right"), "Song", {}, []) == {"left", "right"}


@pytest.mark.parametrize("title", ("Song (with Guest)", "Song (WITH Guest)", "Song [With Guest]", "Song (feat. Guest)"))
def test_title_brackets_are_read_in_any_capitalisation_and_flagged(title):
    result = run([record(f"Host - {title}")], registry_of(guest=("development", ["Guest"])))
    assert result.sources[(f"Host - {title}", "guest")] == ("bracket",)


def test_link_credits_are_flagged():
    aliases = ({}, [(identity.normalise("Host"), identity.normalise("Partner"))])
    result = musdb.match([record("Host - Song")], registry_of(p=("development", ["Partner"])), aliases, {}, {"p": "paired"})
    assert result.sources[("Host - Song", "p")] == ("link",)


def test_results_with_readings_do_not_depend_on_input_order():
    records = [record(f"Clara And Wooldog - Song {i}", "train" if i % 2 else "validation") for i in range(6)]
    readings = {"Clara And Wooldog": "whole-and-pieces"}
    first = run(records, WORLD, readings=readings)
    rng = random.Random(3)
    for _ in range(5):
        shuffled = records[:]
        rng.shuffle(shuffled)
        assert run(shuffled, WORLD, readings=readings) == first


# -- option C: deferring the split -------------------------------------------------------------------------


def random_world(rng, n_groups=12):
    names = [f"artist{i}" for i in range(n_groups)]
    regs = {f"g{i}": ("development" if i < n_groups - 2 else "validation", [names[i]],
                      [sha(f"own{i}")] if rng.random() < 0.3 else []) for i in range(n_groups)}
    records, copied = [], set()
    for j in range(25):
        i = rng.randrange(n_groups + 5)
        split = rng.choice(["train", "validation"])
        if i < n_groups and rng.random() < 0.3 and regs[f"g{i}"][2] and i not in copied:
            copied.add(i)  # each own recording appears in at most one MUSDB record
            records.append(record(f"Someone {j} - Song", split, digest=regs[f"g{i}"][2][0]))
        else:
            records.append(record(f"{names[i] if i < n_groups else 'nobody'} - Song {j}", split))
    registry = registry_of(**regs)
    categories = {k: ("low-rate" if rng.random() < 0.15 else "paired") for k, e in registry["entries"].items()
                  if e["partition"] == "development"}
    return records, registry, categories


def test_deciding_the_split_later_gives_exactly_the_same_results_on_random_inputs():
    rng = random.Random(11)
    for _ in range(60):
        records, registry, categories = random_world(rng)
        early = musdb.match(records, registry, NO_ALIASES, {}, categories)
        deferred = musdb.match(records, registry, NO_ALIASES, {}, None)
        assert deferred.g_t is None and deferred.exposed is None
        assert set(deferred.counts["train"]) == {"development", "validation"}
        late = musdb.exposure(deferred, registry, categories)
        assert late == early


def test_the_deferred_split_must_class_exactly_the_development_groups():
    records, registry, categories = random_world(random.Random(1))
    deferred = musdb.match(records, registry, NO_ALIASES, {}, None)
    missing = dict(categories)
    missing.popitem()
    for bad in (missing, {**categories, "g99": "paired"}, {k: "unknown" for k in categories}):
        with pytest.raises(musdb.MusdbInputError):
            musdb.exposure(deferred, registry, bad)
    with pytest.raises(musdb.MusdbInputError):
        musdb.exposure(musdb.match(records, registry, NO_ALIASES, {}, categories), registry, categories)


# -- the owner's review ------------------------------------------------------------------------------------


REVIEW = registry_of(alpha=("development", ["Alpha"]), beta=("development", ["Beta"], [sha("beta-own")]),
                     val=("validation", ["Valley"]))


def test_a_test_split_match_stops_with_its_pairs_kept_private():
    with pytest.raises(musdb.TestSplitOverlap) as caught:
        run([record(f"Alpha - Song {CANARY}", "test")], REVIEW)
    assert CANARY not in str(caught.value) and "1 record(s), 1 group(s)" in str(caught.value)
    private = caught.value.private
    assert repr(private) == "<private: withheld>"
    assert [(t, k) for t, _s, k, _types in private.matches] == [(f"Alpha - Song {CANARY}", "alpha")]
    assert private.sources[(f"Alpha - Song {CANARY}", "alpha")] == ("artist",)


def test_rejected_test_split_pairs_are_dropped_before_the_test_split_check():
    track = f"Alpha - Song {CANARY}"
    result = run([record(track, "test"), record("Beta - Hit", "train")], REVIEW, test_rejections={(track, "alpha")})
    assert (track, "alpha") not in pairs(result) and ("Beta - Hit", "beta") in pairs(result)


def test_a_file_match_cannot_be_rejected():
    track = "Anyone - Copy"
    with pytest.raises(musdb.ReviewError) as caught:
        run([record(track, "test", digest=sha("beta-own"))], REVIEW, test_rejections={(track, "beta")})
    assert "file match" in str(caught.value)


@pytest.mark.parametrize("rejection", ((f"Nobody {CANARY} - Song", "alpha"), ("Alpha - Song", "beta")))
def test_a_rejection_must_name_a_name_based_pair(rejection):
    with pytest.raises(musdb.ReviewError) as caught:
        run([record("Alpha - Song", "test")], REVIEW, test_rejections={rejection})
    assert CANARY not in str(caught.value)


def test_train_and_validation_rejections_are_applied_at_step_3():
    records = [record("Alpha - Song", "train"), record("Beta - Song", "validation"), record("Valley - Song", "train")]
    deferred = musdb.match(records, REVIEW, NO_ALIASES, {}, None)
    categories = {"alpha": "paired", "beta": "paired"}
    assert musdb.exposure(deferred, REVIEW, categories).g_t == {"alpha"}
    after = musdb.exposure(deferred, REVIEW, categories, rejections={("Alpha - Song", "alpha")})
    assert after.g_t == frozenset() and after.g_v == {"beta"}
    assert after.counts["train"]["paired"]["exact"] == {"groups": 0, "records": 0}
    with pytest.raises(musdb.ReviewError):
        musdb.exposure(deferred, REVIEW, categories, rejections={("Beta - Song", "alpha")})


def test_review_pairs_list_name_based_and_file_matches_with_their_sources():
    records = [record("Alpha - Song (with Valley)", "train"), record("Anyone - Copy", "validation", digest=sha("beta-own"))]
    result = run(records, REVIEW)
    listed = {(t, k): (types, src) for t, _s, k, types, src in musdb.review_pairs(result)}
    assert listed[("Alpha - Song (with Valley)", "alpha")] == (("exact",), ("artist",))
    assert listed[("Alpha - Song (with Valley)", "val")] == (("exact",), ("bracket",))
    assert listed[("Anyone - Copy", "beta")] == (("checksum",), ())
