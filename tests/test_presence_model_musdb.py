"""A5.2 Step 1b: the MUSDB18-HQ overlap matcher, on generated inputs only.

Every corpus record, registry entry, alias and title below is made up in the test. Group keys,
names, titles and track identifiers carry random canaries so that leaks into public output can
be detected. The numbered comments refer to the Step 1b acceptance criteria.
"""

import copy
import hashlib
import json
import random
import secrets
import subprocess
import sys

import pytest

from gmv.character.research.presence_model import musdb

CANARY = "CANARY" + secrets.token_hex(6)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(track_id, split, digest=None, dataset="musdb18hq"):
    return {"dataset": dataset, "dataset_track_id": track_id, "split": split,
            "sha256": digest or sha(f"musdb:{track_id}")}


def entry(partition, names=(), recordings=()):
    return {"partition": partition, "nodes": [f"name:{n}" for n in names] + [f"rec:{r}" for r in recordings]}


def world():
    """A synthetic corpus, registry, aliases, titles and categories with one case per rule."""
    own = {k: sha(f"own:{k}:{CANARY}") for k in ("checksum", "validation-rec", "titled")}
    records = [
        record(f"Alpha {CANARY} - First Song", "train"),  # exact -> paired-alpha
        record(f"Alpha {CANARY} - Second Song", "train"),  # a second record for the same group
        record("Some Artist - Copied Track", "train", digest=own["checksum"]),  # checksum -> paired-check
        record(f"Bravo {CANARY} - Bright Day (feat. Guest)", "validation"),  # exact -> paired-bravo
        record(f"Spelling {CANARY} - Any", "train"),  # alias -> paired-alias
        record(f"Linked {CANARY} - Any", "train"),  # one link step -> paired-link
        record(f"Charlie {CANARY} and Delta - Tune", "train"),  # permissive only -> paired-perm
        record(f"Titled {CANARY} - Night Owl (feat. Someone)", "train"),  # exact + title confirmation
        record(f"Lowrate {CANARY} - Song", "train"),  # exact -> low-rate group (not exposure)
        record(f"Valgroup {CANARY} - Song", "validation"),  # exact -> validation partition
        record(f"Reserved {CANARY} - Song", "train"),  # matches only a reserved group: not assessed
        record("NoSeparatorHere", "test"),  # unparsed, no match
        record("Unrelated - Nothing", "test"),
        {"dataset": "mdbdrums", "dataset_track_id": f"Alpha {CANARY} - Other", "split": "train", "sha256": "x"},  # other dataset
    ]
    registry = {"entries": {
        f"paired-alpha-{CANARY}": entry("development", [f"alpha {CANARY.lower()}"]),
        f"paired-check-{CANARY}": entry("development", [f"someone else {CANARY.lower()}"], [own["checksum"]]),
        f"paired-bravo-{CANARY}": entry("development", [f"bravo {CANARY.lower()}"]),
        f"paired-alias-{CANARY}": entry("development", [f"canonical {CANARY.lower()}"]),
        f"paired-link-{CANARY}": entry("development", [f"partner {CANARY.lower()}"]),
        f"paired-chain-{CANARY}": entry("development", [f"second partner {CANARY.lower()}"]),
        f"paired-perm-{CANARY}": entry("development", ["delta"]),
        f"paired-titled-{CANARY}": entry("development", [f"titled {CANARY.lower()}"], [own["titled"]]),
        f"lowrate-{CANARY}": entry("development", [f"lowrate {CANARY.lower()}"]),
        f"val-{CANARY}": entry("validation", [f"valgroup {CANARY.lower()}"], [own["validation-rec"]]),
        f"reserved-{CANARY}": entry("reserved", [f"reserved {CANARY.lower()}"]),
        f"excluded-{CANARY}": {"partition": "excluded", "nodes": "never read"},
    }}
    aliases = ({f"spelling {CANARY.lower()}": f"canonical {CANARY.lower()}"},
               [(f"linked {CANARY.lower()}", f"partner {CANARY.lower()}", "shared vocalist"),
                (f"partner {CANARY.lower()}", f"second partner {CANARY.lower()}", "chain")])
    titles = {own["titled"]: "Night Owl", own["checksum"]: f"Private Title {CANARY}"}
    categories = {k: "paired" for k in registry["entries"] if k.startswith("paired-")}
    categories[f"lowrate-{CANARY}"] = "low-rate"
    return records, registry, aliases, titles, categories


@pytest.fixture
def inputs():
    return world()


@pytest.fixture
def result(inputs):
    return musdb.match(*inputs)


def private_pairs(result):
    return {(t, k): types for t, _s, k, types in result.matches}


def cell(result, split, partition, kind):
    return result.counts[split][partition][kind]


# -- 1, 2: parsing, unparsed and malformed records ---------------------------------------------------


@pytest.mark.parametrize(("track_id", "expected"), (
    ("Artist - Title", ("Artist", "Title", True)),
    ("Artist - Title - Remix - Live", ("Artist", "Title - Remix - Live", True)),  # first " - " only
    ("Jay-Z - Song", ("Jay-Z", "Song", True)),  # a hyphen inside a name is not the separator
    ("Artist-Title", ("Artist-Title", "Artist-Title", False)),
    ("NoSeparatorHere", ("NoSeparatorHere", "NoSeparatorHere", False)),
    (" - Title", ("", "Title", True)),  # as written: the artist part is empty
))
def test_track_identifiers_split_at_the_first_spaced_hyphen(track_id, expected):
    assert musdb.parse_track_id(track_id) == expected


def test_unparsed_records_are_counted_and_still_matched(inputs):
    records, registry, aliases, titles, categories = inputs
    registry["entries"]["paired-whole"] = entry("development", ["wholename"])
    categories["paired-whole"] = "paired"
    records.append(record("WholeName", "train"))  # unparsed: artist and title are the whole id
    result = musdb.match(records, registry, aliases, titles, categories)
    assert result.unparsed == {"train": 1, "validation": 0, "test": 1}
    assert ("WholeName", "paired-whole") in private_pairs(result)


@pytest.mark.parametrize("bad", (
    "not a record", None,
    {"dataset": "musdb18hq", "split": "train", "sha256": "0" * 64},  # no identifier
    {"dataset": "musdb18hq", "dataset_track_id": "  ", "split": "train", "sha256": "0" * 64},
    {"dataset": "musdb18hq", "dataset_track_id": 7, "split": "train", "sha256": "0" * 64},
    {"dataset": "musdb18hq", "dataset_track_id": "A - B", "split": "unassigned", "sha256": "0" * 64},
    {"dataset": "musdb18hq", "dataset_track_id": "A - B", "sha256": "0" * 64},  # no split
    {"dataset": "musdb18hq", "dataset_track_id": "A - B", "split": "train", "sha256": "ABC"},
    {"dataset": "musdb18hq", "dataset_track_id": "A - B", "split": "train", "sha256": "A" * 64},  # upper case
    {"dataset": "musdb18hq", "dataset_track_id": "A - B", "split": "train"},  # no checksum
))
def test_malformed_records_stop_the_check(inputs, bad):
    records, *rest = inputs
    with pytest.raises(musdb.MusdbInputError) as caught:
        musdb.match([*records, bad], *rest)
    assert CANARY not in str(caught.value)


def test_other_datasets_are_ignored_but_an_empty_musdb_set_stops(inputs):
    _records, *rest = inputs
    with pytest.raises(musdb.MusdbInputError):
        musdb.match([record("A - B", "train", dataset="mdbdrums")], *rest)
    with pytest.raises(musdb.MusdbInputError):
        musdb.match([], *rest)


# -- 3-6: exact credit, aliases, one link step, checksums -------------------------------------------------


def test_exact_credit_matches_include_featured_artists_and_never_generic_names(result, inputs):
    pairs = private_pairs(result)
    assert pairs[(f"Alpha {CANARY} - First Song", f"paired-alpha-{CANARY}")] == ("exact",)
    records, registry, aliases, titles, categories = inputs
    registry["entries"]["paired-guest"] = entry("development", ["guest"])
    categories["paired-guest"] = "paired"
    registry["entries"]["paired-generic"] = entry("development", ["various artists", ""])
    categories["paired-generic"] = "paired"
    records.append(record("Various Artists - Compilation", "train"))
    again = private_pairs(musdb.match(records, registry, aliases, titles, categories))
    assert again[(f"Bravo {CANARY} - Bright Day (feat. Guest)", "paired-guest")] == ("exact",)  # credited in the title
    assert not any(k == "paired-generic" for _t, k in again)


def test_aliases_map_a_spelling_to_its_canonical_name(result):
    assert private_pairs(result)[(f"Spelling {CANARY} - Any", f"paired-alias-{CANARY}")] == ("exact",)


def test_link_expansion_is_exactly_one_step(result):
    pairs = private_pairs(result)
    assert pairs[(f"Linked {CANARY} - Any", f"paired-link-{CANARY}")] == ("exact",)  # linked -> partner
    assert (f"Linked {CANARY} - Any", f"paired-chain-{CANARY}") not in pairs  # partner -> second partner: no
    keys = musdb.exact_keys("A", "x", {}, [("a", "b", "r"), ("b", "c", "r"), ("d", "a", "r")])
    assert keys == {"a", "b", "d"}  # both directions, one step, nothing transitive


def test_checksum_matches_need_the_identical_sha256(result, inputs):
    pairs = private_pairs(result)
    assert pairs[("Some Artist - Copied Track", f"paired-check-{CANARY}")] == ("checksum",)
    records, *rest = inputs
    near = [r if r["dataset_track_id"] != "Some Artist - Copied Track" else dict(r, sha256="f" * 64) for r in records]
    assert ("Some Artist - Copied Track", f"paired-check-{CANARY}") not in private_pairs(musdb.match(near, *rest))


# -- 7, 8: report-only match types ----------------------------------------------------------------------


def test_title_confirmation_is_reported_but_never_needed_or_sufficient_for_exposure(result, inputs):
    pairs = private_pairs(result)
    titled = f"paired-titled-{CANARY}"
    assert pairs[(f"Titled {CANARY} - Night Owl (feat. Someone)", titled)] == ("exact", "title-confirmation")
    assert cell(result, "train", "paired", "title-confirmation") == {"groups": 1, "records": 1}
    # the same title under another artist: no exact match, so neither confirmation nor exposure
    records, registry, aliases, titles, categories = inputs
    swapped = [r for r in records if not r["dataset_track_id"].startswith("Titled")] + [record("Stranger - Night Owl", "train")]
    other = musdb.match(swapped, registry, aliases, titles, categories)
    assert not any(k == titled for _t, k in private_pairs(other)) and titled not in other.g_t


def test_a_checksum_match_with_a_matching_title_gets_no_title_confirmation(inputs):
    """Title confirmation needs an exact credit match; a checksum match is not enough."""
    records, registry, aliases, titles, categories = inputs
    own = next(n[4:] for n in registry["entries"][f"paired-titled-{CANARY}"]["nodes"] if n.startswith("rec:"))
    swapped = [r for r in records if not r["dataset_track_id"].startswith("Titled")]
    swapped.append(record("Stranger - Night Owl", "train", digest=own))  # same title, checksum, other artist
    pairs = private_pairs(musdb.match(swapped, registry, aliases, titles, categories))
    assert pairs[("Stranger - Night Owl", f"paired-titled-{CANARY}")] == ("checksum",)


def test_a_misspelt_partition_cannot_hide_a_test_split_match(inputs):
    records, registry, aliases, titles, categories = inputs
    registry["entries"]["mis"] = entry("Development", ["bravo two"])
    records.append(record("Bravo Two - Song", "test"))
    with pytest.raises(musdb.MusdbInputError):  # refused, not skipped
        musdb.match(records, registry, aliases, titles, categories)


def test_permissive_only_matches_are_reported_but_never_expose(result):
    pairs = private_pairs(result)
    perm = f"paired-perm-{CANARY}"
    assert pairs[(f"Charlie {CANARY} and Delta - Tune", perm)] == ("permissive-only",)
    assert perm not in result.g_t and perm not in result.g_v
    assert cell(result, "train", "paired", "permissive-only") == {"groups": 1, "records": 1}
    assert set(musdb.permissive_keys("Alpha and Beta (Gamma)", frozenset({"alpha beta"}))) == {
        "alpha and beta gamma", "alpha", "beta", "gamma", "alphabeta"}


# -- 9, 10: exposure sets and multiplicity -------------------------------------------------------------------


def test_exposure_sets_hold_paired_groups_with_credit_or_checksum_matches(result):
    assert result.g_t == {f"paired-{k}-{CANARY}" for k in ("alpha", "check", "alias", "link", "titled")}
    assert result.g_v == {f"paired-bravo-{CANARY}"}
    assert f"lowrate-{CANARY}" not in result.g_t and f"val-{CANARY}" not in result.g_v  # paired groups only
    assert cell(result, "train", "low-rate", "exact") == {"groups": 1, "records": 1}
    assert cell(result, "validation", "validation", "exact") == {"groups": 1, "records": 1}
    assert result.exposed == {"training": 5, "validation": 1, "both": 0}


def test_groups_with_several_records_or_match_types_are_counted_once_per_cell(inputs):
    records, registry, aliases, titles, categories = inputs
    alpha = f"paired-alpha-{CANARY}"
    registry["entries"][alpha]["nodes"].append(f"rec:{records[0]['sha256']}")  # exact and checksum at once
    records.append(record(f"Alpha {CANARY} - Third Song", "validation"))  # the same group in G_V too
    result = musdb.match(records, registry, aliases, titles, categories)
    assert private_pairs(result)[(f"Alpha {CANARY} - First Song", alpha)] == ("exact", "checksum")
    # Alpha's two training records, the alias, the link and the titled record
    assert cell(result, "train", "paired", "exact") == {"groups": 4, "records": 5}
    assert cell(result, "train", "paired", "checksum") == {"groups": 2, "records": 2}
    assert alpha in result.g_t and alpha in result.g_v and result.exposed["both"] == 1


# -- 11, 12: reserved groups and the test-split stop ----------------------------------------------------------


def test_reserved_and_other_partitions_are_never_assessed(inputs, result):
    assert not any("reserved" in k or "excluded" in k for _t, k in private_pairs(result))
    records, registry, aliases, titles, categories = inputs
    changed = copy.deepcopy(registry)
    changed["entries"][f"reserved-{CANARY}"] = {"partition": "reserved", "nodes": [f"name:alpha {CANARY.lower()}", "rec:bad"]}
    changed["entries"]["final"] = {"partition": "final_test", "nodes": None}
    assert musdb.match(records, changed, aliases, titles, categories) == result  # nothing outside is read


@pytest.mark.parametrize(("track_id", "digest"), (
    (f"Alpha {CANARY} - Leaked", None),  # credit match to a paired group
    ("Anyone - Leaked", "checksum"),  # checksum match
    (f"Lowrate {CANARY} - Leaked", None),  # a low-rate development group
    (f"Valgroup {CANARY} - Leaked", None),  # an A5.1 validation group
))
def test_a_credit_or_checksum_match_to_a_test_record_stops_everything(inputs, track_id, digest):
    records, registry, aliases, titles, categories = inputs
    # a paired group's own recording that no other MUSDB record carries
    own = next(n[4:] for n in registry["entries"][f"paired-titled-{CANARY}"]["nodes"] if n.startswith("rec:"))
    records.append(record(track_id, "test", digest=own if digest else None))
    with pytest.raises(musdb.TestSplitOverlap) as caught:
        musdb.match(records, registry, aliases, titles, categories)
    assert CANARY not in str(caught.value) and "1 record(s), 1 group(s)" in str(caught.value)


def test_report_only_matches_to_test_records_do_not_stop(inputs):
    records, *rest = inputs
    records.append(record(f"Charlie {CANARY} and Delta - Tune 2", "test"))  # permissive only
    result = musdb.match(records, *rest)
    assert cell(result, "test", "paired", "permissive-only") == {"groups": 1, "records": 1}


# -- 13, 14: public counts and privacy ----------------------------------------------------------------------


FINGERPRINTS = {name: sha(name) for name in musdb.INPUTS}


def test_every_count_cell_exists_including_zeros(result):
    report = musdb.public_report(result, FINGERPRINTS, "5f308be")
    assert set(report["counts"]) == set(musdb.SPLITS)
    for split in musdb.SPLITS:
        assert set(report["counts"][split]) == set(musdb.PARTITIONS)
        for partition in musdb.PARTITIONS:
            assert set(report["counts"][split][partition]) == set(musdb.MATCH_TYPES)
            assert all(set(v) == {"groups", "records"} for v in report["counts"][split][partition].values())
    assert report["counts"]["test"]["validation"]["checksum"] == {"groups": 0, "records": 0}
    assert report["musdb_records"] == {"train": 9, "validation": 2, "test": 2}
    assert report["limitations"] == list(musdb.LIMITATIONS)


def test_public_output_holds_no_private_identifier(result):
    report = musdb.public_report(result, FINGERPRINTS, "5f308be")
    text = json.dumps(report)
    assert CANARY not in text and CANARY.lower() not in text
    assert not any(v in text for v in FINGERPRINTS.values())  # only a digest of them
    assert set(report) == {"format", "commit", "input_digest_sha256", "musdb_records", "unparsed_records", "counts",
                           "exposed_paired_groups", "limitations"}
    assert CANARY not in repr(result) and CANARY.lower() not in repr(result)  # private fields are hidden
    assert repr(result.private) == str(result.private) == "<private: withheld>"
    with pytest.raises(TypeError):
        json.dumps({"private": result.private})


NESTED = '''
import secrets
from pathlib import Path

from gmv.character.research.presence_model import musdb

OUT = Path(@OUT@)


def results():
    canary = "CANARY" + secrets.token_hex(6)  # made at run time, so the source printed in a traceback holds none
    OUT.write_text(OUT.read_text() + canary + "\\n" if OUT.exists() else canary + "\\n")
    key = "g-" + canary
    records = [{"dataset": "musdb18hq", "dataset_track_id": canary + " - Song", "split": "train", "sha256": "a" * 64},
               {"dataset": "musdb18hq", "dataset_track_id": "Other - Song", "split": "train", "sha256": "b" * 64}]
    registry = {"entries": {key: {"partition": "development", "nodes": ["name:" + canary.lower()]}}}
    one = musdb.match(records, registry, ({}, []), {}, {key: "paired"})
    two = musdb.match(records[1:], registry, ({}, []), {}, {key: "paired"})
    assert one.matches and not two.matches
    return one, two


def test_differing_results_fail():
    one, two = results()
    assert one == two


def test_differing_private_parts_fail():
    one, two = results()
    assert one.private == two.private
'''


def test_a_failed_comparison_of_results_prints_no_private_identifier(tmp_path):
    """pytest's failure diff of two dataclasses walks every field; the private part must still
    show nothing, even with the most verbose diff."""
    out = tmp_path / "canary.txt"
    test_file = tmp_path / "test_nested_result.py"
    test_file.write_text(NESTED.replace("@OUT@", repr(str(out))))
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=long", "-vv",
                           str(test_file)], cwd=tmp_path, capture_output=True, text=True, check=False)
    output = done.stdout + done.stderr
    assert done.returncode == 1 and "2 failed" in output
    canaries = out.read_text().split()
    assert len(canaries) == 2
    assert not any(c in output or c.lower() in output for c in canaries)
    assert "<private: withheld>" in output  # the differing private parts were shown only as withheld


# -- 15: determinism --------------------------------------------------------------------------------------


def test_results_do_not_depend_on_input_order(inputs, result):
    records, registry, aliases, titles, categories = inputs
    rng = random.Random(7)
    for _ in range(5):
        shuffled_records = rng.sample(records, len(records))
        items = list(registry["entries"].items())
        rng.shuffle(items)
        links = rng.sample(aliases[1], len(aliases[1]))
        again = musdb.match(shuffled_records, {"entries": dict(items)}, (aliases[0], links), titles,
                            dict(rng.sample(sorted(categories.items()), len(categories))))
        assert again == result and again.matches == result.matches
        assert musdb.public_report(again, FINGERPRINTS, "abc1234") == musdb.public_report(result, FINGERPRINTS, "abc1234")


# -- 16: missing, malformed, duplicated and conflicting inputs ----------------------------------------------


def broken(inputs, what):
    records, registry, aliases, titles, categories = copy.deepcopy(inputs)
    paired = f"paired-alpha-{CANARY}"
    if what == "duplicate track id":
        records.append(dict(records[0], sha256=sha("other")))
    elif what == "duplicate checksum":
        records.append(record("Another - Song", "train", digest=records[0]["sha256"]))
    elif what == "registry without entries":
        registry = {}
    elif what == "entry without partition":
        registry["entries"]["x"] = {"nodes": []}
    elif what == "malformed nodes":
        registry["entries"][paired]["nodes"] = "name:alpha"
    elif what == "malformed recording checksum":
        registry["entries"][paired]["nodes"].append("rec:XYZ")
    elif what == "node shared by two groups":
        registry["entries"][f"paired-bravo-{CANARY}"]["nodes"].append(f"name:alpha {CANARY.lower()}")
    elif what == "missing category":
        del categories[paired]
    elif what == "category for a non-development key":
        categories[f"val-{CANARY}"] = "paired"
    elif what == "unknown category":
        categories[paired] = "elsewhere"
    elif what == "malformed aliases":
        aliases = ({"a": 1}, [])
    elif what == "malformed links":
        aliases = (aliases[0], [("only-one",)])
    elif what == "malformed titles":
        titles = ["not", "a", "mapping"]
    elif what == "malformed title":
        own = next(n[4:] for n in registry["entries"][f"paired-titled-{CANARY}"]["nodes"] if n.startswith("rec:"))
        titles[own] = 12
    elif what == "no assessed group":
        registry = {"entries": {"r": entry("reserved", ["x"])}}
        categories = {}
    elif what == "misspelt partition":
        registry["entries"]["mis"] = entry("Development", [f"alpha {CANARY.lower()} two"])
    elif what == "hyphenated partition":
        registry["entries"]["mis"] = entry("final-test", ["x"])
    elif what == "empty alias value":
        aliases = ({"zed": ""}, aliases[1])
    elif what == "empty link end":
        aliases = (aliases[0], [("", "a", "r")])
    elif what == "unnormalised alias":
        aliases = ({"The Alpha": "alfa"}, aliases[1])
    return records, registry, aliases, titles, categories


@pytest.mark.parametrize("what", (
    "duplicate track id", "duplicate checksum", "registry without entries", "entry without partition",
    "malformed nodes", "malformed recording checksum", "node shared by two groups", "missing category",
    "category for a non-development key", "unknown category", "malformed aliases", "malformed links",
    "malformed titles", "malformed title", "no assessed group", "misspelt partition", "hyphenated partition",
    "empty alias value", "empty link end", "unnormalised alias",
))
def test_missing_malformed_duplicated_or_conflicting_inputs_stop(inputs, what):
    with pytest.raises(musdb.MusdbInputError) as caught:
        musdb.match(*broken(inputs, what))
    assert CANARY not in str(caught.value) and CANARY.lower() not in str(caught.value)


@pytest.mark.parametrize(("fingerprints", "commit"), (
    ({k: v for k, v in FINGERPRINTS.items() if k != "titles"}, "abc1234"),  # an input missing
    ({**FINGERPRINTS, "extra": sha("x")}, "abc1234"),  # an unexpected input
    ({**FINGERPRINTS, "titles": "not-a-hash"}, "abc1234"),
    (FINGERPRINTS, "not a commit"),
))
def test_the_public_report_refuses_malformed_fingerprints_or_commit(result, fingerprints, commit):
    with pytest.raises(musdb.MusdbInputError):
        musdb.public_report(result, fingerprints, commit)


def test_titles_of_recordings_outside_assessed_groups_are_never_used(inputs, result):
    records, registry, aliases, titles, categories = inputs
    extra = dict(titles, **{sha("someone else's recording"): 12, "not-a-checksum": None})  # never looked up
    assert musdb.match(records, registry, aliases, extra, categories) == result
