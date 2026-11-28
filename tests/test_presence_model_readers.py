"""A5.2 Step 2a, S2a.1: the strict step 2 readers (protocol Amendment 3, A3.4 and A3.6), on
generated files only. Every refusal message must carry counts only: canaries go into names,
titles, credits and keys, and no message may contain one."""

import json
import secrets
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step2_synth  # noqa: E402

from gmv.character.research.presence import identity, project  # noqa: E402
from gmv.character.research.presence_model import readers  # noqa: E402
from gmv.character.research.presence_model.readers import ReaderError  # noqa: E402

CANARY = "CANARY-" + secrets.token_hex(8)


def refused(call, *args):
    with pytest.raises(ReaderError) as caught:
        call(*args)
    assert CANARY not in str(caught.value) and CANARY.lower() not in str(caught.value)
    return str(caught.value)


@pytest.fixture
def world(tmp_path):
    records = [step2_synth.corpus_record(f"Artist {CANARY} - Song", "train"),
               step2_synth.corpus_record(f"Other {CANARY} - Tune", "validation"),
               step2_synth.corpus_record(f"Third {CANARY} - Air", "test")]
    return step2_synth.build(tmp_path, records=records, aliases={f"Spelling {CANARY}": f"artist {CANARY}"},
                             links=[(f"left {CANARY}", f"right {CANARY}", "same person")],
                             titles={"group-00": f"Title {CANARY}"})


# -- corpus ---------------------------------------------------------------------------------------------


def test_the_corpus_reader_keeps_musdb_fields_only(world):
    loaded = readers.read_corpus(world["corpus"])
    assert [split for _t, split, _h in loaded.value["musdb"]] == ["train", "validation", "test"]
    assert loaded.value["other"] == 1
    assert all(set(map(type, r)) == {str} for r in loaded.value["musdb"])
    assert loaded.sha256 == step2_synth.sha_of(world["corpus"].read_text())


def test_the_tracked_empty_corpus_manifest_fails_closed(world):
    assert "no MUSDB18-HQ record" in refused(readers.read_corpus, world["tracked_empty"])


def test_an_extra_corpus_field_is_refused(world, tmp_path):
    record = step2_synth.corpus_record(f"Artist {CANARY} - Song", "train")
    record["unexpected"] = CANARY
    path = step2_synth.write_corpus(tmp_path / "bad.json", [record])
    assert "corpus schema" in refused(readers.read_corpus, path)


def test_an_unassigned_musdb_split_is_passed_on_for_the_matcher_to_refuse(tmp_path):
    path = step2_synth.write_corpus(tmp_path / "c.json", [step2_synth.corpus_record("A - B", "unassigned")])
    assert readers.read_corpus(path).value["musdb"][0][1] == "unassigned"


@pytest.mark.parametrize("text", ("not json", '{"recordings": [], "recordings": []}', "\xff"))
def test_malformed_or_ambiguous_json_is_refused(tmp_path, text):
    path = tmp_path / "c.json"
    path.write_bytes(text.encode("latin-1"))
    refused(readers.read_corpus, path)


def test_a_missing_file_is_refused(tmp_path):
    assert "missing" in refused(readers.read_corpus, tmp_path / "absent.json")


# -- registry -------------------------------------------------------------------------------------------


def test_the_registry_reader_keeps_assessed_entries_and_partitions_only(tmp_path):
    w = step2_synth.build(tmp_path, reserved={f"reserved-{CANARY}": [f"reserved {CANARY}"]})
    entries = readers.read_registry(w["root"]).value["entries"]
    assert entries[f"reserved-{CANARY}"] == {"partition": "reserved"}  # nodes of other entries are never passed on
    assert all(set(e) == {"partition", "nodes"} for e in entries.values() if e["partition"] in ("development", "validation"))


def test_a_registry_that_differs_from_its_fingerprint_is_refused(world):
    path = world["registry"]
    path.write_text(path.read_text().replace("development", "developmenT", 1))
    assert "fingerprint" in refused(readers.read_registry, world["root"])


@pytest.mark.parametrize(("change", "words"), (
    (lambda r: r.update(extra=CANARY), "top-level"),
    (lambda r: next(iter(r["entries"].values())).update(colour=CANARY), "unexpected"),
    (lambda r: next(iter(r["entries"].values()))["nodes"].append(f"odd:{CANARY}"), "node type"),
    (lambda r: next(iter(r["entries"].values())).update(partition="developmnt"), "partition"),
    (lambda r: next(iter(r["entries"].values()))["nodes"].append("rec:not-a-checksum"), "checksum"),
))
def test_unexpected_registry_content_is_refused(world, change, words):
    registry = project.read_registry(world["root"])
    change(registry)
    project.write_registry(world["root"], registry)  # a correctly fingerprinted but malformed registry
    assert words in refused(readers.read_registry, world["root"])


# -- aliases --------------------------------------------------------------------------------------------


def test_the_alias_file_is_mandatory(world):
    world["aliases"].unlink()
    assert "missing" in refused(readers.read_aliases, world["aliases"])


def test_an_explicit_empty_alias_map_is_accepted(world):
    world["aliases"].write_text(json.dumps({"aliases": {}, "links": []}))
    assert readers.read_aliases(world["aliases"]).value == ({}, [])


def test_aliases_are_normalised(world):
    alias_map, links = readers.read_aliases(world["aliases"]).value
    n = identity.normalise
    assert alias_map == {n(f"Spelling {CANARY}"): n(f"artist {CANARY}")}
    assert links == [(n(f"left {CANARY}"), n(f"right {CANARY}"))]


@pytest.mark.parametrize(("content", "words"), (
    ({"aliases": {}, "links": [], "note": CANARY}, "exactly aliases and links"),
    ({"aliases": {f"A {CANARY}": "x", f"a {CANARY}": "y"}, "links": []}, "different targets"),
    ({"aliases": {f"a {CANARY}": "b", "b": "c"}, "links": []}, "themselves alias spellings"),
    ({"aliases": {f"{CANARY}": "Various Artists"}, "links": []}, "empty or generic"),
    ({"aliases": {"!!!": f"{CANARY}"}, "links": []}, "empty or generic"),
    ({"aliases": {}, "links": [[f"a {CANARY}", "b"]]}, "malformed"),
    ({"aliases": {}, "links": [[f"a {CANARY}", f"A {CANARY}", "same"]]}, "itself"),
    ({"aliases": {f"s {CANARY}": "t"}, "links": [[f"s {CANARY}", "u", "why"]]}, "alias spellings"),
))
def test_alias_file_faults_stop_step_2(world, content, words):
    world["aliases"].write_text(json.dumps(content))
    assert words in refused(readers.read_aliases, world["aliases"])


def test_a_repeated_alias_key_in_the_file_is_refused(world):
    world["aliases"].write_text('{"aliases": {"x": "y", "x": "z"}, "links": []}')
    assert "repeats a key" in refused(readers.read_aliases, world["aliases"])


# -- titles ---------------------------------------------------------------------------------------------


def test_titles_keep_only_the_assessed_recordings(world):
    keep = world["recordings"]["group-00"][:1]
    assert readers.read_titles(world["titles"], keep).value == {keep[0]: f"Title {CANARY}"}
    assert readers.read_titles(world["titles"], []).value == {}


@pytest.mark.parametrize("content", ({"not-a-key": "t"}, {"a" * 64: 3}))
def test_malformed_titles_are_refused(world, content):
    world["titles"].write_text(json.dumps(content))
    refused(readers.read_titles, world["titles"], [])


# -- readings and decisions -----------------------------------------------------------------------------


def test_readings_are_read_and_checked(tmp_path):
    path = step2_synth.write_readings(tmp_path / "r.json", {f"X and {CANARY}": "whole-and-pieces",
                                                           f"Y And {CANARY}": ("single-name", "one band")})
    assert readers.read_readings(path).value == {f"X and {CANARY}": "whole-and-pieces", f"Y And {CANARY}": "single-name"}


@pytest.mark.parametrize("reading", ({"reading": "single-name"}, {"reading": "single-name", "reason": " "},
                                     {"reading": "guess"}, "whole-and-pieces"))
def test_malformed_readings_are_refused(tmp_path, reading):
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"format": readers.READINGS_FORMAT, "version": 1, "readings": {f"A and {CANARY}": reading}}))
    refused(readers.read_readings, path)


@pytest.mark.parametrize("content", ({"format": "other", "version": 1, "readings": {}},
                                     {"format": readers.READINGS_FORMAT, "version": 2, "readings": {}},
                                     {"format": readers.READINGS_FORMAT, "version": 1, "readings": {}, "x": 1}))
def test_the_readings_format_is_strict(tmp_path, content):
    path = tmp_path / "r.json"
    path.write_text(json.dumps(content))
    refused(readers.read_readings, path)


def test_an_empty_readings_file_is_valid_and_a_missing_one_is_not(tmp_path):
    path = step2_synth.write_readings(tmp_path / "r.json", {})
    assert readers.read_readings(path).value == {}
    refused(readers.read_readings, tmp_path / "absent.json")


def test_review_decisions_are_read(tmp_path):
    path = step2_synth.write_decisions(tmp_path / "d.json", [(f"A {CANARY} - B", "group-00")])
    assert readers.read_decisions(path).value == frozenset({(f"A {CANARY} - B", "group-00")})
    assert readers.read_decisions(step2_synth.write_decisions(tmp_path / "e.json", [])).value == frozenset()


@pytest.mark.parametrize("item", ({"track_id": "t", "group": "g", "reason": "I think so"},
                                  {"track_id": "t", "group": "g"},
                                  {"track_id": "", "group": "g", "reason": "not-the-same-artist"}))
def test_malformed_review_decisions_are_refused(tmp_path, item):
    path = tmp_path / "d.json"
    path.write_text(json.dumps({"format": readers.DECISIONS_FORMAT, "version": 1, "rejections": [item]}))
    refused(readers.read_decisions, path)


def test_a_repeated_review_decision_is_refused(tmp_path):
    path = step2_synth.write_decisions(tmp_path / "d.json", [("t", "g"), ("t", "g")])
    assert "repeated" in refused(readers.read_decisions, path)


def test_loaded_inputs_never_print_their_content(world):
    loaded = readers.read_titles(world["titles"], world["recordings"]["group-00"])
    assert CANARY not in repr(loaded) and CANARY not in str(loaded)
