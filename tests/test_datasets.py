import json

import numpy as np
import pytest
import soundfile as sf

from gmv import cli
from gmv.corpus import registry
from gmv.corpus.cli import validate_corpus
from gmv.corpus.datasets import (
    adapters,
    import_dataset,
    mdbdrums,
    moisesdb,
    musdb,
    read_sum,
)
from gmv.evaluation.real import evaluate_corpus

RATE = 8000


def tone(seconds=4.0, frequency=220.0, gate=None, level=0.2):
    time = np.arange(round(seconds * RATE)) / RATE
    audio = level * np.sin(2 * np.pi * frequency * time)
    if gate:
        audio *= sum(((time >= s) & (time < e)) for s, e in gate)
    return np.column_stack((audio, audio)).astype(np.float32)


def write(path, audio):
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, RATE, subtype="FLOAT")
    return path


def fake_musdb(root, tracks):
    """tracks: {(split, 'Artist - Title'): {stem: gate}}; mixture is the exact stem sum."""
    for index, ((split, name), gates) in enumerate(tracks.items()):
        parts = {
            stem: tone(frequency=110 * (2 + position) + index, gate=gates.get(stem, [(9, 9)]))
            for position, stem in enumerate(musdb.STEMS)
        }
        for stem, audio in parts.items():
            write(root / split / name / f"{stem}.wav", audio)
        write(root / split / name / "mixture.wav", sum(parts.values()))
    return root


def test_musdb_import_registers_stems_references_and_leak_free_splits(tmp_path):
    root = fake_musdb(
        tmp_path / "musdb18hq",
        {
            ("train", "Alpha - One"): {"drums": [(0, 2)], "bass": [(1, 3)]},
            ("train", "Alpha - Two"): {"vocals": [(0, 1)]},
            ("train", "Beta - Song"): {"other": [(2, 4)]},
            ("test", "Gamma - Held"): {"drums": [(0, 4)], "vocals": [(1, 2)]},
            ("test", "Alpha - Hidden"): {"bass": [(0, 1)]},
        },
    )
    manifest_path = tmp_path / "corpus" / "manifest.json"
    summary = import_dataset(manifest_path, musdb.ADAPTER, root, progress=lambda *_: None)
    assert summary["imported"] == 5 and summary["failed"] == []
    assert summary["layers"]["activity.drums"] == {"stem_derived": 5}
    manifest = registry.load_manifest(manifest_path)
    by_track = {r.dataset_track_id: r for r in manifest.recordings}
    # An artist that the dataset places in both train and test goes to test as a whole.
    assert {by_track[n].split for n in ("Alpha - One", "Alpha - Two", "Alpha - Hidden")} == {"test"}
    assert by_track["Gamma - Held"].split == "test"
    assert by_track["Beta - Song"].split in ("train", "validation")
    assert by_track["Alpha - One"].group_id == by_track["Alpha - Two"].group_id
    first = by_track["Alpha - One"]
    assert first.source_kind == "open_dataset" and first.dataset == "musdb18hq"
    assert set(first.stems) == set(musdb.STEMS) and "musdb18hq" in first.tags
    assert "non-commercial" in first.license
    assert str(tmp_path) not in manifest_path.read_text(encoding="utf-8")

    reference = registry.annotations_for(manifest_path, first.recording_id)[0]
    assert reference.reference_kind == "stem_derived" and reference.usable_as_reference
    spans = {a.label: (round(a.start, 1), round(a.end, 1)) for a in reference.activity}
    assert spans == {"drums": (0.0, 2.0), "bass": (1.0, 3.0)}  # silent stems: annotated, absent
    assert reference.covers("activity.vocals")
    assert validate_corpus(manifest_path) == ([], [])

    again = import_dataset(manifest_path, musdb.ADAPTER, root, progress=lambda *_: None)
    assert again["imported"] == 0 and again["already_registered"] == 5


def test_dataset_evaluation_reports_activity_as_stem_derived_and_attacks_as_unavailable(tmp_path):
    root = fake_musdb(tmp_path / "ds", {("test", "Gamma - Held"): {"drums": [(0, 2)]}})
    manifest_path = tmp_path / "corpus" / "manifest.json"
    import_dataset(manifest_path, musdb.ADAPTER, root, progress=lambda *_: None)
    report = evaluate_corpus(manifest_path, "test", final=True, dataset="musdb18hq")
    assert report["evaluated"] == 1 and report["dataset_filter"] == "musdb18hq"
    result = report["recordings"][0]
    assert result["recording"]["dataset_track_id"] == "Gamma - Held"
    assert result["onsets"] == {"status": "not_annotated"}
    assert result["beats"] == {"status": "not_annotated"}
    assert result["instrument_events"] == {}
    assert result["candidates"]["snare_like_candidate"]["status"] == "not_annotated"
    drums = result["instrument_activity"]["drums"]
    assert drums["reference_kind"] == "stem_derived" and drums["status"] == "no_detector_output"
    assert drums["reference_active_seconds"] == pytest.approx(2, abs=0.1)
    assert result["candidates_vs_activity"]["snare_like_candidate"]["checked_against"] == ["drums"]
    assert evaluate_corpus(manifest_path, "test", final=True, dataset="other")["evaluated"] == 0


def test_import_cli_lists_datasets_and_reports_bad_folders(tmp_path, capsys):
    manifest = str(tmp_path / "corpus" / "manifest.json")
    cli.main(["corpus", "--manifest", manifest, "datasets"])
    listed = {d["name"]: d for d in json.loads(capsys.readouterr().out)}
    assert "musdb18hq" in listed and listed["musdb18hq"]["audio_license"]
    assert set(listed) == set(adapters())
    root = fake_musdb(tmp_path / "ds", {("train", "Solo - Track"): {"bass": [(0, 1)]}})
    cli.main(["corpus", "--manifest", manifest, "import-dataset", "musdb18hq", str(root)])
    assert '"imported": 1' in capsys.readouterr().out
    (tmp_path / "empty").mkdir()
    with pytest.raises(SystemExit, match="No MUSDB18-HQ tracks"):
        cli.main(
            [
                "corpus",
                "--manifest",
                manifest,
                "import-dataset",
                "musdb18hq",
                str(tmp_path / "empty"),
            ]
        )
    with pytest.raises(SystemExit, match="Unknown dataset"):
        cli.main(["corpus", "--manifest", manifest, "import-dataset", "nope", str(root)])


def test_read_sum_pads_and_rejects_mismatched_stems(tmp_path):
    a = write(tmp_path / "a.wav", tone(1.0))
    b = write(tmp_path / "b.wav", tone(1.2, level=0.1))
    audio, rate = read_sum([a, b])
    assert rate == RATE and len(audio) == round(1.2 * RATE)
    assert float(np.abs(audio).max()) == pytest.approx(0.3, abs=0.01)
    long = write(tmp_path / "c.wav", tone(3.0))
    with pytest.raises(ValueError, match="lengths differ"):
        read_sum([a, long])


def test_mdbdrums_import_keeps_dataset_attacks_apart_from_stem_activity(tmp_path):
    base = tmp_path / "MDBDrums" / "MDB Drums"
    drum = tone(gate=[(0.5, 2.5)])
    write(base / "audio" / "drum_only" / "MusicDelta_Punk_Drum.wav", drum)
    write(base / "audio" / "full_mix" / "MusicDelta_Punk_MIX.wav", drum + tone(frequency=330))
    for folder, name, text in (
        ("class", "MusicDelta_Punk_class.txt", "0.5\tKD\n1.0\tSD\n1.0\tHH\n1.5\tCY\n2.0\tOT\n"),
        ("beats", "MusicDelta_Punk_MIX.beats", "0.5\t1\n1.0\t2\n1.5\t3\n2.0\t4\n2.5\t1\n"),
    ):
        (base / "annotations" / folder).mkdir(parents=True, exist_ok=True)
        (base / "annotations" / folder / name).write_text(text, encoding="utf-8")
    manifest_path = tmp_path / "corpus" / "manifest.json"
    summary = import_dataset(
        manifest_path, mdbdrums.ADAPTER, tmp_path / "MDBDrums", progress=lambda *_: None
    )
    assert summary["imported"] == 1 and summary["failed"] == []
    assert summary["layers"]["hits.kick"] == {"dataset_annotation": 1}
    assert summary["layers"]["activity.drums"] == {"stem_derived": 1}
    recording = registry.load_manifest(manifest_path).recordings[0]
    kinds = {
        a.annotator_id: a for a in registry.annotations_for(manifest_path, recording.recording_id)
    }
    truth = kinds["mdbdrums"]
    assert truth.reference_kind == "dataset_annotation"
    hits = {}
    for hit in truth.instrument_events:
        hits.setdefault(hit.label, []).append(hit.time)
    assert hits["drums"] == [0.5, 1.0, 1.5, 2.0] and hits["kick"] == [0.5]
    assert hits["hi_hat"] == [1.0] and hits["cymbals"] == [1.5] and "toms" not in hits
    assert truth.covers("hits.toms") and truth.covers("beat") and truth.covers("downbeat")
    assert [e.time for e in truth.events if e.kind == "downbeat"] == [0.5, 2.5]
    assert kinds["stem-energy"].reference_kind == "stem_derived"

    report = evaluate_corpus(manifest_path, recording.split, final=True)
    result = report["recordings"][0]
    assert result["onsets"] == {"status": "not_annotated"}
    assert result["beats"]["reference_kind"] == "dataset_annotation"
    assert result["instrument_events"]["kick"]["reference_kind"] == "dataset_annotation"
    assert "full_mix_onset_recall_50ms" in result["instrument_events"]["kick"]
    assert result["candidates"]["snare_like_candidate"]["compared_with"] == ["snare"]
    assert result["instrument_activity"]["drums"]["reference_kind"] == "stem_derived"
    recall = report["aggregate"]["onset_recall_on_instrument_attacks"]["drums"]
    assert recall["reference_count"] == 4 and recall["reference_kinds"] == ["dataset_annotation"]


def test_same_artist_shares_a_group_across_datasets(tmp_path):
    fake_musdb(tmp_path / "musdb", {("test", "Music Delta - Punk"): {"drums": [(0, 1)]}})
    base = tmp_path / "mdb" / "MDB Drums"
    write(base / "audio" / "full_mix" / "MusicDelta_Rock_MIX.wav", tone(frequency=300))
    (base / "annotations" / "class").mkdir(parents=True)
    (base / "annotations" / "class" / "MusicDelta_Rock_class.txt").write_text("0.5\tKD\n")
    manifest_path = tmp_path / "corpus" / "manifest.json"
    import_dataset(manifest_path, musdb.ADAPTER, tmp_path / "musdb", progress=lambda *_: None)
    import_dataset(manifest_path, mdbdrums.ADAPTER, tmp_path / "mdb", progress=lambda *_: None)
    first, second = registry.load_manifest(manifest_path).recordings
    assert first.group_id == second.group_id and first.split == second.split == "test"
    assert {first.dataset, second.dataset} == {"musdb18hq", "mdbdrums"}


def test_moisesdb_import_renders_mix_maps_drum_mics_and_flags_bleed(tmp_path):
    track = tmp_path / "moisesdb" / "moisesdb_v0.1" / "1111-aaaa"
    stems = {
        "drums": [("k1", "kick drum", False, [(0, 1)]), ("o1", "overheads", True, [(0, 2)])],
        "bass": [("b1", "bass guitar", False, [(1, 3)])],
        "guitar": [("g1", "distorted electric guitar", False, [(2, 4)])],
        "other keys": [("p1", "synth pad", False, [(3, 4)])],
    }
    data = {"artist": "Some Band", "song": "Song", "genre": "Rock", "stems": []}
    for position, (stem, items) in enumerate(stems.items()):
        entry = {"id": f"s{position}", "stemName": stem, "tracks": []}
        for number, (identifier, kind, bleed, gate) in enumerate(items):
            write(
                track / stem.replace(" ", "_") / f"{identifier}.wav",
                tone(frequency=100 * (position + 1) + 10 * number, gate=gate),
            )
            entry["tracks"].append(
                {"id": identifier, "extension": "wav", "trackType": kind, "has_bleed": bleed}
            )
        data["stems"].append(entry)
    (track / "data.json").write_text(json.dumps(data), encoding="utf-8")
    manifest_path = tmp_path / "corpus" / "manifest.json"
    summary = import_dataset(
        manifest_path, moisesdb.ADAPTER, tmp_path / "moisesdb", progress=lambda *_: None
    )
    assert summary["imported"] == 1 and summary["failed"] == []
    recording = registry.load_manifest(manifest_path).recordings[0]
    assert recording.dataset_track_id == "1111-aaaa" and "genre_rock" in recording.tags
    assert set(recording.stems) == {"kick", "bass", "guitar", "other_harmonic"}  # single files
    mix = registry.load_paths(manifest_path)["recordings"][recording.recording_id]
    assert mix.endswith(".flac") and "corpus-local" in mix
    reference = registry.annotations_for(manifest_path, recording.recording_id)[0]
    spans = {a.label: (round(a.start, 1), round(a.end, 1), a.certainty) for a in reference.activity}
    assert spans["kick"] == (0.0, 1.0, "certain")
    assert spans["drums"] == (0.0, 2.0, "uncertain")  # overheads carry bleed
    assert spans["guitar"] == (2.0, 4.0, "certain") and spans["other_harmonic"][0] == 3.0
    assert not reference.covers("activity.snare")  # no snare microphone: not annotated
    assert validate_corpus(manifest_path) == ([], [])
