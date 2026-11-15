import json

import numpy as np
import pytest
import soundfile as sf
from pydantic import ValidationError

from gmv import cli
from gmv.corpus import audacity, registry, stems
from gmv.corpus.cli import validate_corpus
from gmv.corpus.compare import compare, consensus
from gmv.corpus.intervals import intersect, merge, rasterize
from gmv.corpus.schema import (
    Activity,
    AnnotationFile,
    CorpusManifest,
    Coverage,
    MixEvent,
    Recording,
    TimeRange,
)

RATE = 8000


def tone(seconds=4.0, frequency=220.0, gate=None):
    time = np.arange(round(seconds * RATE)) / RATE
    audio = 0.3 * np.sin(2 * np.pi * frequency * time)
    if gate:
        audio *= sum(((time >= s) & (time < e)) for s, e in gate)
    return np.column_stack((audio, audio)).astype(np.float32)


def write(path, audio):
    sf.write(path, audio, RATE, subtype="FLOAT")
    return path


def recording(sha="a" * 64, group="g1", split="unassigned", **extra):
    return Recording(
        recording_id="r-" + sha[:12],
        sha256=sha,
        duration_s=4.0,
        sample_rate=RATE,
        channels=2,
        source_kind="original",
        license="CC0-1.0",
        permission_basis="test fixture",
        group_id=group,
        split=split,
        **extra,
    )


def annotation(rec, annotator="ann1", status="reviewed", **parts):
    layers = parts.pop("layers")
    return AnnotationFile(
        recording_id=rec.recording_id,
        recording_sha256=rec.sha256,
        annotator_id=annotator,
        tool="test",
        provenance="test",
        status=status,
        coverage=Coverage(ranges=(TimeRange(start=0, end=4),), layers=layers),
        **parts,
    )


def run(tmp_path, *argv):
    return cli.main(["corpus", "--manifest", str(tmp_path / "corpus" / "manifest.json"), *argv])


def test_manifest_rejects_leakage_duplicates_and_wrong_ids():
    a, b = recording("a" * 64, split="train"), recording("b" * 64, split="test")
    with pytest.raises(ValidationError, match="leakage"):
        CorpusManifest(corpus_version="0.1.0", recordings=(a, b))
    with pytest.raises(ValidationError, match="Duplicate"):
        CorpusManifest(corpus_version="0.1.0", recordings=(a, a))
    with pytest.raises(ValidationError, match="recording_id"):
        Recording(**{**a.model_dump(), "recording_id": "r-000000000000"})
    assert CorpusManifest(
        corpus_version="0.1.0", recordings=(a, recording("b" * 64, group="g2", split="test"))
    )


def test_annotation_requires_declared_layers_and_valid_ranges():
    rec = recording()
    with pytest.raises(ValidationError, match="undeclared"):
        annotation(rec, layers=("onset",), events=(MixEvent(time=1, kind="beat"),))
    with pytest.raises(ValidationError, match="Unknown annotation layers"):
        annotation(rec, layers=("activity.theremin",))
    with pytest.raises(ValidationError):
        Activity(start=2, end=1, label="bass")
    draft = annotation(rec, status="machine_draft", layers=("onset",))
    assert not draft.usable_as_reference and annotation(rec, layers=("onset",)).usable_as_reference


def test_interval_helpers():
    assert merge([(0, 1), (0.5, 2), (3, 4)]) == [(0, 2), (3, 4)]
    assert merge([(0, 1), (1.1, 2)], gap=0.2) == [(0, 2)]
    assert intersect([(0, 2), (3, 5)], [(1, 4)]) == [(1, 2), (3, 4)]
    assert rasterize([(0.1, 0.2)], 0.3, hop=0.1).tolist() == [False, True, False]


def test_audacity_round_trip_keeps_overlap_points_uncertainty_and_empty_layers(tmp_path):
    rec, folder = recording(), tmp_path / "labels"
    folder.mkdir()
    (folder / "onset.txt").write_text("0.500000\t0.500000\t\n1,250\t1,250\t?\n", encoding="utf-8")
    (folder / "tempo.txt").write_text("0\t0\t174 bpm\n", encoding="utf-8")
    (folder / "sections.txt").write_text("2.0\t3.0\tbreakdown\n3.0\t3.0\ttransition\n")
    (folder / "activity.kick.txt").write_text("0.0\t3.0\t\n")
    (folder / "activity.rhythm_guitar.txt").write_text("1.0\t4.0\t?\n\\\t80\t5000\n")
    (folder / "hits.snare.txt").write_text("1.0\t1.0\tx\n")
    (folder / f"{rec.recording_id}.activity.vocals_harsh.txt").write_text("")
    result = audacity.import_folder(folder, rec, "ann1")
    assert [e.certainty for e in result.events if e.kind == "onset"] == ["certain", "uncertain"]
    assert [e.value for e in result.events if e.kind == "tempo_change"] == [174.0]
    assert {e.kind for e in result.events} >= {"section_change", "transition"}
    spans = {a.label: (a.start, a.end, a.certainty) for a in result.activity}
    assert spans["kick"] == (0.0, 3.0, "certain")
    assert spans["rhythm_guitar"] == (1.0, 4.0, "uncertain")
    assert result.covers("activity.vocals_harsh") and "vocals_harsh" not in spans
    assert not result.covers("activity.bass")
    out = tmp_path / "exported"
    audacity.export_folder(result, out)
    again = audacity.import_folder(out, rec, "ann1")
    assert again.model_dump() == result.model_dump()


def test_audacity_rejects_bad_files(tmp_path):
    rec = recording()
    with pytest.raises(ValueError, match="No label files"):
        audacity.import_folder(tmp_path, rec, "a")
    (tmp_path / "activity.kick.txt").write_text("1.0\t1.0\t\n")
    with pytest.raises(ValueError, match="regions"):
        audacity.import_folder(tmp_path, rec, "a")
    (tmp_path / "activity.kick.txt").write_text("1.0\t9.0\t\n")
    with pytest.raises(ValueError, match="past the recording end"):
        audacity.import_folder(tmp_path, rec, "a")
    (tmp_path / "activity.kick.txt").unlink()
    (tmp_path / "hits.triangle.txt").write_text("")
    with pytest.raises(ValueError, match="unknown layer"):
        audacity.import_folder(tmp_path, rec, "a")


def test_stem_activity_finds_gated_regions_and_is_only_a_draft():
    spans = stems.activity_intervals(tone(gate=[(0.5, 1.5), (2.5, 3.5)]), RATE)
    assert len(spans) == 2
    for (start, end), (want_start, want_end) in zip(spans, [(0.5, 1.5), (2.5, 3.5)], strict=True):
        assert abs(start - want_start) < 0.05 and abs(end - want_end) < 0.05
    assert stems.activity_intervals(np.zeros((RATE, 2), dtype=np.float32), RATE) == []
    draft = stems.draft(recording(), {"bass": (tone(gate=[(1, 2)]), RATE)})
    assert draft.status == "machine_draft" and not draft.usable_as_reference
    assert draft.coverage.layers == ("activity.bass",) and draft.activity[0].label == "bass"


def test_compare_and_consensus_keep_disagreements_out():
    rec = recording()
    a = annotation(
        rec,
        "a",
        layers=("onset", "activity.bass"),
        events=tuple(MixEvent(time=t, kind="onset") for t in (0.5, 1.0, 2.0)),
        activity=(Activity(start=0, end=2, label="bass"),),
    )
    b = annotation(
        rec,
        "b",
        layers=("onset", "activity.bass", "beat"),
        events=tuple(MixEvent(time=t, kind="onset") for t in (0.51, 1.02, 3.0)),
        activity=(Activity(start=1, end=3, label="bass"),),
    )
    report = compare(a, b)
    onset = report["layers"]["onset"]
    assert onset["agreed"] == 2 and onset["only_a"] == [2.0] and onset["only_b"] == [3.0]
    assert report["layers"]["activity.bass"]["iou"] == pytest.approx(1 / 3)
    assert report["layers_only_b"] == ["beat"]
    merged = consensus(a, b)
    assert [round(e.time, 3) for e in merged.events] == [0.505, 1.01]
    assert [(x.start, x.end) for x in merged.activity] == [(1, 2)]
    assert merged.usable_as_reference and "beat" not in merged.coverage.layers
    draft = b.model_copy(update={"status": "in_progress"})
    assert not consensus(a, draft).usable_as_reference


def test_cli_register_split_label_validate_flow(tmp_path, capsys):
    mix = write(tmp_path / "song.wav", tone())
    bass = write(tmp_path / "bass.wav", tone(gate=[(1, 2)]))
    other = write(tmp_path / "other.wav", tone(frequency=330))
    common = ("--source-kind", "original", "--license", "CC0-1.0", "--permission-basis", "mine")
    run(tmp_path, "register", str(mix), *common, "--group", "album-a", "--stem", f"bass={bass}")
    rid = json.loads(capsys.readouterr().out)["recording_id"]
    run(tmp_path, "register", str(other), *common, "--group", "album-b", "--tag", "sparse")
    run(tmp_path, "assign-splits")
    manifest_path = tmp_path / "corpus" / "manifest.json"
    manifest = registry.load_manifest(manifest_path)
    assert {r.split for r in manifest.recordings} <= {"train", "validation", "test"}
    text = manifest_path.read_text(encoding="utf-8")
    assert "song.wav" not in text and str(tmp_path) not in text  # no paths in the manifest
    assert (tmp_path / "corpus-local" / "paths.json").is_file()

    before = {r.recording_id: r.split for r in manifest.recordings}
    third = write(tmp_path / "third.wav", tone(frequency=110))
    run(tmp_path, "register", str(third), *common, "--group", "album-a")
    run(tmp_path, "assign-splits", "--seed", "different")
    after = registry.load_manifest(manifest_path)
    assert {r.recording_id: r.split for r in after.recordings if r.recording_id in before} == before
    assert len({r.split for r in after.recordings if r.group_id == "album-a"}) == 1

    run(tmp_path, "draft-from-stems", rid)
    capsys.readouterr()
    problems, warnings = validate_corpus(manifest_path)
    assert problems == [] and any("machine_draft" in w for w in warnings)
    run(tmp_path, "export-labels", rid, "--annotator", "stem-energy", "--out", str(tmp_path / "l"))
    run(tmp_path, "import-labels", rid, str(tmp_path / "l"), "--annotator", "me")
    run(tmp_path, "review", rid, "--annotator", "me")
    reviewed = registry.load_annotation(registry.annotation_path(manifest_path, rid, "me"))
    assert reviewed.usable_as_reference and reviewed.activity[0].label == "bass"
    capsys.readouterr()
    run(tmp_path, "status")
    listed = json.loads(capsys.readouterr().out)["recordings"]
    assert {r["audio"] for r in listed} == {"available"}


def test_validate_reports_changed_missing_and_mismatched(tmp_path, capsys):
    mix = write(tmp_path / "song.wav", tone())
    common = ("--source-kind", "original", "--license", "CC0-1.0", "--permission-basis", "mine")
    run(tmp_path, "register", str(mix), *common, "--group", "g")
    rid = json.loads(capsys.readouterr().out)["recording_id"]
    manifest_path = tmp_path / "corpus" / "manifest.json"
    rec = registry.load_manifest(manifest_path).get(rid)
    registry.save_annotation(manifest_path, annotation(rec, layers=("onset",)))

    write(mix, tone(frequency=440))
    _, warnings = validate_corpus(manifest_path)
    assert any("hash mismatch" in w for w in warnings)
    mix.unlink()
    _, warnings = validate_corpus(manifest_path)
    assert any("missing" in w for w in warnings)

    stale = annotation(rec, "old", layers=("onset",)).model_copy(
        update={"recording_sha256": "f" * 64}
    )
    registry.save_annotation(manifest_path, stale)
    problems, _ = validate_corpus(manifest_path, check_audio=False)
    assert any("different version" in p for p in problems)
    with pytest.raises(SystemExit):
        run(tmp_path, "validate", "--no-audio")
