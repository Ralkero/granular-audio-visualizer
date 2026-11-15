import hashlib
import json

import numpy as np
import pytest
import soundfile as sf

from gmv.corpus import registry
from gmv.corpus.schema import CorpusManifest, MixEvent, Recording
from gmv.evaluation import run as evaluation_run
from gmv.evaluation.corpus import generate
from gmv.evaluation.metrics import beat_score, drop_uncertain, interval_score
from gmv.evaluation.real import evaluate_corpus, evaluate_recording, render_markdown
from gmv.evaluation.run import evaluate, load_import
from gmv.evaluation.synthetic_corpus import fixture_annotation


def fixture(category="simultaneous"):
    return next(f for f in generate("development") if f.category == category)


def build_corpus(tmp_path, item, split="validation", tags=("dense",)):
    """Register one synthetic fixture exactly as a real recording would be."""
    audio_path = tmp_path / "audio" / "fixture.wav"
    audio_path.parent.mkdir(exist_ok=True)
    sf.write(audio_path, item.audio, item.rate, subtype="FLOAT")
    digest = registry.file_sha256(audio_path)
    recording = Recording(
        recording_id="r-" + digest[:12],
        sha256=digest,
        duration_s=len(item.audio) / item.rate,
        sample_rate=item.rate,
        channels=item.audio.shape[1],
        source_kind="original",
        license="CC0-1.0",
        permission_basis="Original procedural fixture",
        group_id="synthetic-" + item.category,
        split=split,
        tags=tags,
    )
    manifest_path = tmp_path / "corpus" / "manifest.json"
    registry.save_manifest(
        manifest_path, CorpusManifest(corpus_version="0.1.0", recordings=(recording,))
    )
    registry.save_paths(
        manifest_path, {"recordings": {recording.recording_id: str(audio_path)}, "stems": {}}
    )
    annotation = fixture_annotation(item, recording)
    registry.save_annotation(manifest_path, annotation)
    return manifest_path, recording, annotation, audio_path


def test_interval_score_counts_only_covered_certain_time():
    score = interval_score([(1, 3)], [(2, 4)], [(0, 5)])
    assert score["tp_seconds"] == pytest.approx(1) and score["fp_seconds"] == pytest.approx(1)
    assert score["fn_seconds"] == pytest.approx(1) and score["iou"] == pytest.approx(1 / 3)
    assert score["f1"] == pytest.approx(0.5)
    assert score["false_positive_rate"] == pytest.approx(1 / 3)
    assert score["entrances"]["matched_within_500ms"] == 0

    partial = interval_score([(1, 3)], [(2, 4)], [(0, 2.5)])  # prediction after 2.5 s is unknown
    assert partial["fp_seconds"] == pytest.approx(0) and partial["tp_seconds"] == pytest.approx(0.5)

    abstain = interval_score([(1, 2)], [(1.1, 2.0), (3, 4)], [(0, 5)], uncertain=[(3, 4)])
    assert abstain["abstained_seconds"] == pytest.approx(1)
    assert abstain["fp_seconds"] == pytest.approx(0)
    assert abstain["entrances"]["bias_ms"] == pytest.approx(100)
    assert abstain["exits"]["median_absolute_error_ms"] == pytest.approx(0)

    empty = interval_score([], [], [(0, 5)])
    assert empty["correct_empty"] and empty["iou"] is None and empty["scored_seconds"] == 5


def test_beat_score_names_the_metrical_level():
    beats = np.arange(0.5, 20, 0.5)
    assert beat_score(beats, beats + 0.01)["metrical_level"] == "same"
    half = beat_score(beats, beats[::2])
    assert half["metrical_level"] == "half_time" and half["precision"] == 1
    assert half["recall"] == pytest.approx(0.5, abs=0.02)
    assert beat_score(beats, np.arange(0.5, 20, 0.25))["metrical_level"] == "double_time"
    nothing = beat_score(beats, [])
    assert (
        nothing["tp"] == 0 and nothing["continuity"] is None and nothing["metrical_level"] is None
    )


def test_uncertain_references_earn_neither_credit_nor_blame():
    kept, ignored = drop_uncertain(
        [1.0, 2.0, 3.0], certain=[1.0], uncertain=[2.01, 1.02], tolerance=0.05
    )
    assert kept == [1.0, 3.0] and ignored == 1


def test_corpus_path_reproduces_synthetic_scores_end_to_end(tmp_path, capsys):
    item = fixture()
    manifest_path, recording, _, _ = build_corpus(tmp_path, item)
    direct = evaluate(item, "candidate")
    report = evaluate_corpus(manifest_path, "validation", detector="candidate")
    assert report["real_music_status"] == "evaluated" and report["data_kind"] == "real_corpus"
    assert report["evaluated"] == 1 and report["skipped"] == []
    result = report["recordings"][0]
    for tolerance in ("25", "50"):
        for key in ("tp", "fp", "fn", "timing_errors_ms"):
            assert result["onsets"][tolerance][key] == direct["onsets"][tolerance][key]
    assert direct["onsets"]["50"]["tp"] > 0
    snare = result["candidates"]["snare_like_candidate"]["score_50ms"]
    assert snare["tp"] == direct["classes"]["snare_like_candidate"]["tp"]
    assert result["onsets"]["latency"]["p50_ms"] > 0
    assert {v["status"] for v in result["instrument_activity"].values()} == {"no_detector_output"}
    assert result["instrument_events"]["snare"]["status"] == "no_detector_output"
    assert result["downbeats"] == {"status": "not_annotated"}
    assert report["aggregate"]["onsets_50ms_by_tag"]["dense"]["tp"] == direct["onsets"]["50"]["tp"]
    assert report["aggregate"]["instrument_activity"]["drums"]["status"] == "no_detector_output"

    output, markdown = tmp_path / "report.json", tmp_path / "report.md"
    evaluation_run.main(
        ["--detector", "candidate", "--corpus", str(manifest_path), "--corpus-split", "validation"]
        + ["--output", str(output), "--report-md", str(markdown)]
    )
    written = json.loads(output.read_text(encoding="utf-8"))
    assert written["aggregate"]["onsets"]["50"]["tp"] == direct["onsets"]["50"]["tp"]
    assert written["corpus_content_sha256"] == report["corpus_content_sha256"]
    assert "Full-mix onset, ±50 ms" in markdown.read_text(encoding="utf-8")
    assert recording.recording_id not in capsys.readouterr().err


def test_held_out_split_needs_final_and_is_stamped(tmp_path):
    manifest_path, *_ = build_corpus(tmp_path, fixture("isolated_bass"), split="test")
    with pytest.raises(ValueError, match="held out"):
        evaluate_corpus(manifest_path, "test")
    with pytest.raises(SystemExit):
        evaluation_run.main(
            [
                "--corpus",
                str(manifest_path),
                "--corpus-split",
                "test",
                "--output",
                str(tmp_path / "o"),
            ]
        )
    report = evaluate_corpus(manifest_path, "test", final=True)
    assert report["final_test_evaluation"] and report["evaluated"] == 1
    assert evaluate_corpus(manifest_path, "validation")["real_music_status"] == "pending"


def test_unusable_recordings_are_skipped_with_reasons(tmp_path):
    manifest_path, recording, annotation, audio_path = build_corpus(
        tmp_path, fixture("isolated_bass")
    )
    registry.save_annotation(
        manifest_path, annotation.model_copy(update={"status": "machine_draft"})
    )
    report = evaluate_corpus(manifest_path, "validation")
    assert report["real_music_status"] == "pending"
    assert report["skipped"] == [
        {"recording_id": recording.recording_id, "reason": "annotations not reviewed"}
    ]
    assert "pending" in render_markdown(
        {**report, "detector": "candidate", "code_commit": "0" * 40}
    )
    assert evaluate_corpus(manifest_path, "validation", allow_unreviewed=True)["evaluated"] == 1

    registry.save_annotation(manifest_path, annotation)
    assert (
        "no annotation by"
        in evaluate_corpus(manifest_path, "validation", annotator="nobody")["skipped"][0]["reason"]
    )
    sf.write(audio_path, np.zeros((800, 2), dtype=np.float32), 8000, subtype="FLOAT")
    assert "hash mismatch" in evaluate_corpus(manifest_path, "validation")["skipped"][0]["reason"]
    audio_path.unlink()
    assert "missing" in evaluate_corpus(manifest_path, "validation")["skipped"][0]["reason"]


def test_uncertain_labels_and_activity_predictions_are_scored_per_class(tmp_path):
    item = fixture()
    _, recording, annotation, _ = build_corpus(tmp_path, item)
    onsets = [e for e in annotation.events if e.kind == "onset"]
    doubtful = annotation.model_copy(
        update={
            "events": tuple(
                e.model_copy(update={"certainty": "uncertain"}) if e in onsets[:3] else e
                for e in annotation.events
            )
            + (MixEvent(time=0.0123, kind="beat"),)
        }
    )
    drums = [(a.start, a.end) for a in annotation.activity if a.label == "drums"]
    result = evaluate_recording(
        recording,
        doubtful,
        item.audio,
        item.rate,
        activity_predictions={"drums": drums, "bass": []},
    )
    score = result["onsets"]["50"]
    assert score["abstained_references"] == 3 and score["ignored_predictions"] == 3
    assert score["fn"] == 0 and score["fp"] == 0
    assert result["instrument_activity"]["drums"]["status"] == "scored"
    assert result["instrument_activity"]["drums"]["f1"] == pytest.approx(1)
    bass = result["instrument_activity"]["bass"]
    assert bass["recall"] == 0 and bass["fn_seconds"] > 0
    assert result["instrument_activity"]["vocals"]["status"] == "no_detector_output"
    assert result["beats"]["observed"]["tolerance_seconds"] == 0.07


def test_legacy_single_file_import_still_validates(tmp_path):
    audio = tmp_path / "licensed.wav"
    sf.write(audio, np.zeros((4800, 2), dtype=np.float32), 48000, subtype="FLOAT")
    manifest = {
        "audio": "licensed.wav",
        "sha256": hashlib.sha256(audio.read_bytes()).hexdigest(),
        "license": "CC0-1.0",
        "provenance": "test",
        "permission_verified": True,
        "onsets": [0.05],
        "annotation_method": "manual",
        "uncertainty_seconds": 0.02,
        "source_recording_id": "x",
        "split": "development",
    }
    path = tmp_path / "import.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    loaded = load_import(path)
    assert loaded.onsets == [0.05] and loaded.category == "real_music"
    path.write_text(json.dumps({**manifest, "sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_import(path)
    path.write_text(json.dumps({**manifest, "permission_verified": False}), encoding="utf-8")
    with pytest.raises(ValueError, match="permission"):
        load_import(path)
