"""Scoring rules corrected in Goal B: one-to-one attacks, beat uncertainty and off-beats,
tempo coverage across gaps, and point-only annotation layers."""

from types import SimpleNamespace

import numpy as np
import pytest

from gmv.corpus import audacity
from gmv.corpus.schema import Recording
from gmv.evaluation.metrics import attack_recall, beat_score, tempo_span_score


def test_one_detected_onset_cannot_answer_two_attacks():
    # A flam: two snare hits 30 ms apart, and the detector reports one onset between them.
    result = attack_recall([1.000, 1.030, 2.000], onsets=[1.015, 2.004])
    assert result["distinct_attacks"] == 3
    assert result["matched"] == 2 and result["recall"] == pytest.approx(2 / 3)
    assert result["shared_onsets"] == 1  # what nearest-onset counting used to give away
    assert result["median_absolute_error_ms"] == pytest.approx(9.5)
    # Two onsets, two attacks: both are found.
    assert attack_recall([1.000, 1.030], onsets=[1.002, 1.031])["matched"] == 2
    empty = attack_recall([], onsets=[1.0])
    assert empty["recall"] is None and empty["matched"] == 0
    assert attack_recall([1.0], onsets=[])["recall"] == 0


def test_simultaneous_hits_of_a_whole_kit_are_one_acoustic_event():
    # Kick and hi-hat struck together, then a lone snare: three labels, two events to find.
    result = attack_recall([1.000, 1.004, 1.500], onsets=[1.002, 1.501])
    assert result["reference_count"] == 3 and result["distinct_attacks"] == 2
    assert result["matched"] == 2 and result["recall"] == 1


def test_beat_estimates_at_uncertain_beats_earn_neither_credit_nor_blame():
    beats = np.arange(1.0, 9.0, 0.5)
    certain, unsure = beats[:8], beats[8:]
    score = beat_score(certain, beats + 0.01, uncertain=unsure)
    assert score["tp"] == 8 and score["fp"] == 0 and score["ignored_predictions"] == 8
    assert score["abstained_references"] == 8
    blamed = beat_score(certain, beats + 0.01)
    assert blamed["fp"] == 8  # without the uncertainty information they were false positives


def test_off_beat_locking_is_measured_separately_from_tempo():
    beats = np.arange(1.0, 21.0, 0.5)
    on = beat_score(beats, beats + 0.005)
    assert on["phase_lock"] == "on_beat" and on["tempo_correct"] and on["offbeat_f1"] == 0
    assert on["phase"]["median_distance_from_beat_fraction"] < 0.05
    off = beat_score(beats, beats + 0.25)
    # Right tempo, wrong phase: the F-measure alone would only say "bad".
    assert off["f1"] == 0 and off["tempo_correct"] and off["metrical_level"] == "same"
    assert off["phase_lock"] == "off_beat" and off["offbeat_f1"] > 0.95
    assert off["phase"]["share_nearer_the_off_beat"] == 1.0
    half = beat_score(beats, beats[::2])
    assert half["phase_lock"] == "on_beat" and not half["tempo_correct"]
    assert beat_score(beats, [])["phase_lock"] is None


def test_beats_report_how_much_of_the_reference_had_any_tracking():
    beats = np.arange(1.0, 21.0, 0.5)
    late = beat_score(beats, beats[20:])  # the tracker only locked half-way through
    assert late["reference_beats_with_tracking_fraction"] == pytest.approx(0.55, abs=0.05)
    assert late["precision"] == 1 and late["recall"] == pytest.approx(0.5)


def frames(times, bpm, confidence=0.8):
    return [SimpleNamespace(timestamp=t, tempo_bpm=bpm, tempo_confidence=confidence) for t in times]


def test_tempo_gaps_and_untracked_time_are_not_counted_as_tracking():
    hop = 0.02
    steady = frames(np.arange(0, 24, hop), 120.0)
    full = tempo_span_score(0, 24, 120, steady)
    assert full["strict_accuracy_after_4s"] == 1 and full["gap_seconds"] == pytest.approx(
        0, abs=0.05
    )
    assert full["strict_accuracy_of_span"] == pytest.approx(1, abs=0.01)
    assert full["confident_fraction_of_span"] == pytest.approx(1, abs=0.01)
    # Ten seconds of the span have no frames at all (a capture gap).
    holed = [f for f in steady if not 8 <= f.timestamp < 18]
    gap = tempo_span_score(0, 24, 120, holed)
    assert gap["strict_accuracy_after_4s"] == 1  # every frame that exists is right ...
    assert gap["gap_seconds"] == pytest.approx(10, abs=0.1)  # ... but half the span is missing
    assert gap["strict_accuracy_of_span"] == pytest.approx(0.5, abs=0.01)
    assert gap["tracked_fraction_of_span"] == pytest.approx(0.5, abs=0.01)
    # Frames exist but the tracker has no estimate, or only an unsure one.
    silent = frames(np.arange(0, 14, hop), 120.0) + frames(np.arange(14, 24, hop), None)
    lost = tempo_span_score(0, 24, 120, silent)
    assert lost["gap_seconds"] == pytest.approx(0, abs=0.05)
    assert lost["untracked_fraction_of_span"] == pytest.approx(0.5, abs=0.01)
    assert lost["strict_accuracy_on_tracked"] == 1
    unsure = tempo_span_score(0, 24, 120, frames(np.arange(0, 24, hop), 126.0, confidence=0.3))
    assert unsure["confident_fraction_of_span"] == 0 and unsure["strict_accuracy_of_span"] == 0
    assert unsure["median_tempo_error_percent_on_tracked"] == pytest.approx(5.0)


def recording():
    return Recording(
        recording_id="r-000000000000",
        sha256="0" * 64,
        duration_s=5.0,
        sample_rate=48000,
        channels=2,
        source_kind="original",
        license="CC0",
        permission_basis="test",
        group_id="g",
    )


@pytest.mark.parametrize("layer", ["onset", "beat", "downbeat", "tempo", "hits.kick"])
def test_point_layers_reject_region_labels(tmp_path, layer):
    text = "120" if layer == "tempo" else ""
    (tmp_path / f"{layer}.txt").write_text(f"1.0\t1.0\t{text}\n2.0\t2.5\t{text}\n")
    with pytest.raises(ValueError, match="single moments.*2.000s is a region ending at 2.500s"):
        audacity.import_folder(tmp_path, recording(), "a")
    (tmp_path / f"{layer}.txt").write_text(f"1.0\t1.0\t{text}\n2.0\t2.0004\t{text}\n")
    imported = audacity.import_folder(tmp_path, recording(), "a")  # sub-millisecond: a point
    assert len(imported.events) + len(imported.instrument_events) == 2


def test_region_layers_still_accept_regions(tmp_path):
    (tmp_path / "sections.txt").write_text("1.0\t3.0\tverse\n")
    (tmp_path / "activity.kick.txt").write_text("1.0\t2.0\t\n")
    imported = audacity.import_folder(tmp_path, recording(), "a")
    assert imported.events[0].end == 3.0 and imported.activity[0].end == 2.0
