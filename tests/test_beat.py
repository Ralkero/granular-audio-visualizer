import numpy as np
import pytest

from gmv.beat import BeatTracker
from gmv.evaluation.detect import run_detector
from gmv.models import AnalysisConfig

RATE = 48000
TRACKING = AnalysisConfig(experimental_beat_phase=True)


def clicks(bpm, seconds=20.0, first=0.5):
    audio = np.zeros((round(seconds * RATE), 2), dtype=np.float32)
    beats = np.arange(first, seconds - 0.2, 60 / bpm)
    burst = np.random.default_rng(1).standard_normal(480).astype(np.float32)
    burst *= np.exp(-np.arange(480) / 80).astype(np.float32) * 0.5
    for time in beats:
        start = round(time * RATE)
        audio[start : start + 480] += burst[:, None]
    return audio, beats


def predictions(audio, config=TRACKING):
    frames, events, _, emitted = run_detector(audio, RATE, "candidate", config)
    found = [
        (e, at) for e, at in zip(events, emitted, strict=True) if e.event_type == "beat_prediction"
    ]
    return frames, found


@pytest.mark.parametrize("bpm", [90, 120, 150])
def test_steady_pulse_is_tracked_and_announced_ahead(bpm):
    audio, beats = clicks(bpm)
    frames, found = predictions(audio)
    late = [e for e, _ in found if 8 < e.timestamp < beats[-1] + 0.07]
    assert late
    errors = [np.abs(beats - e.timestamp).min() for e in late]
    assert max(errors) < 0.07
    expected = [b for b in beats if 8 < b < late[-1].timestamp + 0.07]
    assert len(late) == pytest.approx(len(expected), abs=1)
    tempos = dict(late[-1].measurements)
    assert tempos["tempo_bpm"] == pytest.approx(bpm, rel=0.04)
    # Each beat is reported before the audio containing it has been analysed.
    assert all(e.timestamp > at for e, at in found)
    assert all(
        e.evidence_kind == "predicted" and e.method == "causal_beat_tracker_v1" for e in late
    )
    assert frames[-1].beat_locked and 0 <= frames[-1].beat_phase < 1


def test_no_beats_without_attacks():
    time = np.arange(12 * RATE) / RATE
    tone = (0.3 * np.sin(2 * np.pi * 220 * time)).astype(np.float32)
    for audio in (np.zeros((12 * RATE, 2), np.float32), np.stack((tone, tone), axis=1)):
        frames, found = predictions(audio)
        assert not found
        assert not any(f.beat_locked for f in frames)


def test_tracker_is_off_by_default():
    audio, _ = clicks(120, seconds=10)
    frames, found = predictions(audio, AnalysisConfig())
    assert not found and not any(f.beat_locked for f in frames)


def test_tracker_is_independent_of_level():
    rng = np.random.default_rng(3)
    pattern = np.where(np.arange(600) % 25 == 0, 1.0, 0.0) + 0.02 * rng.random(600)
    quiet, loud = BeatTracker(0.02), BeatTracker(0.02)
    a = [quiet.step(v * 1e-3) for v in pattern]
    b = [loud.step(v * 10) for v in pattern]
    assert a == pytest.approx(b, nan_ok=True) or [x is None for x in a] == [x is None for x in b]
    assert quiet.tempo_bpm == pytest.approx(120, rel=0.04)
