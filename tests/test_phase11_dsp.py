import numpy as np
import pytest
from test_dsp import analyze


def test_dense_attacks_and_independent_energy_evidence():
    rate = 48000
    audio = np.zeros(rate * 3)
    times = np.arange(0.3, 2.8, 1 / 12)
    pulse = np.sin(2 * np.pi * 90 * np.arange(2400) / rate) * np.exp(-np.arange(2400) / 500)
    for time in times:
        start = round(time * rate)
        audio[start : start + len(pulse)] += pulse
    frames, events = analyze(audio[:, None])
    onsets = [e.timestamp for e in events if e.event_type == "onset"]
    assert len(onsets) == len(times)
    assert max(min(abs(t - o) for o in onsets) for t in times) < 0.025
    assert any(abs(f.transient_strength - f.onset_strength) > 0.05 for f in frames)


def test_fixed_power_mapping_preserves_embedded_and_antiphase_content():
    tone = 0.4 * np.sin(2 * np.pi * 1000 * np.arange(48000) / 48000)
    stereo = np.column_stack((tone, -tone))
    two, _ = analyze(stereo)
    eight, _ = analyze(np.pad(stereo, ((0, 0), (0, 6))))
    full, _ = analyze(np.tile(stereo, (1, 4)))
    assert eight[-1].rms == pytest.approx(two[-1].rms / 2)
    assert eight[-1].analysis_rms == pytest.approx(two[-1].analysis_rms)
    assert full[-1].analysis_rms == pytest.approx(two[-1].analysis_rms * 2)
    assert eight[-1].analysis_weights == (0.5,) * 8


def test_silence_has_no_beat_lock_or_predictions():
    frames, events = analyze(np.zeros((48000 * 5, 1)))
    assert all(not f.beat_locked for f in frames)
    assert not any(e.event_type.startswith("beat") for e in events)
