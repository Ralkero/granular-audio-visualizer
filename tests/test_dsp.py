import numpy as np
import pytest
from gmv.dsp import Analyzer, WindowAssembler
from gmv.models import AnalysisConfig
from gmv.sources.base import AudioBlock


def analyze(audio, rate=48000):
    config = AnalysisConfig()
    assembler = WindowAssembler(config)
    analyzer = Analyzer(rate, audio.shape[1], config, "test", "synthetic")
    frames, events = [], []
    for pos in range(0, len(audio), 480):
        data = audio[pos:pos+480].astype(np.float32)
        block = AudioBlock(data.tobytes(), len(data), rate, data.shape[1], pos/rate, 0, 0)
        for window, start, mono, gap in assembler.push(block):
            frame, new_events = analyzer.process(window, start, mono, block, live=False)
            frames.append(frame)
            events.extend(new_events)
    return frames, events


@pytest.mark.parametrize("rate", [22050, 44100, 48000, 96000])
def test_known_tone_and_opposite_phase_channels(rate):
    x = 0.4 * np.sin(2*np.pi*1000*np.arange(rate)/rate)
    frames, _ = analyze(np.column_stack((x, -x)), rate)
    f = frames[-1]
    assert f.rms == pytest.approx(0.4 / np.sqrt(2), rel=0.015)
    assert f.spectral_centroid_hz == pytest.approx(1000, abs=3)
    assert f.band_energy[3] / sum(f.band_energy) > 0.99
    assert f.band_energy[3] == pytest.approx(0.08, rel=0.01)
    assert f.zero_crossing_rate == pytest.approx(2000/rate, abs=0.002)


def test_silence_finite_and_transition():
    rate = 48000
    x = np.concatenate((np.zeros(rate), 0.2*np.sin(2*np.pi*1000*np.arange(rate)/rate),
                        np.zeros(rate)))[:, None]
    frames, events = analyze(x)
    assert frames[-1].silence and frames[-1].rms == 0
    assert any(e.event_type == "audio_resumed" for e in events)
    assert any(e.event_type == "silence_started" for e in events)
    assert all('NaN' not in f.model_dump_json() for f in frames)


def test_pulses_onset_timing_and_tempo():
    rate = 48000
    x = np.zeros(rate*12)
    for t in np.arange(0.5, 11.6, 0.5):
        n = int(t*rate)
        pulse = np.sin(2*np.pi*90*np.arange(2400)/rate)*np.exp(-np.arange(2400)/500)
        x[n:n+2400] = pulse
    frames, events = analyze(x[:, None])
    onsets = [e.timestamp for e in events if e.event_type == "onset"]
    assert sum(min(abs(t-o) for o in onsets) < 0.05 for t in np.arange(0.5,11.6,0.5)) >= 21
    assert frames[-1].tempo_bpm == pytest.approx(120, abs=4)
    assert frames[-1].tempo_confidence > 0.5


def test_gap_discards_partial_window():
    assembler = WindowAssembler(AnalysisConfig())
    def block(ts):
        return AudioBlock(np.zeros(1000, dtype=np.float32).tobytes(), 1000, 48000, 1, ts, ts, ts)
    assert not list(assembler.push(block(0)))
    assert not list(assembler.push(block(1)))
    assert assembler.resets == 2
