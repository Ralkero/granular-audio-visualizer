"""A5.2 Stage 1, protocol §16 "Alignment": the analysis engine's frame path (B0 and B1) and
model C's front end put frames at the same times. Generated audio only; no weights are
loaded (the engine runs without an activity model)."""

import numpy as np
import pytest

from gmv.character.research import gmvpass
from gmv.character.research.presence_model import frontend

RATES = (22050, 44100, 48000)
TOLERANCE_S = 1e-3  # the registered requirement: centres within 1 ms


def noise(samples, seed=0):
    rng = np.random.default_rng(seed)
    left = 0.1 * rng.standard_normal(samples)
    return np.stack([left, 0.5 * left + 0.05 * rng.standard_normal(samples)], axis=1).astype(np.float32)


def engine_frames(audio, rate):
    _messages, _rows, frames = gmvpass.run(audio, rate)
    return frames


def lengths(rate):
    size, hop = frontend.engine_window(rate), frontend.engine_hop(rate)
    return sorted({size - 1, size, size + 1, size + hop - 1, size + hop, size + 7 * hop + 3,
                   int(0.73 * rate), int(1.0 * rate), int(1.0 * rate) + 1, 2 * size + hop // 2})


@pytest.mark.parametrize("rate", RATES)
def test_frame_counts_and_centres_agree_for_many_lengths(rate):
    for samples in lengths(rate):
        frames = engine_frames(noise(samples, samples), rate)
        mel = frontend.log_mel(noise(samples, samples), rate)
        assert len(mel) == len(frames) == frontend.frame_count(samples, rate), samples
        engine = np.array([f.timestamp for f in frames])
        ours = frontend.frame_centres(len(mel), rate)
        if len(engine):
            assert np.max(np.abs(engine - ours)) < TOLERANCE_S
            # the centre offset itself, not just its consistency: 46.44 ms / 2 after the start
            assert abs(engine[0] - 0.02322) < TOLERANCE_S and abs(ours[0] - 0.02322) < TOLERANCE_S
            assert np.allclose(np.diff(ours), 0.02, atol=1e-9)


@pytest.mark.parametrize("rate", RATES)
def test_an_impulse_peaks_in_the_same_frame_on_both_paths(rate):
    centres = frontend.frame_centres(frontend.frame_count(rate, rate), rate)
    # Offsets from a frame centre, including close to (but not on) the midpoint between frames.
    for k, offset in ((10, 0.0), (11, 0.003), (17, -0.007), (23, 0.0085), (30, -0.0085), (37, 0.0099)):
        audio = np.zeros((rate, 2), np.float32)
        when = centres[k] + offset
        audio[round(when * rate)] = 0.8
        engine = np.array([sum(f.band_energy) for f in engine_frames(audio, rate)])
        ours = frontend.mel_power(audio, rate).sum(axis=1)
        expected = int(np.argmin(np.abs(centres - when)))
        assert int(np.argmax(engine)) == expected == int(np.argmax(ours)), (k, offset)


@pytest.mark.parametrize("rate", (44100, 48000))
def test_resampling_keeps_impulse_timing_within_one_sample(rate):
    for position in (1000, 4321, rate // 2 + 17, rate - 2500):
        mono = np.zeros(rate)
        mono[position] = 1.0
        out = frontend.resample(mono, rate)
        assert abs(int(np.argmax(np.abs(out))) - position * frontend.RATE / rate) <= 1.0


def test_resampling_at_the_model_rate_is_the_identity():
    mono = noise(5000)[:, 0].astype(np.float64)
    assert np.array_equal(frontend.resample(mono, frontend.RATE), mono)


@pytest.mark.parametrize("rate", RATES)
def test_preroll_and_scored_frames_agree(rate):
    recording = noise(int(14.5 * rate), 7)
    for start, end in ((0.0, 4.0), (1.0, 5.0), (3.33, 7.33), (6.0, 10.0), (12.0, 14.5)):
        first, last = frontend.segment(len(recording), rate, start, end)
        assert first == round(max(0.0, start - frontend.PREROLL_S) * rate)
        audio = recording[first:last]
        offset = first / rate
        engine = np.array([f.timestamp for f in engine_frames(audio, rate)]) + offset
        ours = frontend.frame_centres(len(frontend.log_mel(audio, rate)), rate, offset)
        assert len(engine) == len(ours)
        keep_engine, keep_ours = frontend.scored(engine, start, end), frontend.scored(ours, start, end)
        assert np.array_equal(keep_engine, keep_ours)
        assert np.max(np.abs(engine[keep_engine] - ours[keep_ours])) < TOLERANCE_S
        # every scored frame lies in the excerpt, and the excerpt is covered from start to end:
        # the first centre falls within one hop of the start, except at the very start of a
        # recording, where no audio precedes it and the first centre is half a window in
        scored = ours[keep_ours]
        assert scored.min() >= start and scored.max() < end
        # (and symmetrically at the very end of a recording, where the last centre is half a
        # window before the last sample)
        half_window = frontend.engine_window(rate) / rate / 2
        lead = 0.02 if start - offset >= half_window else half_window
        tail = 0.02 if len(recording) / rate - end >= half_window else half_window + 0.02
        assert scored.min() - start <= lead + 1e-9 and end - scored.max() <= tail + 1e-9
        if end < len(recording) / rate - 0.05:
            assert keep_ours.sum() == pytest.approx((end - start) / 0.02, abs=1)


def test_a_misplaced_centre_or_window_would_be_caught():
    """Regression: the tolerance is tight enough to catch a half-window or one-hop offset."""
    rate = 48000
    frames = engine_frames(noise(rate), rate)
    engine = np.array([f.timestamp for f in frames])
    starts = frontend.frame_centres(len(frames), rate) - frontend.engine_window(rate) / rate / 2
    assert np.max(np.abs(engine - starts)) > TOLERANCE_S  # window starts are not centres
    shifted = frontend.frame_centres(len(frames), rate) + 0.02
    assert np.max(np.abs(engine - shifted)) > TOLERANCE_S  # a one-hop shift


def test_invalid_audio_and_segments_are_refused():
    with pytest.raises(ValueError):
        frontend.mid(np.zeros((100, 3)))
    with pytest.raises(ValueError):
        frontend.mid(np.array([0.0, np.nan]))
    with pytest.raises(ValueError):
        frontend.resample(np.zeros(10), 0)
    with pytest.raises(ValueError):
        frontend.segment(48000, 48000, 2.0, 1.0)
    with pytest.raises(ValueError):
        frontend.segment(48000, 48000, 5.0, 6.0)  # outside a one-second recording
    assert frontend.log_mel(noise(100), 22050).shape == (0, frontend.BANDS)


def test_the_mel_bank_covers_40_hz_to_11_khz_with_64_bands():
    bank = frontend.mel_filterbank()
    freqs = np.fft.rfftfreq(frontend.WINDOW, 1 / frontend.RATE)
    assert bank.shape == (64, frontend.WINDOW // 2 + 1)
    used = freqs[bank.sum(axis=0) > 0]
    assert used.min() > 40.0 - 22.0 and used.max() < 11000.0 + 22.0  # within one bin (21.5 Hz)
    assert (bank.sum(axis=1) > 0).all()
