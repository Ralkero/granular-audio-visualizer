"""Model C's audio front end, on the analysis engine's own frame grid (protocol §4).

The engine (``gmv.dsp.WindowAssembler``) emits a frame every ``round(rate * 20 ms)`` samples
while a whole window of ``round(rate * 46.44 ms)`` samples fits, and stamps each frame with
its window's centre. Model C works at 22,050 Hz with a Hann window of 1024 samples and a hop
of 441 (20 ms). Its window is 46.44 ms long too, so its frame centres fall within a few
microseconds of the engine's. ``frame_count`` uses the engine's rule, so both paths always
have the same number of frames; at 48 kHz C may need up to one resampled sample beyond the
end of the audio, which is zero-padded (``MAX_PAD``).

Choices the protocol leaves open, recorded in the Stage 1 report for review: mel bands on
the HTK mel scale, triangular, from the power spectrum; log of (mel power + ``FLOOR``).
"""

from math import gcd

import numpy as np
from scipy.signal import resample_poly

RATE = 22050
WINDOW = 1024
HOP = 441
BANDS = 64
FMIN_HZ, FMAX_HZ = 40.0, 11000.0
FLOOR = 1e-10
ENGINE_WINDOW_S = 0.04644  # models.AnalysisConfig.window_ms
ENGINE_HOP_S = 0.020  # models.AnalysisConfig.hop_ms
PREROLL_S = 5.0
MAX_PAD = 2  # resampled samples that may be missing at the end (window and length rounding)


def engine_window(rate):
    return round(rate * ENGINE_WINDOW_S)


def engine_hop(rate):
    return round(rate * ENGINE_HOP_S)


def frame_count(samples, rate):
    """Frames the engine produces for ``samples`` source samples at ``rate``."""
    size, hop = engine_window(rate), engine_hop(rate)
    return 0 if samples < size else (samples - size) // hop + 1


def frame_centres(count, rate, start_s=0.0):
    """Centre times (s) of the engine's frames for audio whose first sample is at ``start_s``."""
    return start_s + (np.arange(count) * engine_hop(rate) + engine_window(rate) / 2) / rate


def segment(total_samples, rate, excerpt_start_s, excerpt_end_s, preroll_s=PREROLL_S):
    """(first sample, end sample) of the audio both paths process for one excerpt: up to
    ``preroll_s`` before its start, through enough samples that every frame centred before
    its end exists."""
    if not (0 <= excerpt_start_s < excerpt_end_s):
        raise ValueError("an excerpt must start at or after 0 and end after it starts")
    if round(excerpt_end_s * rate) > total_samples + 1:
        raise ValueError("the excerpt runs past the end of the recording")
    first = round(max(0.0, excerpt_start_s - preroll_s) * rate)
    end = min(total_samples, round(excerpt_end_s * rate) + engine_window(rate))
    if end <= first:
        raise ValueError("the excerpt lies outside the recording")
    return first, end


def scored(centres, excerpt_start_s, excerpt_end_s):
    """Frames scored for an excerpt: centres inside [start, end)."""
    centres = np.asarray(centres, dtype=float)
    return (centres >= excerpt_start_s) & (centres < excerpt_end_s)


def mid(audio):
    """(L+R)/2 of stereo audio; mono stays as it is."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim == 1:
        mono = audio
    elif audio.ndim == 2 and audio.shape[1] in (1, 2):
        mono = audio.mean(axis=1)
    else:
        raise ValueError("expected mono or stereo audio shaped (samples,) or (samples, channels)")
    if not np.isfinite(mono).all():
        raise ValueError("audio contains non-finite samples")
    return mono


def resample(mono, rate):
    """``mono`` at ``rate`` resampled to 22,050 Hz (polyphase, zero-phase)."""
    if rate <= 0:
        raise ValueError("the sample rate must be positive")
    if rate == RATE:
        return np.asarray(mono, dtype=np.float64)
    common = gcd(RATE, int(rate))
    return resample_poly(np.asarray(mono, dtype=np.float64), RATE // common, int(rate) // common)


def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + np.asarray(hz, dtype=float) / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (np.asarray(mel, dtype=float) / 2595.0) - 1.0)


def mel_filterbank():
    """(BANDS, WINDOW // 2 + 1) triangular filters, evenly spaced on the HTK mel scale."""
    freqs = np.fft.rfftfreq(WINDOW, 1 / RATE)
    edges = _mel_to_hz(np.linspace(_hz_to_mel(FMIN_HZ), _hz_to_mel(FMAX_HZ), BANDS + 2))
    bank = np.zeros((BANDS, len(freqs)))
    for band in range(BANDS):
        low, centre, high = edges[band : band + 3]
        rising = (freqs - low) / (centre - low)
        falling = (high - freqs) / (high - centre)
        bank[band] = np.clip(np.minimum(rising, falling), 0.0, None)
    if (bank.sum(axis=1) <= 0).any():
        raise ValueError("a mel band covers no frequency bin")
    return bank


_BANK = None
_HANN = np.hanning(WINDOW)


def mel_power(audio, rate):
    """(frames, BANDS) mel power of ``audio`` on the engine's frame grid for ``rate``."""
    global _BANK
    if _BANK is None:
        _BANK = mel_filterbank()
    signal = resample(mid(audio), rate)
    count = frame_count(len(audio), rate)
    if count == 0:
        return np.zeros((0, BANDS))
    need = (count - 1) * HOP + WINDOW
    if need - len(signal) > MAX_PAD:
        raise ValueError("the resampled audio is shorter than the engine's frame grid allows")
    if len(signal) < need:
        signal = np.pad(signal, (0, need - len(signal)))
    index = np.arange(count)[:, None] * HOP + np.arange(WINDOW)[None, :]
    power = np.abs(np.fft.rfft(signal[index] * _HANN[None, :], axis=1)) ** 2
    return power @ _BANK.T


def log_mel(audio, rate):
    """(frames, BANDS) log mel power on the engine's frame grid; frame centres come from
    ``frame_centres(len(result), rate, start_s)``."""
    return np.log(mel_power(audio, rate) + FLOOR)
