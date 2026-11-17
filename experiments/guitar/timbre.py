"""Experimental features for guitar detection: a semitone-resolution log spectrum per frame.

Each frame is 96 semitone bands from a 93 ms window, with the slowly varying overall level
removed causally. ``train_timbre.py`` puts a small stack of causal convolutions over about
1.2 s of these frames. The experiment did not produce a usable detector (see
docs/evaluation/guitar_timbre_experiment.md), so nothing here is used by the application.
"""

import numpy as np
from scipy.signal import lfilter

WINDOW_S = 4096 / 44100
LOW_HZ, BANDS, BANDS_PER_OCTAVE = 55.0, 96, 12
LEVEL_S = 3.0
FLOOR, CEILING = -4.0, 2.0
FEATURE_VERSION = "timbre_features_v1"


def _alpha(hop, seconds):
    return 1 - np.exp(-hop / seconds)


class LogBands:
    """Semitone-spaced triangular bands on the power spectrum of a mono window."""

    def __init__(self, rate):
        self.rate = rate
        self.size = round(rate * WINDOW_S)
        self.window = np.hanning(self.size).astype(np.float32)
        freq = np.fft.rfftfreq(self.size, 1 / rate)
        centres = LOW_HZ * 2 ** (np.arange(-1, BANDS + 1) / BANDS_PER_OCTAVE)
        if centres[-1] > rate / 2:
            raise ValueError("Sample rate is too low for the timbre bands")
        matrix = np.zeros((BANDS, len(freq)), np.float32)
        for band in range(BANDS):
            low, mid, high = centres[band : band + 3]
            rising = (freq - low) / (mid - low)
            falling = (high - freq) / (high - mid)
            matrix[band] = np.clip(np.minimum(rising, falling), 0, None)
            if not matrix[band].any():  # narrower than one bin: take the nearest bin
                matrix[band, np.abs(freq - mid).argmin()] = 1
            matrix[band] /= matrix[band].sum()
        self.matrix = matrix
        # Same gain for every rate and window length: a full-scale sine reads about 0 dB.
        self.gain = 4 / float(self.window.sum()) ** 2

    def frames(self, mono, hop):
        """Log band power (bels) for every hop of a whole signal: (frames, BANDS)."""
        count = (len(mono) - self.size) // hop + 1
        out = np.empty((max(count, 0), BANDS), np.float32)
        for begin in range(0, count, 256):
            index = np.arange(begin, min(count, begin + 256))
            block = mono[index[:, None] * hop + np.arange(self.size)[None, :]] * self.window
            power = np.abs(np.fft.rfft(block, axis=1)) ** 2 * self.gain
            out[index] = np.log10(power @ self.matrix.T + 1e-10)
        return out

    def frame(self, window):
        power = np.abs(np.fft.rfft(window * self.window)) ** 2 * self.gain
        return np.log10(self.matrix @ power + 1e-10).astype(np.float32)


def normalise(frames, hop_s):
    """Remove the slowly varying overall level (causally) and limit the range."""
    frames = np.asarray(frames, np.float32)
    if not len(frames):
        return frames
    a = _alpha(hop_s, LEVEL_S)
    loud = frames.max(axis=1)
    level = lfilter([a], [1, a - 1], loud, zi=(1 - a) * loud[:1])[0]
    return np.clip(frames - level[:, None], FLOOR, CEILING).astype(np.float32)
