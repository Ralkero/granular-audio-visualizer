"""One decode and one spectrogram for all the experimental measurements.

The audio is brought to 22.05 kHz and transformed once (2048-sample Hann windows every 512
samples: 93 ms windows, 43 frames a second, 10.8 Hz a bin). From it:

* ``power``: mean power of the two channels, per frame and bin;
* ``harmonic`` and ``percussive``: a soft split of that power by median filtering
  (Fitzgerald 2010): what is steady along time is harmonic, what is steady along frequency
  is percussive. It is the method ``gmv.advanced`` already uses, at a finer hop;
* ``centre``: what both channels share (``gmv.instruments.common.centre_magnitude``);
* ``level_db`` and ``music``: the frame level, and whether it is above the analysis's
  silence threshold. Silent frames are left out of everything.

Everything measured from these is a ratio, a share or a frequency, so the playback volume
cancels; only ``music`` uses an absolute level, the same -66 dBFS the analysis pass uses.
"""

from math import gcd

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly

RATE = 22050
SIZE, HOP = 2048, 512
FRAME_S = HOP / RATE
SILENT_DBFS = -66.0
TOP_HZ = 8000.0  # the split is made up to here; above it everything counts as percussive
MEDIAN = 17  # frames (0.4 s) and bins (180 Hz) of the two median filters


def load(path, gain_db=0.0, seconds=None):
    """Decoded samples (frames, channels) and their rate. The file is only read."""
    from ...corpus.registry import read_audio

    audio, rate = read_audio(path)
    if seconds:
        audio = audio[: round(seconds * rate)]
    if gain_db:
        audio = audio * np.float32(10.0 ** (gain_db / 20.0))
    return audio, rate


def stft(signal):
    if len(signal) < SIZE:
        signal = np.pad(signal, (0, SIZE - len(signal)))
    frames = np.lib.stride_tricks.sliding_window_view(signal, SIZE)[::HOP]
    return np.fft.rfft(frames * np.hanning(SIZE).astype(np.float32), axis=1).astype(np.complex64)


class Analysis:
    """The shared spectrogram of one recording (or stem)."""

    def __init__(self, audio, rate):
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio[:, None]
        if rate != RATE:
            divisor = gcd(int(rate), RATE)
            audio = resample_poly(audio, RATE // divisor, int(rate) // divisor, axis=0).astype(
                np.float32
            )
        self.duration_s = len(audio) / RATE
        left = stft(audio[:, 0])
        right = stft(audio[:, 1]) if audio.shape[1] > 1 else left
        self.freq = np.fft.rfftfreq(SIZE, 1 / RATE)
        self.times = (np.arange(len(left)) * HOP + SIZE / 2) / RATE
        total = np.abs(left + right) / 2
        side = np.abs(left - right) / 2
        self.stereo = bool(side.sum() > 0.05 * total.sum())
        self.centre = np.maximum(total - side, 0.0)
        self.power = (np.abs(left) ** 2 + np.abs(right) ** 2) / 2
        del left, right
        # Level of each frame as the analysis pass would state it (window power -> dBFS).
        scale = 2.0 / (SIZE * float(np.square(np.hanning(SIZE)).sum()))
        mean_square = self.power.sum(axis=1) * scale
        self.level_db = 10.0 * np.log10(np.maximum(mean_square, 1e-12))
        self.music = self.level_db >= SILENT_DBFS
        top = int(np.searchsorted(self.freq, TOP_HZ))
        magnitude = np.sqrt(self.power[:, :top])
        steady = median_filter(magnitude, size=(MEDIAN, 1), mode="nearest")
        struck = median_filter(magnitude, size=(1, MEDIAN), mode="nearest")
        mask = np.zeros(self.power.shape, dtype=np.float32)
        mask[:, :top] = steady**2 / (steady**2 + struck**2 + 1e-20)
        self.harmonic_mask = mask
        self.harmonic = self.power * mask
        self.percussive = self.power - self.harmonic

    def frames(self, start, end):
        """Frame indices whose centre lies in [start, end) seconds."""
        return slice(int(np.searchsorted(self.times, start)), int(np.searchsorted(self.times, end)))


def result(value, confidence):
    """A measurement: None when unknown, with a confidence held to 0..1. The confidence is
    a heuristic score of signal support, not a calibrated probability of being right."""
    if value is None or not np.isfinite(value):
        return (None, 0.0)
    return (float(value), float(min(1.0, max(0.0, confidence))))
