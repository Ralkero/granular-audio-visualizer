"""Causal instrument-activity estimation from the full mix.

Per analysis frame, a level-independent description of the spectrum is built (band balance,
band rises, tonal peakiness, stereo width), smoothed over short causal time scales, and fed
to a small neural network that outputs one score per instrument: "is this instrument sounding
now". The scores are uncalibrated. They are not a separation of the instruments.

The network's weights are learned from stem-derived activity references and are loaded from a
file; without a weights file there is no activity output.
"""

from pathlib import Path

import numpy as np
from scipy.signal import lfilter

BAND_EDGES_HZ = np.geomspace(40.0, 16000.0, 21)
SMOOTHING_S = (0.25, 1.5)
DEVIATION_S = 1.0
LEVEL_S = 5.0
FEATURE_VERSION = "activity_features_v1"


def _alpha(hop, seconds):
    return 1 - np.exp(-hop / seconds)


class FrameFeatures:
    """Per-frame spectral description. Only ``level`` depends on earlier frames."""

    def __init__(self, rate, size, hop):
        freq = np.fft.rfftfreq(size, 1 / rate)
        edges = BAND_EDGES_HZ
        self.masks = [(freq >= a) & (freq < b) for a, b in zip(edges, edges[1:], strict=False)]
        self.used = [m for m in self.masks if m.any()]
        if len(self.used) != len(self.masks):
            raise ValueError("Analysis window is too short for the activity bands")
        self.hop = hop
        self.previous = None
        self.level = None
        self.count = 4 * len(self.masks) + 1

    def step(self, spectrum):
        """``spectrum``: complex array (bins, channels) of one windowed frame."""
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        if spectrum.shape[1] >= 2:
            side = np.abs(spectrum[:, 0] - spectrum[:, 1]) ** 2 / 4
            mid = np.abs(spectrum[:, 0] + spectrum[:, 1]) ** 2 / 4
        else:
            side, mid = np.zeros_like(power), power
        energy = np.array([power[m].sum() for m in self.masks]) + 1e-12
        total = float(energy.sum())
        log_energy = np.log10(energy)
        balance = log_energy - np.log10(total)
        rise = (
            np.maximum(log_energy - self.previous, 0)
            if self.previous is not None
            else np.zeros(len(energy))
        )
        self.previous = log_energy
        peak = np.array(
            [np.log10((power[m].max() + 1e-12) / (power[m].mean() + 1e-12)) for m in self.masks]
        )
        width = np.array(
            [side[m].sum() / (side[m].sum() + mid[m].sum() + 1e-12) for m in self.masks]
        )
        log_total = np.log10(total)
        if self.level is None:
            self.level = log_total
        self.level += _alpha(self.hop, LEVEL_S) * (log_total - self.level)
        return np.concatenate((balance, rise, peak, width, [log_total - self.level]))


def add_context(frames, hop):
    """Causal smoothing of a (frames, features) array: the same recursion as ``Context``."""
    frames = np.asarray(frames, dtype=np.float64)
    if not len(frames):
        return np.zeros((0, frames.shape[1] * 4))

    def ema(x, seconds):
        a = _alpha(hop, seconds)
        return lfilter([a], [1, a - 1], x, axis=0, zi=(1 - a) * x[:1])[0]

    fast, slow = (ema(frames, s) for s in SMOOTHING_S)
    deviation = ema(np.abs(frames - fast), DEVIATION_S)
    return np.concatenate((frames, fast, slow, deviation), axis=1)


class Context:
    """Frame-by-frame version of ``add_context``."""

    def __init__(self, hop):
        self.hop = hop
        self.fast = self.slow = self.deviation = None

    def step(self, frame):
        if self.fast is None:
            self.fast, self.slow = frame.copy(), frame.copy()
            self.deviation = np.zeros_like(frame)
        else:
            self.fast += _alpha(self.hop, SMOOTHING_S[0]) * (frame - self.fast)
            self.slow += _alpha(self.hop, SMOOTHING_S[1]) * (frame - self.slow)
            self.deviation += _alpha(self.hop, DEVIATION_S) * (
                np.abs(frame - self.fast) - self.deviation
            )
        return np.concatenate((frame, self.fast, self.slow, self.deviation))


class ActivityModel:
    """One hidden layer; weights from an ``.npz`` file written by the training script."""

    def __init__(self, path):
        data = np.load(Path(path), allow_pickle=False)
        if str(data["feature_version"]) != FEATURE_VERSION:
            raise ValueError("Activity weights were trained for a different feature version")
        self.labels = tuple(str(label) for label in data["labels"])
        self.mean, self.scale = data["mean"], data["scale"]
        self.w1, self.b1, self.w2, self.b2 = (data[k] for k in ("w1", "b1", "w2", "b2"))
        self.on, self.off = data["on"], data["off"]
        self.name = str(data["name"])

    def scores(self, features):
        hidden = np.maximum((features - self.mean) / self.scale @ self.w1 + self.b1, 0)
        return 1 / (1 + np.exp(-(hidden @ self.w2 + self.b2)))


class ActivityEstimator:
    """Features, model and on/off hysteresis for one stream."""

    def __init__(self, model, rate, size, hop):
        self.model = model
        self.features = FrameFeatures(rate, size, hop)
        self.context = Context(hop)
        self.active = np.zeros(len(model.labels), dtype=bool)

    def step(self, spectrum, quiet):
        scores = self.model.scores(self.context.step(self.features.step(spectrum)))
        if quiet:
            scores = np.zeros_like(scores)
        self.active = np.where(self.active, scores >= self.model.off, scores >= self.model.on)
        return scores, self.active


def intervals(times, active, hop):
    """Turn per-frame booleans into (start, end) spans."""
    spans, start = [], None
    for time, on in zip(times, active, strict=True):
        if on and start is None:
            start = time - hop / 2
        elif not on and start is not None:
            spans.append((max(0.0, start), time - hop / 2))
            start = None
    if start is not None:
        spans.append((max(0.0, start), times[-1] + hop / 2))
    return spans
