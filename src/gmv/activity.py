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
# Pitch movement is measured on a log-frequency grid, in half-octave bands.
PITCH_RANGE_HZ, PITCH_POINTS_PER_OCTAVE, PITCH_BAND_POINTS, PITCH_SHIFTS = (
    (300.0, 4800.0),
    48,
    24,
    3,
)
FEATURE_VERSION = "activity_features_v2"


def _alpha(hop, seconds):
    return 1 - np.exp(-hop / seconds)


class FrameFeatures:
    """Per-frame spectral description. Only ``level`` depends on earlier frames."""

    def __init__(self, rate, size, hop):
        freq = np.fft.rfftfreq(size, 1 / rate)
        # Bands are runs of neighbouring bins: starts[i] to starts[i + 1].
        self.starts = np.searchsorted(freq, BAND_EDGES_HZ)
        self.widths = np.diff(self.starts)
        self.bands = len(self.widths)
        if (self.widths <= 0).any():
            raise ValueError("Analysis window is too short for the activity bands")
        self.hop = hop
        self.previous = None
        self.level = None
        self.freq = freq
        octaves = np.log2(PITCH_RANGE_HZ[1] / PITCH_RANGE_HZ[0])
        points = round(octaves * PITCH_POINTS_PER_OCTAVE)
        self.grid = PITCH_RANGE_HZ[0] * 2 ** (np.arange(points) / PITCH_POINTS_PER_OCTAVE)
        self.pitch_bands = points // PITCH_BAND_POINTS
        self.previous_grid = None
        self.count = 6 * self.bands + 1 + 2 * self.pitch_bands

    def _sums(self, values):
        return np.add.reduceat(values, self.starts)[: self.bands]

    def _peakiness(self, values):
        top = np.maximum.reduceat(values, self.starts)[: self.bands]
        return np.log10((top + 1e-12) / (self._sums(values) / self.widths + 1e-12))

    def _pitch_movement(self, centre):
        """Per band: how far the spectral pattern slid along log frequency since the last
        frame (positive is upward), and how well it matched after sliding. Sung notes glide
        and wobble; fretted and keyed notes mostly do not."""
        grid = np.interp(self.grid, self.freq, 0.5 * np.log10(centre + 1e-12))
        grid = grid[: self.pitch_bands * PITCH_BAND_POINTS].reshape(self.pitch_bands, -1)
        previous, self.previous_grid = self.previous_grid, grid
        if previous is None:
            return np.zeros(self.pitch_bands), np.zeros(self.pitch_bands)
        edge = PITCH_SHIFTS
        core = grid[:, edge:-edge]
        core = core - core.mean(axis=1, keepdims=True)
        core_norm = np.linalg.norm(core, axis=1)
        matches = np.empty((2 * edge + 1, self.pitch_bands))
        for row, shift in enumerate(range(-edge, edge + 1)):
            other = previous[:, edge + shift : previous.shape[1] - edge + shift]
            other = other - other.mean(axis=1, keepdims=True)
            matches[row] = (core * other).sum(axis=1) / (
                core_norm * np.linalg.norm(other, axis=1) + 1e-12
            )
        best = matches.argmax(axis=0)
        return (edge - best) / edge, matches.max(axis=0)

    def step(self, spectrum):
        """``spectrum``: complex array (bins, channels) of one windowed frame."""
        power = np.mean(np.abs(spectrum) ** 2, axis=1)
        if spectrum.shape[1] >= 2:
            side = np.abs(spectrum[:, 0] - spectrum[:, 1]) ** 2 / 4
            mid = np.abs(spectrum[:, 0] + spectrum[:, 1]) ** 2 / 4
        else:
            side, mid = np.zeros_like(power), power
        energy = self._sums(power) + 1e-12
        total = float(energy.sum())
        log_energy = np.log10(energy)
        balance = log_energy - np.log10(total)
        rise = (
            np.maximum(log_energy - self.previous, 0)
            if self.previous is not None
            else np.zeros(len(energy))
        )
        self.previous = log_energy
        peak = self._peakiness(power)
        side_energy, mid_energy = self._sums(side), self._sums(mid)
        width = side_energy / (side_energy + mid_energy + 1e-12)
        log_total = np.log10(total)
        if self.level is None:
            self.level = log_total
        self.level += _alpha(self.hop, LEVEL_S) * (log_total - self.level)
        # What both channels share: lead vocals, bass and the core of the drum kit sit here.
        centre = mid * mid / (mid + side + 1e-20)
        centre_energy = self._sums(centre) + 1e-12
        centre_balance = np.log10(centre_energy) - np.log10(centre_energy.sum())
        centre_peak = self._peakiness(centre)
        shifts, matches = self._pitch_movement(centre)
        return np.concatenate(
            (
                balance,
                rise,
                peak,
                width,
                [log_total - self.level],
                centre_balance,
                centre_peak,
                shifts,
                matches,
            )
        )


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
