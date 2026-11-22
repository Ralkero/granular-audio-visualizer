"""Pieces the drum, bass and vocal analyzers share. The guitar analyzer predates this module
and keeps its own copies, so that it stays exactly as it was accepted."""

import numpy as np

SLOWEST_HZ, FASTEST_HZ = 2.0, 17.0  # repetition rates that are reported
HISTORY_S = 1.28


def threshold_scale(sensitivity):
    """Sensitivity 0..1 (0.5 is the tuned default) as a factor on thresholds:
    1 at the default, 2 at the least and 0.5 at the most sensitive setting."""
    return float(2.0 ** (1.0 - 2.0 * min(1.0, max(0.0, sensitivity))))


def centre_magnitude(spectrum):
    """Per bin, what the two channels share: |L+R|/2 less |L-R|/2, never below zero.

    A source in the middle (kick, snare, bass, lead voice) is identical in both channels and
    survives. A source panned to one side, or two different takes left and right, has a
    difference about as large as its sum and mostly cancels. A mono recording passes through
    unchanged, so nothing is removed and the analyzers say ``basis`` "mono".
    Returns (centre magnitude, sum magnitude, whether there is a stereo image at all)."""
    left = spectrum[:, 0]
    right = spectrum[:, 1] if spectrum.shape[1] > 1 else left
    total = np.abs(left + right) / 2
    side = np.abs(left - right) / 2
    stereo = float(side.sum()) > 0.05 * float(total.sum())
    return np.maximum(total - side, 0.0), total, stereo


class Level:
    """A band's level against its own recent peak, 0..1 over 30 dB."""

    def __init__(self, hop, hold_s=4.0):
        self.decay = float(np.exp(-hop / hold_s))
        self.reference = 0.0

    def step(self, value):
        self.reference = max(value, self.reference * self.decay)
        if self.reference <= 1e-12:
            return 0.0
        return float(min(1.0, max(0.0, 1.0 + np.log10(max(value, 1e-12) / self.reference) / 1.5)))


class Repetition:
    """Rate and regularity of an articulation signal: its strongest self-similarity lag over
    the last 1.28 s, reported only when at least three attacks fell in that time."""

    def __init__(self, hop):
        self.hop = hop
        self.frames = max(16, round(HISTORY_S / hop))
        self.lags = np.arange(
            max(2, round(1 / (FASTEST_HZ * hop))), round(1 / (SLOWEST_HZ * hop)) + 1
        )
        self.every = max(1, round(0.1 / hop))
        self.reset()

    def reset(self):
        self.flux = np.zeros(self.frames)
        self.struck = np.zeros(self.frames, dtype=bool)
        self.count = 0
        self.rate_hz, self.regularity = None, 0.0

    def step(self, flux, attack):
        self.flux[:-1] = self.flux[1:]
        self.flux[-1] = flux
        self.struck[:-1] = self.struck[1:]
        self.struck[-1] = attack
        self.count += 1
        if self.count % self.every:
            return
        x = self.flux - self.flux.mean()
        power = float(np.dot(x, x))
        if power < 2e-3 * len(x) or self.count < len(x) // 2 or self.struck.sum() < 3:
            self.rate_hz, self.regularity = None, 0.0
            return
        scores = np.array(
            [
                np.dot(x[:-lag], x[lag:])
                / (np.linalg.norm(x[:-lag]) * np.linalg.norm(x[lag:]) + 1e-12)
                for lag in self.lags
            ]
        )
        best = int(np.argmax(scores))
        for index in range(best):  # the shortest lag nearly as good as the best
            if scores[index] >= 0.85 * scores[best] and scores[index] > 0.3:
                best = index
                break
        self.regularity = float(max(0.0, min(1.0, scores[best])))
        self.rate_hz = float(1.0 / (self.lags[best] * self.hop)) if self.regularity >= 0.4 else None
