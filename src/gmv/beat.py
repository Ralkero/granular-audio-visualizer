"""Causal beat and tempo tracking from a full-mix onset-strength signal.

One value per analysis hop goes in. The tracker keeps a few seconds of history, estimates the
beat period from the autocorrelation of that history, and follows beat phase with a running
cumulative score (the approach of Ellis 2007 and of Stark, Davies and Plumbley's real-time
tracker, reimplemented here). Each beat is announced about half a period before it happens.

Everything here is a tempo-and-phase hypothesis about the mixture. It knows nothing about
instruments, bars or downbeats.
"""

import numpy as np


class BeatTracker:
    def __init__(
        self,
        hop,
        history=6.0,
        bpm_range=(60.0, 240.0),
        preferred_bpm=120.0,
        prior_octaves=1.0,
        alpha=0.9,
        tightness=5.0,
        tempo_interval=0.5,
        tempo_inertia=0.08,
        lock_confidence=0.1,
    ):
        self.hop = hop
        self.size = round(history / hop)
        self.alpha, self.tightness = alpha, tightness
        self.odf = np.zeros(self.size)
        self.score = np.zeros(self.size)
        fastest, slowest = 60 / bpm_range[1] / hop, 60 / bpm_range[0] / hop
        # Candidate beat periods in (fractional) frames, evenly spaced in log tempo.
        self.periods = np.exp(np.linspace(np.log(fastest), np.log(slowest), 96))
        octaves = np.log2(60 / (self.periods * hop) / preferred_bpm)
        self.prior = np.exp(-0.5 * (octaves / prior_octaves) ** 2)
        ratio = np.log2(self.periods[:, None] / self.periods[None, :])
        self.transition = np.exp(-0.5 * (ratio / tempo_inertia) ** 2)
        self.delta = np.full(len(self.periods), 1 / len(self.periods))
        self.period = 60 / preferred_bpm / hop
        self.tempo_every = max(1, round(tempo_interval / hop))
        self.frames = 0
        self.until_beat = None
        self.until_predict = round(self.period / 2)
        self.confidence = 0.0
        self.lock_confidence = lock_confidence
        self.scale = 0.0

    @property
    def locked(self):
        return self.confidence >= self.lock_confidence and self.frames >= self.size // 2

    @property
    def tempo_bpm(self):
        return 60 / (self.period * self.hop)

    def _estimate_tempo(self):
        x = self.odf - self.odf.mean()
        x = np.maximum(x, 0)
        energy = float(np.dot(x, x))
        if energy < 1e-9:
            self.confidence = 0.0
            return
        spectrum = np.fft.rfft(x, 2 * self.size)
        acf = np.fft.irfft(np.abs(spectrum) ** 2)[: self.size]
        acf = acf / (self.size - np.arange(self.size)) / (energy / self.size)
        lags = np.arange(self.size, dtype=float)
        # A beat period is supported by peaks at its first four multiples.
        comb = sum(np.interp(k * self.periods, lags, acf) for k in (1, 2, 3, 4)) / 4
        observation = np.maximum(comb, 1e-6) * self.prior
        self.delta = (self.delta[:, None] * self.transition).max(axis=0) * observation
        self.delta /= self.delta.sum()
        best = int(np.argmax(self.delta))
        self.period = float(self.periods[best])
        self.confidence = float(np.clip(comb[best], 0, 1))

    def _past_window(self):
        first, last = round(2 * self.period), max(1, round(self.period / 2))
        first = min(first, self.size)
        offsets = np.arange(first, last - 1, -1)
        weights = np.exp(-0.5 * (self.tightness * np.log(offsets / self.period)) ** 2)
        return offsets, weights

    def _predict(self):
        """Extend the cumulative score one period ahead and pick the next beat."""
        offsets, weights = self._past_window()
        ahead = round(self.period)
        future = np.concatenate((self.score, np.zeros(ahead)))
        for i in range(self.size, self.size + ahead):
            future[i] = (weights * future[i - offsets]).max()
        expected = np.arange(1, ahead + 1)
        window = np.exp(-0.5 * ((expected - self.period / 2) / (self.period / 2)) ** 2)
        candidate = future[self.size :] * window
        best = int(np.argmax(candidate))
        shift = 0.0
        if 0 < best < ahead - 1:
            a, b, c = candidate[best - 1 : best + 2]
            if a - 2 * b + c < 0:
                shift = 0.5 * (a - c) / (a - 2 * b + c)
        return best + 1 + float(np.clip(shift, -0.5, 0.5))

    def step(self, value):
        """Feed one onset-strength value. Returns frames until the next beat when a new
        beat has just been predicted (fractional, measured from this frame), else None."""
        # Slowly decaying peak normalisation keeps the score independent of playback level.
        self.scale = max(self.scale * 0.9995, value)
        value = value / self.scale if self.scale > 0 else 0.0
        offsets, weights = self._past_window()
        best = (weights * self.score[self.size - offsets]).max()
        new = (1 - self.alpha) * value + self.alpha * best
        self.odf = np.roll(self.odf, -1)
        self.odf[-1] = value
        self.score = np.roll(self.score, -1)
        self.score[-1] = new
        self.frames += 1
        if self.frames % self.tempo_every == 0:
            self._estimate_tempo()
        announced = None
        if self.until_beat is not None:
            self.until_beat -= 1
            if self.until_beat <= 0:
                self.until_beat = None
                self.until_predict = max(1, round(self.period / 2))
        if self.until_beat is None:
            self.until_predict -= 1
            if self.until_predict <= 0:
                announced = self._predict()
                self.until_beat = announced
        return announced
