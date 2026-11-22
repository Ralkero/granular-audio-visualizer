"""Synthetic instruments for the layer tests: stereo float32 at 48 kHz, deterministic."""

import numpy as np
from scipy.signal import butter, sosfilt

RATE = 48000


def _place(seconds, times, make, level, pan=(1.0, 1.0)):
    out = np.zeros(round(seconds * RATE), np.float32)
    for index, time in enumerate(times):
        one = make(index)
        start = round(time * RATE)
        one = one[: len(out) - start]
        out[start : start + len(one)] += level * one.astype(np.float32)
    return np.column_stack((out * pan[0], out * pan[1]))


def _band(noise, low, high):
    return sosfilt(butter(4, [low, high], btype="band", fs=RATE, output="sos"), noise)


def kick(seconds, times, level=0.5, seed=0):
    """A falling low tone with a short click, in the centre."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(0.25 * RATE)) / RATE
    tone = np.sin(2 * np.pi * (50 * t + 70 * 0.04 * (1 - np.exp(-t / 0.04)))) * np.exp(-t / 0.09)

    def make(_):
        click = _band(rng.standard_normal(len(t)), 1500, 5000) * np.exp(-t / 0.004)
        return (tone + 0.15 * click) * np.minimum(1.0, t / 0.001)

    return _place(seconds, times, make, level)


def snare(seconds, times, level=0.4, seed=1):
    """A burst of mid-band noise with a short body tone, in the centre."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(0.2 * RATE)) / RATE

    def make(_):
        noise = _band(rng.standard_normal(len(t)), 180, 6000) * np.exp(-t / 0.05)
        return noise + 0.4 * np.sin(2 * np.pi * 190 * t) * np.exp(-t / 0.06)

    return _place(seconds, times, make, level)


def hat(seconds, times, level=0.2, seed=2):
    """A very short burst above 7 kHz, slightly off centre as an overhead would have it."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(0.06 * RATE)) / RATE
    high = butter(4, 7000, btype="high", fs=RATE, output="sos")

    def make(_):
        return sosfilt(high, rng.standard_normal(len(t))) * np.exp(-t / 0.012)

    return _place(seconds, times, make, level, pan=(1.0, 0.7))


def bass(seconds, notes, level=0.4, decay=1.2):
    """Bass notes in the centre: ``notes`` is [(start, MIDI note, length)]; each has a soft
    pluck and five harmonics."""
    out = np.zeros(round(seconds * RATE), np.float32)
    for start, midi, length in notes:
        t = np.arange(round(length * RATE)) / RATE
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        tone = sum(np.sin(2 * np.pi * hz * h * t) / h**1.3 for h in (1, 2, 3, 4, 5))
        tone *= np.exp(-t / decay) * np.minimum(1.0, t / 0.008) * np.minimum(1.0, (length - t) / 0.02)
        begin = round(start * RATE)
        tone = tone[: len(out) - begin]
        out[begin : begin + len(tone)] += level * tone.astype(np.float32)
    return np.column_stack((out, out))


def voice(seconds, phrases, level=0.3):
    """A sung line in the centre: ``phrases`` is [(start, length, MIDI from, MIDI to)]. Ten
    harmonics shaped by a broad vowel-like emphasis around 700 Hz and 1.2 kHz, with a
    little vibrato, swelling in and out."""
    out = np.zeros(round(seconds * RATE), np.float32)
    for start, length, first, last in phrases:
        t = np.arange(round(length * RATE)) / RATE
        midi = first + (last - first) * t / length + 0.15 * np.sin(2 * np.pi * 5.5 * t)
        hz = 440.0 * 2 ** ((midi - 69) / 12)
        phase = 2 * np.pi * np.cumsum(hz) / RATE
        tone = np.zeros(len(t))
        for h in range(1, 11):
            f = hz * h
            shape = np.exp(-(((f - 700) / 500) ** 2)) + 0.6 * np.exp(-(((f - 1200) / 500) ** 2)) + 0.15
            tone += shape * np.sin(h * phase)
        tone *= np.minimum(1.0, t / 0.06) * np.minimum(1.0, (length - t) / 0.08)
        begin = round(start * RATE)
        tone = tone[: len(out) - begin]
        out[begin : begin + len(tone)] += level * tone.astype(np.float32)
    return np.column_stack((out, out))


def pad(seconds, level=0.15, seed=3):
    """A held synthesizer chord, a different voicing on each side (panned wide)."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * RATE)) / RATE
    sides = []
    for notes in ((220.0, 277.2, 329.6, 440.0), (221.0, 276.0, 331.0, 441.5)):
        sides.append(
            sum(np.sin(2 * np.pi * (n * h * t + rng.random())) / h for n in notes for h in (1, 2, 3))
        )
    return (level * np.column_stack(sides)).astype(np.float32)
