"""Rhythmic feel: the pulse, how sure it is, and what happens around it.

The onset signal is the rise in magnitude of the percussive part of the spectrogram, whole
and in a low (under 150 Hz) and a high (above 5 kHz) band. Only its shape is used. Over a
passage of about twelve seconds:

1. its autocorrelation gives the candidate pulse periods; the pulse is the best one between
   50 and 220 a minute, with a broad preference for about 110. **Which metrical level is
   "the beat" is a convention here, not a finding**: half and double the reported tempo are
   always candidates, and ``tempo_ambiguity`` says how nearly as good they are;
2. the phase that puts most of the onset signal on the pulse places the beats;
3. the onset signal is folded onto one beat (24 steps) and read: how much falls on the
   beat, half-way, on the quarters, on the triplets.

Everything after step 1 is relative to the pulse found, so it is only stated when the
pulse is clear (``pulse_clarity``); otherwise it is unknown. A passage with no clear pulse
is "no pulse found", never "irregular".

A passage in which nothing is struck (a held pad) has no pulse to find and is unknown too.

Limits: a tempo that changes inside the passage smears everything; a dense mix leaves a
weak percussive part; the folding assumes a constant pulse and knows nothing of bars,
so "off-beat weight" is weight between pulses, whichever metrical level the pulse is.
"""

import numpy as np

from .frontend import FRAME_S, result

BPM_RANGE = (50.0, 220.0)
PREFERRED_BPM, PREFERENCE_OCTAVES = 110.0, 1.0
STEPS = 24  # steps of one beat the onset signal is folded onto
CLEAR = 0.25  # pulse clarity from which the beat-relative values are stated
NEEDED_S = 8.0
SUBDIVIDED = 0.2  # share of the onset signal on the subdivisions for their kind to be read
STRUCK = (0.05, 0.15)  # share of the percussive part that is newly risen: none .. plenty

DEFINITIONS = {
    "pulse_clarity": ("0..1", "measured", "autocorrelation of the onset signal at the pulse period: how periodic the passage is", "0.1 .. 0.8"),
    "tempo_bpm": ("BPM", "inferred", "the pulse as a tempo; half and double are equally valid readings", "60 .. 200"),
    "tempo_ambiguity": ("0..1+", "measured", "strength of the stronger of half and double the pulse, relative to the pulse: near or above 1, the level is undecided", "0.3 .. 1.2"),
    "on_beat_share": ("0..1", "measured", "share of the onset signal within a twelfth of a beat of the pulse; 0.17 if spread evenly", "0.2 .. 0.6"),
    "events_per_beat": ("count", "measured", "onset peaks per pulse: how busy the rhythm is relative to its own beat", "0.8 .. 4"),
    "offbeat_weight": ("0..1", "measured", "of the onset signal on the beat and half-way between beats, the share half-way", "0.2 .. 0.6"),
    "triplet_share": ("0..1", "measured", "of the onset signal on the subdivisions, the share on the thirds rather than the quarters", "0.2 .. 0.6"),
    "low_events_per_beat": ("count", "measured", "onset peaks under 150 Hz per pulse (where a kick is)", "0.3 .. 2"),
    "bar_repetition": ("0..1", "measured", "autocorrelation of the onset signal at four pulses: how much the pattern repeats", "0 .. 0.7"),
    "accent_alternation": ("-1..1", "measured", "how regularly strong and weak pulses alternate (a backbeat does)", "-0.3 .. 0.6"),
}


def envelopes(analysis):
    """Onset signals per frame: whole, low and high."""
    freq = analysis.freq
    # Magnitude, not its logarithm: accents are what tell the beat from its subdivisions,
    # and a logarithm flattens them. Every use below is a ratio, so the level cancels.
    compressed = np.sqrt(np.maximum(analysis.percussive, 0.0))
    rise = np.maximum(np.diff(compressed, axis=0, prepend=compressed[:1]), 0.0)
    rise[~analysis.music] = 0.0
    return {
        "body": np.where(analysis.music, compressed.sum(axis=1), 0.0),
        "all": rise.sum(axis=1),
        "low": rise[:, freq < 150.0].sum(axis=1),
        "high": rise[:, freq > 5000.0].sum(axis=1),
    }


def autocorrelation(signal, most):
    """Normalised, unbiased autocorrelation for lags 0..most of a mean-removed signal."""
    x = signal - signal.mean()
    energy = float(x @ x)
    if energy <= 0:
        return np.zeros(most + 1)
    spectrum = np.fft.rfft(x, 2 * len(x))
    full = np.fft.irfft(np.abs(spectrum) ** 2)[: most + 1]
    return full / (len(x) - np.arange(most + 1)) / (energy / len(x))


def peaks(signal):
    """How many clear local peaks the onset signal has."""
    if len(signal) < 3 or signal.max() <= 0:
        return 0
    floor = np.median(signal) + 0.3 * (signal.max() - np.median(signal)) * 0.5
    inner = signal[1:-1]
    return int(((inner > signal[:-2]) & (inner >= signal[2:]) & (inner > floor)).sum())


def at(values, lag):
    return float(np.interp(lag, np.arange(len(values)), values))


def measure(analysis, start, end, cache=None):
    """The rhythm measurements of [start, end) seconds."""
    cache = cache if cache is not None else {}
    if "envelopes" not in cache:
        cache["envelopes"] = envelopes(analysis)
    out = {name: (None, 0.0) for name in DEFINITIONS}
    part = analysis.frames(start, end)
    music = analysis.music[part]
    seconds = float(music.sum()) * FRAME_S
    signal = cache["envelopes"]["all"][part]
    if seconds < 4.0 or signal.max() <= 0:
        return out
    # How much of the percussive part is newly risen. A held sound has almost none, and
    # the faint ripple it does have can look perfectly periodic: then nothing is struck,
    # and there is no pulse to speak of.
    struck = float(signal.sum()) / max(float(cache["envelopes"]["body"][part].sum()), 1e-12)
    if struck < STRUCK[0]:
        return out
    enough = (1.0 - np.exp(-seconds / NEEDED_S)) * min(1.0, struck / STRUCK[1])
    shortest, longest = 60.0 / BPM_RANGE[1] / FRAME_S, 60.0 / BPM_RANGE[0] / FRAME_S
    most = min(len(signal) - 2, int(4 * longest) + 2)
    if most < longest:
        return out
    # Eased over three frames first: an onset one frame wide would otherwise only
    # correlate at whole-frame lags, and a pulse period is rarely a whole number of frames.
    eased = np.convolve(signal, (0.25, 0.5, 0.25), mode="same")
    correlation = autocorrelation(eased, most)
    lags = np.arange(int(np.ceil(shortest)), int(longest) + 1)
    preference = np.exp(
        -0.5 * (np.log2(60.0 / (lags * FRAME_S) / PREFERRED_BPM) / PREFERENCE_OCTAVES) ** 2
    )
    best = int(lags[int(np.argmax(np.maximum(correlation[lags], 0.0) * preference))])
    # Refine to a fraction of a frame.
    a, b, c = correlation[best - 1], correlation[best], correlation[best + 1]
    period = best + (0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c < 0 else 0.0)
    clarity = max(0.0, at(correlation, period))
    out["pulse_clarity"] = result(clarity, enough)
    if clarity <= 0.05:
        return out
    sure = enough * min(1.0, clarity / CLEAR)
    out["tempo_bpm"] = result(60.0 / (period * FRAME_S), sure)
    other = max(at(correlation, period / 2) if period / 2 >= 2 else 0.0, at(correlation, 2 * period))
    out["tempo_ambiguity"] = result(max(0.0, other) / clarity, enough)
    if clarity < CLEAR:
        return out  # no clear pulse: what happens around it is unknown
    # Fold onto one beat, at the phase that puts most on the pulse.
    position = (np.arange(len(signal)) / period) % 1.0
    step = np.minimum((position * STEPS).astype(int), STEPS - 1)
    folded = np.bincount(step, weights=signal, minlength=STEPS)
    folded = np.roll(folded, -int(np.argmax(folded + np.roll(folded, 1) + np.roll(folded, -1))))
    total = float(folded.sum())
    near = lambda index: float(folded[(index - 1) % STEPS] + folded[index] + folded[(index + 1) % STEPS])  # noqa: E731
    on, half = near(0), near(12)
    quarters, thirds = near(6) + near(18), near(8) + near(16)
    out["on_beat_share"] = result(on / total, sure)
    out["offbeat_weight"] = result(half / max(on + half, 1e-12), sure)
    if (thirds + quarters) / total >= SUBDIVIDED:  # otherwise nothing is subdivided
        out["triplet_share"] = result(thirds / (thirds + quarters), sure)
    beats = len(signal) / period
    out["events_per_beat"] = result(peaks(signal) / beats, sure)
    out["low_events_per_beat"] = result(peaks(cache["envelopes"]["low"][part]) / beats, sure)
    if 4 * period < most:
        out["bar_repetition"] = result(max(0.0, at(correlation, 4 * period)), sure)
    # Strength of each pulse, then whether strong and weak alternate.
    phase = float(np.argmax(np.bincount(step, weights=signal, minlength=STEPS))) / STEPS * period
    indices = np.arange(phase, len(signal) - 1, period)
    if len(indices) >= 8:
        # Summed over the frames round each pulse: how an onset falls between two frames
        # must not look like an accent.
        total_to = np.concatenate(([0.0], np.cumsum(signal)))
        low = np.clip(np.round(indices).astype(int) - 2, 0, len(signal))
        strength = total_to[np.clip(low + 5, 0, len(signal))] - total_to[low]
        x = strength - strength.mean()
        if float(x @ x) > 0:
            lag1 = float(x[:-1] @ x[1:]) / float(x @ x)
            lag2 = float(x[:-2] @ x[2:]) / float(x @ x)
            # Scaled by how much the pulses differ at all: among equal pulses the tiny
            # leftover differences alternate or not by chance.
            varied = min(1.0, float(strength.std() / max(strength.mean(), 1e-12)) / 0.2)
            out["accent_alternation"] = result(varied * (lag2 - lag1) / 2.0, sure)
    return out
