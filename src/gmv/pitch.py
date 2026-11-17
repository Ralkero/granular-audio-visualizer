"""Causal multi-pitch candidates in the guitar register (E2 to E6).

Each hop, a longer window than the main analysis (about 85 ms, needed to tell low notes
apart) is transformed, the spectrum is flattened, and every semitone is scored by how much
energy sits on its harmonic series. The strongest notes are picked one at a time, removing
each one's harmonics before the next, so that several simultaneous notes can be reported.
Notes are then followed over time: a note must persist to count as sounding, a fresh rise
on a sounding note is a new attack, and anything else is sustain.

What this is not: it does not know which instrument plays a note. On a full mix it reports
the pitched content of that register, whatever produced it. Scores are uncalibrated. Known
weaknesses: octave and fifth confusions on chords, distorted chords (their intermodulation
products are real spectral peaks), bends reported as the nearest semitone, and at most
``MAX_NOTES`` notes at once.
"""

from dataclasses import dataclass

import numpy as np

METHOD = "harmonic_salience_v1"
LOW_MIDI, HIGH_MIDI = 40, 88  # E2 .. E6
STEPS = 3  # candidates per semitone
HARMONICS = 10
MAX_NOTES = 4
WINDOW_S = 0.085


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69) / 12)


def note_name(midi):
    names = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
    return f"{names[int(round(midi)) % 12]}{int(round(midi)) // 12 - 1}"


@dataclass
class Note:
    midi: int
    score: float
    age: int = 1  # frames since the note started
    missed: int = 0
    sounding: bool = False  # persisted long enough to be shown
    energy: float = 0.0  # recent level of the note's own harmonics
    since_attack: int = 0
    attack: bool = False  # set for the one frame in which an attack is reported
    attack_back: int = 0  # how many frames before this one the attack happened


class PitchTracker:
    def __init__(
        self,
        rate,
        hop,
        *,
        threshold=0.2,
        relative=0.4,
        need=4,
        floor=0.01,
        gate=0.01,
        confirm=3,
        release=3,
        reattack=1.8,
        overtone=1.25,
        cancel=2,
        cancel_gain=0.4,
        require=0.5,
    ):
        self.rate, self.hop = rate, hop
        self.size = 1 << round(np.log2(WINDOW_S * rate))
        self.window = np.hanning(self.size)
        self.buffer = np.zeros(self.size)
        self.filled = 0
        self.threshold, self.relative = threshold, relative
        self.confirm, self.release, self.reattack = confirm, release, reattack
        self.cancel, self.cancel_gain, self.require = cancel, cancel_gain, require
        self.need, self.floor, self.gate = need, floor, gate
        self.overtone = overtone
        self.reference = 0.0  # loudest recent partial; fades over about ten seconds
        self.reference_decay = float(np.exp(-hop / 10.0))
        freq = np.fft.rfftfreq(self.size, 1 / rate)
        self.bin_hz = freq[1]
        self.top = int(np.searchsorted(freq, min(6000.0, rate / 2 - 1)))
        self.candidates = LOW_MIDI + np.arange((HIGH_MIDI - LOW_MIDI) * STEPS + 1) / STEPS
        f0 = midi_to_hz(self.candidates)
        # weights[c, b]: how much bin b counts towards candidate c (its harmonic series,
        # each harmonic spread over the two nearest bins).
        self.weights = np.zeros((len(f0), self.top))
        self.harmonic_bins = []
        for row, fundamental in enumerate(f0):
            bins = []
            for harmonic in range(1, HARMONICS + 1):
                position = harmonic * fundamental / self.bin_hz
                low = int(position)
                if low + 1 >= self.top:
                    break
                share = position - low
                gain = 1.0 / harmonic**0.7
                self.weights[row, low] += gain * (1 - share)
                self.weights[row, low + 1] += gain * share
                bins.append(low)
            self.weights[row] /= self.weights[row].sum()
            self.harmonic_bins.append(np.array(bins))
        # Spectral flattening: each bin relative to its neighbourhood (about a third octave).
        self.smooth_width = np.maximum(3, (freq[: self.top] * 0.12 / self.bin_hz).astype(int))
        self.notes = {}
        # The window is long, so what it describes is centred this far before its end.
        self.lag_s = self.size / rate / 2

    def _flatten(self, magnitude):
        """Peak strength per bin, 0 to 1: how far a spectral peak stands above its
        neighbourhood. Anything that is not a local peak, or is far below the loudest partial
        of the frame, counts as nothing, so noise and silence give no support to any note."""
        cumulative = np.concatenate(([0.0], np.cumsum(magnitude)))
        index = np.arange(self.top)
        low = np.maximum(0, index - self.smooth_width)
        high = np.minimum(self.top, index + self.smooth_width + 1)
        local = (cumulative[high] - cumulative[low]) / (high - low)
        strength = np.clip(np.log2(magnitude / (local + 1e-12) + 1e-12) / 3.0, 0, 1)
        peak = np.zeros(self.top, dtype=bool)
        peak[1:-1] = (magnitude[1:-1] >= magnitude[:-2]) & (magnitude[1:-1] > magnitude[2:])
        strength *= peak & (magnitude > magnitude.max() * self.floor)
        # A partial rarely sits exactly on a bin: let it count a little for its neighbours.
        spread = strength.copy()
        spread[1:] = np.maximum(spread[1:], 0.7 * strength[:-1])
        spread[:-1] = np.maximum(spread[:-1], 0.7 * strength[1:])
        return spread

    def salience(self, audio):
        """Peak strengths for one window of mono audio and its magnitudes, or (None, None)
        when the window is silent or far below the loudest recent level."""
        magnitude = np.abs(np.fft.rfft(audio * self.window))[: self.top]
        loudest = float(magnitude.max())
        self.reference = max(loudest, self.reference * self.reference_decay)
        if loudest <= 1e-7 or loudest < self.reference * self.gate:
            return None, None
        return self._flatten(magnitude), magnitude

    def pick(self, flat, magnitude):
        """Up to ``MAX_NOTES`` (midi, score, energy) triples, strongest first."""
        original = self.weights @ flat
        flat = flat.copy()
        found = []
        strongest = None
        for _ in range(MAX_NOTES):
            scores = self.weights @ flat
            best = int(scores.argmax())
            score = float(scores[best])
            if strongest is None:
                strongest = score
            if score < self.threshold or score < self.relative * strongest:
                break
            midi = float(self.candidates[best])
            bins = self.harmonic_bins[best]
            support = np.maximum(flat[bins], flat[np.minimum(bins + 1, self.top - 1)])
            present = int((support[:6] > 0.2).sum())
            # A note needs several of its low harmonics, and its own fundamental or octave:
            # without those it is usually the octave below, or the fifth of, a real note.
            real = present >= min(self.need, len(bins)) and support[:2].max() > 0.2
            if real:
                energy = float(np.maximum(magnitude[bins[:6]], magnitude[bins[:6] + 1]).sum())
                found.append((midi, score, energy, float(original[best])))
            for position in bins:
                flat[max(0, position - self.cancel + 1) : position + self.cancel + 1] *= (
                    self.cancel_gain
                )
        # Every harmonic of a note an octave (or a twelfth, or two octaves) above another
        # note is also a harmonic of that lower note. The upper one is kept only if it is
        # clearly stronger than the lower note would explain, whichever was picked first.
        kept = []
        for midi, score, energy, own in found:
            overtone = any(
                min(abs(midi - lower - gap) for gap in (12, 19, 24, 28)) < 0.5
                and own <= self.overtone * lower_own
                for lower, _, _, lower_own in found
            )
            if not overtone:
                kept.append((midi, score, energy))
        return kept

    def step(self, mono):
        """Feed one hop of mono samples; returns the notes currently followed.

        Each returned ``Note`` has ``sounding`` (persisted long enough to show) and
        ``attack`` (true for the frame in which a new attack of that note is reported).
        """
        mono = np.asarray(mono, dtype=float)
        if len(mono) >= self.size:
            self.buffer[:] = mono[-self.size :]
        else:
            self.buffer = np.concatenate((self.buffer[len(mono) :], mono))
        self.filled = min(self.size, self.filled + len(mono))
        flat = magnitude = None
        if self.filled >= self.size:
            flat, magnitude = self.salience(self.buffer)
        picked = self.pick(flat, magnitude) if flat is not None else []
        seen = {}
        for midi, score, energy in picked:
            key = int(round(midi))
            if key not in seen or score > seen[key][0]:
                seen[key] = (score, energy)
        for key, note in list(self.notes.items()):
            note.attack = False
            note.since_attack += 1
            if key in seen:
                score, energy = seen.pop(key)
                note.age += 1
                note.missed = 0
                # The note's own harmonics getting clearly louder again while it is still
                # sounding is the string being struck again; anything else is sustain.
                if note.sounding and note.since_attack > 5 and energy > self.reattack * note.energy:
                    note.attack, note.attack_back, note.since_attack = True, 0, 0
                note.energy = 0.6 * note.energy + 0.4 * energy if note.age > 2 else energy
                note.score = score
                if not note.sounding and note.age >= self.confirm:
                    note.sounding = note.attack = True
                    note.attack_back, note.since_attack = note.age - 1, 0
            else:
                note.missed += 1
                note.score *= 0.7
                if note.missed >= self.release or not note.sounding:
                    del self.notes[key]
        for key, (score, energy) in seen.items():
            self.notes[key] = Note(midi=key, score=score, energy=energy)
        return [note for note in self.notes.values() if note.sounding]

    def reset(self):
        self.buffer[:] = 0
        self.filled = 0
        self.notes = {}
        self.reference = 0.0
