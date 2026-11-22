"""Bass notes from the mixed signal: pitch, weight and note changes, every analysis frame.

The analysis pass's own window (46 ms) cannot tell low bass notes apart: E1 and F1 are
2.5 Hz apart and its bins are 21.5 Hz wide. So this analyzer keeps the last 170 ms of the
mono signal, low-passed at 180 Hz and thinned to about 3 kHz (a few hundred samples, cheap),
and measures the period of that directly by autocorrelation. This is the one place where an
instrument analyzer looks at samples as well as at the shared spectrum.

* ``pitch_midi``: the note, when the low register is periodic enough to have one
  (``pitch_confidence``); otherwise None. A kick drum makes the low register briefly
  aperiodic, so during a hit the last note is held rather than replaced.
* ``level``: how much periodic low-register sound there is, against its recent peak.
* ``attack``: a new note (the pitch settled at least a semitone away) or the same note
  struck again (its level rose sharply while it stayed periodic).

It follows the lowest periodic sound in the centre of the mix. That is usually the bass; it
can be a low guitar riff, a synthesizer or a tuned kick. Whether a bass is playing is the
activity model's judgement. Notes above about C3 are not followed, and the weight is measured
against the whole mix, so that a voice or a guitar with a little energy down there does not
count as a heavy bass.
"""

import numpy as np
from scipy.signal import butter, sosfilt

from . import Activity
from .common import Level, Repetition, threshold_scale

METHOD = "bass_pitch_v1"
LOW_HZ, HIGH_HZ = 38.0, 135.0  # just under E1 to just over C3
WINDOW_S = 0.17
CONFIDENT = 0.55  # periodicity needed to name a note at default sensitivity
HEAVY = 0.25  # weight below which the low register is too faint to be a bass note
LOW_MIDI, HIGH_MIDI = 26.0, 56.0  # display range of ``centre``


def hz_to_midi(hz):
    return 69.0 + 12.0 * np.log2(hz / 440.0)


class BassPitch:
    key = "bass"
    method = METHOD
    needs_samples = True

    def __init__(self, rate, size, hop_s, sensitivity=0.5):
        self.hop = hop_s
        freq = np.fft.rfftfreq(size, 1 / rate)
        self.low = slice(int(np.searchsorted(freq, 38.0)), int(np.searchsorted(freq, 300.0)))
        self.step_down = max(1, round(rate / 3000))
        self.rate = rate / self.step_down
        self.filter = butter(4, 180.0, btype="low", fs=rate, output="sos")
        self.size = int(2 ** np.ceil(np.log2(WINDOW_S * self.rate)))
        self.window = np.hanning(self.size)
        self.lags = np.arange(int(self.rate / HIGH_HZ), int(np.ceil(self.rate / LOW_HZ)) + 1)
        # What a perfectly periodic signal scores at each lag under this window.
        envelope = np.fft.irfft(np.abs(np.fft.rfft(self.window, 2 * self.size)) ** 2)
        self.ideal = envelope[self.lags] / envelope[0]
        self.sensitivity = sensitivity
        # The window describes its middle: answers are back-dated by half of it.
        self.delay_frames = max(1, round(self.size / self.rate / 2 / hop_s))
        self.delay_s = self.delay_frames * hop_s
        self.repetition = Repetition(hop_s)
        self.reset()

    def reset(self):
        self.state = np.zeros((self.filter.shape[0], 2))
        self.phase = 0
        self.buffer = np.zeros(self.size)
        self.filled = 0
        self.loudness = Level(self.hop)
        self.full = 0.0  # recent peak of the whole signal's level
        self.hops = [0.0, 0.0, 0.0]  # low-register level of the three frames before
        self.top = 0.0
        self.note = None  # the note being held, as a MIDI number
        self.candidate, self.candidate_frames = None, 0
        self.smooth = 0.0  # level, eased
        self.previous_loud = 0.0
        self.since_attack = 1e6
        self.repetition.reset()

    def _pitch(self):
        """(MIDI note or None, periodicity 0..1, rms) of the samples held now."""
        x = self.buffer * self.window
        power = np.abs(np.fft.rfft(x, 2 * self.size)) ** 2
        auto = np.fft.irfft(power)
        rms = float(np.sqrt(np.mean(self.buffer**2)))
        if auto[0] <= 1e-14:
            return None, 0.0, rms
        scores = auto[self.lags] / auto[0] / self.ideal
        best = int(np.argmax(scores))
        # Twice the period fits as well as the period: take the shortest lag that is nearly
        # as good, which is the higher octave when both fit.
        for index in range(best):
            if (
                scores[index] >= 0.9 * scores[best]
                and (index == 0 or scores[index] >= scores[index - 1])
                and scores[index] >= scores[min(len(scores) - 1, index + 1)]
            ):
                best = index
                break
        clarity = float(min(1.0, max(0.0, scores[best])))
        lag = float(self.lags[best])
        if 0 < best < len(scores) - 1:  # refine between whole samples
            a, b, c = scores[best - 1], scores[best], scores[best + 1]
            if a + c - 2 * b < 0:
                lag += 0.5 * (a - c) / (a - 2 * b + c)
        return float(hz_to_midi(self.rate / lag)), clarity, rms

    def step(self, spectrum, quiet=False, samples=None):
        if samples is None:
            raise ValueError("The bass analyzer needs the frame's new samples")
        mono = samples[:, : min(2, samples.shape[1])].mean(axis=1)
        low, self.state = sosfilt(self.filter, mono, zi=self.state)
        thinned = low[(-self.phase) % self.step_down :: self.step_down]
        self.phase = (self.phase + len(mono)) % self.step_down
        count = len(thinned)
        if count:
            self.buffer = np.concatenate((self.buffer[count:], thinned))[-self.size :]
            self.filled = min(self.size, self.filled + count)
        if self.filled < self.size:
            return None
        scale = threshold_scale(self.sensitivity)
        midi, clarity, rms = self._pitch()
        # Weight is measured against the whole mix as well as against the low register's own
        # recent peak: a little energy down there under a loud voice is not a heavy bass.
        self.full = max(float(np.sqrt(np.mean(mono**2))), self.full * float(np.exp(-self.hop / 4.0)))
        self.loudness.reference = max(self.loudness.reference, 0.35 * self.full)
        loud = 0.0 if quiet else self.loudness.step(rms)
        # Struck again: the low register of the shared spectrum (which sees 46 ms, not
        # 170) clearly above where it was in the three frames before.
        newest = float(np.abs(spectrum[self.low, : min(2, spectrum.shape[1])].sum(axis=1)).sum())
        self.top = max(newest, self.top * float(np.exp(-self.hop / 2.0)))
        struck = self.top > 0 and (newest - max(self.hops)) / self.top > 0.18 * scale
        self.hops = self.hops[1:] + [newest]
        confident = not quiet and clarity >= min(0.9, CONFIDENT * scale**0.5) and loud > HEAVY
        attack = 0.0
        self.since_attack += self.hop
        if confident:
            if self.note is not None and abs(midi - self.note) < 0.7:
                self.note = 0.7 * self.note + 0.3 * midi
                self.candidate, self.candidate_frames = None, 0
                # The same note struck again: its newest 20 ms clearly louder than the
                # three hops before, while it stays periodic.
                if struck and self.since_attack > 0.12:
                    attack = min(1.0, max(0.3, loud))
            elif self.candidate is not None and abs(midi - self.candidate) < 0.7:
                self.candidate_frames += 1
                if self.candidate_frames >= 2:  # settled on a note for 60 ms
                    self.note, self.candidate, self.candidate_frames = midi, None, 0
                    attack = max(0.3, loud)
            else:
                self.candidate, self.candidate_frames = midi, 0
        elif quiet or loud < HEAVY:
            self.note, self.candidate = None, None
        flux = max(0.0, loud - self.previous_loud) * 4.0
        self.previous_loud = 0.6 * self.previous_loud + 0.4 * loud
        # Weight: periodic low sound. A hit (aperiodic) does not add weight; it only
        # stops the note from being re-read for a moment.
        target = loud * min(1.0, clarity / 0.7) if not quiet else 0.0
        self.smooth += (target - self.smooth) * (1.0 - float(np.exp(-self.hop / 0.06)))
        if attack:
            self.since_attack = 0.0
        self.repetition.step(min(1.0, flux), attack > 0)
        note = self.note if confident or (self.note is not None and loud > HEAVY) else None
        return Activity(
            delay_frames=self.delay_frames,
            level=float(min(1.0, self.smooth)),
            flux=float(min(1.0, flux)),
            attack=float(attack),
            rate_hz=self.repetition.rate_hz,
            regularity=self.repetition.regularity,
            centre=0.5
            if note is None
            else float(min(1.0, max(0.0, (note - LOW_MIDI) / (HIGH_MIDI - LOW_MIDI)))),
            spread=0.0,
            basis="mono",
            pitch_midi=note,
            pitch_confidence=clarity if not quiet else 0.0,
        )
