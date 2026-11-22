"""The lead voice from the mixed signal: how strongly it sounds, its pitch contour when it
has one, and where phrases begin and end. Every analysis frame.

A lead vocal is almost always in the centre of the stereo image, so everything here works
on the centre extraction (what both channels share), between 150 Hz and 4 kHz.

* ``pitch_midi``: the pitch whose harmonic series best explains the spectral peaks of the
  centre, between A2 and A5, when it explains them well enough (``pitch_confidence``).
  A pitch close to the one before is preferred, so the contour does not jump between a
  note and its octave. Harsh or whispered vocals have no clear pitch: then None.
* ``level``: how loud the centre is in the range where a voice carries (300 Hz to
  3.4 kHz), counted in full when it is pitched and at a third when it is not.
* ``boundary``: "start" when a phrase begins, "end" when it has been over for a quarter of
  a second. A phrase is sound in the centre while the activity model reports a voice in
  that frame (the analysis pass hands that judgement over); without a model it is sound in
  the centre, which in a full band never stops. ``attack`` is set at a start.

None of this proves a voice. A snare, a centred guitar solo or a synthesizer lead also
live in the centre. Whether anybody is singing is the activity model's judgement; a phrase
drawn on screen is a phrase of *the centre while the model reports vocals*.
"""

import numpy as np
from scipy.ndimage import median_filter

from . import Activity
from .common import Level, Repetition, centre_magnitude, threshold_scale

METHOD = "vocal_contour_v1"
BAND_HZ = (150.0, 4000.0)
VOICE_HZ = (300.0, 3400.0)
LOW_MIDI, HIGH_MIDI = 45.0, 81.0  # A2 .. A5
STEPS = 2  # pitch candidates per semitone
HARMONICS = 6
CONFIDENT = 0.5
ON, OFF = 0.3, 0.15  # level at which a phrase starts and below which it may end
GAP_S = 0.25


def midi_to_hz(midi):
    return 440.0 * 2.0 ** ((np.asarray(midi, dtype=float) - 69.0) / 12.0)


class VocalContour:
    key = "vocals"
    method = METHOD
    delay_frames = 0

    def __init__(self, rate, size, hop_s, sensitivity=0.5):
        self.hop = hop_s
        freq = np.fft.rfftfreq(size, 1 / rate)
        resolution = rate / size
        self.lo = int(np.searchsorted(freq, BAND_HZ[0]))
        self.hi = int(np.searchsorted(freq, min(BAND_HZ[1], rate / 2)))
        self.voice = slice(
            int(np.searchsorted(freq, VOICE_HZ[0])),
            int(np.searchsorted(freq, min(VOICE_HZ[1], rate / 2))),
        )
        if self.hi - self.lo < 32:
            raise ValueError("The analysis window is too short for the vocal contour")
        self.width = max(5, int(round(250.0 / resolution)) | 1)
        self.candidates = LOW_MIDI + np.arange(int((HIGH_MIDI - LOW_MIDI) * STEPS) + 1) / STEPS
        bins = self.hi - self.lo
        # weights[c, b]: how much bin b counts for candidate c: its harmonics, each shared
        # between the two nearest bins, lower harmonics counting more.
        self.weights = np.zeros((len(self.candidates), bins))
        self.member = np.zeros((len(self.candidates), bins), dtype=bool)
        self.reach = np.zeros(len(self.candidates), dtype=int)
        for row, f0 in enumerate(midi_to_hz(self.candidates)):
            for harmonic in range(1, HARMONICS + 1):
                position = harmonic * f0 / resolution - self.lo
                low = int(np.floor(position))
                if low < 0 or low + 1 >= bins:
                    continue
                share = position - low
                gain = 1.0 / harmonic**0.5
                self.weights[row, low] += gain * (1 - share)
                self.weights[row, low + 1] += gain * share
            total = self.weights[row].sum()
            if total > 0:
                self.weights[row] /= total
            # Bins that belong to the candidate at all, and the top of its series: used to
            # ask how much of what is there the candidate explains.
            on = self.weights[row] > 0
            on[1:] |= on[:-1].copy()  # a partial is a few bins wide
            on[:-1] |= on[1:].copy()
            self.member[row] = on
            self.reach[row] = min(bins, int((HARMONICS + 0.5) * f0 / resolution - self.lo))
        self.sensitivity = sensitivity
        self.present = None  # the activity model's judgement for this frame, when there is one
        self.delay_s = self.delay_frames * hop_s
        self.repetition = Repetition(hop_s)
        self.reset()

    def reset(self):
        self.loudness = Level(self.hop)
        self.reference = 0.0
        self.pitch = None
        self.whole = 0.0
        self.eased = None
        self.smooth = 0.0
        self.sounding = False
        self.below = 0.0  # seconds the level has been under OFF
        self.above = 0
        self.repetition.reset()

    def step(self, spectrum, quiet=False, samples=None):
        centre, total, stereo = centre_magnitude(spectrum)
        # Two different takes left and right cancel only on average: ease over a few frames
        # so that their leftovers do not flicker through as centre.
        self.eased = centre if self.eased is None else 0.5 * self.eased + 0.5 * centre
        centre = np.minimum(centre, self.eased) if stereo else centre
        band = centre[self.lo : self.hi]
        mean = float(band.mean())
        self.reference = max(mean, self.reference * float(np.exp(-self.hop / 4.0)))
        scale = threshold_scale(self.sensitivity)
        pitch, clarity = None, 0.0
        # Loud means loud against the whole mix in the voice's range, not only against what
        # the centre itself has recently held: a faint leftover in the middle is not a voice.
        self.whole = max(float(total[self.voice].mean()), self.whole * float(np.exp(-self.hop / 4.0)))
        self.loudness.reference = max(self.loudness.reference, 0.3 * self.whole)
        loud = 0.0 if quiet else self.loudness.step(float(centre[self.voice].mean()))
        if not quiet and self.reference > 1e-12:
            log = np.log(band + 0.05 * self.reference)
            peaks = np.clip(log - median_filter(log, self.width, mode="nearest"), 0.0, 2.0) / 2.0
            scores = self.weights @ peaks
            if self.pitch is not None:  # stay near the contour: within a few semitones
                scores = scores * (
                    0.75 + 0.25 * np.exp(-(((self.candidates - self.pitch) / 4.0) ** 2))
                )
            best = int(np.argmax(scores))
            # Confidence: the share of all peak strength up to the top of the series that sits
            # on the series. One voice explains most of it; a dense mix does not, however
            # well some pitch happens to fit.
            within = float(peaks[: max(1, self.reach[best])].sum())
            clarity = float(peaks[self.member[best]].sum() / within) if within > 1e-6 else 0.0
            clarity = min(1.0, clarity * min(1.0, scores[best] / 0.25))
            if clarity >= CONFIDENT * scale**0.5:
                pitch = float(self.candidates[best])
                self.pitch = pitch if self.pitch is None else 0.5 * self.pitch + 0.5 * pitch
                pitch = self.pitch
        if pitch is None and (quiet or loud < OFF):
            self.pitch = None
        target = loud * (1.0 if pitch is not None else 0.35)
        self.smooth += (target - self.smooth) * (1.0 - float(np.exp(-self.hop / 0.05)))
        level = float(min(1.0, self.smooth))
        boundary, attack = "", 0.0
        # A phrase is sound in the centre *while the activity model reports a voice*. In a
        # full band the centre is never empty, so without the model's judgement a phrase
        # would never end; with no model, the level alone has to do.
        voiced = level if self.present is not False else 0.0
        if not self.sounding:
            self.above = self.above + 1 if voiced >= ON * scale else 0
            if self.above >= 5:  # 100 ms: longer than a drum hit lasts
                self.sounding, self.below, self.above = True, 0.0, 0
                boundary, attack = "start", max(0.3, level)
        else:
            self.below = self.below + self.hop if voiced < OFF * scale else 0.0
            if self.below >= GAP_S:
                self.sounding = False
                boundary = "end"
        flux = float(min(1.0, max(0.0, target - self.smooth) * 3.0))
        self.repetition.step(flux, attack > 0)
        return Activity(
            delay_frames=self.delay_frames,
            level=level if self.sounding or boundary == "end" else level * 0.5,
            flux=flux,
            attack=attack,
            rate_hz=None,
            regularity=0.0,
            centre=0.5
            if pitch is None
            else float(min(1.0, max(0.0, (pitch - LOW_MIDI) / (HIGH_MIDI - LOW_MIDI)))),
            spread=0.0,
            basis="centre" if stereo else "mono",
            pitch_midi=pitch,
            pitch_confidence=clarity,
            boundary=boundary,
        )
