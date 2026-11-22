"""How intense the music is as a whole, 0 (quiet passage) to 1 (very energetic).

One number for the whole mix, for colouring the main spectrum. It is computed here, in the
visualizer, from fields every ``FeatureFrame`` already carries, so it needs no second
analysis pass, works the same live and in replay, and does not depend on any instrument
being recognised or shown.

**What is measured.** Three things, blended:

* *Level* (65%): the loudness of the mix in decibels, averaged over about 0.4 s so that
  a single snare or kick barely moves it, placed between a quiet reference and a
  loud reference that adapt to what has been heard. The loud reference follows the loudest
  recent passage (it rises within a second or two and falls over about a minute); the
  quiet reference follows the quietest (falling within a few seconds, rising over minutes).
  For the first four seconds of music both simply follow the level, so a fade-in is not
  taken for a quiet passage.
  The scale runs from 6 dB below the quiet reference to the loud reference, and is never
  taken as narrower than 18 dB or wider than 30 dB. So music that has been evenly loud
  throughout sits a third of the way up, not at the top: without contrast there is no
  telling from level alone whether it is a loud passage or just a loud recording. The
  playback volume cancels out.
* *Weight* (25%): the share of the energy below 250 Hz. A full band with bass and kick drum
  has most of its energy there; a voice, a lone guitar or a quiet break does not. It does
  not depend on volume, so it is what tells a soft opening from a heavy one before anything
  else has been heard.
* *Activity* (10%): onsets per second.

**Smoothing.** The blend is eased with separate time constants, 0.7 s when rising and
2.5 s when falling, applied by elapsed audio time, so the result does not depend on the
frame rate, follows a build promptly and lets go gradually. Silence eases it to zero.
"""

import math

LEVEL_TAU_S = 0.4  # averaging of the level
ATTACK_S, RELEASE_S = 0.7, 2.5  # easing of the result
SILENCE_S = 1.2  # easing to zero in silence
LOUD_RISE_S, LOUD_FALL_S = 1.5, 60.0  # the loud reference
QUIET_FALL_S, QUIET_RISE_S = 4.0, 150.0  # the quiet reference
SETTLE_S, SETTLE_TAU_S = 4.0, 0.5  # how long, and how fast, both references follow at first
BELOW_QUIET_DB = 6.0  # the scale starts this far under the quiet reference
SPAN_DB = (18.0, 30.0)  # least and most the scale may cover
WEIGHTS = (0.65, 0.25, 0.10)  # level, weight, activity
LOW_SHARE = (0.1, 0.6)  # share of energy below 250 Hz: none .. full
LOW_BANDS = 2  # the bands below 250 Hz
ONSETS_PER_S = (1.0, 6.0)
SILENT_DBFS = -66.0
TINT = (0.3, 0.9)  # intensity where colouring begins, and where it is complete


def toward(value, target, elapsed, seconds):
    """``value`` moved towards ``target`` as an exponential with time constant ``seconds``."""
    return target + (value - target) * math.exp(-elapsed / seconds)


def unit(value, low, high):
    return min(1.0, max(0.0, (value - low) / (high - low)))


def number(value, default=0.0):
    return float(value) if isinstance(value, (int, float)) and math.isfinite(value) else default


def tint(intensity):
    """How far the spectrum is coloured, 0..1, for an intensity: nothing at all in quiet
    passages, fully only in the most intense ones, eased at both ends."""
    share = unit(intensity, *TINT)
    return share * share * (3.0 - 2.0 * share)


class MusicalIntensity:
    """Feed it every feature frame in order; read ``value``."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0.0  # the eased result, 0..1
        self.raw = 0.0  # before easing
        self.level = None  # short-term level of the mix, dB
        self.loud = self.quiet = None  # reference levels, dB
        self.time = None
        self.heard = 0.0  # seconds of music so far

    def step(self, frame):
        """Take one ``FeatureFrame``; returns the eased intensity."""
        time = number(frame.timestamp)
        elapsed = 0.02 if self.time is None else time - self.time
        if not 0.0 < elapsed < 1.0:  # the timeline restarted or jumped
            if elapsed < 0.0 or elapsed > 10.0:
                self.reset()
            elapsed = 0.02
        self.time = time
        level = number(frame.loudness_dbfs, -120.0)
        if frame.silence or level < SILENT_DBFS:
            self.raw = 0.0
            self.value = toward(self.value, 0.0, elapsed, SILENCE_S)
            return self.value
        # Averaged in decibels, not as energy: one hit 20 dB up for a few hundredths of a
        # second then counts for its duration, not for a hundred times it.
        short = self.level = (
            level if self.level is None else toward(self.level, level, elapsed, LEVEL_TAU_S)
        )
        if self.loud is None:
            self.loud = self.quiet = short
        else:
            # A fade-in is not a quiet passage: for the first seconds both simply follow.
            settling = self.heard < SETTLE_S
            up, down = (SETTLE_TAU_S,) * 2 if settling else (LOUD_RISE_S, LOUD_FALL_S)
            self.loud = toward(self.loud, short, elapsed, up if short > self.loud else down)
            down, up = (SETTLE_TAU_S,) * 2 if settling else (QUIET_FALL_S, QUIET_RISE_S)
            self.quiet = toward(self.quiet, short, elapsed, down if short < self.quiet else up)
        self.heard += elapsed
        floor = self.quiet - BELOW_QUIET_DB
        top = max(self.loud, floor + SPAN_DB[0])
        floor = max(floor, top - SPAN_DB[1])
        bands = frame.band_energy or ()
        total = sum(bands)
        weight = unit(sum(bands[:LOW_BANDS]) / total, *LOW_SHARE) if total > 0 else 0.0
        self.raw = (
            WEIGHTS[0] * unit(short, floor, top)
            + WEIGHTS[1] * weight
            + WEIGHTS[2] * unit(number(frame.rhythmic_density), *ONSETS_PER_S)
        )
        rising = self.raw > self.value
        self.value = toward(self.value, self.raw, elapsed, ATTACK_S if rising else RELEASE_S)
        return self.value
