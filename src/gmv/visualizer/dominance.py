"""Which instrument is most prominent right now, for the colour of the background glow.

**Why not compare the instruments' levels.** Each analyzer measures its level against its
own recent peak, so a level of 0.8 means "near this instrument's own loudest of the last
seconds", not "louder than the others". Measured on 59 multitrack songs, picking the
instrument with the highest level (times its presence weight) named the truly loudest of
the four stems in about 40% of the clear cases, no better than always answering "bass".

**The common scale.** Every instrument gets a score that estimates the same thing: the
chance that it is the loudest of the four. The score is a logistic function of

* its own sustained activity: its level (bass: its energy), averaged over half a second;
* its own transient activity: its attacks (bass: note pulses), each dying away over half
  a second;
* the share of the whole mix's energy below 250 Hz and between 250 Hz and 4 kHz, from the
  band energies every feature frame already carries, averaged the same way.

The coefficients (``MODELS``) were fitted on those songs against which stem really was the
loudest (by at least 3 dB, over 0.4 s), with every instrument given the same chance
beforehand, so a score is evidence from the signals and not how often an instrument
happens to be loudest in that collection. Fitted on two thirds of the songs and asked
about the other third, the highest score named the loudest stem in 71% of the clear cases
(chance: 25%); bass and vocals most reliably (about 77% and 79% of their cases), drums
and guitar less (about 55% and 45%). It is an estimate from mixed audio, not a measurement
of the stems.

The score is then multiplied by the presence weight (what the recognition models let
through: 1, 0.65, 0.4 tentative, 0) and by a gate that is zero while the instrument itself
reports nothing, so an instrument that is not playing cannot win on the mix's spectrum alone.
Nothing here is a new analysis: it reads rows and frames the visual state already takes in.

**Choosing.** The instrument with the highest score leads once it has stayed at least
``MARGIN`` above the current leader for ``HOLD_S`` (a single hit does not take over, a
real change does within about a third of a second). Below ``FLOOR`` nobody leads. The
colour does not switch: ``mix`` holds how much of each instrument's colour (and of the
neutral one) the glow has, eased towards the leader over ``BLEND_S``.

**Clock.** Everything advances with the timestamps of the frames and rows themselves, in
the visual state, not in the renderer: the same messages give the same colour at any frame
rate, whether or not anything is drawn, and again in replay. A new audio segment starts it
afresh.
"""

import math

KEYS = ("vocals", "guitar", "bass", "drums")
# Per instrument: constant, own sustained activity, own transient activity (capped at 4),
# share of mix energy below 250 Hz, share between 250 Hz and 4 kHz.
MODELS = {
    "vocals": (-1.008, 3.268, 1.033, -4.011, 3.037),
    "guitar": (1.221, -3.209, 0.226, -0.581, 2.875),
    "bass": (-2.149, 3.076, -1.089, 3.606, -3.187),
    "drums": (1.436, -3.669, 0.964, -0.073, -0.306),
}
AVERAGE_S = 0.5  # the listening interval of every input
TRANSIENT_CAP = 4.0
STALE_S = 0.3  # an instrument with no row for this long is taken to have stopped
GATE = (0.03, 0.15)  # own sustained activity: no say .. full say
FLOOR = 0.2  # a score under this leads nothing
MARGIN = 0.08  # how far ahead a challenger must be
HOLD_S = 0.45  # and for how long, before it takes the lead
BLEND_S = 0.35  # easing of the colour mix


def toward(value, target, elapsed, seconds):
    return target + (value - target) * math.exp(-elapsed / seconds)


class Dominance:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sustain = dict.fromkeys(KEYS, 0.0)
        self.transient = dict.fromkeys(KEYS, 0.0)
        self.seen = dict.fromkeys(KEYS)  # time of each instrument's newest row
        self.low = self.mid = None  # averaged band shares of the mix
        self.scores = dict.fromkeys(KEYS, 0.0)
        self.leader = None
        self.challenger, self.challenged = None, 0.0
        self.mix = dict.fromkeys(KEYS, 0.0)  # share of each instrument's colour in the glow
        self.time = None

    def row(self, key, time, drive, hit):
        """One activity row of ``key``: its level or energy, and an attack or pulse strength."""
        if key not in self.sustain:
            return
        last = self.seen[key]
        elapsed = 0.02 if last is None else min(1.0, max(0.0, time - last))
        self.sustain[key] = (
            drive if last is None else toward(self.sustain[key], drive, elapsed, AVERAGE_S)
        )
        self.transient[key] = self.transient[key] * math.exp(-elapsed / AVERAGE_S) + hit
        self.seen[key] = time

    def step(self, time, bands, silent, weights):
        """One feature frame at audio ``time``: the mix's band energies, whether it is
        silent, and each instrument's presence weight. Returns the leader or None."""
        elapsed = 0.02 if self.time is None else min(1.0, max(0.0, time - self.time))
        self.time = time
        total = sum(bands) if bands else 0.0
        if total > 0 and len(bands) >= 5:
            low, mid = (bands[0] + bands[1]) / total, (bands[2] + bands[3] + bands[4]) / total
            if self.low is None:
                self.low, self.mid = low, mid
            else:
                self.low = toward(self.low, low, elapsed, AVERAGE_S)
                self.mid = toward(self.mid, mid, elapsed, AVERAGE_S)
        for key in KEYS:
            seen = self.seen[key]
            if seen is None or time - seen > STALE_S:  # it stopped: let its activity go
                self.sustain[key] = toward(self.sustain[key], 0.0, elapsed, AVERAGE_S)
                self.transient[key] *= math.exp(-elapsed / AVERAGE_S)
            score = 0.0
            if not silent and self.low is not None:
                a, b, c, d, e = MODELS[key]
                z = (
                    a
                    + b * self.sustain[key]
                    + c * min(TRANSIENT_CAP, self.transient[key])
                    + d * self.low
                    + e * self.mid
                )
                gate = min(1.0, max(0.0, (self.sustain[key] - GATE[0]) / (GATE[1] - GATE[0])))
                score = gate * weights.get(key, 0.0) / (1.0 + math.exp(-z))
            self.scores[key] = score
        return self.choose(elapsed)

    def choose(self, elapsed):
        """From ``scores``, ``elapsed`` seconds of audio after the last choice: who leads, and
        the colour mix eased towards that. Returns the leader or None."""
        best = max(KEYS, key=self.scores.__getitem__)
        top = self.scores[best]
        if top < FLOOR:
            best = None
        current = self.scores[self.leader] if self.leader is not None else 0.0
        if best == self.leader:
            self.challenger, self.challenged = None, 0.0
        elif best is None:
            # The leader has fallen under the floor: it keeps the lead for HOLD_S, no longer.
            self.challenged = self.challenged + elapsed if self.challenger is None else elapsed
            self.challenger = None
            if self.challenged >= HOLD_S:
                self.leader, self.challenged = None, 0.0
        elif self.leader is None or top >= current + MARGIN or current < FLOOR:
            self.challenged = self.challenged + elapsed if self.challenger == best else elapsed
            self.challenger = best
            if self.challenged >= HOLD_S:
                self.leader, self.challenger, self.challenged = best, None, 0.0
        else:
            self.challenger, self.challenged = None, 0.0
        for key in KEYS:
            self.mix[key] = toward(self.mix[key], float(key == self.leader), elapsed, BLEND_S)
        return self.leader
