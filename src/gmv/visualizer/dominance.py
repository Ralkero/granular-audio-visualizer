"""Which instrument is most prominent right now, for the colour of the background glow.

**The common scale.** Each analyzer measures its level against its own recent peak, so the
instruments' levels cannot be compared with one another. Here every instrument gets a
score that estimates the same thing: the chance that it is the loudest of the four. The
score is a logistic function of

* its own sustained activity: its level (bass: its energy), averaged over half a second;
* its own transient activity: its attacks (bass: note pulses), each dying away over half
  a second;
* the share of the whole mix's energy below 250 Hz and between 250 Hz and 4 kHz, from the
  band energies every feature frame already carries (``Bands``), averaged the same way.

The coefficients (``MODELS``) were fitted against which stem really is the loudest in
multitrack songs, every instrument given the same chance beforehand. "Loudest stem" is a
measurable stand-in, not what a listener would necessarily call the lead. How they were
fitted, on which songs, and how well the result does on songs it had not seen (and how
poorly for guitar) is in docs/VISUAL_PRESENTATION.md, the one place for those figures.

The score is multiplied by the presence weight (what the recognition models let through:
1, 0.65, 0.4 tentative, 0) and by a gate that is zero while the instrument itself reports
nothing, so an instrument that is not playing cannot win on the mix's spectrum alone.
Nothing here is a new analysis: it reads rows and frames the visual state already takes in.

**Choosing.** The instrument with the highest score leads once it has stayed at least
``MARGIN`` above the current leader for ``HOLD_S`` (a single hit does not take over, a
real change does within about half a second). Below ``FLOOR`` nobody leads. The colour
does not switch: ``mix`` holds how much of each instrument's colour (and, for the rest, of
the neutral one) the glow has, eased towards the leader over ``BLEND_S``.

**Clock.** Everything advances with the timestamps of the frames and rows themselves, in
the visual state, not in the renderer: the same messages give the same colour at any frame
rate, whether or not anything is drawn, and again in replay. A new audio segment starts it
afresh.
"""

import math

from ..models import AnalysisConfig

KEYS = ("vocals", "guitar", "bass", "drums")
DEFAULT_EDGES = AnalysisConfig().band_edges  # the edges the coefficients were fitted with
LOW_HZ, MID_HZ = 250.0, 4000.0  # the two cutoffs the mix shares are about
COVER_HZ = (45.0, 8000.0)  # the edges must reach at least this far down and up
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


class Bands:
    """The two mix shares the scores need (energy below 250 Hz, and from 250 Hz to 4 kHz, as
    shares of all band energy) from a frame's ``band_energy``, whatever the band edges are.

    ``band_energy`` holds the energy between consecutive ``edges``; the edges are a setting.
    A band wholly on one side of a cutoff counts there in full, which is exact. A band that
    lies across 250 Hz or 4 kHz is divided by how that band's power is spread in the display
    spectrum of the same frame (48 narrower bands of the same power, same units), each
    display band counted by how much of it, in octaves, lies on each side. Without a display
    spectrum such a band cannot be divided, and no shares are given.

    No shares are given either (``usable`` is False) when the edges are unknown, or do not
    reach down to 45 Hz and up to 8 kHz: the shares would then not be shares of the same
    whole the scores were fitted on. With no shares nobody leads and the glow stays neutral.
    """

    def __init__(self, edges):
        self.edges = tuple(float(edge) for edge in edges) if edges else None
        edges = self.edges
        self.usable = bool(
            edges
            and len(edges) >= 3
            and all(a < b for a, b in zip(edges, edges[1:], strict=False))
            and edges[0] <= COVER_HZ[0]
            and edges[-1] >= COVER_HZ[1]
        )
        self.whole, self.split = [], []  # (band, part of low, part of mid) / bands to divide
        if self.usable:
            for index, (a, b) in enumerate(zip(edges, edges[1:], strict=False)):
                if b <= LOW_HZ + 0.5:
                    self.whole.append((index, 1.0, 0.0))
                elif a >= LOW_HZ - 0.5 and b <= MID_HZ + 0.5:
                    self.whole.append((index, 0.0, 1.0))
                elif a >= MID_HZ - 0.5:
                    self.whole.append((index, 0.0, 0.0))
                else:
                    self.split.append((index, a, b))
        self.exact = self.usable and not self.split
        self._overlaps = {}  # (count, low, high) -> per split band: (low, mid, all) weights

    def _weights(self, count, low, high):
        key = (count, low, high)
        if key not in self._overlaps:
            marks = [low * (high / low) ** (i / count) for i in range(count + 1)]

            def inside(a, b):  # how much of each display band, in octaves, lies in [a, b]
                return [
                    max(0.0, math.log2(min(b, d) / max(a, c))) / math.log2(d / c)
                    if min(b, d) > max(a, c)
                    else 0.0
                    for c, d in zip(marks, marks[1:], strict=False)
                ]

            self._overlaps = {
                key: [
                    (inside(a, min(b, LOW_HZ)), inside(max(a, LOW_HZ), min(b, MID_HZ)), inside(a, b))
                    for _, a, b in self.split
                ]
            }
        return self._overlaps[key]

    def shares(self, bands, spectrum_db=(), spectrum_range=None):
        """(low share, mid share) of the frame, or None when they cannot be had."""
        if not self.usable or not bands or len(bands) != len(self.edges) - 1:
            return None
        total = sum(bands)
        if total <= 0:
            return None
        low = sum(bands[index] * part for index, part, _ in self.whole)
        mid = sum(bands[index] * part for index, _, part in self.whole)
        if self.split:
            if not spectrum_db or not spectrum_range or not spectrum_range[0] or not spectrum_range[1]:
                return None
            power = [10.0 ** (value / 10.0) for value in spectrum_db]
            weights = self._weights(len(power), float(spectrum_range[0]), float(spectrum_range[1]))
            for (index, _, _), (below, between, whole) in zip(self.split, weights, strict=True):
                inside = sum(p * w for p, w in zip(power, whole, strict=True))
                if inside <= 0:
                    return None  # the display spectrum does not reach this band
                low += bands[index] * sum(p * w for p, w in zip(power, below, strict=True)) / inside
                mid += bands[index] * sum(p * w for p, w in zip(power, between, strict=True)) / inside
        return low / total, mid / total


class Dominance:
    def __init__(self, edges=DEFAULT_EDGES):
        self.bands = Bands(edges)
        self.reset()

    def configure(self, edges):
        """The band edges ``band_energy`` is measured between (None: unknown). Starts
        afresh: averages made with other edges mean something else."""
        self.bands = Bands(edges)
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

    def step(self, time, bands, silent, weights, spectrum_db=(), spectrum_range=None):
        """One feature frame at audio ``time``: the mix's band energies (and its display
        spectrum, needed only when a band lies across a cutoff), whether it is silent,
        and each instrument's presence weight. Returns the leader or None."""
        elapsed = 0.02 if self.time is None else min(1.0, max(0.0, time - self.time))
        self.time = time
        shares = self.bands.shares(bands, spectrum_db, spectrum_range)
        if shares is None:
            # Unknown or unsuitable band edges, or a band that cannot be divided: there is
            # nothing to score with, so nobody leads (checked below through ``self.low``).
            self.low = self.mid = None
        elif self.low is None:
            self.low, self.mid = shares
        else:
            self.low = toward(self.low, shares[0], elapsed, AVERAGE_S)
            self.mid = toward(self.mid, shares[1], elapsed, AVERAGE_S)
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
