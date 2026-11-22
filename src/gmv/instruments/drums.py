"""Fast drum impacts from the mixed signal, every analysis frame (20 ms).

A drum hit is a sudden gain of energy that covers a whole region of the spectrum at once
and is gone again quickly. Three regions are followed separately:

* ``low``  (40 to 140 Hz), from the centre of the stereo image, where a kick is;
* ``mid``  (180 Hz to 2.5 kHz), from the centre, where a snare's body and toms are;
* ``high`` (6 to 14 kHz), from the whole image, where cymbals and hi-hats are (overhead
  microphones are stereo, so the centre alone would lose them).

They are regions, not kit pieces: a hit in ``mid`` is "something struck in the middle
register", most often a snare, sometimes a tom or a loud consonant. Nothing here names a
kick or a snare.

What keeps other instruments out, as far as it goes:

* centre extraction removes most of what is panned (rhythm guitars, keyboards);
* a hit is measured as level gained against the region's own recent peak, with a floor
  tied to the whole mix, so a region that holds almost nothing cannot hit;
* in ``mid`` and ``high`` a hit must raise the region evenly. A chord or a note raises a
  few partials and is discounted by how concentrated its gain is;
* in ``low`` there are too few bins for that. There a gain counts in full only together
  with a click higher up (1.5 to 5 kHz) in the same frames, which a kick beater makes and a
  bass note mostly does not, and it must be gone again: 60 ms later the region must have
  given back a good part of what it gained. A kick does; a bass note stays. This is why
  every answer is three frames late, and why a soft kick with no click is missed.

It cannot tell a drum from another short broadband sound in the centre, and it is not an
isolated drum signal. Whether drums are playing at all is the activity model's judgement.
"""

from collections import deque

import numpy as np

from . import Activity
from .common import Level, Repetition, centre_magnitude, threshold_scale

METHOD = "drum_impacts_v1"
REGIONS = (("low", 40.0, 140.0), ("mid", 180.0, 2500.0), ("high", 6000.0, 14000.0))
FLAT = (0.8, 0.5)  # share of a gain in the top fifth of bins: not a hit .. fully a hit
GAIN_SCALE = {"low": 0.25, "mid": 0.16, "high": 0.3}  # gain (share of the region) -> 0.63
FLOOR = {"low": 0.04, "mid": 0.08, "high": 0.025, "click": 0.04}  # of the whole centre
CLICK_HZ = (1500.0, 5000.0)
CLICK = 0.12  # click gain that makes a low hit count in full
LOW_ALONE = 0.2  # how much a low gain counts without a click above it
GIVEN_BACK = 0.35  # share of a low gain that must be gone again after CONFIRM frames
CONFIRM = 3
THRESHOLD = 0.28
MIN_GAP_S = 0.055
PLACE = {"low": 0.1, "mid": 0.5, "high": 0.9}


class DrumImpacts:
    key = "drums"
    method = METHOD
    delay_frames = CONFIRM  # a low hit is confirmed by its decay

    def __init__(self, rate, size, hop_s, sensitivity=0.5):
        self.hop = hop_s
        freq = np.fft.rfftfreq(size, 1 / rate)
        top = rate / 2

        def band(low, high):
            return slice(
                int(np.searchsorted(freq, low)), int(np.searchsorted(freq, min(high, top)))
            )

        self.bands = {name: band(low, high) for name, low, high in REGIONS}
        self.everything = band(40.0, 14000.0)
        self.click = band(*CLICK_HZ)
        if any(b.stop - b.start < 2 for b in self.bands.values()):
            raise ValueError("The analysis window is too short for drum impacts")
        self.fade = float(np.exp(-hop_s / 4.0))
        self.fade_region = float(np.exp(-hop_s / 2.0))
        self.sensitivity = sensitivity
        self.delay_s = self.delay_frames * hop_s
        self.repetition = Repetition(hop_s)
        self.reset()

    def reset(self):
        self.previous = {}  # region -> (magnitudes, their sum, the sum one frame earlier)
        self.loud = 0.0  # recent peak of the whole centre's summed magnitude
        self.peak = {}
        self.levels = {name: Level(self.hop, 2.0) for name in self.bands}
        size = CONFIRM + 2
        self.strength = {name: deque([0.0] * size, maxlen=size) for name in self.bands}
        self.low_power = deque([0.0] * (CONFIRM + 3), maxlen=CONFIRM + 3)
        self.loudness = deque([0.0] * (CONFIRM + 1), maxlen=CONFIRM + 1)
        self.since = {name: 1e6 for name in self.bands}
        self.count = 0
        self.repetition.reset()
        self.basis = "centre"

    def _gain(self, name, magnitude):
        """What a region gained since the two frames before, as a share of its recent
        peak (never less than a small share of the whole mix), and how evenly the gain is
        spread over the region's bins: 0 concentrated in a few partials .. 1 flat."""
        total = float(magnitude.sum())
        before = self.previous.get(name)
        self.previous[name] = (magnitude, total, before[1] if before else total)
        self.peak[name] = max(total, self.peak.get(name, 0.0) * self.fade_region)
        reference = max(self.peak[name], FLOOR[name] * self.loud)
        if before is None or reference <= 1e-20:
            return 0.0, 0.0, total
        gain = max(0.0, total - max(before[1], before[2])) / reference
        rise = np.maximum(magnitude - before[0], 0.0)
        whole = float(rise.sum())
        if whole <= 1e-20:
            return 0.0, 0.0, total
        fifth = max(1, len(rise) // 5)
        share = float(np.partition(rise, -fifth)[-fifth:].sum() / whole)
        return gain, min(1.0, max(0.0, (FLAT[0] - share) / (FLAT[0] - FLAT[1]))), total

    def step(self, spectrum, quiet=False, samples=None):
        centre, total, stereo = centre_magnitude(spectrum)
        self.basis = "centre" if stereo else "mono"
        scale = threshold_scale(self.sensitivity)
        self.loud = max(float(centre[self.everything].sum()), self.loud * self.fade)
        loud = 0.0
        click, click_flat, _ = self._gain("click", centre[self.click])
        beaten = LOW_ALONE + (1 - LOW_ALONE) * min(1.0, click * click_flat / CLICK)
        for name, band in self.bands.items():
            source = total if name == "high" else centre
            gain, flat, amount = self._gain(name, source[band])
            loud = max(loud, self.levels[name].step(amount))
            value = 0.0 if quiet else gain * (beaten if name == "low" else flat)
            self.strength[name].append(float(1.0 - np.exp(-value / (GAIN_SCALE[name] * scale))))
            if name == "low":
                self.low_power.append(amount)
        self.loudness.append(0.0 if quiet else loud)
        self.count += 1
        if self.count < CONFIRM + 3:
            return None
        # The frame decided now is CONFIRM frames back: its neighbours on both sides are
        # known, and so is what the low region did afterwards.
        hits, flux = {}, 0.0
        for name, history in self.strength.items():
            before, peak, after = history[0], history[1], history[2]
            self.since[name] += self.hop
            flux = max(flux, peak)
            if not (
                peak > before
                and peak >= after
                and peak >= THRESHOLD * scale
                and self.since[name] >= MIN_GAP_S
            ):
                continue
            if name == "low":
                # low_power: [1] is the frame before the hit, [2] the hit, [-1] now.
                base = min(self.low_power[0], self.low_power[1])
                top = max(self.low_power[2], self.low_power[3])
                if top - self.low_power[-1] < GIVEN_BACK * (top - base):
                    continue  # still there 60 ms on: a note, not a hit
            hits[name] = peak
            self.since[name] = 0.0
        attack = max(hits.values(), default=0.0)
        self.repetition.step(flux, attack > 0)
        return Activity(
            delay_frames=self.delay_frames,
            level=self.loudness[0],
            flux=flux,
            attack=attack,
            rate_hz=self.repetition.rate_hz,
            regularity=self.repetition.regularity,
            centre=sum(PLACE[n] * s for n, s in hits.items()) / sum(hits.values())
            if hits
            else 0.5,
            spread=len(hits) / 3,
            basis=self.basis,
            components=tuple((name, hits.get(name, 0.0)) for name in self.bands),
        )
