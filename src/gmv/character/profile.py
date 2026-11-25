"""The character profile: seven values between 0 and 1, each with the measurements it was
made from and a confidence.

A dimension is a weighted average of a few measurements, each first placed on a 0..1 scale
between two stated anchors (``DIMENSIONS`` below; a value at or beyond an anchor is 0 or 1).
The anchors are fixed numbers in the measurement's own unit, chosen as round values near
the 5th and 95th percentile of a sample of real recordings (docs/SONIC_CHARACTER.md), so
0 and 1 mean "about as far as real music goes", not a mathematical limit. Nothing adapts
to the recording being heard: the same sound always gets the same value, and two
recordings can be compared.

No dimension uses the level of the recording. They use shares, frequencies, ratios and
differences in decibels, all of which are the same at any playback volume.

These are properties of the sound, not of how it is felt. "Roughness" is a noise-like,
percussive spectrum, not aggression; nothing here measures mood or emotion, and nothing
names a genre.
"""

import math
from typing import Literal

from ..models import FrozenModel
from .measures import INDEX, MEASURES, NAMES

PROFILE_VERSION = "0.1"

# Measurements made from a window as a whole, beside the per-frame ones of ``measures``.
DERIVED = {
    "level_range_db": ("dB", "from the quiet tenth to the loud twentieth of the 0.4 s level"),
    "level_top_db": ("dB", "how far the loud twentieth stands above the middle level"),
    "interval_spread_oct": ("octaves", "spread of the time between onsets, as a ratio"),
}

# dimension -> (what 0 and 1 mean, seconds of music it needs, parts, gate).
# A part is (measurement, value that gives 0, value that gives 1, weight). With a gate the
# weighted average is multiplied with the gate's own 0..1 value and the square root taken:
# both have to be there.
DIMENSIONS = {
    "force": (
        "gentle .. forceful and unrelenting",
        8.0,
        (
            ("crest_db", 10.8, 8.2, 0.45),  # a saturated, compressed waveform
            ("level_top_db", 6.0, 1.0, 0.35),  # the level stays at its top
            ("low_share", 0.25, 0.65, 0.20),  # weight in the low end
        ),
        ("flux", 0.13, 0.21),  # ...and the sound keeps being struck anew: a held pad is not force
    ),
    "brightness": (
        "dark, warm .. bright",
        2.0,
        (("centroid_oct", 8.0, 10.1, 0.6), ("tilt_db_oct", -5.0, -0.8, 0.4)),
        None,
    ),
    "pace": (
        "few onsets .. many onsets a second",
        4.0,
        (("onset_rate", 1.0, 5.0, 1.0),),
        None,
    ),
    "regularity": (
        "loose, irregular .. a tight steady pulse",
        12.0,
        (("pulse_clarity", 0.2, 0.65, 0.6), ("interval_spread_oct", 1.3, 0.7, 0.4)),
        None,
    ),
    "roughness": (
        "smooth, tonal .. noise-like, percussive",
        2.0,
        (("contrast_db", 19.0, 11.0, 0.5), ("attack", 0.02, 0.07, 0.25), ("flux", 0.12, 0.24, 0.25)),
        None,
    ),
    "density": (
        "open, sparse .. crowded, filled",
        4.0,
        (
            ("occupancy", 0.45, 0.97, 0.30),  # how much of the spectrum is filled at once
            ("ripple_db", 11.0, 4.8, 0.20),  # no isolated peaks standing out of it
            ("punch_db", 1.8, 0.4, 0.25),  # single sounds do not stand out of the rest
            ("level_top_db", 8.0, 1.0, 0.25),  # no long stretches far under the top
        ),
        None,
    ),
    "dynamics": (
        "one even level .. very quiet and very powerful passages",
        20.0,
        (("level_range_db", 2.0, 22.0, 1.0),),
        None,
    ),
}
DIMENSION_NAMES = tuple(DIMENSIONS)
QUIET_CAPTURE_DBFS = -50.0  # a middle level under this is close to the silence threshold
INTERVALS_NEEDED = 8.0  # onset intervals for a full say on their spread


class Component(FrozenModel):
    measurement: str
    unit: str
    value: float | None  # the measurement itself, averaged over the window
    low: float  # the value that gives 0
    high: float  # the value that gives 1
    normalized: float | None  # 0..1 between the two
    weight: float


class Dimension(FrozenModel):
    name: str
    meaning: str
    value: float | None  # 0..1, None when nothing it needs has been measured
    confidence: float  # 0..1: how much the value can be relied on, not how strong it is
    components: tuple[Component, ...]
    gate: Component | None = None


class Measurement(FrozenModel):
    name: str
    unit: str
    mean: float | None
    spread: float | None  # standard deviation inside the window
    seconds: float  # music it was measured over, as weighted by the window


class CharacterProfile(FrozenModel):
    profile_version: Literal["0.1"] = PROFILE_VERSION
    window: str
    window_s: float | None  # the window's time constant; None for the whole recording
    time: float | None  # audio time of the newest frame taken in
    segment: int | None
    active: bool  # whether music was sounding at ``time`` (false: held through silence)
    heard_s: float  # seconds of music taken in since the last fresh start
    evidence_s: float  # of which the window still weighs this much
    dimensions: tuple[Dimension, ...]
    measurements: tuple[Measurement, ...]
    tempo_bpm: float | None  # the engine's tempo, averaged on a ratio scale; context only
    warnings: tuple[str, ...] = ()

    def dimension(self, name):
        return next(d for d in self.dimensions if d.name == name)

    def values(self):
        """The dimension values in the order of ``DIMENSION_NAMES`` (None where unknown)."""
        return tuple(d.value for d in self.dimensions)

    def confidences(self):
        return tuple(d.confidence for d in self.dimensions)


def unit(value, low, high):
    """``value`` on the scale that is 0 at ``low`` and 1 at ``high`` (either may be the
    larger), held inside 0..1."""
    return min(1.0, max(0.0, (value - low) / (high - low)))


def known(value):
    value = float(value)
    return value if math.isfinite(value) else None


def build(
    *,
    window,
    window_s,
    time,
    segment,
    active,
    heard_s,
    weight,
    mean,
    spread,
    level_quantiles,
    spectrum_missing=0.0,
):
    """Make a ``CharacterProfile`` of one window's statistics. ``weight``, ``mean`` and
    ``spread`` are per measurement, in the order of ``measures.NAMES``;
    ``level_quantiles`` are the 10%, 50% and 95% levels in dB; ``spectrum_missing`` is the
    share of the music frames that carried no display spectrum."""
    raw = {name: (float(mean[i]), float(weight[i])) for name, i in INDEX.items()}
    low, middle, top = level_quantiles
    seconds = float(weight[INDEX["level_db"]])
    raw["level_range_db"] = (top - low, seconds)
    raw["level_top_db"] = (top - middle, seconds)
    interval = INDEX["interval_oct"]
    onsets = float(mean[INDEX["onset_rate"]])
    counted = onsets * seconds if math.isfinite(onsets) else 0.0  # about how many onsets
    raw["interval_spread_oct"] = (
        float(spread[interval]) if weight[interval] > 0 and counted >= 3 else math.nan,
        seconds,
    )
    quiet = math.isfinite(middle) and middle < QUIET_CAPTURE_DBFS

    def component(part):
        name, zero, one = part[:3]
        value, _ = raw[name]
        unit_name = (MEASURES.get(name) or DERIVED[name])[0]
        present = math.isfinite(value)
        return Component(
            measurement=name,
            unit=unit_name,
            value=value if present else None,
            low=zero,
            high=one,
            normalized=unit(value, zero, one) if present else None,
            weight=part[3] if len(part) > 3 else 1.0,
        )

    dimensions = []
    for name, (meaning, needed, parts, gate) in DIMENSIONS.items():
        components = tuple(component(part) for part in parts)
        gated = component(gate) if gate else None
        have = [c for c in components if c.normalized is not None]
        share = sum(c.weight for c in have)
        value = sum(c.normalized * c.weight for c in have) / share if share > 0 else None
        if gated is not None:
            value = (
                None
                if value is None or gated.normalized is None
                else math.sqrt(value * gated.normalized)
            )
        # Confidence: enough music, of the measurements it needs, and (for the spacing of
        # onsets) enough onsets. A quiet capture halves it: its quiet passages are lost.
        used = [raw[c.measurement][1] for c in have] + ([raw[gate[0]][1]] if gate else [])
        evidence = 1.0 - math.exp(-min(used) / needed) if used and value is not None else 0.0
        confidence = evidence * share / sum(c.weight for c in components)
        if name == "regularity":
            confidence *= 1.0 - math.exp(-counted / INTERVALS_NEEDED)
        if quiet:
            confidence *= 0.5
        dimensions.append(
            Dimension(
                name=name,
                meaning=meaning,
                value=value,
                confidence=min(1.0, max(0.0, confidence)),
                components=components,
                gate=gated,
            )
        )
    measurements = tuple(
        Measurement(
            name=name,
            unit=MEASURES[name][0],
            mean=known(mean[i]),
            spread=known(spread[i]),
            seconds=float(weight[i]),
        )
        for i, name in enumerate(NAMES)
    ) + tuple(
        Measurement(
            name=name, unit=DERIVED[name][0], mean=known(raw[name][0]), spread=None, seconds=seconds
        )
        for name in DERIVED
    )
    warnings = []
    if heard_s < 5.0:
        warnings.append("little_music: under five seconds heard; every value is provisional")
    if not active:
        warnings.append("held: no music at this moment; the profile is held as it was")
    if quiet:
        warnings.append(
            "quiet_capture: the level is close to the silence threshold; quiet passages "
            "may have been taken for silence"
        )
    if spectrum_missing > 0.5:
        warnings.append(
            "no_display_spectrum: these frames carry no display spectrum; brightness and "
            "density rest on fewer measurements"
        )
    if seconds > 10.0 and counted >= 8 and not weight[interval] > 0:
        warnings.append(
            "no_onset_events: onset events were not supplied; regularity rests on the "
            "pulse alone"
        )
    tempo = float(mean[INDEX["tempo_oct"]])
    return CharacterProfile(
        window=window,
        window_s=window_s if math.isfinite(window_s) else None,
        time=time,
        segment=segment,
        active=active,
        heard_s=heard_s,
        evidence_s=seconds,
        dimensions=tuple(dimensions),
        measurements=measurements,
        tempo_bpm=2.0**tempo if math.isfinite(tempo) else None,
        warnings=tuple(warnings),
    )
