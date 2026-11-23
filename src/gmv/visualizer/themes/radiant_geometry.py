"""Where everything in the Radiant theme is, and what moves it. Numbers only: no Qt and
no colours, so the same geometry can be drawn in any palette, by any renderer.

Angles are clockwise from the top (x = cx + r sin a, y = cy - r cos a), as in the
instrument layers. Everything is a function of the visual state and one audio time
(``now``): nothing here remembers a frame, so the same recording gives the same shapes at
any frame rate and in replay.

Where each structure gets its movement from:

* core, halo, rays: the level and musical intensity of the newest analysis frames, the
  onset markers (``flash``) and the predicted-beat markers (``pulse``);
* spikes: the display spectrum (the same values Spectrum Bars draws), lows at the top,
  highs at the bottom, mirrored left and right;
* contour: the envelope of the last ``CONTOUR_S`` seconds of analysis frames
  (level and onset strength), newest at the top, flowing down both sides;
* segments: the display spectrum averaged over a few regions, plus ``flash``; a predicted
  beat pushes the ring out a little;
* streaks: onset markers of the whole mix and the attacks, pulses and hits the
  instruments' live states hold. No event, no streak.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

SUB = 3  # spikes drawn per spectrum band
# Height of the spikes between two bands, as a share of the straight line between them:
# the fine teeth of the picture. Never above one, so no spike is longer than the line.
PATTERN = (1.0, 0.55, 0.78)
INWARD = 0.2  # how far a spike also reaches inwards, as a share of its length
CONTOUR_S = 3.0  # seconds of envelope round each half of the contour
SEGMENTS = 40
SEGMENT_FILL = 0.7  # share of its slot a segment covers; the rest is the gap
FLASH_S, PULSE_S = 0.14, 0.22  # how fast an onset's flash and a beat's pulse die away
ONSET_LIFE, ATTACK_LIFE = 0.45, 0.36  # seconds a streak lasts
SHOCK_LIFE = 0.6  # seconds an onset's widening circle lasts
MOST_STREAKS = 64
BEAT_LIFE = 2.6  # seconds a beat goes on disturbing the smoke
MOST_BEATS = 16  # beats that can be disturbing it at once
BEAT_LEAST = 0.45  # strength of a beat the tracker was least sure of
PARTICLES = 150


@dataclass(frozen=True)
class Layout:
    """Radii of the structures round (``cx``, ``cy``), in pixels, inside ``reach``."""

    cx: float
    cy: float
    reach: float
    zone: float  # radius of the inner composition (core, spectrum, contour, segments)
    core: float  # radius of the core at rest
    core_most: float  # and the largest it may get
    base: float  # baseline circle of the spikes
    thickness: float  # of the baseline
    tips: float  # how far out a full spike reaches
    contour: float  # middle of the contour
    swing: float  # how far the contour may go either side
    segments: float  # middle of the segmented ring
    depth: float  # its thickness
    frame: float  # the primary ring between the inner composition and the instruments
    rim: float  # the outer primary ring
    bands: tuple  # (inner radius, outer radius) per instrument, the first outermost


def layout(cx, cy, reach, weights=(), gap=0.0):
    """The composition in a circle of radius ``reach``. ``weights``: the relative room of
    each instrument layer to draw, outermost first; none gives the inner composition the
    whole circle."""
    reach = max(40.0, float(reach))
    zone = reach * (0.56 if weights else 0.9)
    rim = reach * 0.985
    bands = []
    if weights:
        inner, outer = zone + 2.0 * gap + 0.012 * reach, reach * 0.955
        unit = (outer - inner - gap * (len(weights) - 1)) / sum(weights)
        edge = outer
        for weight in weights:
            bands.append((edge - unit * weight, edge))
            edge -= unit * weight + gap
    return Layout(
        cx=cx,
        cy=cy,
        reach=reach,
        zone=zone,
        core=zone * 0.15,
        core_most=zone * 0.21,
        base=zone * 0.33,
        thickness=max(1.5, zone * 0.012),
        tips=zone * 0.73,
        contour=zone * 0.815,
        swing=zone * 0.065,
        segments=zone * 0.915,
        depth=max(2.0, zone * 0.032),
        frame=zone,
        rim=rim,
        bands=tuple(bands),
    )


def xy(lay, angles, radii):
    """Pixel coordinates of points at ``radii`` and ``angles`` round the layout's centre."""
    return lay.cx + radii * np.sin(angles), lay.cy - radii * np.cos(angles)


# -- spikes -------------------------------------------------------------------------------


def spike_lengths(values, sub=SUB):
    """``sub`` spike lengths (0..1) per band, in band order. The first of each band is the
    band's own value; the others lie on the straight line to the next band, shortened by
    ``PATTERN``. Nothing is invented: every spike is at or below that line."""
    values = np.clip(np.nan_to_num(np.asarray(values, dtype=float)), 0.0, 1.0)
    count = len(values)
    if not count:
        return np.zeros(0)
    places = np.arange(count * sub) / sub
    return np.interp(places, np.arange(count), values) * np.tile(PATTERN[:sub], count)


@lru_cache(maxsize=8)
def half_angles(count):
    """Angles of ``count`` things down the right half, top to bottom."""
    angles = np.pi * (np.arange(count) + 0.5) / count
    angles.flags.writeable = False
    return angles


def mirrored(right):
    """Once round the circle from values down the right half: the left half mirrors it."""
    return np.concatenate((right, right[::-1]))


def round_angles(count):
    right = half_angles(count)
    return np.concatenate((right, 2 * np.pi - right[::-1]))


def spikes(lay, values, curve=1.0, sub=SUB):
    """The spike ring: (angles, outer radii, inner radii), one spike per entry, once
    round. A spike runs from its inner radius (a little inside the baseline) to its outer
    one. ``values`` None or empty: bare stubs on the baseline. ``curve`` is the response's
    exponent on the heights."""
    if values is None or not len(values):
        values = np.zeros(48)
    values = np.clip(np.nan_to_num(np.asarray(values, dtype=float)), 0.0, 1.0)
    lengths = mirrored(spike_lengths(values**curve, sub))
    reach = lay.tips - lay.base
    outer = lay.base + np.maximum(1.0, lengths * reach)
    inner = lay.base - lay.thickness - lengths * reach * INWARD
    return round_angles(len(lengths) // 2), outer, inner


def peak_marks(lay, peaks, curve=1.0):
    """(angles, radii) of the peak marks, one per band and side, where a peak is held."""
    if peaks is None or not len(peaks):
        return np.zeros(0), np.zeros(0)
    peaks = np.clip(np.asarray(peaks, dtype=float), 0.0, 1.0) ** curve
    count = len(peaks)
    # A band's own spike is the first of its ``SUB``.
    right = half_angles(count * SUB)[::SUB]
    angles = np.concatenate((right, 2 * np.pi - right[::-1]))
    radii = lay.base + mirrored(peaks) * (lay.tips - lay.base)
    keep = mirrored(peaks) > 0.04
    return angles[keep], radii[keep] + 2.0


# -- whole-mix signals ----------------------------------------------------------------------


def mix_levels(frames, now):
    """(level, intensity) of the whole mix at audio time ``now``, each 0..1: the mean of
    the last tenth of a second and of the last three tenths of the analysis frames at or
    before ``now``. ``frames`` is ``VisualState.frames``."""
    if now is None or frames is None or not len(frames):
        return 0.0, 0.0
    rows = frames.since(now - 0.3)
    rows = rows[rows[:, 0] <= now]
    if not len(rows):
        return 0.0, 0.0
    recent = rows[rows[:, 0] >= now - 0.1]
    level = float((recent if len(recent) else rows[-1:])[:, 1].mean())
    return min(1.0, max(0.0, level)), min(1.0, max(0.0, float(rows[:, 2].mean())))


def impulses(markers, now, since=None, beats=True):
    """Onsets and predicted beats of the whole mix that audio time ``now`` has reached:
    ``(flash, pulse, onsets, beats)``. ``flash`` and ``pulse`` are 0..1 envelopes that jump
    at an event and die away; the lists hold ``(age, strength, time)``, newest first.
    Marks before ``since`` (the start of the audio segment) are over and done with, and
    delayed context marks never flash. Only what the analysis reported is here."""
    flash = pulse = 0.0
    onsets, predicted = [], []
    if now is None:
        return flash, pulse, onsets, predicted
    for scanned, mark in enumerate(reversed(markers)):
        age = now - mark.time
        if scanned > 400 or age > 4.0:
            break
        if age < 0 or age > 1.0 or mark.style == "context":
            continue
        if since is not None and mark.time < since:
            continue
        if mark.kind == "onset":
            strength = 0.3 + 0.7 * min(1.0, max(0.0, mark.intensity))
            flash = max(flash, strength * math.exp(-age / FLASH_S))
            onsets.append((age, strength, mark.time))
        elif mark.style == "predicted" and beats:
            pulse = max(pulse, math.exp(-age / PULSE_S))
            predicted.append((age, 1.0, mark.time))
    return flash, pulse, onsets, predicted


def beats(markers, now, since=None, shown=True, life=BEAT_LIFE, most=MOST_BEATS):
    """The predicted beats that audio time ``now`` has reached within the last ``life``
    seconds: ``(age, strength, time)``, newest first, at most ``most``, each beat once.

    A beat here is exactly what the beat tracker published as ``beat_prediction`` (drawn
    as "predicted"), and only once its time has come. Nothing else is a beat: not an
    onset, not a drum hit, not a transient. With the tracker off, or not locked, it
    publishes none and this is empty. ``shown`` is the "Show beats" setting, which
    switches every beat-driven part of the picture, as it always has.

    ``strength`` is the tracker's own confidence for that beat (which the tracker caps at
    0.7), scaled so that a beat it was sure of is 1 and the least sure is ``BEAT_LEAST``:
    a predicted beat always does something. Marks before ``since`` (the start of the
    audio segment) are over."""
    found, seen = [], set()
    if now is None or not shown:
        return found
    for scanned, mark in enumerate(reversed(markers)):
        age = now - mark.time
        if scanned > 600 or age > life + 4.0:
            break
        if age < 0 or age > life or mark.kind != "beat_prediction" or mark.style != "predicted":
            continue
        if since is not None and mark.time < since:
            continue
        key = round(mark.time * 1000)
        if key in seen:
            continue
        seen.add(key)
        sure = min(1.0, max(0.0, mark.intensity) / 0.7)
        found.append((age, BEAT_LEAST + (1.0 - BEAT_LEAST) * sure, mark.time))
    found.sort(key=lambda beat: beat[0])
    return found[:most]


def core(lay, level, intensity, flash, pulse, breath=1.0):
    """(core radius, halo radius, ray reach) for the music now. The core swells with the
    level, an onset and a beat, and never reaches the spikes; the rays are light and go
    out over the rings, further in intense music and at an onset."""
    radius = lay.core + lay.zone * breath * (0.035 * level + 0.03 * flash + 0.02 * pulse)
    halo = lay.zone * (0.34 + 0.24 * intensity + min(0.2, 0.1 * breath * flash))
    rays = lay.zone * min(1.3, 0.72 + 0.34 * intensity + 0.2 * breath * flash)
    return min(lay.core_most, radius), halo, rays


# -- contour ----------------------------------------------------------------------------------


def contour(lay, frames, now, motion=1.0, seconds=CONTOUR_S):
    """The secondary ring: (angles, radii) once round. Each analysis frame of the last
    ``seconds`` is a point, the newest at the top and older ones further down both sides;
    how far it stands off the circle is that frame's envelope (0.35 level + 0.65 onset
    strength), to one side or the other by turns, which is what makes it look like a
    waveform. It is the envelope the analysis published, not the audio waveform: the
    visualizer never has audio samples. Without frames it is a plain circle."""
    theta, radii = np.zeros(0), np.zeros(0)
    if now is not None and frames is not None and len(frames):
        rows = frames.since(now - seconds)
        rows = rows[rows[:, 0] <= now][::-1]  # newest first: going down from the top
        if len(rows):
            times = rows[:, 0]
            theta = np.pi * (now - times) / seconds
            envelope = np.clip(0.35 * rows[:, 1] + 0.65 * rows[:, 3], 0.0, 1.0)
            side = np.where(np.round(times * 50.0) % 2 == 0, 1.0, -1.0)
            radii = lay.contour + np.clip(
                lay.swing * motion * envelope * side, -lay.swing, lay.swing
            )
    theta = np.concatenate(([0.0], theta, [np.pi]))
    radii = np.concatenate(([lay.contour], radii, [lay.contour]))
    return np.concatenate((theta, 2 * np.pi - theta[::-1])), mirrored(radii)


# -- segments ---------------------------------------------------------------------------------


def segment_levels(values, count=SEGMENTS):
    """How lit each of ``count`` segments is (0..1), once round: the mean of the display
    spectrum over the stretch of it the segment stands for, lows at the top."""
    half = count // 2
    if values is None or not len(values):
        return np.zeros(2 * half)
    values = np.clip(np.asarray(values, dtype=float), 0.0, 1.0)
    edges = np.linspace(0, len(values), half + 1).round().astype(int)
    right = np.array(
        [values[a : max(b, a + 1)].mean() for a, b in zip(edges[:-1], edges[1:], strict=True)]
    )
    return mirrored(right)


@lru_cache(maxsize=8)
def segment_arcs(count=SEGMENTS, fill=SEGMENT_FILL):
    """Where each segment starts and ends: ``count`` rows of (angle, angle)."""
    half = count // 2
    centres = np.concatenate((half_angles(half), 2 * np.pi - half_angles(half)[::-1]))
    span = np.pi / half * fill
    arcs = np.column_stack((centres - span / 2, centres + span / 2))
    arcs.flags.writeable = False
    return arcs


def segment_ring(lay, pulse):
    """(inner radius, outer radius) of the segmented ring; a predicted beat pushes it out."""
    middle = lay.segments + lay.zone * 0.014 * min(1.0, pulse)
    return middle - lay.depth / 2, middle + lay.depth / 2


# -- streaks ----------------------------------------------------------------------------------


@lru_cache(maxsize=512)
def needles(seed, count):
    """Where the ``count`` streaks of the event with ``seed`` are: rows of (turn 0..1,
    length share, width share). The same every frame for the same event."""
    spots = np.random.default_rng(seed).random((count, 3))
    spots[:, 1] = 0.55 + 0.45 * spots[:, 1]
    spots[:, 2] = 0.6 + 0.4 * spots[:, 2]
    spots.flags.writeable = False
    return spots


def seed_of(time, kind):
    """A stable number for the event of ``kind`` at audio ``time`` (milliseconds count)."""
    return (int(round(time * 1000.0)) * 31 + kind * 7919) & 0xFFFFFFFF


def streaks(lay, onsets, attacks, impact=1.0):
    """Radial streaks for events that happened: rows of
    ``(angle, inner radius, outer radius, width, strength 0..1, source)``.

    ``onsets``: ``(age, strength, time)`` of whole-mix onsets; they shoot from the spike
    tips towards the rim (``source`` None). ``attacks``: instrument -> (band index,
    ``[(age, strength, time)]``); they cross the instrument's own band (``source`` is the
    instrument). A stronger event gives more, longer and wider streaks; each fades and
    moves outwards with its age. Bounded, and nothing is random between frames."""
    found = []
    for age, strength, time in onsets:
        progress = age / ONSET_LIFE
        if progress >= 1.0:
            continue
        fade = (1.0 - progress) ** 1.5
        spots = needles(seed_of(time, 0), 2 + round(4 * strength))
        low, high = lay.tips, lay.rim
        for turn, length, width in spots:
            start = low + (high - low) * 0.35 * progress
            end = min(high, start + (high - low) * length * min(1.0, 0.45 + 0.5 * strength * impact))
            found.append((2 * math.pi * turn, start, end, width * (1.0 + 2.0 * strength), fade * strength, None))
    for kind, (key, (band, events)) in enumerate(attacks.items(), start=1):
        if band >= len(lay.bands):
            continue
        low, high = lay.bands[band]
        height = high - low
        for age, strength, time in events:
            progress = age / ATTACK_LIFE
            if progress >= 1.0 or strength <= 0.05:
                continue
            strength = min(1.0, strength)
            fade = (1.0 - progress) ** 1.5
            spots = needles(seed_of(time, kind), 1 + (strength > 0.55) + (strength > 0.8))
            for turn, length, width in spots:
                start = max(lay.frame, low - 0.25 * height) + 0.3 * height * progress
                end = min(lay.rim, high + height * length * min(1.2, 0.3 + 0.7 * strength * impact))
                found.append((2 * math.pi * turn, start, end, width * (1.0 + 1.6 * strength), fade * strength, key))
    return found[:MOST_STREAKS]


def shocks(lay, onsets):
    """Widening circles of recent onsets: (radius, strength 0..1)."""
    rings = []
    for age, strength, _time in onsets:
        progress = age / SHOCK_LIFE
        if progress < 1.0:
            rings.append((lay.base + (lay.frame - lay.base) * progress, strength * (1.0 - progress)))
    return rings


# -- atmosphere -------------------------------------------------------------------------------


@lru_cache(maxsize=2)
def particle_field(count=PARTICLES, seed=7):
    """Fixed points of light: rows of (turn, radius as a share of the reach, turns per
    second, size class 0..2). Worked out once."""
    random = np.random.default_rng(seed)
    field = np.column_stack(
        (
            random.random(count),
            0.2 + 1.9 * random.random(count) ** 0.8,
            (0.002 + 0.006 * random.random(count)) * np.where(random.random(count) < 0.5, -1, 1),
            random.integers(0, 3, count),
        )
    )
    field.flags.writeable = False
    return field


def particles(lay, now):
    """(x, y, size class) of the particles at audio time ``now``: each drifts round the
    centre at its own slow rate. ``now`` None: where they are at rest."""
    field = particle_field()
    angles = 2 * np.pi * (field[:, 0] + field[:, 2] * (now or 0.0))
    x, y = xy(lay, angles, field[:, 1] * lay.reach)
    return x, y, field[:, 3].astype(int)


@lru_cache(maxsize=2)
def ambient_arcs(count=7, seed=11):
    """Faint fixed arcs of the atmosphere: rows of (radius share, start turn, span turn)."""
    random = np.random.default_rng(seed)
    arcs = np.column_stack(
        (1.04 + 0.75 * random.random(count), random.random(count), 0.06 + 0.2 * random.random(count))
    )
    arcs.flags.writeable = False
    return arcs
