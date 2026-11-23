"""What the shaders are told for one frame: numbers only. No Qt and no OpenGL.

This is the rendering contract of GPU Radiant. ``values`` turns the shared visual state
(through the scene: a function of the state and audio time), the composition
(``scene.layout``), the palette and the response into a dictionary of shader uniforms.
Everything the picture does in answer to the music is decided here and in ``scene``; the
shaders only draw it. So the same recording at the same audio time gives the same
dictionary whatever the frame rate, and the same dictionary gives the same picture.

Nothing in it comes from another renderer: no Classic geometry, no QPainter layer.

Lengths are in units of the composition's reach. Colours are linear light (the palette's
``#rrggbb`` decoded from sRGB), which is what the shaders add and mix in.
"""

import math

from .. import radiant_geometry as geometry
from ..radiant_palette import KEYS
from . import scene as composition

SEED = 7.0
# Audio time is wrapped once a day so that it stays a usable 32-bit float: the picture
# jumps once then. Within a day a float still tells times 8 ms apart, and 2 ms apart for
# the first four and a half hours.
WRAP_S = 86400.0
ONSETS, BOLTS = 8, 16  # most onsets and bolts a frame carries
BEATS = geometry.MOST_BEATS  # most beats disturbing the smoke at once
SPECTRUM = 48  # bands of the display spectrum the shaders are given
EXPOSURE = 1.0
COLOURS = KEYS  # every role of the palette is a uniform: ``c_<role>``


def linear(colour):
    """``#rrggbb`` as three linear-light numbers 0..1. Anything else is black, so that no
    saved or typed value can put a NaN into a shader."""
    try:
        text = colour.lstrip("#")
        channels = [int(text[index : index + 2], 16) / 255.0 for index in (0, 2, 4)]
    except (AttributeError, ValueError, IndexError):
        return (0.0, 0.0, 0.0)
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in channels)


def number(value, low=0.0, high=1.0):
    """A finite number within bounds: what a uniform may be."""
    value = float(value)
    return low if not math.isfinite(value) else min(high, max(low, value))


def own_turn(time):
    """A number 0..1 that belongs to the event at audio ``time`` (to the millisecond) and
    is unlike that of events a round interval away: beats half a second or a second
    apart must not all push the smoke the same way.

    A full integer mix (the "lowbias32" finaliser), not a single multiplication: a single
    multiplication adds the same step for every same-sized interval, so the beats of a
    steady tempo would get turns a constant, small step apart and all look alike."""
    x = int(round(time * 1000.0)) & 0xFFFFFFFF
    x ^= x >> 16
    x = (x * 0x7FEB352D) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x846CA68B) & 0xFFFFFFFF
    x ^= x >> 16
    return x / 4294967296.0


def bands(spectrum):
    """(lows, mids, highs) of the display spectrum, each 0..1: the mean of its lowest
    quarter (to about 150 Hz), its middle and its top third (from about 2 kHz)."""
    if spectrum is None or not len(spectrum):
        return 0.0, 0.0, 0.0
    count = len(spectrum)
    low, high = max(1, count // 4), max(1, count // 3)
    parts = spectrum[:low], spectrum[low : count - high], spectrum[count - high :]
    return tuple(number(sum(part) / len(part)) if len(part) else 0.0 for part in parts)


def spectrum_bands(spectrum, curve=1.0, count=SPECTRUM):
    """The display spectrum as exactly ``count`` numbers 0..1, lows first, lifted by the
    response's curve. A spectrum of another length is resampled along its own order; none
    is all zeros. Nothing is added: every number is a measured band or lies between two."""
    if spectrum is None or not len(spectrum):
        return [0.0] * count
    measured = [number(value) ** curve for value in spectrum]
    if len(measured) == count:
        return measured
    last = len(measured) - 1
    found = []
    for index in range(count):
        place = index * last / (count - 1)
        lower = int(place)
        upper = min(last, lower + 1)
        found.append(measured[lower] + (measured[upper] - measured[lower]) * (place - lower))
    return found


def values(lay, view, palette, response, *, size, area=None, spectrum=None, glow=None,
           instrument=None, state=None, keys=(), show=0):  # fmt: skip
    """The uniforms of one frame.

    ``lay``: the composition (``scene.layout``). ``view``: the scene from
    ``Radiant.scene``. ``size``: the view in logical pixels. ``area``: top and bottom of
    the visualizer's own area, in pixels. ``spectrum``: the display spectrum. ``glow``:
    (colour, strength) of the background glow. ``instrument``: key -> colour, for bolts.
    ``state`` and ``keys``: the visual state and the instruments shown, for what each
    instrument is doing (``scene.instruments``); without a state every instrument is at
    rest."""
    reach = lay.reach
    width, height = size
    low, mid, high = bands(spectrum)
    top, bottom = area if area is not None else (0.0, height)
    found = {
        "u_size": (float(width), float(height)),
        "u_centre": (float(lay.cx), float(lay.cy)),
        "u_reach": float(reach),
        "u_time": float((view.now or 0.0) % WRAP_S),
        "u_seed": SEED,
        "u_zone": lay.zone / reach,
        "u_frame": lay.frame / reach,
        "u_rim": lay.rim / reach,
        "u_base": lay.base / reach,
        "u_tips": lay.tips / reach,
        "u_contour": lay.contour / reach,
        "u_swing": lay.swing / reach,
        "u_seg_low": view.segment_ring[0] / reach,
        "u_seg_high": view.segment_ring[1] / reach,
        "u_level": number(view.level),
        "u_intensity": number(max(view.intensity, view.musical)),
        "u_musical": number(view.musical),
        "u_flash": number(view.flash),
        "u_pulse": number(view.pulse),
        "u_low": low,
        "u_mid": mid,
        "u_high": high,
        "u_plasma": number(response.plasma, 0.0, 3.0),
        "u_motion": number(response.motion, 0.0, 2.0),
        "u_impact": number(response.impact, 0.0, 3.0),
        "u_shine": number(response.shine, 0.0, 2.0),
        "u_bloom": number(response.bloom, 0.0, 3.0),
        "u_exposure": EXPOSURE,
        "u_core": view.core / reach,
        "u_halo": view.halo / reach,
        "u_rays": view.rays / reach,
        "u_ray_turn": float(view.ray_turn % (2 * math.pi)),
        "u_segments": [number(level) for level in view.segments],
        "u_spectrum": spectrum_bands(spectrum, response.spectrum_curve),
        "u_area": (top / height, bottom / height),
        "u_show": int(show),
        "u_glow": number(glow[1]) if glow else 0.0,
        "c_glow": linear(glow[0]) if glow else (0.0, 0.0, 0.0),
    }
    for role in COLOURS:
        found["c_" + role] = linear(palette[role])
    for key, (inner, outer) in composition.BANDS.items():
        found["u_band_" + key] = (inner, outer)
    # What each instrument is doing: zeros for one that is hidden or not heard.
    running = bool(getattr(view, "running", True))
    if state is not None:
        found["u_wave"] = composition.wave(state.frames if running else None, view.now)
        found.update(composition.instruments(state, keys, running))
    else:
        found["u_wave"] = [0.0] * composition.WAVE
        found.update(composition.instruments(None, (), False))
    # Onsets of the whole mix, newest first: age, strength, a number of its own.
    onsets = [(number(age, 0.0, 10.0), number(strength), float(geometry.seed_of(time, 0) % 1000) / 1000.0)
              for age, strength, time in view.onsets[:ONSETS]]  # fmt: skip
    found["u_onset_count"] = len(onsets)
    found["u_onsets"] = onsets + [(0.0, 0.0, 0.0)] * (ONSETS - len(onsets))
    # Beats still disturbing the smoke, newest first: age, strength, a turn of its own. They
    # are rebuilt from the beats' timestamps every frame; nothing is carried over.
    struck = [(number(age, 0.0, 10.0), number(strength), own_turn(time), 0.0)
              for age, strength, time in getattr(view, "beats", ())[:BEATS]]  # fmt: skip
    found["u_beat_count"] = len(struck)
    found["u_beat_life"] = float(geometry.BEAT_LIFE)
    found["u_beats"] = struck + [(0.0, 0.0, 0.0, 0.0)] * (BEATS - len(struck))
    # Bolts: one per streak of the scene, which is one per reported event or a few.
    accents = palette["energy_accents"]
    strongest = sorted(view.streaks, key=lambda row: -row[4])[:BOLTS]
    bolts, colours = [], []
    for angle, start, end, _width, strength, source in strongest:
        colour = accents if source is None or instrument is None else instrument(source)
        bolts.append((float(angle % (2 * math.pi)), start / reach, end / reach, number(strength)))
        colours.append((*linear(colour), float(angle * 7.13 % 1.0)))
    found["u_bolt_count"] = len(bolts)
    found["u_bolts"] = bolts + [(0.0, 0.0, 0.0, 0.0)] * (BOLTS - len(bolts))
    found["u_bolt_colours"] = colours + [(0.0, 0.0, 0.0, 0.0)] * (BOLTS - len(colours))
    return found
