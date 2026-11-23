"""The composition of GPU Radiant, and what each instrument is doing in it. Numbers only.

When the graphics card draws Radiant it draws all of it: this module is where its
structures are, which is not where phase 1's were. Nothing here comes from the Classic
renderer or the QPainter instrument layers. What it reads is the shared visual state: the
same live instrument states, frames and markers every renderer reads.

From the centre outwards, as shares of the reach (half the smaller side of the area):

    core                 to 0.085 (0.13 at most)
    spectrum formation   0.20 to 0.44: a plasma annulus whose outer surface is the spectrum
    waveform             0.475: the envelope of the last seconds
    segments             0.52
    middle formation     0.555
    vocals               0.60 to 0.68: ribbons
    guitar               0.68 to 0.76: electrical strands
    bass                 0.76 to 0.86: a pressure ring
    drums                0.86 to 0.94: impacts
    outer formation      0.965, and its corona beyond

An instrument keeps its place whether or not the others are shown: hiding one removes its
structure and moves nothing.
"""

import math

import numpy as np

from .. import radiant_geometry as geometry

BANDS = {  # instrument -> (inner, outer) as shares of the reach
    "vocals": (0.60, 0.68),
    "guitar": (0.68, 0.76),
    "bass": (0.76, 0.86),
    "drums": (0.86, 0.94),
}
WAVE = 48  # points of the waveform ring, newest first
PULSES, HITS = 4, 8  # bass pulses and drum hits a frame carries
BASS_FLOW = (0.5, 3.0)  # radians a second at rest and per unit of energy, as the bass layer's


def layout(cx, cy, reach, keys=()):
    """The GPU composition in a circle of radius ``reach``, as a ``geometry.Layout`` (so
    that the scene's arithmetic is shared with phase 1). ``keys``: the instruments shown;
    ``bands`` lists their zones in that order."""
    reach = max(40.0, float(reach))
    return geometry.Layout(
        cx=cx,
        cy=cy,
        reach=reach,
        zone=reach * 0.555,
        core=reach * 0.085,
        core_most=reach * 0.13,
        base=reach * 0.20,
        thickness=reach * 0.035,
        tips=reach * 0.44,
        contour=reach * 0.475,
        swing=reach * 0.02,
        segments=reach * 0.52,
        depth=reach * 0.026,
        frame=reach * 0.555,
        rim=reach * 0.965,
        bands=tuple((BANDS[key][0] * reach, BANDS[key][1] * reach) for key in keys if key in BANDS),
    )


def advance(state, now, keys):
    """Bring the live state of every instrument shown to audio time ``now``, with the
    presence model's weight, exactly as the instrument layers do before they draw."""
    if now is None:
        return
    for key in keys:
        live = state.live.get(key)
        if live is None or live.time is None:
            continue
        weights, tentative = state.attribution(key, (live.time,))
        live.advance(now, float(weights[0]), bool(tentative[0]))


def wave(frames, now, seconds=geometry.CONTOUR_S, count=WAVE):
    """The envelope (0.35 level + 0.65 onset strength) of the last ``seconds`` at ``count``
    evenly spaced ages, newest first: what the waveform ring is displaced by. A function
    of the frames and ``now`` alone."""
    if now is None or frames is None or not len(frames):
        return [0.0] * count
    rows = frames.since(now - seconds - 0.1)
    rows = rows[rows[:, 0] <= now]
    if not len(rows):
        return [0.0] * count
    envelope = np.clip(0.35 * rows[:, 1] + 0.65 * rows[:, 3], 0.0, 1.0)
    ages = np.linspace(0.0, seconds, count)
    found = np.interp(now - ages, rows[:, 0], envelope, left=0.0, right=float(envelope[-1]))
    return [float(value) for value in found]


def instruments(state, keys, running=True):
    """What each instrument is doing now, for the shaders: plain numbers from the live
    states (which ``advance`` has brought to the present). An instrument that is hidden,
    has no data, or that the presence model does not hear, is all zeros."""

    def live_of(key):
        if not running or key not in keys:
            return None
        live = state.live.get(key)
        return live if live is not None and live.t is not None and live.weight > 0.02 else None

    found = {}
    # Vocals: how open the ribbons are, where the pitch puts them, a bloom at a phrase start.
    live = live_of("vocals")
    if live is None:
        found["u_vocals"], found["u_vocals_more"] = (0.0,) * 4, (0.0,) * 4
    else:
        starts = [(age, hit.strength) for age, hit in live.recent(2.0) if hit.start]
        bloom = sum(strength * math.exp(-age / 0.35) for age, strength in starts)
        newest = min((age for age, _ in starts), default=9.0)
        found["u_vocals"] = (min(1.0, live.weight), min(1.0, live.level), live.place, min(1.5, bloom))
        found["u_vocals_more"] = (live.t % 600.0, float(live.pitched), float(live.tentative), min(9.0, newest))
    # Guitar: how hard the strands ring, and the last stroke (when, where, how wide, how hard).
    live = live_of("guitar")
    if live is None:
        found["u_guitar"], found["u_guitar_more"] = (0.0,) * 4, (9.0, 0.5, 0.0, 0.0)
    else:
        last = live.last
        since = max(0.0, live.t - last.time) if last is not None else 9.0
        struck = last.strength if last is not None else 0.0
        found["u_guitar"] = (min(1.0, live.weight), min(1.0, live.level), min(1.5, live.ring), min(1.0, struck))
        found["u_guitar_more"] = (
            min(9.0, since),
            last.centre if last is not None else 0.5,
            last.spread if last is not None else 0.0,
            min(1.0, (live.rate_hz or 0.0) / 10.0) * live.regularity,
        )
    # Bass: weight of the ring, where the note puts it, its running phase, the note onsets.
    live = live_of("bass")
    pulses = []
    if live is None:
        found["u_bass"], found["u_bass_more"] = (0.0,) * 4, (0.0,) * 4
    else:
        pulses = [(min(9.0, age), min(1.0, strength) * live.weight) for age, strength in live.beats(1.2)][-PULSES:]
        found["u_bass"] = (min(1.0, live.weight), min(1.0, live.energy), live.place, min(2.0, live.swell(0.18)))
        found["u_bass_more"] = (live.phase(*BASS_FLOW) % (2 * math.pi), float(live.pitched), float(live.tentative), 0.0)
    found["u_bass_pulses"] = pulses + [(9.0, 0.0)] * (PULSES - len(pulses))
    # Drums: the recent hits, each with its age and its strength in the three regions.
    live = live_of("drums")
    hits = []
    if live is None:
        found["u_drums"] = (0.0,) * 4
    else:
        found["u_drums"] = (min(1.0, live.weight), min(1.0, live.level), 0.0, 0.0)
        for age, hit in live.recent(0.6)[-HITS:]:
            turn = (geometry.seed_of(hit.time, 4) % 1000) / 1000.0
            hits.append(((min(9.0, age), turn), (hit.low * live.weight, hit.mid * live.weight, hit.high * live.weight)))
    found["u_drum_hits"] = [hit[0] for hit in hits] + [(9.0, 0.0)] * (HITS - len(hits))
    found["u_drum_regions"] = [hit[1] for hit in hits] + [(0.0, 0.0, 0.0)] * (HITS - len(hits))
    return found
