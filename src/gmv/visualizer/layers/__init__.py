"""Instrument layers: one independent picture per instrument, drawn inside whichever
visualizer is selected, on a shared audio timeline.

A layer reads one instrument's activity history (``VisualState.activity[key]``, filled from
``InstrumentActivity`` messages) and how much of it the presence model attributes to the
instrument (``VisualState.attribution``), and draws it in its own visual language. Layers
do not read the whole-mix spectrum and do not read each other.

Responsibilities are kept apart:

* the visualizer mode decides *where* each instrument's band is and hands every layer a
  ``Stage`` with a ``surface`` (``layers.surface``): a strip across the bars, a ring round
  the radial spectrum;
* the layer decides *what* is drawn, in band space, and never touches mode geometry;
* colours come from ``gmv.visualizer.palette``; visibility, intensity and style are the
  layer's own and never reach an analyzer.

``profiles()`` is the design contract: the character each instrument has and what its
analyzer provides. Every instrument has its own ``Layer`` subclass and its own picture.
There is deliberately no generic renderer: an instrument is never drawn with another
instrument's picture in a different colour, and one without its own analysis source is not
drawn at all.

Adding an instrument: an analyzer in ``gmv.instruments`` that publishes
``InstrumentActivity`` for its key; a ``Layer`` subclass with ``paint_history`` and
``paint_lane``; ``renderer=`` on its profile; its colour in the palette.
Adding a visualizer mode: a surface, and a ``Stage`` per layer from the mode's geometry.
"""

import math
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from PySide6.QtCore import QRectF

from .. import palette
from ..response import BASELINE
from ..strip import Strip, Sweep
from .surface import Linear, Polar

GAP = 0.06  # share of a ring left empty ahead of "now"


@lru_cache(maxsize=8)
def around(count):
    """(angles, sines, cosines) of ``count`` + 1 points once round a circle from the top,
    the last on the first, so a polyline through them closes."""
    angles = np.linspace(0.0, 2 * math.pi, count + 1)
    return angles, np.sin(angles), np.cos(angles)


def circle(stage, radii, sines, cosines):
    """A closed curve round the stage's centre at ``radii`` (one per angle), kept inside
    the instrument's own ring."""
    from ..strip import polygon

    radii = np.clip(radii, stage.inner, stage.outer)
    return polygon(stage.cx + radii * sines, stage.cy - radii * cosines)


@dataclass(frozen=True)
class Profile:
    key: str
    title: str
    colour: str
    character: str  # the visual identity to aim for
    signals: str  # what its analyzer has to provide for that identity
    renderer: type | None = None
    styles: tuple[tuple[str, str], ...] = ()  # (key, title); the first is the default
    weight: float = 1.0  # share of the layers' room, relative to the other layers


class Stage:
    """What a layer needs for one paint: where its band is and how time runs along it.

    ``kind`` "linear": ``rect`` is the band; time runs left to right, ``seconds`` wide.
    ``kind`` "polar": the band is the ring between ``inner`` and ``outer`` round
    (``cx``, ``cy``); time runs clockwise from the top, one turn per ``seconds``.
    ``height`` is the band's thickness and ``per_second`` its length per second, in pixels,
    in both kinds."""

    def __init__(
        self, now, seconds, scale, generation, font, *, rect=None, ring=None, live=False, gain=1.0,
        response=BASELINE,
    ):  # fmt: skip
        # ``response``: how strongly everything is drawn (``visualizer.response``).
        self.response = response
        # ``live``: a polar stage is drawn as a fixed, animated object from the
        # instrument's present state instead of as a history swept by a hand.
        # ``gain``: how much larger everything moves in intense music, about 0.5..1.
        self.live, self.gain = live and ring is not None, gain
        self.now, self.seconds = now, seconds
        self.scale, self.generation, self.font = scale, generation, font
        self.rect, self.ring = rect, ring
        self.kind = "polar" if ring is not None else "linear"
        self.start = None if now is None else now - seconds
        if ring is not None:
            self.cx, self.cy, self.inner, self.outer = ring
            self.height = self.outer - self.inner
            self.per_turn = math.pi * (self.inner + self.outer)  # length of the middle circle
            self.per_second = self.per_turn / seconds
        else:
            self.height = rect.height()
            self.per_second = rect.width() / seconds
        self.surface = None  # set for the duration of a history paint


class Layer:
    """Base of every instrument layer. Subclasses draw; this holds what they share.

    ``intensity`` scales how strongly the layer is drawn and nothing else: it is never
    passed to an analyzer. ``enabled`` hides the layer; analysis carries on regardless.
    """

    def __init__(self, profile):
        self.profile = profile
        self.enabled = True
        self.intensity = 1.0
        self.response = BASELINE  # of the stage being painted; see ``drawn``
        self.style = self.profile.styles[0][0] if self.profile.styles else ""
        self.strip = Strip()  # history as a scrolling strip (linear stages)
        self.sweep = Sweep()  # history as a ring that a hand sweeps round (polar stages)
        self.drawn_until = None  # time of the last history row already on the picture
        self._kind = None

    @property
    def key(self):
        return self.profile.key

    @property
    def drawn(self):
        """The layer's own intensity times the response's: what history pictures scale by."""
        return self.intensity * self.response.motion

    def rows(self, state):
        track = state.activity.get(self.key)
        return None if track is None or not len(track) else track

    # -- to implement --------------------------------------------------------------------

    def restart(self):
        """The history is about to be drawn again from its start: forget whatever a
        renderer carries from one slice to the next."""

    def paint_static(self, painter, rect, stage, state):
        """Labels of a linear band: drawn into the rarely-changing picture."""

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        """Draw ``rows`` (ACTIVITY columns, oldest first; the first row was drawn before
        and is there to join on to) in band space, through ``stage.surface``."""
        raise NotImplementedError

    def paint_lane(self, artist, to_x, rows, top, bottom):
        """The same rows in the detailed lane, between ``top`` and ``bottom``."""
        raise NotImplementedError

    def paint_live(self, painter, stage, live):
        """Draw the instrument as it is now, in its ring (``stage.inner`` to
        ``stage.outer`` round ``stage.cx``, ``stage.cy``), from ``live`` (``live.Live``,
        already advanced to this frame). Called every frame; nothing is kept."""
        raise NotImplementedError

    def reading(self, state):
        """One line saying how the newest activity is being read, or ''."""
        return ""

    # -- shared machinery ------------------------------------------------------------------

    def paint_stage(self, painter, stage, state):
        """History through a cached picture: only rows that arrived since the last paint
        are drawn. It is drawn again from its start only when *this* instrument's presence
        judgement changes, or the layer's style or intensity do."""
        track = self.rows(state)
        if track is None or stage.now is None:
            return
        self.response = stage.response
        if stage.live:
            live = state.live.get(self.key)
            if live is None or live.time is None:
                return
            weights, tentative = state.attribution(self.key, (live.time,))
            live.advance(stage.now, float(weights[0]), bool(tentative[0]))
            if live.weight > 0.02:
                self.paint_live(painter, stage, live)
            return
        key = (
            stage.generation,
            state.board.judged.get(self.key, 0),
            self.style,
            round(self.intensity, 3),
            self.response.name,
        )
        if stage.kind != self._kind:  # the mode changed: the other picture starts afresh
            self._kind, self.drawn_until = stage.kind, None
        latest = float(track.last[0])
        if stage.kind == "polar":
            centre = math.ceil(stage.outer) + 2  # of the ring's own picture
            stage.surface = Polar(centre, centre, stage.outer, stage.per_turn)
            oldest = stage.now - stage.seconds * (1 - GAP)

            def to_x(time):
                return time * stage.per_second

            def draw(artist, since):
                self._slice(artist, to_x, track, since, oldest, state, stage)

            self.sweep.paint(
                painter, stage.cx, stage.cy, stage.outer, stage.seconds, GAP, key, latest, draw
            )
        else:
            stage.surface = Linear()

            def draw(artist, to_x, since, _height):
                self._slice(artist, to_x, track, since, stage.start, state, stage)

            self.strip.paint(painter, stage.rect, stage.start, stage.per_second, key, latest, draw)
        stage.surface = None

    def _slice(self, artist, to_x, track, since, oldest, state, stage):
        if since is None:
            self.restart()
        first = oldest if since is None or self.drawn_until is None else self.drawn_until
        rows = track.since(first)
        if len(rows) < 2:
            return
        weights, tentative = state.attribution(self.key, rows[:, 0])
        self.paint_history(artist, to_x, rows, weights, tentative, stage.height, stage)
        self.drawn_until = float(rows[-1, 0])


def profiles():
    from .bass import BassLayer
    from .drums import DrumsLayer
    from .guitar import GuitarLayer
    from .vocals import VocalsLayer

    colour = palette.INSTRUMENTS
    return {
        profile.key: profile
        for profile in (
            Profile(
                "vocals",
                "Vocals",
                colour["vocals"],
                "Organic, flowing, phrase-oriented: strands that open with a phrase, bend with "
                "its pitch and close between phrases.",
                "Level of the centre in the voice's range, pitch contour where there is one, "
                "phrase starts and ends.",
                renderer=VocalsLayer,
                styles=(("ribbon", "Ribbon"), ("strands", "Strands only")),
                weight=1.0,
            ),
            Profile(
                "guitar",
                "Guitar",
                colour["guitar"],
                "Articulated and string-like: strings that are struck, ring and die away; "
                "sharp strokes for picking and strumming, beamed when they repeat.",
                "Attacks, ringing level, repetition rate, where in the range a stroke sat.",
                renderer=GuitarLayer,
                styles=(("strings", "Strings"), ("strokes", "Strokes")),
                weight=1.25,
            ),
            Profile(
                "bass",
                "Bass",
                colour["bass"],
                "Deep, heavy, flowing: one broad ribbon whose height is the note and whose "
                "thickness is its weight; it moves in steps and holds.",
                "Low-register pitch, weight of the periodic low sound, new and repeated notes.",
                renderer=BassLayer,
                weight=0.9,
            ),
            Profile(
                "drums",
                "Drums",
                colour["drums"],
                "Sharp, percussive, impact-driven: marks that hit and are gone, a different "
                "mark and height for each region of the kit's range.",
                "Hit times and strengths in three regions (low, mid, high).",
                renderer=DrumsLayer,
                styles=(("all", "All regions"), ("low and mid", "Low and mid only")),
                weight=1.0,
            ),
        )
    }


def build():
    """One layer per implemented profile, in composition order (top to bottom, or outside
    to inside)."""
    return {
        key: profile.renderer(profile) for key, profile in profiles().items() if profile.renderer
    }


def slots(rect, layers, gap):
    """A horizontal band of ``rect`` for each layer, top to bottom, sized by the weight of
    its profile: an instrument whose picture needs room (six strings) gets more than one
    that does not."""
    if not layers:
        return []
    weights = [layer.profile.weight for layer in layers]
    unit = (rect.height() - gap * (len(layers) - 1)) / sum(weights)
    bands, top = [], rect.top()
    for weight in weights:
        bands.append(QRectF(rect.left(), top, rect.width(), unit * weight))
        top += unit * weight + gap
    return bands


def rings(inner, outer, layers, gap):
    """(inner radius, outer radius) for each layer, the first layer outermost, sized by
    weight like ``slots``."""
    if not layers:
        return []
    weights = [layer.profile.weight for layer in layers]
    unit = (outer - inner - gap * (len(layers) - 1)) / sum(weights)
    found, edge = [], outer
    for weight in weights:
        found.append((edge - unit * weight, edge))
        edge -= unit * weight + gap
    return found
