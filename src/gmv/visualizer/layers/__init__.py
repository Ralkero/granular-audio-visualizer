"""Instrument layers: one independent picture per instrument, on a shared audio timeline.

The main view is a composition of layers. A layer reads one instrument's activity history
(``VisualState.activity[key]``, filled from ``InstrumentActivity`` messages) and how much of
it the recognition model attributes to the instrument (``VisualState.attribution``), and
draws it in its own visual language. Layers do not read the whole-mix spectrum and do not
read each other.

``PROFILES`` is the design contract: the character each instrument is meant to have. A
profile with a ``renderer`` is implemented; one without is a plan, and nothing is drawn for
it. There is deliberately no generic fallback renderer: an instrument without its own
analysis source and its own picture is not shown at all.

Adding an instrument:

1. an analyzer in ``gmv.instruments`` that publishes ``InstrumentActivity`` for its key;
2. a ``Layer`` subclass here with ``paint_stage`` (main view) and ``paint_lane`` (detail);
3. ``renderer=`` on its profile.

Settings, composition, the lane and session logs pick it up from the registry.
"""

from dataclasses import dataclass

from PySide6.QtCore import QRectF

from ..strip import Strip


@dataclass(frozen=True)
class Profile:
    key: str
    title: str
    colour: str
    character: str  # the visual identity to aim for
    signals: str  # what its analyzer has to provide for that identity
    renderer: type | None = None
    styles: tuple[tuple[str, str], ...] = ()  # (key, title); the first is the default


class Stage:
    """What a layer needs to place things in time, for one paint."""

    def __init__(self, rect, now, seconds, scale, generation, font):
        self.rect, self.now, self.seconds = rect, now, seconds
        self.scale, self.generation, self.font = scale, generation, font
        self.start = None if now is None else now - seconds
        self.per_second = rect.width() / seconds


class Layer:
    """Base of every instrument layer. Subclasses draw; this holds what they share.

    ``intensity`` scales how strongly the layer is drawn and nothing else: it is never
    passed to an analyzer. ``enabled`` hides the layer; analysis carries on regardless.
    """

    def __init__(self, profile):
        self.profile = profile
        self.enabled = True
        self.intensity = 1.0
        self.style = self.profile.styles[0][0] if self.profile.styles else ""
        self.strip = Strip()
        self.drawn_until = None  # time of the last history row already on the strip

    @property
    def key(self):
        return self.profile.key

    def rows(self, state):
        track = state.activity.get(self.key)
        return None if track is None or not len(track) else track

    # -- to implement --------------------------------------------------------------------

    def paint_static(self, painter, rect, stage, state):
        """Labels and backgrounds: drawn into the rarely-changing picture."""

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        """Draw ``rows`` (ACTIVITY columns, oldest first; the first row was drawn before
        and is there to join on to) into the scrolling strip."""
        raise NotImplementedError

    def paint_live(self, painter, rect, stage, state):
        """What moves every frame and is not history (drawn after the strip)."""

    def paint_lane(self, artist, to_x, rows, top, bottom):
        """The same rows in the detailed lane, between ``top`` and ``bottom``."""
        raise NotImplementedError

    # -- shared machinery ------------------------------------------------------------------

    def paint_stage(self, painter, rect, stage, state):
        """History through a cached strip (only new rows are drawn), then the live part."""
        track = self.rows(state)
        if track is None or stage.now is None:
            return
        # A changed judgement of the recognition model changes how past rows should look,
        # and so do style and intensity: then the strip is drawn again from its start.
        key = (stage.generation, state.board.judgement, self.style, round(self.intensity, 3))

        def draw(artist, to_x, since, height):
            first = stage.start if since is None or self.drawn_until is None else self.drawn_until
            rows = track.since(first)
            if len(rows) < 2:
                return
            weights, tentative = state.attribution(self.key, rows[:, 0])
            self.paint_history(artist, to_x, rows, weights, tentative, height, stage)
            self.drawn_until = float(rows[-1, 0])

        self.strip.paint(
            painter, rect, stage.start, stage.per_second, key, float(track.last[0]), draw
        )
        self.paint_live(painter, rect, stage, state)


def profiles():
    from .guitar import GuitarLayer

    return {
        profile.key: profile
        for profile in (
            Profile(
                "vocals",
                "Vocals",
                "#f472b6",
                "Organic, flowing, phrase-oriented: a ribbon that swells and bends with a "
                "phrase and closes between phrases.",
                "Voiced level and pitch contour of the lead voice, phrase boundaries.",
            ),
            Profile(
                "guitar",
                "Guitar",
                "#f5c451",
                "Articulated and string-like: strings that are struck, ring and die away; "
                "sharp strokes for picking and strumming, beamed when they repeat.",
                "Attacks, ringing level, repetition rate, where in the range a stroke sat.",
                renderer=GuitarLayer,
                styles=(("strings", "Strings"), ("strokes", "Strokes")),
            ),
            Profile(
                "bass",
                "Bass",
                "#7ad7f5",
                "Deep, heavy, flowing: one broad, slow wave whose weight follows the note "
                "and whose height follows the pitch.",
                "Low-register note level and pitch, note changes.",
            ),
            Profile(
                "drums",
                "Drums",
                "#f87171",
                "Sharp, percussive, impact-driven: bursts that hit and are gone, placed by "
                "the part of the kit.",
                "Hit times and strengths per kit region (low, mid, high).",
            ),
        )
    }


def build():
    """One layer per implemented profile, in composition order (top to bottom)."""
    return {
        key: profile.renderer(profile) for key, profile in profiles().items() if profile.renderer
    }


def slots(rect, count, gap):
    """Equal horizontal bands of ``rect`` for ``count`` layers, top to bottom."""
    if not count:
        return []
    height = (rect.height() - gap * (count - 1)) / count
    return [
        QRectF(rect.left(), rect.top() + index * (height + gap), rect.width(), height)
        for index in range(count)
    ]
