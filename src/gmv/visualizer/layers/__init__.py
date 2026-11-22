"""Instrument layers: one independent picture per instrument, on a shared audio timeline.

The main view is a composition of layers. A layer reads one instrument's activity history
(``VisualState.activity[key]``, filled from ``InstrumentActivity`` messages) and how much of
it the recognition model attributes to the instrument (``VisualState.attribution``), and
draws it in its own visual language. Layers do not read the whole-mix spectrum and do not
read each other.

``profiles()`` is the design contract: the character each instrument has and what its
analyzer provides. Every instrument has its own ``Layer`` subclass and its own picture.
There is deliberately no generic renderer: an instrument is never drawn with another
instrument's picture in a different colour, and one without its own analysis source is not
drawn at all.

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
    weight: float = 1.0  # share of the layers' height, relative to the other layers


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

    def restart(self):
        """The history strip is about to be drawn again from its start: forget whatever a
        renderer carries from one slice to the next."""

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
            if since is None:
                self.restart()
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
    from .bass import BassLayer
    from .drums import DrumsLayer
    from .guitar import GuitarLayer
    from .vocals import VocalsLayer

    return {
        profile.key: profile
        for profile in (
            Profile(
                "vocals",
                "Vocals",
                "#f472b6",
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
                "#f5c451",
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
                "#7ad7f5",
                "Deep, heavy, flowing: one broad ribbon whose height is the note and whose "
                "thickness is its weight; it moves in steps and holds.",
                "Low-register pitch, weight of the periodic low sound, new and repeated notes.",
                renderer=BassLayer,
                weight=0.9,
            ),
            Profile(
                "drums",
                "Drums",
                "#f87171",
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
    """One layer per implemented profile, in composition order (top to bottom)."""
    return {
        key: profile.renderer(profile) for key, profile in profiles().items() if profile.renderer
    }


SHARE = {1: 0.5, 2: 0.62, 3: 0.72}  # of the main area, by number of layers; more: 0.8


def share(count):
    """How much of the main area the layers take while the spectrum is shown below them."""
    return SHARE.get(count, 0.8)


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
