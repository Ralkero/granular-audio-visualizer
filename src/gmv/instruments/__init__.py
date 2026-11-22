"""Instrument-specific activity: fast analyzers and how their output meets recognition.

An *instrument analyzer* turns the analysis stream into an ``Activity`` per frame for one
instrument: how much of it is ringing, how sharply it was just articulated, and whether
that articulation repeats. Analyzers run inside the one analysis pass (they are handed the
per-channel spectrum the analyzer already computed) and must stay cheap; anything expensive
belongs in its own worker with a bounded queue, publishing the same message.

``ANALYZERS`` is the registry. To add an instrument: write a class with ``key``, ``method``,
``delay_s``, ``sensitivity``, ``reset()`` and ``step(spectrum, quiet, samples) -> Activity |
None`` (``samples`` are the frame's new samples, for the rare analyzer that needs them),
and register it here. Nothing else in the engine changes.

Fast activity says *what is happening*; it does not say *which instrument*. ``combine``
is the single place where it meets the slower presence estimate of a recognition model.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Activity:
    """One frame of one instrument, ``delay_frames`` frames before the newest audio."""

    delay_frames: int
    level: float  # 0..1: how much of the instrument's sound is ringing
    flux: float  # 0..1: how sharply it is being articulated at this frame
    attack: float  # 0, or the strength (0..1) of an attack placed at this frame
    rate_hz: float | None  # repetition rate of the articulation, when there is one
    regularity: float  # 0..1: how periodic the articulation has been
    centre: float  # 0..1: where in the instrument's range the articulation sat (low to high)
    spread: float  # 0..1: how much of the range it covered (single note to full chord)
    basis: str = ""  # which signal the analyzer could use, when it has more than one
    components: tuple = ()  # drums: (("low", s), ("mid", s), ("high", s)) hit strengths
    pitch_midi: float | None = None  # bass, vocals: the note, when there is one
    pitch_confidence: float = 0.0
    boundary: str = ""  # vocals: "start" / "end" of a phrase
    # A faster reading that some analyzers have beside the one above, which describes the
    # frame ``lead_frames`` frames *newer* than this message's own (bass: its pitch needs
    # 170 ms and is back-dated; its energy does not and is not).
    energy: float | None = None  # 0..1 against a reference with a long memory
    pulse: float = 0.0  # 0, or the strength of a note onset in that newer frame
    lead_frames: int = 0


def analyzers():
    from .bass import BassPitch
    from .drums import DrumImpacts
    from .guitar import GuitarArticulation
    from .vocals import VocalContour

    return {
        "guitar": GuitarArticulation,
        "drums": DrumImpacts,
        "bass": BassPitch,
        "vocals": VocalContour,
    }


def build(names, rate, size, hop_s, sensitivity=None):
    """Analyzers for ``names``; unknown names are an error, raised before a session starts."""
    known = analyzers()
    unknown = [name for name in names if name not in known]
    if unknown:
        raise ValueError(f"No instrument analyzer for: {', '.join(unknown)}")
    sensitivity = sensitivity or {}
    return [known[name](rate, size, hop_s, sensitivity.get(name, 0.5)) for name in names]


# Weight of fast activity for each reading of the recognition model.
PRESENCE = {"detected": 1.0, "candidate": 0.65, "absent": 0.0}
UNCONFIRMED = 0.4  # no judgement available: shown, visibly tentative


def combine(state, warming_up=False):
    """How much of an instrument's fast activity to attribute to it, 0..1, and whether that
    is a tentative reading.

    ``state`` is what the recognition model said about that moment: "detected", "candidate",
    "absent", or None when there is no model or it has not reported on that moment.

    * detected / candidate: the activity is shown, fully or weakened.
    * absent: nothing is shown, however lively the articulation. A recognition model that
      judges guitar absent overrules the fast analysis, which cannot tell instruments apart.
    * absent while ``warming_up`` (the model has heard only a few seconds of this music and
      its context is still mostly silence), or no judgement at all: shown as tentative.
      Without this the first seconds of every song would be blank.
    """
    if state is None or (state == "absent" and warming_up):
        return UNCONFIRMED, True
    return PRESENCE[state], False
