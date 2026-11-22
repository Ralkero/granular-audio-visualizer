"""How strongly the picture answers the music: one place for every presentation gain.

A ``Response`` only says how large, bright and far things are drawn. It never reaches an
analyzer, a model, a threshold, a timestamp or the list of events: the same recording
gives the same attacks, pulses, pitches and presence judgements whatever the response is.
Renderers read it from the canvas (``Canvas.response``, handed to layers as
``stage.response``); they do not keep gains of their own.

``BASELINE`` is all ones and reproduces the picture as it was before this existed.
``EXPRESSIVE`` is the "Expressive Mode" of the settings: the same events, drawn larger and
brighter. Every value is a bounded multiplier (or an exponent below one, for curves that
must stay within 0..1), and every renderer clips what it draws to its own region, so no
response can push a shape out of its ring or a level past full.

Timing is not part of a response on purpose: how fast things swing, how long an impact
lasts and when it starts stay with the audio clock.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Response:
    name: str = "baseline"
    # Instrument layers
    motion: float = 1.0  # displacement of strings, strands and the bass ring; reach of impacts
    impact: float = 1.0  # extra size and thickness at an attack, pulse or hit
    shine: float = 1.0  # brightness and opacity of instrument shapes and their flashes
    # Whole-mix spectrum
    spectrum_curve: float = 1.0  # exponent on bar heights (0..1): below one lifts them
    flash: float = 1.0  # how strongly an onset whitens the bars
    breath: float = 1.0  # how far the radial core swells with the level and the beat
    heat: float = 1.0  # how far the intensity colouring goes for the same intensity
    # Background
    glow: float = 1.0  # strength of the background glow and of the beat glow

    def lifted(self, values):
        """Bar heights (an array or a number in 0..1) through the spectrum curve."""
        return values if self.spectrum_curve == 1.0 else values**self.spectrum_curve


BASELINE = Response()
EXPRESSIVE = Response(
    name="expressive",
    motion=1.5,
    impact=1.5,
    shine=1.35,
    spectrum_curve=0.78,
    flash=1.5,
    breath=1.8,
    heat=1.25,
    glow=1.7,
)
