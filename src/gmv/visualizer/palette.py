"""The one place colours with a meaning are defined. No Qt here.

Every instrument has one colour, used by its layer, its lane, its label, its button and
its settings. Regional or sub-instrument variations are shades of that colour (``shade``),
never a different hue. The whole-mix spectrum and the event markers stay out of the
instruments' hues: a muted indigo-to-lavender for the spectrum, lime for predictions.
"""

INSTRUMENTS = {
    "vocals": "#f472b6",  # pink
    "guitar": "#f5c451",  # yellow / gold
    "bass": "#4cc9f0",  # blue / teal
    "drums": "#ff8a5b",  # orange
}
# Whole mix, low to high frequencies. Deliberately not teal (bass) and not warm.
SPECTRUM = ("#5b62a8", "#8f8fd6", "#cfd0f2")
# The same spectrum when the music as a whole is at its most intense (see
# ``visualizer.intensity``): a dark, deep red, low to high. Darker and bluer than the
# drums' orange, so the two cannot be confused; instruments are never tinted with it.
INTENSE = ("#74101e", "#9a1828", "#b52f40")
# Predicted events (beats). Lime: unlike guitar's gold and every other instrument.
PREDICTED = "#a3e635"
CONTEXT = "#8e86c9"
NEUTRAL = "#9aa6c4"  # unattributed or tentative


def rgb(colour):
    colour = colour.lstrip("#")
    return tuple(int(colour[index : index + 2], 16) for index in (0, 2, 4))


def shade(colour, towards="#ffffff", share=0.0):
    """``colour`` moved ``share`` of the way towards another colour, as ``#rrggbb``."""
    a, b = rgb(colour), rgb(towards)
    return "#" + "".join(f"{round(x + (y - x) * share):02x}" for x, y in zip(a, b, strict=True))


def instrument(key, fallback=NEUTRAL):
    return INSTRUMENTS.get(key, fallback)


# Drum regions: three shades of the drums' orange, darker for low, lighter for high.
DRUM_REGIONS = {
    "low": shade(INSTRUMENTS["drums"], "#b3261e", 0.35),
    "mid": INSTRUMENTS["drums"],
    "high": shade(INSTRUMENTS["drums"], "#fff1d6", 0.55),
}
