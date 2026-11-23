"""Radial themes: different pictures of the same visual state.

A theme decides how the radial view is drawn and nothing else. It reads what the canvas
and the ``VisualState`` already hold (spectrum, musical intensity, dominance, live
instrument states, markers, audio time); it takes in no analysis messages of its own and
keeps no clock.

``classic`` is the radial view as it always was, drawn by the canvas itself.
``radiant`` is ``themes.radiant.Radiant``: concentric luminous structures with a palette
of its own (``themes.radiant_palette``) that the user can change colour by colour.

Spectrum Bars is a mode, not a radial theme, and no theme touches it.
"""

THEMES = (("classic", "Classic"), ("radiant", "Radiant"))
DEFAULT = "classic"


def known(name):
    """``name`` if it is a theme, otherwise the default (old or damaged settings)."""
    return name if name in dict(THEMES) else DEFAULT
