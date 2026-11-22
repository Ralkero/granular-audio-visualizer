"""The drum layer: impacts.

Every hit the drum analyzer reports is one mark, placed at its time and gone again; nothing
is joined up and nothing rings. The three regions of the analyzer are three kinds of mark
at three heights, so a pattern reads as a pattern:

* ``low``   a filled disc with a ring around it, at the bottom: a thump;
* ``mid``   a burst of rays, in the middle: a crack;
* ``high``  a small spark, at the top: a tick.

Size and brightness are the strength of the hit. They are regions of the spectrum, not kit
pieces, and the layer does not label them otherwise. While the activity model says drums
are absent nothing is drawn; without a judgement the marks are hollow and pale.
"""

import math

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from .. import palette
from ..state import COLUMN as C
from . import Layer
from .guitar import NEUTRAL, tint

ROWS = (("low", 0.76), ("mid", 0.47), ("high", 0.17))  # region, height in the band (0 top)
SHADES = palette.DRUM_REGIONS  # three shades of the drums' own colour
RAYS = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]


class DrumsLayer(Layer):
    def paint_static(self, painter, rect, stage, state):
        painter.setFont(stage.font(10, QFont.Weight.DemiBold, 1.2))
        painter.setPen(tint(self.profile.colour, "#ffffff", 0.1, 0.95))
        painter.drawText(
            QPointF(rect.left() + 10 * stage.scale, rect.top() + 16 * stage.scale),
            self.profile.title.upper(),
        )

    def mark(self, surface, artist, name, x, y, size, strength, opacity, hollow):
        """One impact of region ``name`` centred on (x, y)."""
        shade = SHADES[name]
        colour = (
            tint(shade, NEUTRAL, 0.6, opacity * 0.8)
            if hollow
            else tint(shade, "#ffffff", 0.15 + 0.5 * strength, opacity)
        )
        if name == "low":
            radius = size * (0.45 + 0.75 * strength)
            artist.setPen(QPen(colour, 1.2))
            artist.setBrush(
                Qt.BrushStyle.NoBrush if hollow else tint(shade, "#ffffff", 0.1, opacity * 0.75)
            )
            surface.ellipse(artist, x, y, radius)
            artist.setBrush(Qt.BrushStyle.NoBrush)
            artist.setPen(QPen(tint(shade, "#ffffff", 0.2, opacity * 0.45), 1))
            surface.ellipse(artist, x, y, radius * 1.55)
        elif name == "mid":
            reach = size * (0.5 + 1.0 * strength)
            artist.setPen(QPen(colour, 1.0 + 1.2 * strength))
            surface.lines(
                artist,
                [
                    (x + 0.25 * reach * c, y + 0.25 * reach * s, x + reach * c, y + reach * s)
                    for c, s in RAYS
                ],
            )
        else:
            reach = size * (0.3 + 0.5 * strength)
            artist.setPen(QPen(colour, 1.0))
            surface.line(artist, x, y - reach, x, y + reach)
            surface.line(artist, x - reach * 0.6, y, x + reach * 0.6, y)

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        size = height * 0.16 * min(1.5, 0.5 + 0.5 * self.intensity)
        only = None if self.style != "low and mid" else ("low", "mid")
        for row, weight, pale in zip(rows[1:], weights[1:], tentative[1:], strict=True):
            if row[C["attack"]] <= 0 or weight <= 0:
                continue
            x = float(to_x(row[0]))
            for name, place in ROWS:
                strength = row[C[name]]
                if strength > 0 and (only is None or name in only):
                    opacity = (0.4 + 0.6 * strength) * weight * min(1.0, 0.4 + 0.6 * self.intensity)
                    self.mark(
                        stage.surface,
                        artist,
                        name,
                        x,
                        height * place,
                        size,
                        float(strength),
                        opacity,
                        bool(pale),
                    )

    def reading(self, state):
        track = self.rows(state)
        if track is None:
            return ""
        row = track.last
        weights, tentative = state.attribution(self.key, row[:1])
        if weights[0] <= 0:
            return "drums not detected: impacts not shown"
        parts = ["unconfirmed: no activity model is running"] if tentative[0] else []
        rate = row[C["rate_hz"]]
        if np.isfinite(rate) and row[C["regularity"]] >= 0.5:
            parts.append(f"steady, about {rate:.0f} hits per second")
        return "  ·  ".join(parts)

    def paint_lane(self, artist, to_x, rows, top, bottom):
        """Detail lane: a tick per hit, in the third of the lane that belongs to its region."""
        third = (bottom - top) / 3
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        xs = to_x(rows[:, 0])
        for x, row in zip(xs[1:].tolist(), rows[1:], strict=True):
            if row[C["attack"]] <= 0:
                continue
            for index, name in enumerate(("high", "mid", "low")):
                strength = row[C[name]]
                if strength > 0:
                    foot = top + (index + 1) * third
                    artist.setPen(QPen(QColor(tint(SHADES[name], "#ffffff", 0.35, 0.95)), 1))
                    artist.drawLine(
                        QLineF(x, foot - 1, x, foot - 1 - (0.35 + 0.65 * strength) * (third - 2))
                    )
