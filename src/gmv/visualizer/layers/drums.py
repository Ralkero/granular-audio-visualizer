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
from functools import lru_cache

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from .. import palette
from ..state import COLUMN as C
from . import Layer
from .guitar import NEUTRAL, tint

ROWS = (("low", 0.76), ("mid", 0.47), ("high", 0.17))  # region, height in the band (0 top)
SHADES = palette.DRUM_REGIONS  # three shades of the drums' own colour


@lru_cache(maxsize=32)
def shades(colour):
    """Three shades of ``colour`` for the three regions. The drums' own colour has its
    chosen shades (``palette.DRUM_REGIONS``); any other is darkened and lightened."""
    if colour == palette.INSTRUMENTS["drums"]:
        return SHADES
    return {
        "low": palette.shade(colour, "#000000", 0.3),
        "mid": colour,
        "high": palette.shade(colour, "#ffffff", 0.55),
    }


RAYS = [(math.cos(a), math.sin(a)) for a in np.linspace(0, 2 * math.pi, 8, endpoint=False)]


# Live ring: how long each region's impact lasts, in seconds.
LIFE = {"low": 0.5, "mid": 0.2, "high": 0.13}
SPOKES = 28


@lru_cache(maxsize=128)
def sparks(seed):
    """Where the sparks of the high-region hit with ``seed`` are: 18 x (turn, depth), each
    0..1. The same for a hit every frame, so worked out once (bounded cache)."""
    spots = np.random.default_rng(seed).random((18, 2))
    spots.flags.writeable = False
    return spots


class DrumsLayer(Layer):
    def paint_live(self, painter, stage, live):
        """Every hit is an impact that happens and is gone, and any number overlap.
        ``low``: a heavy ring that swells outwards and thins, slowly. ``mid``: a sharp
        burst of rays all the way round, quickly gone. ``high``: a light scatter of
        sparks at the outer edge, gone almost at once. Size and brightness are the
        strength of the hit. Between hits there is only a faint circle."""
        height = stage.height
        respond = stage.response
        strength = min(1.5, self.intensity) * stage.gain * respond.motion * respond.impact
        shown = min(1.0, live.weight * 1.2) * respond.shine
        hollow = live.tentative
        cx, cy = stage.cx, stage.cy
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        rest = stage.inner + height * 0.2
        painter.setPen(QPen(tint(self.tone, NEUTRAL, 0.3, 0.16 * min(1.0, shown)), 1))
        painter.drawEllipse(QPointF(cx, cy), rest, rest)
        regions = ("low", "mid") if self.style == "low and mid" else ("low", "mid", "high")
        for age, hit in live.recent(max(LIFE.values())):
            for name in regions:
                power = getattr(hit, name)
                progress = age / LIFE[name]
                if power <= 0 or progress >= 1:
                    continue
                fade = (1.0 - progress) ** 1.6
                shade = shades(self.tone)[name]
                colour = (
                    tint(shade, NEUTRAL, 0.6, 0.7 * fade * shown)
                    if hollow
                    else tint(shade, "#ffffff", 0.1 + 0.5 * power * fade, fade * shown)
                )
                if name == "low":
                    out = 1.0 - (1.0 - progress) ** 2  # fast at first, then slowing
                    radius = rest + height * (0.05 + 0.55 * out) * min(
                        1.2 * respond.motion, 0.6 + 0.5 * power * strength
                    )
                    width = (1.0 + height * 0.16 * power * strength * (1.0 - progress)) * (0.5 if hollow else 1.0)
                    painter.setPen(QPen(colour, width))
                    painter.drawEllipse(QPointF(cx, cy), min(radius, stage.outer - width / 2), min(radius, stage.outer - width / 2))
                elif name == "mid":
                    centre = stage.inner + height * (0.5 + 0.08 * progress)
                    reach = height * (0.12 + 0.26 * power * strength) * (0.7 + 0.3 * progress)
                    turn = (hit.time * 7.3) % (2 * math.pi / SPOKES)  # each hit sits a little differently
                    painter.setPen(QPen(colour, (1.2 + 1.6 * power) * max(1.0, stage.scale)))
                    lines = []
                    for spoke in range(SPOKES):
                        angle = turn + 2 * math.pi * spoke / SPOKES
                        sin, cos = math.sin(angle), math.cos(angle)
                        a, b = max(stage.inner, centre - reach), min(stage.outer, centre + reach)
                        lines.append(QLineF(cx + a * sin, cy - a * cos, cx + b * sin, cy - b * cos))
                    painter.drawLines(lines)
                else:
                    seed = int(hit.time * 1000) & 0xFFFF
                    spots = sparks(seed)
                    angle = 2 * math.pi * spots[:, 0]
                    radius = stage.outer - height * (0.04 + 0.16 * spots[:, 1]) + height * 0.05 * progress
                    radius = np.minimum(radius, stage.outer)
                    size = (1.0 + 1.8 * power * strength) * max(1.0, stage.scale) * (1.0 - 0.5 * progress)
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(colour)
                    for x, y in zip(
                        (cx + radius * np.sin(angle)).tolist(), (cy - radius * np.cos(angle)).tolist(), strict=True
                    ):
                        painter.drawEllipse(QPointF(x, y), size, size)
                    painter.setBrush(Qt.BrushStyle.NoBrush)

    def paint_static(self, painter, rect, stage, state):
        painter.setFont(stage.font(10, QFont.Weight.DemiBold, 1.2))
        painter.setPen(tint(self.profile.colour, "#ffffff", 0.1, 0.95))
        painter.drawText(
            QPointF(rect.left() + 10 * stage.scale, rect.top() + 16 * stage.scale),
            self.profile.title.upper(),
        )

    def mark(self, surface, artist, name, x, y, size, strength, opacity, hollow):
        """One impact of region ``name`` centred on (x, y)."""
        shade = shades(self.tone)[name]
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
        size = height * 0.16 * min(1.5, 0.5 + 0.5 * self.drawn)
        only = None if self.style != "low and mid" else ("low", "mid")
        for row, weight, pale in zip(rows[1:], weights[1:], tentative[1:], strict=True):
            if row[C["attack"]] <= 0 or weight <= 0:
                continue
            x = float(to_x(row[0]))
            for name, place in ROWS:
                strength = row[C[name]]
                if strength > 0 and (only is None or name in only):
                    opacity = (0.4 + 0.6 * strength) * weight * min(1.0, 0.4 + 0.6 * self.drawn)
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
