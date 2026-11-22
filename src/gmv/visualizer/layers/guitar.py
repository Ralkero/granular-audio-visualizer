"""The guitar layer: six strings on the audio timeline.

Everything that moves is read from the guitar activity history:

* a string is displaced by the ``ring`` envelope (it jumps at an articulation and dies
  away, slowly while a chord is held, quickly after a muted stroke), so a sustained chord
  is a long, calm wave and silence is a straight line;
* each attack is a stroke across the strings it covered: low or high by where the
  articulation sat in the band, few strings or all of them by how wide it was, heavier and
  whiter the stronger it was; the wave of a string starts again at each stroke;
* attacks that repeat at the measured repetition rate are joined by a beam, like beamed
  notes, so fast even strumming reads as one repeating figure;
* how much of all this is shown is the recognition model's judgement: nothing while it
  says guitar is absent, pale and dashed while it has not judged yet.

Nothing is animated on its own: with no activity rows there is no picture.
"""

import numpy as np
from PySide6.QtCore import QLineF, QPointF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from .. import palette
from ..state import COLUMN as C
from ..strip import polygon
from . import Layer

STRINGS = 6
NEUTRAL = QColor(palette.NEUTRAL)
# Waves per second of timeline for each string, low to high: only a look, like the gauge
# of a string. The amplitude and where a wave restarts come from the analysis.
WAVES = (7.0, 8.5, 10.5, 12.5, 15.0, 18.0)
BEAM_TOLERANCE = 0.35  # how far a gap may be from the repetition period and still be beamed
BEAM_REGULARITY = 0.5


def tint(colour, other, share, opacity):
    a, b = QColor(colour), QColor(other)
    mixed = QColor(
        round(a.red() + (b.red() - a.red()) * share),
        round(a.green() + (b.green() - a.green()) * share),
        round(a.blue() + (b.blue() - a.blue()) * share),
    )
    mixed.setAlphaF(max(0.0, min(1.0, opacity)))
    return mixed


def runs(flags):
    """(start, stop, value) for each stretch of equal values in a boolean array."""
    edges = np.flatnonzero(np.diff(flags.astype(np.int8))) + 1
    starts = np.concatenate(([0], edges))
    stops = np.concatenate((edges, [len(flags)]))
    return [(int(a), int(b), bool(flags[a])) for a, b in zip(starts, stops, strict=True)]


def covered(centre, spread):
    """Index of the lowest and highest string an articulation at ``centre`` (0 low .. 1
    high) of width ``spread`` reaches."""
    half = 0.1 + 0.55 * spread
    low = int(np.clip(np.floor((centre - half) * (STRINGS - 1) + 0.5), 0, STRINGS - 1))
    high = int(np.clip(np.floor((centre + half) * (STRINGS - 1) + 0.5), 0, STRINGS - 1))
    return low, max(low, high)


def beamed(row):
    """Whether the attack in ``row`` continues a repeating figure: it follows the attack
    before it by about one period of the measured repetition rate."""
    rate, gap = row[C["rate_hz"]], row[C["gap"]]
    return bool(
        gap > 0
        and np.isfinite(rate)
        and row[C["regularity"]] >= BEAM_REGULARITY
        and abs(gap * rate - 1.0) <= BEAM_TOLERANCE
    )


class GuitarLayer(Layer):
    def geometry(self, height):
        pad = height * 0.16
        ys = np.linspace(height - pad, pad + height * 0.06, STRINGS)  # low string lowest
        return ys, float(ys[0] - ys[1]), pad * 0.55

    def paint_static(self, painter, rect, stage, state):
        colour = QColor(self.profile.colour)
        painter.setFont(stage.font(10, QFont.Weight.DemiBold, 1.2))
        painter.setPen(tint(colour, "#ffffff", 0.1, 0.95))
        painter.drawText(
            QPointF(rect.left() + 10 * stage.scale, rect.top() + 16 * stage.scale),
            self.profile.title.upper(),
        )

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        colour = self.profile.colour
        strength = min(1.6, self.intensity)
        ys, spacing, rail = self.geometry(height)
        times = rows[:, 0]
        x0, x1 = float(to_x(times[0])), float(to_x(times[-1]))
        count = max(2, int((x1 - x0) / 1.5) + 1)
        xs = np.linspace(x0, x1, count)
        dense = times[0] + (xs - x0) / stage.per_second
        index = np.clip(np.searchsorted(times, dense, side="right") - 1, 0, len(times) - 1)
        shown = np.interp(dense, times, weights)
        pale = tentative[index]
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        artist.setBrush(Qt.BrushStyle.NoBrush)
        if self.style == "strokes":
            # One ribbon instead of strings: its half-height is the ringing envelope.
            middle = height / 2
            half = np.interp(dense, times, rows[:, C["ring"]]) * shown * height * 0.3 * strength
            artist.setPen(Qt.PenStyle.NoPen)
            for a, b, faint in runs(pale):
                b = min(len(xs), b + 1)
                artist.setBrush(
                    tint(colour, NEUTRAL, 0.6 if faint else 0.0, 0.16 if faint else 0.3)
                )
                stage.surface.polygon(
                    artist,
                    np.concatenate((xs[a:b], xs[a:b][::-1])),
                    np.concatenate((middle - half[a:b], (middle + half[a:b])[::-1])),
                )
            artist.setBrush(Qt.BrushStyle.NoBrush)
        else:
            ring = np.interp(dense, times, rows[:, C["ring"]]) * shown
            since = dense - rows[index, C["struck_time"]]
            centre, spread = rows[index, C["struck_centre"]], rows[index, C["struck_spread"]]
            reach = 0.14 + 0.7 * spread
            for string in range(STRINGS):
                part = np.exp(-(((string / (STRINGS - 1) - centre) / reach) ** 2))
                wave = np.sin(2 * np.pi * WAVES[string] * since)
                line = ys[string] + spacing * 0.46 * strength * ring * part * wave
                for a, b, faint in runs(pale):
                    b = min(len(xs), b + 1)
                    pen = QPen(
                        tint(colour, NEUTRAL, 0.65, 0.45)
                        if faint
                        else tint(colour, "#ffffff", 0.0, 0.8),
                        1,
                    )
                    if faint:
                        pen.setStyle(Qt.PenStyle.DashLine)
                    artist.setPen(pen)
                    stage.surface.polyline(artist, xs[a:b], line[a:b])
        # Strokes and beams. The first row was drawn with the slice before this one.
        lean = spacing * 0.3
        for row, weight, faint in zip(rows[1:], weights[1:], tentative[1:], strict=True):
            attack = row[C["attack"]]
            if attack <= 0 or weight <= 0:
                continue
            x = float(to_x(row[0]))
            low, high = covered(row[C["centre"]], row[C["spread"]])
            if self.style == "strokes":
                bottom, top = height * (0.5 + 0.38 * attack), height * (0.5 - 0.38 * attack)
            else:
                bottom, top = ys[low] + spacing * 0.42, ys[high] - spacing * 0.42
            opacity = (0.35 + 0.65 * attack) * weight * min(1.0, 0.4 + 0.6 * self.intensity)
            shade = (
                tint(colour, NEUTRAL, 0.6, opacity * 0.75)
                if faint
                else tint(colour, "#ffffff", 0.2 + 0.6 * attack, opacity)
            )
            pen = QPen(shade, 1.0 + 2.4 * attack * min(1.5, self.intensity))
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            artist.setPen(pen)
            stage.surface.line(artist, x - lean / 2, bottom, x + lean / 2, top)
            if beamed(row):
                before = float(to_x(row[0] - row[C["gap"]]))
                artist.setPen(QPen(shade, 1))
                stage.surface.line(artist, x + lean / 2, top, x + lean / 2, rail)
                stage.surface.line(artist, before + lean / 2, top, before + lean / 2, rail)
                artist.setPen(QPen(shade, 2.2))
                stage.surface.line(artist, before + lean / 2, rail, x + lean / 2, rail)

    def reading(self, state):
        """One line saying how the newest activity is being read, or ''."""
        track = self.rows(state)
        if track is None:
            return ""
        row = track.last
        weights, tentative = state.attribution(self.key, row[:1])
        info = state.activity_info.get(self.key, {})
        parts = []
        if weights[0] <= 0:
            parts.append("guitar not detected: articulation not shown")
        else:
            if tentative[0]:
                parts.append(
                    "unconfirmed: the recognition model has not judged this yet"
                    if self.key in state.board.history
                    else "unconfirmed: no recognition model is running"
                )
            rate = row[C["rate_hz"]]
            if np.isfinite(rate) and row[C["regularity"]] >= BEAM_REGULARITY:
                parts.append(f"repeating, about {rate:.0f} strokes per second")
            elif row[C["level"]] > 0.45 and row[C["ring"]] < 0.2:
                parts.append("sustained")
        if info.get("basis") == "mono":
            parts.append("mono source: weaker analysis")
        return "  ·  ".join(parts)

    def paint_lane(self, artist, to_x, rows, top, bottom):
        """Detail lane: the ringing envelope as a line and every attack as a tick. The lane's
        filled band underneath is the recognition model's judgement, so nothing is hidden
        here: this is the raw activity."""
        tall = bottom - top - 3
        xs = to_x(rows[:, 0])
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        artist.setBrush(Qt.BrushStyle.NoBrush)
        artist.setPen(QPen(tint(self.profile.colour, "#ffffff", 0.55, 0.9), 1))
        artist.drawPolyline(polygon(xs, bottom - 1 - rows[:, C["ring"]] * tall))
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for x, row in zip(xs[1:].tolist(), rows[1:], strict=True):
            attack = row[C["attack"]]
            if attack > 0:
                artist.setPen(
                    QPen(tint("#ffffff", self.profile.colour, 0.15, 0.95), 2 if beamed(row) else 1)
                )
                artist.drawLine(QLineF(x, bottom - 1, x, bottom - 1 - (0.3 + 0.7 * attack) * tall))
