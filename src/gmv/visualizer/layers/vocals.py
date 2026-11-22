"""The vocal layer: a ribbon of strands that opens with a phrase and closes after it.

* How far the strands open is the level of the voice: a held note is a long open shape, a
  breath between phrases pinches it shut. A phrase therefore looks like a leaf or an eye,
  and a line of singing like a chain of them.
* The height of the ribbon is the pitch contour where the analyzer could read one, eased so
  that it glides rather than steps. Where it could not (harsh or breathy vocals, a dense
  mix) the ribbon drifts back to the middle and its strands are dotted: level only.
* A phrase start is a small open circle; a phrase end a short closing tick.

While the activity model says vocals are absent nothing is drawn; without a judgement the
strands are pale.
"""

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QPainter, QPen

from ..state import COLUMN as C
from ..strip import polygon
from . import Layer
from .guitar import NEUTRAL, runs, tint

LOW_MIDI, HIGH_MIDI = 45.0, 81.0
STRANDS = (-1.0, -0.55, 0.0, 0.55, 1.0)


class VocalsLayer(Layer):
    def __init__(self, profile):
        super().__init__(profile)
        self.place = 0.5  # eased height of the ribbon, carried from slice to slice

    def restart(self):
        self.place = 0.5

    def paint_static(self, painter, rect, stage, state):
        painter.fillRect(rect, tint(self.profile.colour, "#000000", 0.0, 0.03))
        painter.setFont(stage.font(10, QFont.Weight.DemiBold, 1.2))
        painter.setPen(tint(self.profile.colour, "#ffffff", 0.1, 0.95))
        painter.drawText(
            QPointF(rect.left() + 10 * stage.scale, rect.top() + 16 * stage.scale),
            self.profile.title.upper(),
        )

    def contour(self, rows, start):
        """Eased height (0 low .. 1 high) per row: towards the pitch where there is one,
        slowly back to the middle where there is none."""
        pitch = rows[:, C["pitch"]]
        target = np.where(
            np.isfinite(pitch), np.clip((pitch - LOW_MIDI) / (HIGH_MIDI - LOW_MIDI), 0, 1), np.nan
        )
        out = np.empty(len(rows))
        place = start
        for index, wanted in enumerate(target.tolist()):
            if wanted == wanted:  # a pitch
                place += (wanted - place) * 0.35
            else:
                place += (0.5 - place) * 0.04
            out[index] = place
        return out

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        colour = self.profile.colour
        # The first row was drawn with the slice before: start from where that left off.
        place = self.contour(rows, self.place)
        xs = np.asarray(to_x(rows[:, 0]), dtype=float)
        middle = height * (0.84 - 0.68 * place)
        opening = height * 0.3 * rows[:, C["level"]] * weights * min(1.5, self.intensity)
        unread = ~np.isfinite(rows[:, C["pitch"]])
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.style != "strands":
            artist.setPen(Qt.PenStyle.NoPen)
            artist.setBrush(tint(colour, "#ffffff", 0.0, 0.16))
            artist.drawPolygon(
                polygon(
                    np.concatenate((xs, xs[::-1])),
                    np.concatenate((middle - opening, (middle + opening)[::-1])),
                )
            )
        artist.setBrush(Qt.BrushStyle.NoBrush)
        for offset in STRANDS:
            line = middle + offset * opening
            for a, b, pale in runs(tentative | unread):
                b = min(len(xs), b + 1)
                if b - a < 2:
                    continue
                faint = bool(tentative[a])
                pen = QPen(
                    tint(colour, NEUTRAL, 0.6, 0.5)
                    if faint
                    else tint(colour, "#ffffff", 0.2 + 0.3 * (1 - abs(offset)), 0.9),
                    1,
                )
                if pale:
                    pen.setStyle(Qt.PenStyle.DotLine)
                artist.setPen(pen)
                artist.drawPolyline(polygon(xs[a:b], line[a:b]))
        for index in range(1, len(rows)):
            kind = rows[index, C["boundary"]]
            if kind and weights[index] > 0:
                x, y = xs[index], middle[index]
                artist.setPen(QPen(tint(colour, "#ffffff", 0.7, 0.95), 1.4))
                if kind == 1:
                    artist.drawEllipse(QPointF(x, y), height * 0.045, height * 0.045)
                else:
                    artist.drawLine(QLineF(x, y - height * 0.06, x, y + height * 0.06))
        self.place = float(place[-1])

    def reading(self, state):
        track = self.rows(state)
        if track is None:
            return ""
        row = track.last
        weights, tentative = state.attribution(self.key, row[:1])
        if weights[0] <= 0:
            return "vocals not detected: not shown"
        parts = ["unconfirmed: no activity model is running"] if tentative[0] else []
        if row[C["level"]] > 0.1:
            parts.append(
                "pitched" if np.isfinite(row[C["pitch"]]) else "no clear pitch: level only"
            )
        return "  ·  ".join(parts)

    def paint_live(self, painter, rect, stage, state):
        text = self.reading(state)
        if text:
            painter.setFont(stage.font(11))
            painter.setPen(tint(NEUTRAL, "#ffffff", 0.0, 0.9))
            painter.drawText(
                QRectF(
                    rect.left(),
                    rect.bottom() - 20 * stage.scale,
                    rect.width() - 10 * stage.scale,
                    20 * stage.scale,
                ),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

    def paint_lane(self, artist, to_x, rows, top, bottom):
        """Detail lane: the level as a line, and the phrase boundaries as ticks (a start
        tall, an end short)."""
        tall = bottom - top - 3
        xs = np.asarray(to_x(rows[:, 0]), dtype=float)
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        artist.setBrush(Qt.BrushStyle.NoBrush)
        artist.setPen(QPen(tint(self.profile.colour, "#ffffff", 0.6, 0.95), 1))
        artist.drawPolyline(polygon(xs, bottom - 1 - rows[:, C["level"]] * tall))
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        artist.setPen(QPen(tint("#ffffff", self.profile.colour, 0.1, 0.95), 1))
        for x, row in zip(xs[1:].tolist(), rows[1:], strict=True):
            kind = row[C["boundary"]]
            if kind:
                artist.drawLine(
                    QLineF(x, bottom - 1, x, bottom - 1 - (0.9 if kind == 1 else 0.4) * tall)
                )
