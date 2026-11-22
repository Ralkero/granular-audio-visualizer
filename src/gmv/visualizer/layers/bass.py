"""The bass layer: one broad, heavy ribbon.

* Its height is the note: low notes low, an octave up clearly higher. It moves in steps,
  because a bass plays notes; while no note can be read (a dense passage, a kick) it stays
  where the last note was and is drawn paler, so nothing is invented.
* Its thickness is the weight of the periodic low-register sound. A held note is a long
  slab; silence closes it to nothing.
* A new note, or the same note struck again, is a bright edge across the ribbon.

While the activity model says bass is absent nothing is drawn; without a judgement the
ribbon is an outline.
"""

import numpy as np
from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QFont, QPainter, QPen

from ..state import COLUMN as C
from ..strip import polygon
from . import Layer
from .guitar import NEUTRAL, runs, tint

LOW_MIDI, HIGH_MIDI = 26.0, 56.0
NAMES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")


def note_name(midi):
    nearest = int(round(midi))
    return f"{NAMES[nearest % 12]}{nearest // 12 - 1}"


def held(values, start):
    """``values`` with every NaN replaced by the last number before it (``start`` first)."""
    out = values.copy()
    known = np.isfinite(out)
    index = np.where(known, np.arange(len(out)), -1)
    np.maximum.accumulate(index, out=index)
    filled = np.where(index >= 0, out[np.maximum(index, 0)], start)
    return filled


class BassLayer(Layer):
    def __init__(self, profile):
        super().__init__(profile)
        self.last_pitch = (LOW_MIDI + HIGH_MIDI) / 2  # where the ribbon rests between slices

    def restart(self):
        self.last_pitch = (LOW_MIDI + HIGH_MIDI) / 2

    def paint_static(self, painter, rect, stage, state):
        painter.fillRect(rect, tint(self.profile.colour, "#000000", 0.0, 0.03))
        painter.setFont(stage.font(10, QFont.Weight.DemiBold, 1.2))
        painter.setPen(tint(self.profile.colour, "#ffffff", 0.1, 0.95))
        painter.drawText(
            QPointF(rect.left() + 10 * stage.scale, rect.top() + 16 * stage.scale),
            self.profile.title.upper(),
        )

    def shape(self, rows, weights, height):
        """Centre line and half-thickness of the ribbon for ``rows``."""
        pitch = held(rows[:, C["pitch"]], self.last_pitch)
        place = np.clip((pitch - LOW_MIDI) / (HIGH_MIDI - LOW_MIDI), 0.0, 1.0)
        middle = height * (0.82 - 0.64 * place)
        half = height * 0.22 * rows[:, C["level"]] * weights * min(1.5, self.intensity)
        return pitch, middle, half

    def paint_history(self, artist, to_x, rows, weights, tentative, height, stage):
        colour = self.profile.colour
        pitch, middle, half = self.shape(rows, weights, height)
        xs = np.asarray(to_x(rows[:, 0]), dtype=float)
        # Steps, not slopes: each row holds its height until the next one.
        step_x = np.repeat(xs, 2)[1:]
        step_mid = np.repeat(middle, 2)[:-1]
        step_half = np.repeat(half, 2)[:-1]
        unread = ~np.isfinite(rows[:, C["pitch"]])
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        for a, b, pale in runs(tentative | unread):
            lo, hi = 2 * a, min(len(step_x), 2 * b + 1)
            if hi - lo < 2:
                continue
            outline_only = bool(tentative[a])
            artist.setPen(Qt.PenStyle.NoPen)
            if outline_only:
                artist.setBrush(Qt.BrushStyle.NoBrush)
            else:
                artist.setBrush(tint(colour, NEUTRAL, 0.45 if pale else 0.0, 0.3 if pale else 0.6))
                artist.drawPolygon(
                    polygon(
                        np.concatenate((step_x[lo:hi], step_x[lo:hi][::-1])),
                        np.concatenate(
                            ((step_mid - step_half)[lo:hi], (step_mid + step_half)[lo:hi][::-1])
                        ),
                    )
                )
            pen = QPen(tint(colour, "#ffffff", 0.25, 0.55 if pale else 0.95), 1)
            if outline_only:
                pen.setStyle(Qt.PenStyle.DashLine)
            artist.setBrush(Qt.BrushStyle.NoBrush)
            artist.setPen(pen)
            artist.drawPolyline(polygon(step_x[lo:hi], (step_mid - step_half)[lo:hi]))
            artist.drawPolyline(polygon(step_x[lo:hi], (step_mid + step_half)[lo:hi]))
        for index in range(1, len(rows)):
            attack = rows[index, C["attack"]]
            if attack > 0 and weights[index] > 0 and half[index] > 0.5:
                artist.setPen(QPen(tint(colour, "#ffffff", 0.75, 0.5 + 0.5 * attack), 2.0))
                artist.drawLine(
                    QLineF(
                        xs[index],
                        middle[index] - half[index] - 2,
                        xs[index],
                        middle[index] + half[index] + 2,
                    )
                )
        self.last_pitch = float(pitch[-1])

    def reading(self, state):
        track = self.rows(state)
        if track is None:
            return ""
        row = track.last
        weights, tentative = state.attribution(self.key, row[:1])
        if weights[0] <= 0:
            return "bass not detected: not shown"
        parts = ["unconfirmed: no activity model is running"] if tentative[0] else []
        if np.isfinite(row[C["pitch"]]) and row[C["level"]] > 0.1:
            parts.append(f"around {note_name(row[C['pitch']])}")
        elif row[C["level"]] > 0.1:
            parts.append("no clear note")
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
        """Detail lane: the note as a stepped line (low notes low) and a tick per new note."""
        tall = bottom - top - 3
        xs = np.asarray(to_x(rows[:, 0]), dtype=float)
        pitch = rows[:, C["pitch"]]
        place = np.clip((pitch - LOW_MIDI) / (HIGH_MIDI - LOW_MIDI), 0.0, 1.0)
        artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        artist.setBrush(Qt.BrushStyle.NoBrush)
        artist.setPen(QPen(tint(self.profile.colour, "#ffffff", 0.6, 0.95), 2))
        for a, b, known in runs(np.isfinite(pitch)):
            if known and b - a >= 2:
                artist.drawPolyline(
                    polygon(
                        np.repeat(xs[a:b], 2)[1:], np.repeat(bottom - 2 - place[a:b] * tall, 2)[:-1]
                    )
                )
        artist.setPen(QPen(tint("#ffffff", self.profile.colour, 0.1, 0.95), 1))
        for x, row in zip(xs[1:].tolist(), rows[1:], strict=True):
            if row[C["attack"]] > 0:
                artist.drawLine(QLineF(x, bottom - 1, x, bottom - 1 - 0.3 * tall))
