"""The visualizer window: one more consumer of the engine's EventBus.

It owns a bounded subscription, drains it from a GUI-thread timer and paints with QPainter.
It never touches capture or DSP; if painting falls behind, the bounded queue drops the
oldest messages and the display carries on from the newest.
"""

import math
from collections import deque

import numpy as np
from time import perf_counter

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (
    QColor,
    QFont,
    QLinearGradient,
    QPainter,
    QPen,
    QPixmap,
    QPolygonF,
    QRadialGradient,
)
from PySide6.QtWidgets import (
    QVBoxLayout,
    QWidget,
)

from .state import ROWS, SpectrumDisplay, VisualState
from .strip import Strip, polygon

TEXT, DIM, FAINT = QColor("#e8ebf4"), QColor("#8089a0"), QColor("#262b3a")
PREDICTED = QColor("#ffb454")  # amber is reserved for predictions
CONTEXT = QColor("#8e86c9")
SPECTRUM = ("#5b6cff", "#22d3ee", "#9ef0c9")  # low, middle, high frequencies
CONDITIONS = {
    "idle": ("IDLE", "#6b7280"),
    "waiting": ("WAITING", "#f5c451"),
    "live": ("LIVE", "#34d399"),
    "replay": ("REPLAY", "#a78bfa"),
    "stopped": ("STOPPED", "#6b7280"),
}
AHEAD = 0.08  # share of the time axis kept to the right of "now" for predictions
QUEUE = 2048


def mix(a, b, t):
    a, b = QColor(a), QColor(b)
    return QColor(
        round(a.red() + (b.red() - a.red()) * t),
        round(a.green() + (b.green() - a.green()) * t),
        round(a.blue() + (b.blue() - a.blue()) * t),
    )


def alpha(color, value):
    color = QColor(color)
    color.setAlphaF(max(0.0, min(1.0, value)))
    return color


def round_pen(color, width):
    pen = QPen(color, width)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    return pen


def spectrum_color(position):
    low, middle, high = SPECTRUM
    return mix(low, middle, position * 2) if position < 0.5 else mix(middle, high, position * 2 - 1)


class Canvas(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.display = SpectrumDisplay()
        self.mode = "bars"
        self.show_beats = True
        self.hidden = set()  # timeline rows and lane keys the user switched off
        self.info = {"source": "", "beat_tracker": None, "activity_model": None}
        self.level = self.intensity = self.flash = self.pulse = 0.0
        self.peak_level = 0.0
        self.rings = []  # [age_s, strength, predicted]
        self.now = None
        self.condition = "idle"
        self.paint_ms = deque(maxlen=240)
        self.painted = 0
        self.show_stats = False
        self._background = None
        self._colors = []
        self.lane_strip, self.level_strip, self.mark_strip = Strip(), Strip(), Strip()
        self.setMinimumSize(720, 480)

    # -- animation -------------------------------------------------------------------------

    def advance(self, dt, clock):
        state = self.state
        self.condition = state.condition(clock)
        running = self.condition in ("live", "replay")
        self.now = state.now(clock) if state.latest is not None else None
        if running and state.spectrum is not None:
            self.display.step(state.spectrum, dt, state.spectrum_range)
        else:
            self.display.rest(dt)
        newest = state.frames.last if running else None
        target, intensity = (newest[1], newest[2]) if newest is not None else (0.0, 0.0)
        rate = 0.03 if target > self.level else 0.25
        self.level += (target - self.level) * (1 - math.exp(-dt / rate))
        self.intensity += (intensity - self.intensity) * (1 - math.exp(-dt / 0.2))
        self.peak_level = max(self.level, self.peak_level - 0.25 * dt)
        self.flash *= math.exp(-dt / 0.14)
        self.pulse *= math.exp(-dt / 0.22)
        self.rings = [[age + dt, s, p] for age, s, p in self.rings if age + dt < 0.7]
        if self.now is not None:
            for mark in state.due(self.now):
                if mark.kind == "onset":
                    self.flash = max(self.flash, 0.3 + 0.7 * mark.intensity)
                    self.rings.append([0.0, 0.3 + 0.7 * mark.intensity, False])
                elif mark.style == "predicted" and self.show_beats:
                    self.pulse = 1.0
                    self.rings.append([0.0, 1.0, True])

    def settled(self):
        """True when nothing is moving any more, so the timer can slow down."""
        values = self.display.values
        return (
            self.condition not in ("live", "replay")
            and self.flash < 0.01
            and self.pulse < 0.01
            and self.level < 0.005
            and (values is None or float(values.max()) < 0.005)
        )

    # -- painting --------------------------------------------------------------------------

    def background(self):
        size = self.size() * self.devicePixelRatioF()
        if self._background is None or self._background.size() != size:
            pixmap = QPixmap(size)
            painter = QPainter(pixmap)
            fill = QLinearGradient(0, 0, 0, size.height())
            fill.setColorAt(0, QColor("#0a0c13"))
            fill.setColorAt(1, QColor("#05060a"))
            painter.fillRect(pixmap.rect(), fill)
            painter.end()
            self._background = pixmap
        return self._background

    def font(self, size, weight=QFont.Weight.Normal, spacing=0.0):
        font = QFont("Segoe UI")
        font.setPixelSize(round(size * self.scale))
        font.setWeight(weight)
        if spacing:
            font.setLetterSpacing(QFont.SpacingType.AbsoluteSpacing, spacing)
        return font

    def paintEvent(self, _event):
        began = perf_counter()
        painter = QPainter(self)
        painter.drawPixmap(self.rect(), self.background())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        width, height = self.width(), self.height()
        self.scale = max(0.85, min(1.6, height / 900))
        margin = round(28 * self.scale)
        header = round(58 * self.scale)
        rows = [row for row in ROWS if row[0] not in self.hidden and self.row_in_use(row[0])]
        lanes = [item for item in self.state.board.lanes() if item[0].key not in self.hidden]
        body = height - header - margin
        timeline_h = round(min(body * 0.24, (30 * len(rows) + 26) * self.scale)) if rows else 0
        lanes_h = round(min(body * 0.30, (30 * len(lanes) + 26) * self.scale)) if lanes else 0
        gap = round(18 * self.scale)
        spectrum = QRectF(
            margin,
            header,
            width - 2 * margin,
            body - timeline_h - lanes_h - gap * (bool(rows) + bool(lanes)),
        )
        self.draw_header(painter, QRectF(margin, 0, width - 2 * margin, header))
        self.draw_glow(painter, spectrum)
        if self.mode == "radial":
            self.draw_radial(painter, spectrum)
        else:
            self.draw_bars(painter, spectrum)
        if self.now is None:
            self.draw_idle(painter, spectrum)
        gutter = round(150 * self.scale)
        top = spectrum.bottom() + gap
        axis = QRectF(margin + gutter, top, width - 2 * margin - gutter, timeline_h + lanes_h)
        if rows:
            self.draw_timeline(painter, QRectF(margin, top, width - 2 * margin, timeline_h), rows)
            top += timeline_h + gap
            axis.setHeight(timeline_h + gap + lanes_h)
        if lanes:
            self.draw_lanes(painter, QRectF(margin, top, width - 2 * margin, lanes_h), lanes)
        if (rows or lanes) and self.now is not None:
            self.draw_now(painter, axis)
        if self.show_stats and self.paint_ms:
            painter.setFont(self.font(11))
            painter.setPen(DIM)
            painter.drawText(
                QPointF(margin, height - 8),
                f"paint {sum(self.paint_ms) / len(self.paint_ms):.1f} ms"
                f"  |  dropped by this window: {self.info.get('dropped', 0)}",
            )
        painter.end()
        self.paint_ms.append((perf_counter() - began) * 1000)
        self.painted += 1

    def row_in_use(self, row):
        # The delayed-context row only appears once such analysis has produced something.
        return row != "context" or any(mark.row == "context" for mark in self.state.markers)

    def draw_header(self, painter, rect):
        name, color = CONDITIONS[self.condition]
        middle = rect.center().y() + 4 * self.scale
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(rect.left() + 6, middle - 4 * self.scale), 4.5, 4.5)
        painter.setFont(self.font(12, QFont.Weight.DemiBold, 1.5))
        painter.setPen(QColor(color))
        painter.drawText(QPointF(rect.left() + 20, middle), name)
        offset = painter.fontMetrics().horizontalAdvance(name) + 34
        painter.setFont(self.font(13))
        painter.setPen(DIM)
        detail = self.info["source"]
        if self.condition == "waiting":
            detail = "No audio arriving" + (f"  ·  {detail}" if detail else "")
        elif self.condition == "replay":
            detail = "Recorded session, original timestamps" + (f"  ·  {detail}" if detail else "")
        elif self.condition == "stopped" and self.state.detail:
            detail = self.state.detail
        painter.drawText(QPointF(rect.left() + offset, middle), detail)
        if not self.show_beats:
            return
        right = rect.right()
        latest = self.state.latest
        locked = bool(latest is not None and latest.beat_locked and self.state.beat_tempo)
        if locked and self.condition in ("live", "replay"):
            painter.setFont(self.font(11))
            painter.setPen(DIM)
            note = f"experimental beat tracker  ·  score {self.state.beat_score:.2f} (uncalibrated)"
            painter.drawText(
                QRectF(rect.left(), middle + 2, rect.width(), 18 * self.scale),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                note,
            )
            painter.setFont(self.font(26, QFont.Weight.DemiBold))
            painter.setPen(TEXT)
            text = f"{self.state.beat_tempo:.0f} BPM"
            painter.drawText(
                QRectF(rect.left(), rect.top(), rect.width(), middle - rect.top()),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                text,
            )
            reach = painter.fontMetrics().horizontalAdvance(text) + 26 * self.scale
            radius = (7 + 5 * self.pulse) * self.scale
            painter.setPen(
                QPen(alpha(PREDICTED, 0.45 + 0.55 * self.pulse), 2, Qt.PenStyle.DashLine)
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(right - reach, middle - 12 * self.scale), radius, radius)
        else:
            painter.setFont(self.font(12))
            painter.setPen(DIM)
            if self.info["beat_tracker"] is False:
                text = "Beat tracker off  ·  it can be enabled in Settings"
            elif self.condition in ("live", "replay"):
                text = "Experimental beat tracker  ·  listening for a pulse"
            else:
                text = ""
            painter.drawText(
                QRectF(rect.left(), middle - 16 * self.scale, rect.width(), 20 * self.scale),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                text,
            )

    def colors(self, count):
        if len(self._colors) != count:
            self._colors = [spectrum_color(i / max(1, count - 1)) for i in range(count)]
        return self._colors

    def draw_glow(self, painter, rect):
        strength = 0.05 + 0.20 * self.intensity + 0.16 * self.flash
        centre = QPointF(
            rect.center().x(), rect.bottom() if self.mode == "bars" else rect.center().y()
        )
        glow = QRadialGradient(centre, rect.width() * 0.55)
        glow.setColorAt(0, alpha(mix(SPECTRUM[1], PREDICTED, 0.5 * self.pulse), strength))
        glow.setColorAt(1, alpha(SPECTRUM[0], 0))
        painter.fillRect(self.rect(), glow)

    def draw_bars(self, painter, rect):
        values, peaks = self.display.values, self.display.peaks
        count = len(values) if values is not None else 48
        mirror = rect.height() * 0.14
        base = rect.bottom() - mirror
        reach = base - rect.top() - 6
        meter = 22 * self.scale
        slot = (rect.width() - meter) / count
        bar = slot * 0.7
        radius = min(bar / 2, 4 * self.scale)
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(count):
            value = float(values[index]) if values is not None else 0.0
            color = self.colors(count)[index]
            x = rect.left() + index * slot + (slot - bar) / 2
            tall = max(2.0, value * reach)
            fill = QLinearGradient(0, base - tall, 0, base)
            fill.setColorAt(0, mix(color, "#ffffff", min(0.75, 0.12 + 0.6 * self.flash * value)))
            fill.setColorAt(1, alpha(color, 0.30))
            painter.setBrush(fill)
            painter.drawRoundedRect(QRectF(x, base - tall, bar, tall), radius, radius)
            if peaks is not None and peaks[index] > 0.01:
                painter.fillRect(
                    QRectF(x, base - float(peaks[index]) * reach - 5, bar, 2), alpha(color, 0.85)
                )
            painter.fillRect(QRectF(x, base + 3, bar, min(mirror, tall * 0.22)), alpha(color, 0.09))
        # Baseline: brightens with onset strength, turns amber on a predicted beat.
        line = mix(SPECTRUM[1], PREDICTED, self.pulse)
        painter.fillRect(
            QRectF(rect.left(), base + 0.5, rect.width() - meter, 1.5),
            alpha(line, 0.25 + 0.5 * max(self.flash, self.pulse)),
        )
        self.draw_frequencies(painter, rect, base + mirror * 0.55, slot, count)
        # Level meter: short-term level (filled) and its recent peak (tick).
        x = rect.right() - 6 * self.scale
        painter.fillRect(QRectF(x, rect.top() + 6, 4 * self.scale, reach), alpha(FAINT, 0.7))
        painter.fillRect(
            QRectF(x, base - self.level * reach, 4 * self.scale, self.level * reach),
            alpha(TEXT, 0.75),
        )
        painter.fillRect(
            QRectF(x - 2, base - self.peak_level * reach - 1, 4 * self.scale + 4, 2), TEXT
        )

    def draw_frequencies(self, painter, rect, y, slot, count):
        span = self.state.spectrum_range
        if not span:
            return
        painter.setFont(self.font(10))
        painter.setPen(alpha(DIM, 0.7))
        low, high = span
        for hertz, text in (
            (60, "60 Hz"),
            (250, "250"),
            (1000, "1 kHz"),
            (4000, "4 kHz"),
            (12000, "12 kHz"),
        ):
            if low < hertz < high:
                x = rect.left() + math.log(hertz / low) / math.log(high / low) * count * slot
                painter.drawText(QPointF(x - 14, y), text)

    def draw_radial(self, painter, rect):
        values = self.display.values
        count = len(values) if values is not None else 48
        centre = rect.center()
        outer = min(rect.width(), rect.height()) * 0.54
        inner = outer * (0.27 + 0.07 * self.level + 0.03 * self.pulse)
        reach = outer - inner - 10
        bar = math.pi * inner / count * 0.62
        core = QRadialGradient(centre, inner)
        core.setColorAt(
            0, alpha(mix(SPECTRUM[1], "#ffffff", 0.5 * self.flash), 0.10 + 0.5 * self.intensity)
        )
        core.setColorAt(1, alpha(SPECTRUM[0], 0.04))
        painter.setPen(QPen(alpha(SPECTRUM[1], 0.35 + 0.4 * self.flash), 1.5))
        painter.setBrush(core)
        painter.drawEllipse(centre, inner - 6, inner - 6)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for age, strength, predicted in self.rings:
            progress = age / 0.7
            radius = inner + (outer - inner) * progress
            if predicted:
                pen = QPen(alpha(PREDICTED, 0.8 * (1 - progress)), 2, Qt.PenStyle.DashLine)
            else:
                pen = QPen(alpha(TEXT, 0.5 * strength * (1 - progress)), 1.5)
            painter.setPen(pen)
            painter.drawEllipse(centre, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(count):
            value = float(values[index]) if values is not None else 0.0
            color = spectrum_color(index / max(1, count - 1))
            painter.setBrush(mix(color, "#ffffff", min(0.7, 0.5 * self.flash * value)))
            angle = 180 * (index + 0.5) / count  # lows at the top, highs at the bottom
            for side in (1, -1):
                painter.save()
                painter.translate(centre)
                painter.rotate(-90 + side * angle)
                painter.drawRoundedRect(
                    QRectF(inner, -bar / 2, max(2.0, value * reach), bar), bar / 2, bar / 2
                )
                painter.restore()

    def draw_idle(self, painter, rect):
        painter.setFont(self.font(20, QFont.Weight.DemiBold))
        painter.setPen(alpha(TEXT, 0.85))
        stopped = self.condition == "stopped"
        painter.drawText(
            rect.adjusted(0, 0, 0, -rect.height() * 0.2),
            Qt.AlignmentFlag.AlignCenter,
            "Stopped" if stopped else "Ready",
        )
        painter.setFont(self.font(13))
        painter.setPen(DIM)
        painter.drawText(
            rect.adjusted(0, 56 * self.scale, 0, -rect.height() * 0.2),
            Qt.AlignmentFlag.AlignCenter,
            "Analysis stopped. Press Start to begin again"
            if stopped
            else "Press Start to visualize what is playing on this PC",
        )

    # -- time axis ---------------------------------------------------------------------------

    def axis(self, rect):
        """Map audio time to x inside ``rect``; returns (function, earliest time, x of now)."""
        window = self.state.window_s
        start = self.now - window
        per_second = rect.width() / (window * (1 + AHEAD))
        return (
            (lambda t: rect.left() + (t - start) * per_second),
            start,
            rect.left() + window * per_second,
        )

    def draw_now(self, painter, rect):
        _, _, x = self.axis(rect)
        painter.fillRect(
            QRectF(x, rect.top(), rect.right() - x, rect.height()), alpha(PREDICTED, 0.035)
        )
        painter.setPen(QPen(alpha(TEXT, 0.55), 1))
        painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
        painter.setFont(self.font(10))
        painter.setPen(DIM)
        painter.drawText(QPointF(x - 26 * self.scale, rect.top() - 5), "now")
        painter.setPen(alpha(PREDICTED, 0.8))
        painter.drawText(QPointF(x + 6, rect.top() - 5), "ahead")

    def caption(self, painter, rect, text):
        painter.setFont(self.font(10, QFont.Weight.DemiBold, 1.2))
        painter.setPen(DIM)
        painter.drawText(QPointF(rect.left(), rect.top() + 11 * self.scale), text)

    def row_label(self, painter, rect, text, color=DIM):
        painter.setFont(self.font(12))
        painter.setPen(color)
        painter.drawText(
            QRectF(rect.left(), rect.top(), 140 * self.scale, rect.height()),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            text,
        )

    def draw_timeline(self, painter, rect, rows):
        self.caption(painter, rect, "TIMELINE")
        self.draw_legend(painter, rect)
        head = 22 * self.scale
        gutter = 150 * self.scale
        area = QRectF(
            rect.left() + gutter, rect.top() + head, rect.width() - gutter, rect.height() - head
        )
        row_height = area.height() / len(rows)
        painter.fillRect(area, alpha(FAINT, 0.28))
        for index, (_key, title) in enumerate(rows):
            band = QRectF(area.left(), area.top() + index * row_height, area.width(), row_height)
            self.row_label(painter, QRectF(rect.left(), band.top(), gutter, row_height), title)
            if index:
                painter.fillRect(
                    QRectF(band.left(), band.top(), band.width(), 1), alpha(FAINT, 0.9)
                )
        if self.now is None:
            return
        to_x, start, _ = self.axis(area)
        per_second = area.width() / (self.state.window_s * (1 + AHEAD))
        painter.save()
        painter.setClipRect(area)
        # Loudness over time, as a quiet backdrop for the marks.
        frames = self.state.frames

        def draw_level(artist, at, since, tall):
            history = frames.since(start if since is None else since)
            if len(history) < 2:
                return
            xs = at(history[:, 0])
            artist.setPen(Qt.PenStyle.NoPen)
            artist.setBrush(alpha(SPECTRUM[1], 0.08))
            artist.drawPolygon(
                polygon(
                    np.concatenate(([xs[0]], xs, [xs[-1]])),
                    np.concatenate(([tall], tall - history[:, 1] * tall, [tall])),
                )
            )

        self.level_strip.paint(
            painter, area, start, per_second, self.state.generation, frames.last[0], draw_level
        )
        for second in range(5 * math.ceil(start / 5), math.floor(self.now) + 1, 5):
            painter.fillRect(QRectF(to_x(second), area.top(), 1, area.height()), alpha(FAINT, 0.9))
        for first, last in self.state.gaps:
            if last >= start:
                painter.fillRect(
                    QRectF(to_x(first), area.top(), to_x(last) - to_x(first), area.height()),
                    QColor(0, 0, 0, 150),
                )
        # Marks are drawn once, when they arrive, at their own audio time.
        markers, shown = self.state.markers, [key for key, _ in rows]

        def draw_marks(artist, at, since, tall):
            artist.setRenderHint(QPainter.RenderHint.Antialiasing)
            fresh = len(markers) if since is None else min(len(markers), self.state.marked - since)
            for mark in list(markers)[len(markers) - fresh :]:
                x = at(mark.time)
                if mark.row == "state":
                    artist.setPen(QPen(alpha(DIM, 0.5), 1, Qt.PenStyle.DotLine))
                    artist.drawLine(QPointF(x, 0), QPointF(x, tall))
                elif mark.row in shown and (self.show_beats or mark.row != "beats"):
                    band = QRectF(0, shown.index(mark.row) * row_height, area.width(), row_height)
                    self.draw_mark(artist, mark, x, band, at)

        key = (self.state.generation, tuple(shown), self.show_beats)
        self.mark_strip.paint(painter, area, start, per_second, key, self.state.marked, draw_marks)
        painter.restore()

    def draw_mark(self, painter, mark, x, band, to_x):
        middle = band.center().y()
        size = band.height() * (0.12 + 0.08 * mark.intensity)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        if mark.style == "context":
            end = to_x(mark.end) if mark.end else x + 6
            box = QRectF(
                x, band.top() + band.height() * 0.25, max(4.0, end - x), band.height() * 0.5
            )
            painter.setPen(QPen(alpha(CONTEXT, 0.8), 1, Qt.PenStyle.DotLine))
            painter.setBrush(alpha(CONTEXT, 0.10 + 0.25 * mark.intensity))
            painter.drawRect(box)
        elif mark.style == "predicted":
            painter.setPen(QPen(alpha(PREDICTED, 0.75), 1.2, Qt.PenStyle.DashLine))
            painter.drawLine(QPointF(x, band.top() + 2), QPointF(x, band.bottom() - 2))
            reach = band.height() * 0.22
            painter.setPen(QPen(PREDICTED, 1.4))
            painter.setBrush(QColor("#0a0c13"))
            painter.drawPolygon(
                QPolygonF(
                    [
                        QPointF(x, middle - reach),
                        QPointF(x + reach, middle),
                        QPointF(x, middle + reach),
                        QPointF(x - reach, middle),
                    ]
                )
            )
        elif mark.style == "observed":
            tall = band.height() * (0.3 + 0.6 * mark.intensity)
            painter.setPen(round_pen(alpha(TEXT, 0.55 + 0.45 * mark.intensity), 2))
            painter.drawLine(QPointF(x, middle - tall / 2), QPointF(x, middle + tall / 2))
        else:  # candidate: always hollow
            if mark.region is not None:
                middle = band.bottom() - band.height() * (0.22 + 0.28 * mark.region)
                color = QColor(SPECTRUM[mark.region])
                size = band.height() * (0.09 + 0.07 * mark.intensity)
            elif mark.row == "beats":
                color = PREDICTED
            else:
                color = QColor(SPECTRUM[1]) if mark.kind == "spectral_change" else TEXT
            painter.setPen(QPen(alpha(color, 0.8), 1.3))
            if mark.kind == "energy_rise":
                painter.drawPolygon(
                    QPolygonF(
                        [
                            QPointF(x, middle - size),
                            QPointF(x + size, middle + size),
                            QPointF(x - size, middle + size),
                        ]
                    )
                )
            elif mark.kind == "spectral_change":
                painter.drawRect(QRectF(x - size, middle - size, 2 * size, 2 * size))
            else:
                painter.drawEllipse(QPointF(x, middle), size, size)

    def draw_legend(self, painter, rect):
        painter.setFont(self.font(10))
        y = rect.top() + 7 * self.scale
        x = rect.left() + 158 * self.scale
        for text, style in (
            ("observed", "observed"),
            ("candidate (heuristic)", "candidate"),
            ("predicted", "predicted"),
            ("delayed context", "context"),
        ):
            painter.setPen(DIM)
            painter.drawText(QPointF(x + 12 * self.scale, y + 4 * self.scale), text)
            advance = painter.fontMetrics().horizontalAdvance(text) + 34 * self.scale
            painter.setBrush(Qt.BrushStyle.NoBrush)
            if style == "observed":
                painter.setPen(round_pen(TEXT, 2))
                painter.drawLine(QPointF(x, y - 5), QPointF(x, y + 5))
            elif style == "candidate":
                painter.setPen(QPen(TEXT, 1.3))
                painter.drawEllipse(QPointF(x, y), 4, 4)
            elif style == "predicted":
                painter.setPen(QPen(PREDICTED, 1.4))
                painter.drawPolygon(
                    QPolygonF(
                        [QPointF(x, y - 5), QPointF(x + 5, y), QPointF(x, y + 5), QPointF(x - 5, y)]
                    )
                )
            else:
                painter.setPen(QPen(CONTEXT, 1, Qt.PenStyle.DotLine))
                painter.setBrush(alpha(CONTEXT, 0.25))
                painter.drawRect(QRectF(x - 6, y - 4, 12, 8))
            x += advance

    def draw_lanes(self, painter, rect, lanes):
        method = self.state.board.method
        self.caption(
            painter,
            rect,
            "INSTRUMENT ACTIVITY"
            + (f"   ·   {method}   ·   scores are uncalibrated" if method else ""),
        )
        head = 22 * self.scale
        gutter = 150 * self.scale
        area = QRectF(
            rect.left() + gutter, rect.top() + head, rect.width() - gutter, rect.height() - head
        )
        lane_height = area.height() / len(lanes)
        for index, (lane, live) in enumerate(lanes):
            band = QRectF(
                area.left(), area.top() + index * lane_height + 2, area.width(), lane_height - 4
            )
            label = QRectF(
                rect.left() + (14 * self.scale if lane.parent else 0),
                band.top(),
                gutter,
                band.height(),
            )
            self.row_label(painter, label, lane.title, TEXT if live else DIM)
            painter.fillRect(band, alpha(FAINT, 0.28))
            if live:
                continue
            painter.setFont(self.font(11))
            painter.setPen(alpha(DIM, 0.8))
            if lane.key in ("drums", "bass", "vocals") and not method:
                reason = (
                    "no detector output  ·  choose a weights file in Settings"
                    if not self.info["activity_model"]
                    else "no detector output yet"
                )
            else:
                reason = "no detector for this instrument yet"
            painter.drawText(band.adjusted(12, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, reason)
        if self.now is None or not any(live for _, live in lanes):
            return
        history = self.state.board.history
        _, start, _ = self.axis(area)

        def draw(artist, to_x, since, _height):
            for index, (lane, live) in enumerate(lanes):
                rows = history[lane.key].since(start if since is None else since) if live else ()
                if len(rows) < 2:
                    continue
                bottom = (index + 1) * lane_height - 2
                xs = to_x(rows[:, 0])
                ys = bottom - np.maximum(0.03, rows[:, 1]) * (lane_height - 7)
                edge_x = np.concatenate(([xs[0]], xs, [xs[-1]]))
                color = QColor(lane.color)
                # Fills without smoothing so that consecutive slices meet without a seam.
                artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                artist.setPen(Qt.PenStyle.NoPen)
                for heights, strength in ((ys, 0.14), (np.where(rows[:, 2] > 0, ys, bottom), 0.5)):
                    artist.setBrush(alpha(color, strength))
                    artist.drawPolygon(
                        polygon(edge_x, np.concatenate(([bottom], heights, [bottom])))
                    )
                artist.setRenderHint(QPainter.RenderHint.Antialiasing, True)
                artist.setPen(QPen(alpha(color, 0.9), 1))
                artist.drawPolyline(polygon(xs, ys))

        key = (self.state.generation, tuple((lane.key, live) for lane, live in lanes))
        per_second = area.width() / (self.state.window_s * (1 + AHEAD))
        self.lane_strip.paint(
            painter, area, start, per_second, key, self.state.latest.timestamp, draw
        )


class VisualizerWindow(QWidget):
    """The visualizer surface. ``attach`` it to any EventBus: a running engine or a replay.

    It is the main view of the application and also works on its own as a window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Granular Music Visualizer")
        self.resize(1440, 900)
        self.setObjectName("visualizer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet("QWidget#visualizer { background: #05060a; }")
        self.state = VisualState()
        self.canvas = Canvas(self.state)
        self.bus = self.subscription = None
        self.dropped = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.timer.timeout.connect(self.tick)
        self.last_tick = perf_counter()
        self.timer.start(16)

    # -- display options -----------------------------------------------------------------------

    def set_mode(self, mode):
        self.canvas.mode = mode
        self.canvas.update()

    def set_beats(self, shown):
        self.canvas.show_beats = bool(shown)
        self.canvas.update()

    def set_smoothing(self, seconds):
        self.canvas.display.release_s = seconds

    def set_gain(self, decibels):
        self.canvas.display.gain_db = decibels

    def set_window(self, seconds):
        """Length of the timeline and lanes."""
        self.state.window_s = float(seconds)
        self.canvas.update()

    def set_shown(self, key, shown):
        (self.canvas.hidden.discard if shown else self.canvas.hidden.add)(key)
        self.canvas.update()

    def toggle_fullscreen(self):
        top = self.window()
        if top.isFullScreen():
            top.showNormal()
        else:
            top.showFullScreen()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape and self.window().isFullScreen():
            self.window().showNormal()
        elif event.key() == Qt.Key.Key_D:
            self.canvas.show_stats = not self.canvas.show_stats
        else:
            super().keyPressEvent(event)

    # -- stream ------------------------------------------------------------------------------

    def attach(self, bus, *, source="", beat_tracker=None, activity_model=None):
        """Follow ``bus`` with a subscription of this window's own. Replaces any earlier one."""
        self.detach()
        self.bus, self.subscription = bus, bus.subscribe(QUEUE)
        self.state.status, self.state.detail = "starting", ""
        self.state.reset()
        self.canvas.info.update(
            source=source, beat_tracker=beat_tracker, activity_model=activity_model, dropped=0
        )
        self.timer.start(16)

    def detach(self):
        if self.subscription is not None:
            self.dropped += self.subscription.dropped
            self.bus.unsubscribe(self.subscription)
        self.bus = self.subscription = None

    def tick(self):
        clock = perf_counter()
        dt, self.last_tick = min(0.25, clock - self.last_tick), clock
        if self.subscription is not None:
            for _ in range(QUEUE):
                message = self.subscription.get(timeout=0)
                if message is None:
                    break
                self.state.consume(message)
            self.canvas.info["dropped"] = self.dropped + self.subscription.dropped
        self.canvas.advance(dt, clock)
        self.canvas.update()
        # Nothing arriving and nothing moving: check a few times a second instead of 60.
        interval = 250 if self.canvas.settled() else 16
        if self.timer.interval() != interval:
            self.timer.start(interval)

    def closeEvent(self, event):
        self.detach()
        self.timer.stop()
        event.accept()

    def showEvent(self, event):
        self.last_tick = perf_counter()
        if not self.timer.isActive():
            self.timer.start(16)
        super().showEvent(event)
