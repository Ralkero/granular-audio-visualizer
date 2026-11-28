"""The Sonic Character Monitor: a diagnostic window that shows what the live analysis measures.

It reads ``LiveCharacter`` snapshots four times a second while it is open and draws:

* the seven dimensions of one window (short 4 s, medium 20 s or long 90 s), each as a bar on a
  labelled 0-1 scale with its value written beside it, and its confidence as a second, thinner
  bar and a number;
* "not enough signal" in place of a value that is missing or whose confidence is under 0.3
  (a display threshold only; the measurements are Phase A's, unchanged);
* a rolling history of the last two minutes of audio, one line per dimension, drawn faint
  where its confidence is low and broken where there was no value;
* what state the stream is in (waiting, measuring, held through silence, stopped) and the
  profile's warnings in words.

It never touches the visualizer, the analysis or the audio: it only reads snapshots. Closing
it changes nothing else; the live analysis keeps its history while it is closed.
"""

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget

from ..character.live import HISTORY_S, LOW_CONFIDENCE
from ..character.profile import DIMENSION_NAMES, DIMENSIONS

REFRESH_MS = 250
WINDOW_TITLES = {
    "short": "Short (last 4 s)",
    "medium": "Medium (last 20 s)",
    "long": "Long (last 90 s)",
}
WARNING_TEXT = {
    "little_music": "Under five seconds of music so far: every value is provisional.",
    "held": "No music at this moment: the values are held as they were.",
    "quiet_capture": "The level is close to the silence threshold: quiet passages may be missed.",
    "no_display_spectrum": "The frames carry no display spectrum: brightness and density rest on less.",
    "no_onset_events": "No onset events: regularity rests on the pulse alone.",
}
BACKGROUND, TRACK, GRID = QColor("#0d1018"), QColor("#1c2030"), QColor("#262b3a")
TEXT, MUTED, ACCENT, FAINT = (
    QColor("#e8ebf4"),
    QColor("#8089a0"),
    QColor("#22d3ee"),
    QColor("#3b4460"),
)
LINE_COLOURS = ("#f97316", "#facc15", "#4ade80", "#22d3ee", "#a78bfa", "#f472b6", "#94a3b8")


def title(name):
    return name.capitalize()


class Readout(QWidget):
    """The seven current values and confidences, drawn on a 0-1 scale."""

    ROW = 34

    def __init__(self, parent=None):
        super().__init__(parent)
        self.reading = None
        self.setMinimumHeight(28 + self.ROW * len(DIMENSION_NAMES))
        self.setMinimumWidth(460)

    def set_reading(self, reading):
        self.reading = reading
        self.setAccessibleDescription(self.describe())
        self.update()

    def describe(self):
        """The same content as text (screen readers, tooltips, tests)."""
        if self.reading is None:
            return "No values yet."
        parts = []
        for index, name in enumerate(DIMENSION_NAMES):
            value, confidence = self.reading.values[index], self.reading.confidences[index]
            if self.reading.usable(index):
                parts.append(f"{title(name)} {value:.2f}, confidence {confidence:.2f}")
            else:
                parts.append(f"{title(name)}: not enough signal (confidence {confidence:.2f})")
        return "; ".join(parts) + "."

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), BACKGROUND)
        width = self.width()
        left, right = 104, width - 150
        span = max(40, right - left)
        small = QFont(self.font())
        small.setPointSizeF(max(7.0, small.pointSizeF() - 1))
        painter.setFont(small)
        painter.setPen(MUTED)
        for tick, label in ((0.0, "0"), (0.25, ""), (0.5, "0.5"), (0.75, ""), (1.0, "1")):
            x = left + tick * span
            if label:
                painter.drawText(QRectF(x - 20, 2, 40, 16), Qt.AlignmentFlag.AlignCenter, label)
            painter.setPen(QPen(GRID, 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(x, 20), QPointF(x, self.height() - 4))
            painter.setPen(MUTED)
        painter.setFont(self.font())
        for index, name in enumerate(DIMENSION_NAMES):
            top = 24 + index * self.ROW
            painter.setPen(TEXT)
            painter.drawText(
                QRectF(8, top, left - 14, 18), Qt.AlignmentFlag.AlignVCenter, title(name)
            )
            painter.fillRect(QRectF(left, top + 3, span, 12), TRACK)
            painter.fillRect(QRectF(left, top + 20, span, 4), TRACK)
            reading = self.reading
            if reading is None:
                painter.setPen(MUTED)
                painter.drawText(
                    QRectF(right + 10, top, 140, 18), Qt.AlignmentFlag.AlignVCenter, "waiting"
                )
                continue
            value, confidence = reading.values[index], reading.confidences[index]
            painter.fillRect(
                QRectF(left, top + 20, span * confidence, 4),
                FAINT if confidence < LOW_CONFIDENCE else MUTED,
            )
            if reading.usable(index):
                painter.fillRect(QRectF(left, top + 3, span * value, 12), ACCENT)
                painter.setPen(TEXT)
                text = f"{value:.2f}"
            else:
                painter.setPen(QPen(FAINT, 1, Qt.PenStyle.DashLine))
                painter.drawRect(QRectF(left, top + 3, span, 12))
                painter.setPen(MUTED)
                text = "not enough signal"
            painter.drawText(QRectF(right + 10, top, 140, 18), Qt.AlignmentFlag.AlignVCenter, text)
            painter.setPen(MUTED)
            painter.setFont(small)
            painter.drawText(
                QRectF(right + 10, top + 15, 140, 14),
                Qt.AlignmentFlag.AlignVCenter,
                f"confidence {confidence:.2f}",
            )
            painter.setFont(self.font())
        painter.end()


class HistoryPlot(QWidget):
    """The last two minutes of audio, one line per dimension on a 0-1 scale."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.history, self.window = (), "short"
        self.setMinimumHeight(190)

    def set_history(self, history, window):
        self.history, self.window = history, window
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), BACKGROUND)
        legend_w = 104
        plot = QRectF(
            legend_w, 8, max(40, self.width() - legend_w - 12), max(40, self.height() - 30)
        )
        painter.setPen(QPen(GRID, 1))
        painter.drawRect(plot)
        for level in (0.25, 0.5, 0.75):
            y = plot.bottom() - level * plot.height()
            painter.setPen(QPen(GRID, 1, Qt.PenStyle.DotLine))
            painter.drawLine(QPointF(plot.left(), y), QPointF(plot.right(), y))
        painter.setPen(MUTED)
        painter.drawText(
            QRectF(plot.left() - 30, plot.top() - 6, 26, 14), Qt.AlignmentFlag.AlignRight, "1"
        )
        painter.drawText(
            QRectF(plot.left() - 30, plot.bottom() - 8, 26, 14), Qt.AlignmentFlag.AlignRight, "0"
        )
        painter.drawText(
            QRectF(plot.left(), plot.bottom() + 4, 120, 16),
            Qt.AlignmentFlag.AlignLeft,
            f"{int(HISTORY_S)} s ago",
        )
        painter.drawText(
            QRectF(plot.right() - 60, plot.bottom() + 4, 60, 16), Qt.AlignmentFlag.AlignRight, "now"
        )
        for index, name in enumerate(DIMENSION_NAMES):
            colour = QColor(LINE_COLOURS[index])
            painter.setPen(QPen(colour, 2))
            painter.drawLine(QPointF(8, 14 + index * 18), QPointF(22, 14 + index * 18))
            painter.setPen(TEXT)
            painter.drawText(
                QRectF(28, 6 + index * 18, legend_w - 30, 16),
                Qt.AlignmentFlag.AlignVCenter,
                title(name),
            )
        if not self.history:
            painter.setPen(MUTED)
            painter.drawText(plot, Qt.AlignmentFlag.AlignCenter, "No history yet")
            painter.end()
            return
        now = self.history[-1].time
        for index in range(len(DIMENSION_NAMES)):
            colour = QColor(LINE_COLOURS[index])
            strong, weak = QPainterPath(), QPainterPath()
            pen_down = {"strong": False, "weak": False}
            for snap in self.history:
                reading = snap.readings.get(self.window)
                if reading is None or snap.time is None:
                    continue
                x = plot.right() - (now - snap.time) / HISTORY_S * plot.width()
                if x < plot.left():
                    continue
                value, confidence = reading.values[index], reading.confidences[index]
                for key, path in (("strong", strong), ("weak", weak)):
                    wanted = value is not None and (
                        (confidence >= LOW_CONFIDENCE) == (key == "strong")
                    )
                    if not wanted:
                        pen_down[key] = False
                        continue
                    point = QPointF(x, plot.bottom() - value * plot.height())
                    if pen_down[key]:
                        path.lineTo(point)
                    else:
                        path.moveTo(point)
                        pen_down[key] = True
            painter.setPen(QPen(colour, 2))
            painter.drawPath(strong)
            faint = QColor(colour)
            faint.setAlpha(70)
            painter.setPen(QPen(faint, 1, Qt.PenStyle.DashLine))
            painter.drawPath(weak)
        painter.end()


class CharacterMonitor(QWidget):
    """The monitor window. ``live`` is the ``LiveCharacter`` to read."""

    visibility_changed = Signal(bool)

    def __init__(self, live, parent=None, style=""):
        super().__init__(parent, Qt.WindowType.Window)
        self.setObjectName("app")
        self.setWindowTitle("Sonic Character Monitor")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        if style:
            self.setStyleSheet(style)
        self.live = live
        self.shown_snapshot = self.shown_history_end = None
        self.refreshes = self.redraws = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(8)
        head = QHBoxLayout()
        head.addWidget(QLabel("Sonic Character", objectName="title"))
        self.status = QLabel(objectName="value")
        head.addWidget(self.status, 1)
        self.window_choice = QComboBox()
        for key, text in WINDOW_TITLES.items():
            self.window_choice.addItem(text, key)
        self.window_choice.setToolTip(
            "Which running average to show: the last few seconds, a passage, or the track's identity"
        )
        self.window_choice.currentIndexChanged.connect(self.force_refresh)
        head.addWidget(self.window_choice)
        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.close)
        head.addWidget(self.close_button)
        layout.addLayout(head)
        self.readout = Readout()
        self.readout.setAccessibleName("Current Sonic Character values")
        layout.addWidget(self.readout)
        self.plot = HistoryPlot()
        self.plot.setAccessibleName("Sonic Character history, last two minutes")
        layout.addWidget(self.plot, 1)
        self.warnings = QLabel(objectName="value")
        self.warnings.setWordWrap(True)
        layout.addWidget(self.warnings)
        note = QLabel(
            "Values are Phase A's measurements of the sound (0 to 1 between fixed anchors). Confidence is "
            "how much signal supports a value, not the chance that it is right. Diagnostic only: nothing "
            "here changes the visualizer."
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        self.counters = QLabel()
        layout.addWidget(self.counters)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.resize(640, 640)

    @property
    def window_key(self):
        return self.window_choice.currentData()

    def showEvent(self, event):
        super().showEvent(event)
        self.force_refresh()
        self.timer.start(REFRESH_MS)
        self.visibility_changed.emit(True)

    def hideEvent(self, event):
        super().hideEvent(event)
        self.timer.stop()
        self.visibility_changed.emit(False)

    def force_refresh(self, *_):
        self.shown_snapshot = self.shown_history_end = None
        self.refresh()

    def state_text(self, snapshot):
        if snapshot is None:
            return (
                "Waiting for music"
                if self.live.attached
                else "Not running: start analysis or a replay"
            )
        reading = snapshot.readings[self.window_key]
        prefix = "" if self.live.attached else "Stopped, last values: "
        if not reading.active:
            return prefix + "No music at the moment (values held)"
        return prefix + f"Measuring · {reading.heard_s:.0f} s of music heard"

    def refresh(self):
        """Timer slot: redraw only when there is something new to show."""
        self.refreshes += 1
        snapshot, history = self.live.snapshot(), self.live.history()
        end = history[-1] if history else None
        self.status.setText(self.state_text(snapshot))
        stats = self.live.stats
        per_frame = stats.update_s / stats.frames * 1e6 if stats.frames else 0.0
        self.counters.setText(
            f"frames {stats.frames} · dropped {self.live.dropped} · waiting {self.live.backlog} · "
            f"{per_frame:.0f} µs per frame"
        )
        if snapshot is self.shown_snapshot and end is self.shown_history_end:
            return
        self.shown_snapshot, self.shown_history_end = snapshot, end
        self.redraws += 1
        reading = None if snapshot is None else snapshot.readings[self.window_key]
        self.readout.set_reading(reading)
        self.plot.set_history(history, self.window_key)
        notes = []
        for warning in reading.warnings if reading is not None else ():
            key = warning.split(":", 1)[0]
            notes.append(WARNING_TEXT.get(key, warning))
        self.warnings.setText("\n".join(notes))
        self.readout.setToolTip(
            "\n".join(f"{title(n)}: {DIMENSIONS[n][0]}" for n in DIMENSION_NAMES)
        )
