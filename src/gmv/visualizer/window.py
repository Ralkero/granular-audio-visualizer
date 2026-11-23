"""The visualizer window: one more consumer of the engine's EventBus.

It owns a bounded subscription, drains it from a GUI-thread timer and paints with QPainter.
It never touches capture or DSP; if painting falls behind, the bounded queue drops the
oldest messages and the display carries on from the newest.
"""

import math
import os
from collections import OrderedDict, deque
from contextlib import contextmanager

import numpy as np
from time import perf_counter

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer
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
    QApplication,
    QStackedLayout,
    QWidget,
)

from . import layers as layer_registry
from . import palette
from . import themes
from .intensity import tint
from .dominance import DEFAULT_EDGES
from .perf import Timings
from .response import BASELINE, EXPRESSIVE
from .state import ROWS, SpectrumDisplay, VisualState
from .strip import Strip, polygon
from .themes.radiant import Radiant

TEXT, DIM, FAINT = QColor("#e8ebf4"), QColor("#8089a0"), QColor("#262b3a")
PREDICTED = QColor(palette.PREDICTED)  # reserved for predictions; no instrument has it
CONTEXT = QColor(palette.CONTEXT)
SPECTRUM = palette.SPECTRUM  # whole mix: low, middle, high frequencies
INTENSE = palette.INTENSE  # the same, when the music as a whole is most intense
BAR_STRIPE = 3  # pixels of width a bar has in its cached picture
HEAT_STEPS = 24  # the colouring moves in this many steps; each has its colours cached
CONDITIONS = {
    "idle": ("IDLE", "#6b7280"),
    "waiting": ("WAITING", "#f5c451"),
    "live": ("LIVE", "#34d399"),
    "replay": ("REPLAY", "#a78bfa"),
    "stopped": ("STOPPED", "#6b7280"),
}
AHEAD = 0.08  # share of the time axis kept to the right of "now" for predictions
QUEUE = 2048
FRAME_S = 1 / 60  # the cadence frames are drawn at
ACTIVE_MS = 16  # the longest the timer sleeps while frames are being drawn
DRAIN_S = 0.006  # most time one frame may spend taking in messages
DRAIN_MOST = 1024  # and most messages
SECTIONS = ("timeline", "details", "notes")
DEFAULT_SECTIONS = {"timeline": True, "details": False, "notes": False}
MIN_MAIN, MIN_SECTION = 260, 56  # smallest useful heights, before scaling
GLOW_STEPS = 12  # the glow's colour mix is taken in this many steps of each share
GLOW_KEPT = 4  # mixed-colour glow pictures kept (each is as large as the view)
RADIAL_SPECTRUM = 0.5  # share of the radius the spectrum keeps when instrument rings are shown
RING_SECONDS = 6.0  # one turn of the instrument rings


@contextmanager
def _untimed(_name):
    yield


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


def spectrum_colors(count, share=0.0):
    """The colours of ``count`` spectrum bars, low to high, ``share`` of the way from the
    neutral palette to the intense one. At 0 they are exactly the neutral colours."""
    neutral = [spectrum_color(i / max(1, count - 1)) for i in range(count)]
    if share <= 0:
        return neutral
    low, middle, high = INTENSE
    colours = []
    for index, colour in enumerate(neutral):
        position = index / max(1, count - 1)
        if position < 0.5:
            intense = mix(low, middle, position * 2)
        else:
            intense = mix(middle, high, position * 2 - 1)
        colours.append(mix(colour, intense, min(1.0, share)))
    return colours


def automatic_rate(refresh):
    """Frames per second for a display refreshing ``refresh`` times a second: that rate
    divided by a whole number, as near 66 as it gets. 60 Hz gives 60, 120 Hz 60, 144 Hz 72,
    180 Hz 60, 240 Hz 60. An unusable report from the system gives 60."""
    if not 30 <= refresh <= 1000:
        return 60.0
    return refresh / max(1, round(refresh / 66))


def effects_unavailable():
    """'' when the effects layer may be tried here, otherwise why not. Whether the card
    can really draw it is only known once its context exists (``effects_failed``)."""
    if os.environ.get("GMV_RADIANT_GPU", "").lower() in ("0", "off", "no"):
        return "switched off by GMV_RADIANT_GPU"
    if QApplication.platformName() == "offscreen":
        return "Qt's offscreen platform has no OpenGL"
    try:
        from .themes.radiant_gpu import view  # noqa: F401
    except ImportError as error:
        return f"Qt's OpenGL modules are missing: {error}"
    return ""


def canvas_base():
    """The widget class the canvas is built on. ``GMV_RENDERER=opengl`` selects an OpenGL
    surface (QPainter then draws through the graphics card); anything else, or a failure
    to import it, the ordinary CPU-painted widget. Experimental: see docs/PERFORMANCE.md."""
    if os.environ.get("GMV_RENDERER", "").lower() == "opengl":
        try:
            from PySide6.QtOpenGLWidgets import QOpenGLWidget

            return QOpenGLWidget
        except ImportError:
            pass
    return QWidget


class Canvas(canvas_base()):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self.display = SpectrumDisplay()
        self.mode = "bars"
        self.show_beats = True
        self.hidden = set()  # timeline rows and lane keys the user switched off
        self.info = {
            "source": "",
            "beat_tracker": None,
            "activity_model": None,
            "guitar_model": None,
            "recognition_model": None,
            "instruments": (),
        }
        # Instrument layers of the main view, in composition order. Each draws one
        # instrument from its own activity history; none reads the whole-mix spectrum.
        self.layers = layer_registry.build()
        self.stage_seconds = 4.0  # how much time the layers show; short, so strokes separate
        # The whole-mix spectrum while layers are shown: beside (as it is), behind (dimmed)
        # or hidden. The names are those of saved settings; all three share one area now.
        self.spectrum_place = "behind"
        # Sections under the main visualizer: shown or not, and a height the user chose by
        # dragging its upper edge (None: sized automatically).
        self.sections = {
            name: {"shown": on, "height": None} for name, on in DEFAULT_SECTIONS.items()
        }
        self._edges, self._heights, self._drag = {}, {}, None
        self.setMouseTracking(True)
        self.level = self.intensity = self.flash = self.pulse = 0.0
        self.peak_level = 0.0
        self.rings = []  # [age_s, strength, predicted]
        self.now = None
        self.condition = "idle"
        self.paint_ms = deque(maxlen=240)
        # Rolling timings of every part of a paint, in ms (see ``perf.Timings``).
        self.perf = Timings()
        self._last_paint = None
        self.painted = 0
        self.show_stats = False
        self._background = None
        self._colors = {}
        self._cache = {}
        self._glow_colours = {}  # colour-mix steps -> colour
        self._glow_pictures = OrderedDict()  # (colour, size) -> picture, newest last
        # Colouring of the spectrum by the intensity of the music as a whole: how far it
        # is coloured now (0 neutral .. 1), and whether it is coloured at all.
        self.heat = 0.0
        self.colour_by_intensity = True
        # Radial instrument rings: live objects drawn from each instrument's present
        # state (the default), or the six-second history swept by a hand.
        self.radial_live = True
        # The radial background glow takes the colour of the most prominent instrument
        # (``state.dominance``); off, it keeps the spectrum's neutral colour.
        self.reactive_background = True
        # How strongly everything is drawn: the baseline, or Expressive Mode.
        self.response = BASELINE
        # The radial theme: "classic" is drawn here, as ever; "radiant" by ``self.radiant``,
        # which also holds its palette. The choice stays while Spectrum Bars is shown.
        self.theme = themes.DEFAULT
        self.radiant = Radiant()
        # True while the graphics-card effects layer is showing behind this canvas: the
        # canvas then leaves its background clear and draws Radiant's geometry only.
        self.gpu = False
        self.path = ""  # who drew the central view at the last paint: see ``paintEvent``
        self.static = False  # True while the rarely-changing picture is being drawn
        self.ring_offset = 0.0
        edges = np.linspace(0, 2 * np.pi, 97)
        self._dashes = [  # 48 short arcs of a unit circle, as straight strokes
            (math.cos(a), math.sin(a), math.cos(b), math.sin(b))
            for a, b in zip(edges[0::2].tolist(), edges[1::2].tolist(), strict=False)
        ]
        self.lane_strip, self.level_strip, self.mark_strip = Strip(), Strip(), Strip()
        self.note_strip = Strip()
        self._lane_drawn = {}  # lane -> time of its last row already on the lane strip
        self._notes_until, self._attack_count, self._new_attacks = 0.0, 0, 0
        self._attack_seen = False
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
        heat = (
            min(1.0, tint(state.musical.value) * self.response.heat)
            if running and self.colour_by_intensity
            else 0.0
        )
        self.heat += (heat - self.heat) * (1 - math.exp(-dt / 0.12))
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
        """True when nothing is moving any more, so the timer can slow down. While there is
        a time axis the history keeps scrolling, with or without audio, so it never settles."""
        values = self.display.values
        return (
            self.now is None
            and self.condition not in ("live", "replay")
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
        width, height = self.width(), self.height()
        self.scale = max(0.85, min(1.6, height / 900))
        margin = round(28 * self.scale)
        header = round(58 * self.scale)
        sections = self.sections
        rows = (
            [row for row in ROWS if row[0] not in self.hidden and self.row_in_use(row[0])]
            if sections["timeline"]["shown"]
            else []
        )
        # A lane is live when a detector fills it or an instrument analyzer does.
        lanes = (
            [
                (lane, live or (lane.key in self.layers and lane.key in self.state.activity))
                for lane, live in self.state.board.lanes()
                if lane.key not in self.hidden
            ]
            if sections["details"]["shown"]
            else []
        )
        notes = (
            sections["notes"]["shown"]
            and self.state.pitch_method is not None
            and "notes" not in self.hidden
        )
        body = height - header - margin
        gap = round(18 * self.scale)
        wanted = {
            "timeline": round(min(body * 0.20, (26 * len(rows) + 26) * self.scale)) if rows else 0,
            "details": round(min(body * 0.30, (26 * len(lanes) + 26) * self.scale)) if lanes else 0,
            "notes": round(min(body * 0.14, 110 * self.scale)) if notes else 0,
        }
        present = [
            name
            for name, on in (("timeline", bool(rows)), ("details", bool(lanes)), ("notes", notes))
            if on
        ]
        least = MIN_SECTION * self.scale
        for name in present:
            if sections[name]["height"]:
                wanted[name] = max(least, sections[name]["height"])
        # The main visualizer keeps a useful height: sections give way, the lowest first.
        over = sum(wanted[n] for n in present) - (body - MIN_MAIN * self.scale - gap * len(present))
        for name in reversed(present):
            if over <= 0:
                break
            less = min(over, max(0.0, wanted[name] - least))
            wanted[name] -= less
            over -= less
        timeline_h, lanes_h, notes_h = wanted["timeline"], wanted["details"], wanted["notes"]
        spectrum = QRectF(
            margin,
            header,
            width - 2 * margin,
            body - timeline_h - lanes_h - notes_h - gap * len(present),
        )
        # Instrument layers are drawn inside the selected visualizer, over its spectrum.
        active = self.active_layers()
        show_spectrum = not active or self.spectrum_place != "hidden"
        gutter = round(150 * self.scale)
        top = spectrum.bottom() + gap
        timeline = QRectF(margin, top, width - 2 * margin, timeline_h)
        lane_block = QRectF(
            margin, top + (timeline_h + gap) * bool(rows), width - 2 * margin, lanes_h
        )
        note_block = QRectF(
            margin, lane_block.top() + (lanes_h + gap) * bool(lanes), width - 2 * margin, notes_h
        )
        bottom = note_block.bottom() if notes else lane_block.bottom()
        axis = QRectF(margin + gutter, top, width - 2 * margin - gutter, bottom - top)
        # Where each section's upper edge is: the handle that resizes it.
        blocks = {"timeline": timeline, "details": lane_block, "notes": note_block}
        self._edges = {name: blocks[name].top() - gap / 2 for name in present}
        self._heights = {name: wanted[name] for name in present}

        radiant = self.themed()
        gpu = self.gpu and radiant is not None

        # Who draws the central view: "radiant-gpu" (the graphics card, all of it; this
        # canvas draws none of it), "radiant-cpu" (the reduced QPainter Radiant), "classic"
        # or "bars" (drawn here, as ever).
        self.path = (
            "radiant-gpu" if gpu else "radiant-cpu" if radiant is not None
            else "classic" if self.mode == "radial" else "bars"
        )  # fmt: skip

        def everything(painter):
            if radiant is not None and self.static and not gpu:
                radiant.paint_static(self, painter, spectrum, active)  # under everything
            self.draw_header(painter, QRectF(margin, 0, width - 2 * margin, header))
            timed = self.perf.measure if not self.static else _untimed
            if gpu:
                if not self.static:
                    with timed("radiant frame"):  # numbers for the card; nothing is drawn
                        radiant.present(self, spectrum, active)
            elif radiant is not None:
                if not self.static:
                    with timed("spectrum"):
                        self.draw_glow(painter, spectrum)
                    with timed("radiant"):
                        radiant.paint(self, painter, spectrum, active, show_spectrum)
            elif show_spectrum:
                with timed("spectrum"):
                    if not self.static:
                        self.draw_glow(painter, spectrum)
                    if self.mode == "radial":
                        if not self.static:
                            self.draw_radial(painter, spectrum)
                    else:
                        self.draw_bars(painter, spectrum)
            if active and radiant is None:
                if show_spectrum and self.spectrum_place == "behind" and not self.static:
                    painter.fillRect(spectrum, QColor(5, 6, 10, 150))  # the spectrum steps back
                with timed("layers"):
                    self.draw_layers(painter, spectrum, active)
            if self.static:
                painter.setPen(QPen(alpha(DIM, 0.45), 2))
                for y in self._edges.values():  # a grip on each section's upper edge
                    painter.drawLine(QPointF(width / 2 - 22, y), QPointF(width / 2 + 22, y))
            if self.now is None and self.static:
                self.draw_idle(painter, spectrum)
            if rows:
                with timed("timeline"):
                    self.draw_timeline(painter, timeline, rows)
            if lanes:
                with timed("details"):
                    self.draw_lanes(painter, lane_block, lanes)
            if notes:
                with timed("notes"):
                    self.draw_notes(painter, note_block)
            if (rows or lanes or notes) and self.now is not None:
                self.draw_now(painter, axis)

        def still_picture():
            picture = QPixmap(self.background())
            if gpu:  # clear, so that the effects layer behind the canvas shows through
                picture.fill(Qt.GlobalColor.transparent)
            artist = QPainter(picture)
            artist.scale(self.devicePixelRatioF(), self.devicePixelRatioF())
            artist.setRenderHint(QPainter.RenderHint.Antialiasing)
            self.static = True
            try:
                everything(artist)
            finally:
                self.static = False
                artist.end()
            return picture

        # Labels, legends, row backgrounds and header text change rarely: they are drawn into
        # one picture and only redrawn when something they show changes.
        key = (
            width,
            height,
            self.devicePixelRatioF(),
            self.mode,
            tuple(rows),
            tuple((lane.key, live) for lane, live in lanes),
            notes,
            self.state.board.method,
            self.header_text(),
            self.state.spectrum_range,
            self.now is None,
            self.state.window_s,
            bool(self.info["activity_model"]),
            bool(self.info["recognition_model"] or self.info["guitar_model"]),
            (self.state.board.recognition or {}).get("detail"),
            len(self.display.values) if self.display.values is not None else 0,
            tuple(layer.key for layer in active),
            self.spectrum_place,
            tuple(sorted(self._edges.items())),
            (radiant.name, radiant.palette.version, gpu) if radiant is not None else None,
        )
        still = self.cached("still", key, still_picture)
        painter = QPainter(self)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.drawPixmap(self.rect(), still)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        everything(painter)
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
        self.perf.add("paint", self.paint_ms[-1])
        if self._last_paint is not None:
            self.perf.add("frame interval", (began - self._last_paint) * 1000)
        self._last_paint = began
        self.painted += 1

    def active_layers(self):
        """Layers to draw: switched on, and with an analyzer behind them in this session.
        A layer without data from its own analyzer is not drawn at all."""
        return [
            layer
            for layer in self.layers.values()
            if layer.enabled
            and (layer.key in self.state.activity or layer.key in self.info["instruments"])
        ]

    def themed(self):
        """The theme that draws the radial view instead of the canvas, or None: the classic
        radial view and Spectrum Bars are drawn here."""
        return self.radiant if self.mode == "radial" and self.theme == "radiant" else None

    def plasma_values(self):
        """What the effects layer's shaders are told for the frame last painted
        (``radiant_gpu.frame.values``), or None when there is nothing for it to draw."""
        radiant = self.themed()
        if radiant is None or not self.gpu or radiant.shown is None or len(radiant.shown) != 5:
            return None
        from .themes.radiant_gpu import frame

        lay, view, rect, keys, spectrum = radiant.shown
        # The glow's strength from the scene (audio time), not from the per-paint easing.
        strength = (0.05 + 0.20 * view.intensity + 0.16 * view.flash) * self.response.glow
        return frame.values(
            lay, view, radiant.palette, self.response, size=(self.width(), self.height()),
            area=(rect.top(), rect.bottom()), spectrum=spectrum, state=self.state, keys=keys,
            glow=(self.glow_colour(), min(1.0, strength)), instrument=radiant.instrument,
        )  # fmt: skip

    def radial_geometry(self, rect):
        """Centre and full radius of the radial visualizer in ``rect``."""
        centre = rect.center()
        return centre.x(), centre.y(), min(rect.width(), rect.height()) / 2 - 4

    def draw_layers(self, painter, rect, active):
        """Every active instrument inside the selected visualizer: a ring each round the
        radial spectrum, or a strip each across the bars. The mode only decides where the
        bands are; what is drawn in them is each layer's own business."""
        state = self.state
        if self.mode == "radial":
            if self.static or self.now is None:
                return
            cx, cy, reach = self.radial_geometry(rect)
            inner = reach * RADIAL_SPECTRUM + 8 * self.scale
            bands = layer_registry.rings(inner, reach, active, 5 * self.scale)
            for layer, (low, high) in zip(active, bands, strict=True):
                stage = layer_registry.Stage(
                    self.now, RING_SECONDS, self.scale, state.generation, self.font,
                    ring=(cx, cy, low, high), live=self.radial_live,
                    gain=0.55 + 0.45 * state.musical.value, response=self.response,
                )  # fmt: skip
                with self.perf.measure("layer " + layer.key):
                    layer.paint_stage(painter, stage, state)
            if self.radial_live:
                return  # live rings have no timeline, so no hand
            # The hand: where "now" is on the rings.
            angle = 2 * math.pi * (self.now % RING_SECONDS) / RING_SECONDS
            sin, cos = math.sin(angle), math.cos(angle)
            painter.setPen(QPen(alpha(TEXT, 0.6), 1))
            painter.drawLine(
                QPointF(cx + inner * sin, cy - inner * cos),
                QPointF(cx + reach * sin, cy - reach * cos),
            )
            return
        # Bars: the strips lie across the field the bars rise in.
        field = QRectF(
            rect.left(), rect.top() + 4, rect.width() - 22 * self.scale, rect.height() * 0.86 - 10
        )
        slots = layer_registry.slots(field, active, 5 * self.scale)
        for layer, slot in zip(active, slots, strict=True):
            stage = layer_registry.Stage(
                self.now, self.stage_seconds, self.scale, state.generation, self.font, rect=slot,
                response=self.response,
            )
            if self.static:
                layer.paint_static(painter, slot, stage, state)
            elif self.now is not None:
                painter.save()
                painter.setClipRect(slot)
                with self.perf.measure("layer " + layer.key):
                    layer.paint_stage(painter, stage, state)
                painter.restore()

    # -- sections --------------------------------------------------------------------------

    def edge_at(self, y):
        """The section whose upper edge is at ``y`` (within a few pixels), or None."""
        for name, edge in self._edges.items():
            if abs(y - edge) <= 6:
                return name
        return None

    def mouseMoveEvent(self, event):
        y = event.position().y()
        if self._drag is not None:
            name, start_y, start_height = self._drag
            self.sections[name]["height"] = max(
                MIN_SECTION * self.scale, start_height + start_y - y
            )
            self.update()
        else:
            self.setCursor(
                Qt.CursorShape.SizeVerCursor if self.edge_at(y) else Qt.CursorShape.ArrowCursor
            )

    def mousePressEvent(self, event):
        name = self.edge_at(event.position().y())
        if name is not None and event.button() == Qt.MouseButton.LeftButton:
            self._drag = (name, event.position().y(), self._heights[name])
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        self._drag = None
        super().mouseReleaseEvent(event)

    def row_in_use(self, row):
        # The delayed-context row only appears once such analysis has produced something.
        return row != "context" or any(mark.row == "context" for mark in self.state.markers)

    def header_text(self):
        """(state name, colour, detail, tempo, note) exactly as the header shows them."""
        name, color = CONDITIONS[self.condition]
        detail = self.info["source"]
        if self.condition == "waiting":
            detail = "No audio arriving" + (f"  ·  {detail}" if detail else "")
        elif self.condition == "replay":
            detail = "Recorded session, original timestamps" + (f"  ·  {detail}" if detail else "")
        elif self.condition == "stopped" and self.state.detail:
            detail = self.state.detail
        tempo = note = ""
        if self.show_beats:
            latest = self.state.latest
            locked = bool(latest is not None and latest.beat_locked and self.state.beat_tempo)
            if locked and self.condition in ("live", "replay"):
                tempo = f"{self.state.beat_tempo:.0f} BPM"
                note = (
                    "experimental beat tracker  ·  "
                    f"score {round(self.state.beat_score, 1):.1f} (uncalibrated)"
                )
            elif self.info["beat_tracker"] is False:
                note = "Beat tracker off  ·  it can be enabled in Settings"
            elif self.condition in ("live", "replay"):
                note = "Experimental beat tracker  ·  listening for a pulse"
        return name, color, detail, tempo, note

    def draw_header(self, painter, rect):
        name, color, detail, tempo, note = self.header_text()
        middle = rect.center().y() + 4 * self.scale
        if not self.static:
            if tempo:  # the only moving part: a dashed ring that swells on each predicted beat
                radius = (7 + 5 * self.pulse) * self.scale
                painter.setPen(
                    QPen(alpha(PREDICTED, 0.45 + 0.55 * self.pulse), 2, Qt.PenStyle.DashLine)
                )
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawEllipse(
                    QPointF(rect.right() - self.ring_offset, middle - 12 * self.scale),
                    radius,
                    radius,
                )
            return
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(color))
        painter.drawEllipse(QPointF(rect.left() + 6, middle - 4 * self.scale), 4.5, 4.5)
        painter.setFont(self.font(12, QFont.Weight.DemiBold, 1.5))
        painter.setPen(QColor(color))
        painter.drawText(QPointF(rect.left() + 20, middle), name)
        offset = painter.fontMetrics().horizontalAdvance(name) + 34
        painter.setFont(self.font(13))
        painter.setPen(DIM)
        painter.drawText(QPointF(rect.left() + offset, middle), detail)
        if tempo:
            painter.setFont(self.font(11))
            painter.drawText(
                QRectF(rect.left(), middle + 2, rect.width(), 18 * self.scale),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                note,
            )
            painter.setFont(self.font(26, QFont.Weight.DemiBold))
            painter.setPen(TEXT)
            painter.drawText(
                QRectF(rect.left(), rect.top(), rect.width(), middle - rect.top()),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                tempo,
            )
            self.ring_offset = painter.fontMetrics().horizontalAdvance(tempo) + 26 * self.scale
        elif note:
            painter.setFont(self.font(12))
            painter.drawText(
                QRectF(rect.left(), middle - 16 * self.scale, rect.width(), 20 * self.scale),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                note,
            )

    @property
    def heat_step(self):
        """The colouring in whole steps: every bar takes the same step, so the spectrum
        changes colour as one thing, and each step's colours are worked out once."""
        return round(self.heat * HEAT_STEPS)

    def colors(self, count):
        """The bars' colours for the music's present intensity."""
        key = (count, self.heat_step)
        colours = self._colors.get(key)
        if colours is None:
            if len(self._colors) > 2 * HEAT_STEPS + 2:
                self._colors.clear()
            colours = self._colors[key] = spectrum_colors(count, key[1] / HEAT_STEPS)
        return colours

    def cached(self, name, key, build):
        """Pictures that only change with the window size are drawn once and reused."""
        held = self._cache.get(name)
        if held is None or held[0] != key:
            held = self._cache[name] = (key, build())
        return held[1]

    def glow_picture(self, size, colour):
        """A soft elliptical glow, drawn once; each frame only sets how strongly it shows."""
        ratio = self.devicePixelRatioF()
        picture = QPixmap(round(size.width() * ratio), round(size.height() * ratio))
        picture.fill(Qt.GlobalColor.transparent)
        artist = QPainter(picture)
        artist.scale(ratio * size.width() / 2, ratio * size.height() / 2)
        glow = QRadialGradient(QPointF(1, 1), 1)
        glow.setColorAt(0, QColor(colour))
        glow.setColorAt(1, alpha(colour, 0))
        artist.fillRect(QRectF(0, 0, 2, 2), glow)
        artist.end()
        return picture

    def glow_colour(self):
        """The colour of the background glow now: the neutral one, or, in the radial view
        with a stream running, the neutral and the instruments' colours mixed in the shares
        ``state.dominance.mix`` gives. The shares are taken in ``GLOW_STEPS`` steps, so only
        so many colours ever occur and each has its picture made once."""
        theme = self.themed()  # a theme with a palette has its own neutral and instruments
        neutral = SPECTRUM[1] if theme is None else theme.neutral
        if not (
            self.reactive_background
            and self.mode == "radial"
            and self.condition in ("live", "replay")
        ):
            return neutral
        steps = tuple(
            min(GLOW_STEPS, round(share * GLOW_STEPS)) for share in self.state.dominance.mix.values()
        )
        key = steps if theme is None else (steps, theme.name, theme.palette.version)
        colour = self._glow_colours.get(key)
        if colour is None:
            if len(self._glow_colours) > 512:
                self._glow_colours.clear()
            left = max(0, GLOW_STEPS - sum(steps))
            colour = self._glow_colours[key] = palette.blend(
                [(neutral, left)]
                + [
                    (palette.INSTRUMENTS[name] if theme is None else theme.instrument(name), step)
                    for name, step in zip(self.state.dominance.mix, steps, strict=True)
                ]
            )
        return colour

    def glow_parts(self):
        """What the background glow is made of now: (name, colour, strength 0..1). One glow
        in one colour at the strength the music gives it (the response only scales that),
        and the beat glow, which is its own thing and keeps its own colour."""
        strength = (0.05 + 0.20 * self.intensity + 0.16 * self.flash) * self.response.glow
        theme = self.themed()
        beat = PREDICTED if theme is None else QColor(theme.beat)
        return [
            ("glow", self.glow_colour(), min(1.0, strength)),
            ("beat glow", beat, min(1.0, 0.14 * self.pulse * self.response.glow)),
        ]

    def glow_for(self, colour, key, size):
        """The glow picture in ``colour``. The two fixed colours are kept for good; mixed
        ones in a small most-recently-used store, since a picture is as large as the view."""
        if colour in (SPECTRUM[1], PREDICTED.name()):
            name = "glow" if colour == SPECTRUM[1] else "beat glow"
            return self.cached(name, key, lambda: self.glow_picture(size, colour))
        held = self._glow_pictures.get((colour, key))
        if held is None:
            with self.perf.measure("glow picture"):
                held = self.glow_picture(size, colour)
            while len(self._glow_pictures) >= GLOW_KEPT:
                self._glow_pictures.popitem(last=False)
        else:
            del self._glow_pictures[(colour, key)]
        self._glow_pictures[(colour, key)] = held  # newest last
        return held

    def draw_glow(self, painter, rect):
        centre_y = rect.bottom() if self.mode == "bars" else rect.center().y()
        area = QRectF(0, centre_y - rect.height(), self.width(), rect.height() * 2)
        key = (round(area.width()), round(area.height()), self.devicePixelRatioF())
        for _name, colour, strength in self.glow_parts():
            if strength > 0.01:
                picture = self.glow_for(QColor(colour).name(), key, area.size())
                painter.setOpacity(min(1.0, strength))
                part = 0.5 if self.mode == "bars" else 1.0  # below the baseline is not needed
                painter.drawPixmap(
                    QRectF(area.left(), area.top(), area.width(), area.height() * part),
                    picture,
                    QRectF(0, 0, picture.width(), picture.height() * part),
                )
        painter.setOpacity(1.0)

    def bar_pictures(self, count, reach):
        """Every bar at full height, plain and brightened; frames copy the part they need.
        A bar is the same all the way across, so three pixels of width each are enough (the
        middle one is copied, stretched), which keeps one pair per colour step small."""
        ratio = self.devicePixelRatioF()
        pictures = []
        for whiten in (0.12, 0.8):
            picture = QPixmap(count * BAR_STRIPE, math.ceil(reach * ratio))
            picture.fill(Qt.GlobalColor.transparent)
            artist = QPainter(picture)
            artist.scale(1, ratio)
            artist.setPen(Qt.PenStyle.NoPen)
            for index, colour in enumerate(self.colors(count)):
                fill = QLinearGradient(0, 0, 0, reach)
                fill.setColorAt(0, mix(colour, "#ffffff", whiten))
                fill.setColorAt(1, alpha(mix(colour, "#ffffff", whiten / 2), 0.45))
                artist.fillRect(QRectF(index * BAR_STRIPE, 0, BAR_STRIPE, reach), fill)
            artist.end()
            pictures.append(picture)
        return pictures

    def draw_bars(self, painter, rect):
        values, peaks = self.display.values, self.display.peaks
        if values is not None:  # the response may lift the bars; never past full
            values, peaks = self.response.lifted(values), self.response.lifted(peaks)
        count = len(values) if values is not None else 48
        ratio = self.devicePixelRatioF()
        mirror = rect.height() * 0.14
        base = rect.bottom() - mirror
        reach = base - rect.top() - 6
        meter = 22 * self.scale
        slot = (rect.width() - meter) / count
        bar = slot * 0.7
        if self.static:
            self.draw_frequencies(painter, rect, base + mirror * 0.55, slot, count)
            painter.fillRect(
                QRectF(rect.right() - 6 * self.scale, rect.top() + 6, 4 * self.scale, reach),
                alpha(FAINT, 0.7),
            )
            return
        plain, bright = self.cached(
            f"bars {self.heat_step}",  # each colour step keeps its pair: drawn once
            (count, round(reach), ratio),
            lambda: self.bar_pictures(count, reach),
        )
        caps, mirrors = self.cached(
            "bar colours",
            (count, self.heat_step),
            lambda: (
                [alpha(c, 0.85) for c in self.colors(count)],
                [alpha(c, 0.09) for c in self.colors(count)],
            ),
        )
        heights = np.maximum(2.0, values * reach).tolist() if values is not None else [2.0] * count
        tops = (peaks * reach).tolist() if peaks is not None else [0.0] * count
        flash = min(1.0, self.flash * self.response.flash)
        flashing = flash > 0.03
        left = rect.left() + (slot - bar) / 2
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        for index, tall in enumerate(heights):
            x = left + index * slot
            target = QRectF(x, base - tall, bar, tall)
            source = QRectF(index * BAR_STRIPE + 1, (reach - tall) * ratio, 1, tall * ratio)
            painter.drawPixmap(target, plain, source)
            if flashing:
                painter.setOpacity(min(1.0, flash * tall / reach))
                painter.drawPixmap(target, bright, source)
                painter.setOpacity(1.0)
            if tops[index] > 2:
                painter.fillRect(QRectF(x, base - tops[index] - 5, bar, 2), caps[index])
            painter.fillRect(QRectF(x, base + 3, bar, min(mirror, tall * 0.22)), mirrors[index])
        # Baseline: brightens with onset strength, turns amber on a predicted beat.
        line = mix(SPECTRUM[1], PREDICTED, self.pulse)
        painter.fillRect(
            QRectF(rect.left(), base + 0.5, rect.width() - meter, 1.5),
            alpha(line, 0.25 + 0.5 * max(self.flash, self.pulse)),
        )
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        # Level meter: short-term level (filled) and its recent peak (tick).
        x = rect.right() - 6 * self.scale
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

    def core_picture(self, size):
        ratio = self.devicePixelRatioF()
        picture = QPixmap(round(size * ratio), round(size * ratio))
        picture.fill(Qt.GlobalColor.transparent)
        artist = QPainter(picture)
        artist.setRenderHint(QPainter.RenderHint.Antialiasing)
        artist.scale(ratio, ratio)
        core = QRadialGradient(QPointF(size / 2, size / 2), size / 2)
        core.setColorAt(0, alpha(SPECTRUM[1], 0.6))
        core.setColorAt(1, alpha(SPECTRUM[0], 0.05))
        artist.setPen(Qt.PenStyle.NoPen)
        artist.setBrush(core)
        artist.drawEllipse(QRectF(1, 1, size - 2, size - 2))
        artist.end()
        return picture

    def draw_radial(self, painter, rect):
        values = self.display.values
        count = len(values) if values is not None else 48
        centre = rect.center()
        cx, cy = centre.x(), centre.y()
        # With instrument rings round it, the spectrum keeps the inner half of the radius.
        reach = self.radial_geometry(rect)[2]
        outer = reach * (RADIAL_SPECTRUM if self.active_layers() else 1.07)
        inner = outer * (
            0.27 + self.response.breath * (0.07 * self.level + 0.03 * self.pulse)
        )
        reach = outer - inner - 10
        bar = max(2.0, math.pi * inner / count * 0.62)
        # Core: one cached picture, resized as it breathes with the level.
        size = round(outer * 0.8)
        picture = self.cached(
            "core", (size, self.devicePixelRatioF()), lambda: self.core_picture(size)
        )
        radius = inner - 6
        painter.setOpacity(min(1.0, 0.25 + 0.75 * self.intensity + 0.3 * self.flash))
        painter.drawPixmap(
            QRectF(cx - radius, cy - radius, 2 * radius, 2 * radius),
            picture,
            QRectF(picture.rect()),
        )
        painter.setOpacity(1.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(alpha(SPECTRUM[1], 0.35 + 0.4 * self.flash), 1))
        painter.drawEllipse(centre, radius, radius)
        for age, strength, predicted in self.rings:
            progress = age / 0.7
            ring = inner + (outer - inner) * progress
            if predicted:  # dashed, amber: drawn as separate short strokes
                painter.setPen(QPen(alpha(PREDICTED, 0.8 * (1 - progress)), 1))
                angles = self._dashes
                painter.drawLines(
                    [
                        QLineF(cx + ring * c0, cy + ring * s0, cx + ring * c1, cy + ring * s1)
                        for c0, s0, c1, s1 in angles
                    ]
                )
            else:
                painter.setPen(QPen(alpha(TEXT, 0.5 * strength * (1 - progress)), 1))
                painter.drawEllipse(centre, ring, ring)
        # Bars: lows at the top, highs at the bottom, mirrored left and right.
        cos, sin = self.cached("spokes", count, lambda: self.spokes(count))
        lengths = self.response.lifted(values) * reach if values is not None else np.zeros(count)
        far = inner + np.maximum(2.0, np.concatenate((lengths, lengths[::-1])))
        half = bar / 2
        xs = np.empty((2 * count, 4))
        ys = np.empty((2 * count, 4))
        xs[:, 0], xs[:, 3] = inner * cos + half * sin, inner * cos - half * sin
        xs[:, 1], xs[:, 2] = far * cos + half * sin, far * cos - half * sin
        ys[:, 0], ys[:, 3] = inner * sin - half * cos, inner * sin + half * cos
        ys[:, 1], ys[:, 2] = far * sin - half * cos, far * sin + half * cos
        plain, bright = self.cached(
            "spoke colours",
            (count, self.heat_step),
            lambda: tuple(
                [mix(c, "#ffffff", whiten) for c in self.colors(count) + self.colors(count)[::-1]]
                for whiten in (0.0, 0.45)
            ),
        )
        # An onset briefly whitens the bars.
        colours = bright if self.flash * self.response.flash > 0.35 else plain
        xs, ys = xs + cx, ys + cy
        painter.setPen(Qt.PenStyle.NoPen)
        for index in range(2 * count):
            painter.setBrush(colours[index])
            painter.drawConvexPolygon(polygon(xs[index], ys[index]))

    def spokes(self, count):
        """Unit directions of the bars, going once round the circle from the top."""
        right = np.pi * (np.arange(count) + 0.5) / count - np.pi / 2
        angles = np.concatenate((right, np.pi - right[::-1]))
        return np.cos(angles), np.sin(angles)

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
        if not self.static:
            painter.setPen(QPen(alpha(TEXT, 0.55), 1))
            painter.drawLine(QPointF(x, rect.top()), QPointF(x, rect.bottom()))
            return
        painter.fillRect(
            QRectF(x, rect.top(), rect.right() - x, rect.height()), alpha(PREDICTED, 0.035)
        )
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
        if self.static:
            self.caption(painter, rect, "TIMELINE")
            self.draw_legend(painter, rect)
        head = 22 * self.scale
        gutter = 150 * self.scale
        area = QRectF(
            rect.left() + gutter, rect.top() + head, rect.width() - gutter, rect.height() - head
        )
        row_height = area.height() / len(rows)
        if self.static:
            painter.fillRect(area, alpha(FAINT, 0.28))
            for index, (_key, title) in enumerate(rows):
                band = QRectF(
                    area.left(), area.top() + index * row_height, area.width(), row_height
                )
                self.row_label(painter, QRectF(rect.left(), band.top(), gutter, row_height), title)
                if index:
                    painter.fillRect(
                        QRectF(band.left(), band.top(), band.width(), 1), alpha(FAINT, 0.9)
                    )
            return
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

        # no frame yet right after a session starts in a window that already drew one
        newest = frames.last[0] if frames.last is not None else start
        self.level_strip.paint(
            painter, area, start, per_second, self.state.generation, newest, draw_level
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

    def draw_notes(self, painter, rect):
        """Piano-roll of note candidates: height is pitch, strength is the score.

        Hollow-looking, thin marks: these are candidates. They are tinted with the guitar
        colour only while a guitar detector judges guitar to be playing (fully when it reports
        "detected", faintly for "candidate"); otherwise they are neutral, because the tracker
        cannot tell which instrument a note belongs to. A context recognition model reports
        late, so notes first appear neutral and take their colour when its estimate arrives.
        """
        head = 22 * self.scale
        gutter = 150 * self.scale
        area = QRectF(
            rect.left() + gutter, rect.top() + head, rect.width() - gutter, rect.height() - head
        )
        guitar_known = "guitar" in self.state.board.history
        if self.static:
            self.caption(
                painter,
                rect,
                "NOTES   ·   pitch candidates E2–E6   ·   "
                + (
                    "amber while guitar is detected, pale amber for a guitar candidate"
                    if guitar_known
                    else "any pitched instrument, not attributed to guitar"
                ),
            )
            painter.fillRect(area, alpha(FAINT, 0.28))
            painter.setFont(self.font(10))
            painter.setPen(alpha(DIM, 0.8))
            for midi, name in ((40, "E2"), (52, "E3"), (64, "E4"), (76, "E5"), (88, "E6")):
                y = area.bottom() - (midi - 39) / 50 * area.height()
                painter.drawText(
                    QRectF(rect.left(), y - 8, gutter - 10, 16),
                    Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                    name,
                )
                painter.fillRect(QRectF(area.left(), y, area.width(), 1), alpha(FAINT, 0.7))
            return
        if self.now is None:
            return
        state = self.state
        _, start, _ = self.axis(area)
        per_second = area.width() / (state.window_s * (1 + AHEAD))
        step = area.height() / 50
        tall = max(3.0, step * 1.5)
        wide = max(2.0, per_second * 0.02 + 0.5)
        neutral, guitar = QColor(palette.NEUTRAL), QColor(palette.INSTRUMENTS["guitar"])
        maybe_guitar = mix(neutral, guitar, 0.45)
        attacks = list(state.note_attacks)

        def draw(artist, at, since, height):
            first = start if since is None else self._notes_until
            rows = state.notes.since(first)
            artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            judged = np.nan_to_num(state.board.judged_at("guitar", rows[:, 0]))
            levels = np.maximum(rows[:, 3], judged)
            for (time, midi, score, _), level in zip(rows.tolist(), levels.tolist(), strict=True):
                y = height - (midi - 39) * step
                colour = guitar if level >= 1 else maybe_guitar if level > 0 else neutral
                artist.fillRect(
                    QRectF(at(time), y - tall / 2, wide, tall), alpha(colour, 0.45 + 0.55 * score)
                )
            fresh = attacks if since is None else attacks[len(attacks) - self._new_attacks :]
            for time, midi, _score in fresh if (since is None or self._new_attacks) else ():
                y = height - (midi - 39) * step
                artist.fillRect(QRectF(at(time) - 1, y - tall, 2, 2 * tall), TEXT)
            if len(rows):
                self._notes_until = float(rows[-1, 0]) + 1e-6

        if self._attack_count > len(attacks):
            self._attack_count = 0
        self._new_attacks = len(attacks) - self._attack_count if self._attack_seen else 0
        # A late guitar estimate changes the colour of notes already drawn: draw them again.
        key = (state.generation, guitar_known, state.board.judged.get("guitar", 0))
        if self.note_strip.key is None or self.note_strip.key[0] != key:
            self._notes_until = start
        self.note_strip.paint(painter, area, start, per_second, key, state.note_serial, draw)
        self._attack_count, self._attack_seen = len(attacks), True

    def draw_lanes(self, painter, rect, lanes):
        method = self.state.board.method
        if self.static:
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
        for index, (lane, live) in enumerate(lanes if self.static else ()):
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
            elif lane.key == "guitar":
                recognition = self.state.board.recognition
                if recognition and not recognition["available"]:
                    reason = "recognition unavailable  ·  " + recognition["detail"]
                elif self.info["recognition_model"] or self.info["guitar_model"]:
                    reason = "no detector output yet"
                else:
                    reason = "no detector output  ·  guitar recognition is off in Settings"
            else:
                reason = "no detector for this instrument yet"
            painter.drawText(band.adjusted(12, 0, 0, 0), Qt.AlignmentFlag.AlignVCenter, reason)
        if self.static or self.now is None or not any(live for _, live in lanes):
            return
        history = self.state.board.history
        _, start, _ = self.axis(area)

        def draw(artist, to_x, since, _height):
            if since is None:
                self._lane_drawn = {}
            for index, (lane, _) in enumerate(lanes):
                # Each lane continues from its own last drawn row: detectors report at
                # different rates (frames every 20 ms, recognition twice a second), so the
                # time of the last paint says nothing about what a lane still has to draw.
                bottom = (index + 1) * lane_height - 2
                # The same activity rows the instrument's layer draws, at lane resolution.
                track = self.state.activity.get(lane.key)
                if lane.key in self.layers and track is not None:
                    name = lane.key + "#activity"
                    rows = track.since(self._lane_drawn.get(name, start))
                    if len(rows) >= 2:
                        self._lane_drawn[name] = rows[-1, 0]
                        self.layers[lane.key].paint_lane(
                            artist, to_x, rows, index * lane_height + 2, bottom
                        )
                if lane.key not in history:
                    continue
                rows = history[lane.key].since(self._lane_drawn.get(lane.key, start))
                if len(rows) < 2:
                    continue
                self._lane_drawn[lane.key] = rows[-1, 0]
                xs = to_x(rows[:, 0])
                ys = bottom - np.maximum(0.03, rows[:, 1]) * (lane_height - 7)
                edge_x = np.concatenate(([xs[0]], xs, [xs[-1]]))
                color = QColor(lane.color)
                # Fills without smoothing so that consecutive slices meet without a seam.
                artist.setRenderHint(QPainter.RenderHint.Antialiasing, False)
                artist.setPen(Qt.PenStyle.NoPen)
                sure, maybe = rows[:, 2] >= 1, (rows[:, 2] > 0) & (rows[:, 2] < 1)
                for heights, strength in (
                    (ys, 0.14),
                    (np.where(maybe, ys, bottom), 0.2),  # candidate: visibly weaker
                    (np.where(sure, ys, bottom), 0.5),
                ):
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
        # The canvas, and behind it (made only when first needed) the effects layer.
        self.stack = QStackedLayout(self)
        self.stack.setContentsMargins(0, 0, 0, 0)
        self.stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self.stack.addWidget(self.canvas)
        self.plasma = None
        self.effects = "auto"  # "auto": the graphics card when it can; "off": phase 1 drawing
        self.effects_problem = ""  # why the effects layer is not in use, when it is wanted
        self.timer = QTimer(self)
        self.timer.setTimerType(Qt.TimerType.PreciseTimer)
        self.frame_s = FRAME_S  # seconds per frame; see set_frame_rate
        self.frame_rate = None  # None: chosen from the display
        self.timer.timeout.connect(self.poll)
        self.last_tick = self.next_frame = perf_counter()
        self.timer.start(ACTIVE_MS)

    # -- display options -----------------------------------------------------------------------

    def set_mode(self, mode):
        self.canvas.mode = mode
        self.sync_effects()
        self.canvas.update()

    # -- the graphics-card effects layer ---------------------------------------------------------

    def set_radiant_effects(self, choice):
        """Who draws Radiant's light: "auto" (shaders on the graphics card when an OpenGL
        3.3 context can be had, otherwise the phase 1 drawing) or "off" (always the phase 1
        drawing). Only the drawing."""
        if choice in ("auto", "off"):
            self.effects = choice
            self.sync_effects()
            self.canvas.update()

    def sync_effects(self):
        """Show the effects layer exactly while Radiant is on screen and the card can draw
        it. It is made the first time it is needed, so a session that never shows Radiant
        never makes an OpenGL context."""
        wanted = self.canvas.themed() is not None and self.effects == "auto"
        # Only once the window is on screen: an OpenGL widget in a window that is only
        # ever drawn into a picture (tests, offscreen renders) has no surface to draw on.
        if wanted and self.plasma is None and not self.effects_problem:
            self.effects_problem = effects_unavailable()
            if not self.effects_problem and self.isVisible():
                from .themes.radiant_gpu.view import PlasmaView

                self.plasma = PlasmaView(self.canvas.plasma_values, self)
                self.plasma.failed.connect(self.effects_failed)
                self.stack.addWidget(self.plasma)
                self.stack.setCurrentWidget(self.canvas)  # the canvas stays on top
        showing = wanted and self.plasma is not None and not self.effects_problem
        if self.plasma is not None:
            self.plasma.setVisible(showing)
        self.canvas.gpu = showing

    def effects_failed(self, problem):
        """The card could not draw the effects (no context, a shader that did not compile):
        go back to the phase 1 drawing and say why."""
        self.effects_problem = problem or "the OpenGL effects could not be set up"
        self.sync_effects()
        self.canvas.update()

    def renderer_report(self):
        """Who is drawing Radiant's light, and at what cost: for diagnostics."""
        if self.canvas.gpu and self.plasma is not None:
            renderer = self.plasma.renderer
            return {
                "backend": "gpu",
                "detail": f"{renderer.info.get('renderer', '?')}, OpenGL {renderer.info.get('version', '?')}, "
                f"GLSL {renderer.info.get('glsl', '?')}",
                "submit_ms": renderer.cpu_ms,
                "gpu_ms": renderer.gpu_ms,
                "buffers_mib": round(renderer.memory() / 2**20, 1),
                "frames": renderer.frames,
            }
        if self.effects == "off":
            reason = "effects switched off in Settings"
        else:
            reason = self.effects_problem or (
                "Radiant is not on screen" if self.canvas.themed() is None else "the window is not shown yet"
            )
        return {"backend": "cpu", "detail": "reduced QPainter picture (no plasma): " + reason}

    def set_beats(self, shown):
        self.canvas.show_beats = bool(shown)
        self.canvas.update()

    def set_smoothing(self, seconds):
        self.canvas.display.release_s = seconds
        self.state.spectrum_release_s = seconds

    def set_gain(self, decibels):
        self.canvas.display.gain_db = decibels
        self.state.spectrum_gain_db = decibels

    def set_window(self, seconds):
        """Length of the timeline and lanes."""
        self.state.window_s = float(seconds)
        self.canvas.update()

    def set_layer(self, key, *, enabled=None, intensity=None, style=None):
        """Display options of one instrument layer. None of them reaches an analyzer."""
        layer = self.canvas.layers[key]
        if enabled is not None:
            layer.enabled = bool(enabled)
        if intensity is not None:
            layer.intensity = max(0.0, float(intensity))
        if style is not None and style in dict(layer.profile.styles):
            layer.style = style
        self.canvas.update()

    def set_spectrum_place(self, place):
        """Where the whole-mix spectrum goes while instrument layers are shown."""
        if place in ("beside", "behind", "hidden"):
            self.canvas.spectrum_place = place
            self.canvas.update()

    def set_reactive_background(self, enabled):
        """Whether the radial background glow takes the colour of the most prominent
        instrument. Off: the neutral glow, as before. Only the drawing."""
        self.canvas.reactive_background = bool(enabled)
        self.canvas.update()

    def set_expressive(self, enabled):
        """Expressive Mode: the same music, events and timing, drawn larger and brighter
        (``response.EXPRESSIVE``). Nothing about the analysis changes."""
        self.canvas.response = EXPRESSIVE if enabled else BASELINE
        self.canvas.update()

    def set_radial_theme(self, name):
        """How the radial view is drawn: "classic" or "radiant" (``visualizer.themes``).
        Only the drawing: the stream, the analysis and the history are untouched, and
        Spectrum Bars is not a radial theme."""
        self.canvas.theme = themes.known(name)
        self.sync_effects()
        self.canvas.update()

    @property
    def radiant_palette(self):
        """The Radiant theme's colours (``themes.radiant_palette.RadiantPalette``). They
        are its own: no other theme or mode reads them."""
        return self.canvas.radiant.palette

    def set_radiant_colour(self, role, colour):
        """Give one role of the Radiant palette a colour (``#rrggbb``); the next frame is
        drawn with it. Returns False for an unknown role or something that is no colour."""
        taken = self.radiant_palette.set(role, colour)
        self.canvas.update()
        return taken

    def set_radiant_preset(self, name):
        """Take all the colours of a named Radiant preset."""
        taken = self.radiant_palette.apply(name)
        self.canvas.update()
        return taken

    def instrument_colour(self, key):
        """The colour an instrument is drawn in by what is on screen now."""
        theme = self.canvas.themed()
        return palette.instrument(key) if theme is None else theme.instrument(key)

    def set_radial_rings(self, kind):
        """How the instrument rings of the radial view are drawn: "live" (fixed objects
        animated by what each instrument is doing now) or "history" (the last six
        seconds, swept by a hand). The timeline and the lanes keep their history either
        way."""
        if kind in ("live", "history"):
            self.canvas.radial_live = kind == "live"
            self.canvas.update()

    def set_intensity_colour(self, enabled):
        """Whether the spectrum reddens with the intensity of the music as a whole. Only
        how it is drawn: the intensity is still worked out, the analysis is untouched."""
        self.canvas.colour_by_intensity = bool(enabled)
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

    def attach(
        self,
        bus,
        *,
        source="",
        beat_tracker=None,
        activity_model=None,
        guitar_model=None,
        recognition_model=None,
        instruments=(),
        band_edges=DEFAULT_EDGES,
    ):
        """Follow ``bus`` with a subscription of this window's own. Replaces any earlier one."""
        self.detach()
        self.bus, self.subscription = bus, bus.subscribe(QUEUE)
        self.state.status, self.state.detail = "starting", ""
        # The band edges of this stream's analysis (None: not known, as for a recording
        # whose configuration is missing). Not passing any means the analysis default.
        self.state.set_band_edges(band_edges)
        self.state.reset()
        self.canvas.info.update(
            source=source,
            beat_tracker=beat_tracker,
            activity_model=activity_model,
            guitar_model=guitar_model,
            recognition_model=recognition_model,
            instruments=tuple(instruments),
            dropped=0,
        )
        self.timer.start(ACTIVE_MS)

    def detach(self):
        if self.subscription is not None:
            self.dropped += self.subscription.dropped
            self.bus.unsubscribe(self.subscription)
        self.bus = self.subscription = None

    def poll(self):
        """Timer slot. Frames are due on a fixed 60 Hz grid; the timer is set to wake just
        when the next one is. A wake-up that comes early goes back to sleep for the rest; one
        that comes late (the analysis thread held the interpreter, Windows ran something
        else) draws at once and the grid is kept, so lateness does not accumulate."""
        clock = perf_counter()
        if self.timer.interval() < 250 and clock < self.next_frame - 0.0007:
            self.timer.start(max(1, int((self.next_frame - clock) * 1000)))
            return
        # After a stall the grid starts again from now: the next frame is at least half a
        # frame away, so lateness is never made up with a burst of frames.
        self.next_frame = max(self.next_frame + self.frame_s, clock + self.frame_s / 2)
        self.tick()

    def tick(self):
        """One frame: take in messages (bounded), advance the animation, ask for a paint."""
        clock = perf_counter()
        dt, self.last_tick = min(0.25, clock - self.last_tick), clock
        perf = self.canvas.perf
        perf.add("tick interval", dt * 1000)
        if self.subscription is not None:
            perf.add("backlog", len(self.subscription))
            drained = 0
            # Every message is taken in, in order; none is skipped. A burst is spread over
            # frames instead of holding this one up: the rest stays queued for the next.
            while drained < DRAIN_MOST:
                message = self.subscription.get(timeout=0)
                if message is None:
                    break
                self.state.consume(message)
                drained += 1
                if drained % 32 == 0 and perf_counter() - clock > DRAIN_S:
                    perf.count("frames that left messages queued")
                    break
            perf.add("messages per tick", drained)
            perf.add("consume", (perf_counter() - clock) * 1000)
            self.canvas.info["dropped"] = self.dropped + self.subscription.dropped
        self.canvas.advance(dt, clock)
        self.canvas.update()
        if self.canvas.gpu:
            self.plasma.update()
        perf.add("tick", (perf_counter() - clock) * 1000)
        # Nothing arriving and nothing moving: check a few times a second instead.
        if self.canvas.settled():
            interval = 250
        else:  # until the next frame is due
            interval = max(1, min(ACTIVE_MS, int((self.next_frame - perf_counter()) * 1000)))
        if self.timer.interval() != interval:
            self.timer.start(interval)

    # -- sections and layout -------------------------------------------------------------------

    def set_frame_rate(self, rate=None):
        """Frames per second to draw at. None chooses from the display the window is on:
        the refresh rate divided by a whole number, as near 66 as that gets (60 on a 60,
        120, 180 or 240 Hz display, 72 on 144 Hz). A rate that divides the refresh rate
        gives every frame the same time on screen; one that does not cannot."""
        self.frame_rate = rate
        if rate is None:
            screen = self.screen()
            rate = automatic_rate(screen.refreshRate() if screen is not None else 60.0)
        self.frame_s = 1.0 / min(240.0, max(20.0, float(rate)))

    def set_section(self, name, *, shown=None, height=None):
        """Show or hide a section under the main visualizer, or set its height in pixels
        (0 or None: automatic). A hidden section is not painted at all."""
        section = self.canvas.sections[name]
        if shown is not None:
            section["shown"] = bool(shown)
        if height is not None:
            section["height"] = float(height) or None
        self.canvas.update()

    def layout_state(self):
        """What is worth remembering about the layout."""
        return {name: dict(values) for name, values in self.canvas.sections.items()}

    def restore_layout(self, saved):
        for name in SECTIONS:
            values = (saved or {}).get(name)
            if isinstance(values, dict):
                self.set_section(
                    name,
                    shown=values.get("shown", DEFAULT_SECTIONS[name]),
                    height=values.get("height") or 0,
                )

    def reset_layout(self):
        for name, shown in DEFAULT_SECTIONS.items():
            self.set_section(name, shown=shown, height=0)

    def performance(self):
        """Rolling timings of message intake and painting (see ``perf.Timings``)."""
        report = self.canvas.perf.stats()
        report["radiant renderer"] = self.renderer_report()
        if self.subscription is not None:
            report["dropped by this window"] = self.dropped + self.subscription.dropped
        return report

    def closeEvent(self, event):
        self.detach()
        self.timer.stop()
        self.close_effects()
        event.accept()

    def close_effects(self):
        """Release the graphics card's resources and let the effects view go, now, in an
        order we choose. The reduced picture is what is left until the window is shown again."""
        if self.plasma is not None:
            self.canvas.gpu = False
            self.plasma.shutdown()
            self.stack.removeWidget(self.plasma)
            self.plasma.hide()
            self.plasma.deleteLater()
            self.plasma = None

    def showEvent(self, event):
        if self.frame_rate is None:
            self.set_frame_rate(None)
            handle = self.window().windowHandle()
            if handle is not None and not getattr(self, "_watching_screen", False):
                self._watching_screen = True  # moved to another display: choose again
                handle.screenChanged.connect(
                    lambda _screen: self.frame_rate is None and self.set_frame_rate(None)
                )
        self.last_tick = perf_counter()
        if not self.timer.isActive():
            self.timer.start(ACTIVE_MS)
        super().showEvent(event)
        QTimer.singleShot(0, self.sync_effects)  # the effects layer, now that there is a window
