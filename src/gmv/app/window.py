"""The application: one window. The visualizer fills it; settings slide in from the right."""

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from threading import Thread

from PySide6.QtCore import (
    QEasingCurve,
    QPoint,
    QPropertyAnimation,
    QSettings,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from ..visualizer.state import ROWS
from ..visualizer.window import VisualizerWindow
from .controller import DEFAULT_EDGES, SOURCES, SessionController, Settings, discover

ROOT = Path(__file__).resolve().parents[3]
DRAWER_WIDTH = 440
STYLE = """
QWidget#app { background: #05060a; }
QWidget#topbar { background: #0a0c13; border-bottom: 1px solid #1c2030; }
QFrame#drawer { background: #0d1018; border-left: 1px solid #262b3a; }
QScrollArea, QWidget#drawerBody { background: transparent; border: none; }
QLabel { color: #8089a0; font-size: 12px; }
QLabel#title { color: #e8ebf4; font-size: 15px; font-weight: 600; }
QLabel#heading { color: #e8ebf4; font-size: 11px; font-weight: 600; letter-spacing: 2px;
    padding-top: 14px; }
QLabel#value { color: #c9cfdd; }
QLabel#problem { color: #f59e8b; }
QCheckBox { color: #c9cfdd; font-size: 12px; }
QPushButton, QToolButton, QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {
    color: #e8ebf4; background: #161a26; border: 1px solid #262b3a;
    border-radius: 6px; padding: 5px 10px; font-size: 12px;
}
QPushButton:hover, QToolButton:hover, QComboBox:hover { border-color: #3b4460; }
QPushButton:disabled, QComboBox:disabled, QLineEdit:disabled { color: #5b6377; }
QPushButton#primary { background: #22d3ee; color: #05060a; border: none; font-weight: 600;
    padding: 6px 18px; }
QPushButton#primary[running="true"] { background: #1c2030; color: #e8ebf4;
    border: 1px solid #3b4460; }
QPushButton#apply { border-color: #f5c451; color: #f5c451; }
QToolButton::menu-indicator { image: none; }
QComboBox QAbstractItemView, QMenu {
    color: #e8ebf4; background: #161a26; border: 1px solid #262b3a;
    selection-background-color: #262b3a;
}
QSlider::groove:horizontal { height: 4px; background: #262b3a; border-radius: 2px; }
QSlider::handle:horizontal { width: 12px; margin: -5px 0; border-radius: 6px;
    background: #22d3ee; }
"""


def row(*widgets):
    box = QHBoxLayout()
    box.setSpacing(8)
    for widget in widgets:
        box.addWidget(widget, 1 if isinstance(widget, (QLineEdit, QComboBox, QSlider)) else 0)
    return box


class SettingsDrawer(QFrame):
    """Every control that is not Start/Stop. Reading ``settings()`` never has side effects."""

    engine_changed = Signal()  # something that needs a restart to take effect
    refresh_requested = Signal()

    def __init__(self, view, parent=None):
        super().__init__(parent, objectName="drawer")
        self.view = view
        self.setFixedWidth(DRAWER_WIDTH)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        head = QHBoxLayout()
        head.setContentsMargins(22, 16, 16, 6)
        head.addWidget(QLabel("Settings", objectName="title"), 1)
        self.close_button = QPushButton("Close")
        head.addWidget(self.close_button)
        outer.addLayout(head)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget(objectName="drawerBody")
        self.form = QFormLayout(body)
        self.form.setContentsMargins(22, 0, 18, 22)
        self.form.setVerticalSpacing(9)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)
        self.build_source()
        self.build_analysis()
        self.build_visualizer()
        self.build_sessions()
        self.build_diagnostics()
        self.build_advanced()
        self.source_changed()

    def heading(self, text, note=None):
        self.form.addRow(QLabel(text.upper(), objectName="heading"))
        if note:
            label = QLabel(note)
            label.setWordWrap(True)
            self.form.addRow(label)

    # -- sections ------------------------------------------------------------------------------

    def build_source(self):
        self.heading("Audio source")
        self.source = QComboBox()
        for key, title in SOURCES:
            self.source.addItem(title, key)
        self.devices, self.applications = QComboBox(), QComboBox()
        self.file = QLineEdit()
        self.file.setPlaceholderText("WAV, FLAC, OGG or another supported file")
        self.browse_file = QPushButton("Choose…")
        self.browse_file.clicked.connect(self.choose_file)
        self.refresh = QPushButton("Refresh devices and applications")
        self.refresh.clicked.connect(self.refresh_requested)
        self.source_note = QLabel()
        self.source_note.setWordWrap(True)
        self.form.addRow("Capture", self.source)
        self.form.addRow("Output device", self.devices)
        self.form.addRow("Application", self.applications)
        self.form.addRow("File", row(self.file, self.browse_file))
        self.form.addRow(self.refresh)
        self.form.addRow(self.source_note)
        self.source.currentIndexChanged.connect(self.source_changed)
        for signal in (
            self.source.currentIndexChanged,
            self.devices.currentIndexChanged,
            self.applications.currentIndexChanged,
            self.file.textChanged,
        ):
            signal.connect(self.engine_changed)

    def build_analysis(self):
        self.heading("Analysis")
        self.activity_model, self.guitar_model = QLineEdit(), QLineEdit()
        self.activity_model.setPlaceholderText("Drums / bass / vocals weights (.npz) — none")
        self.guitar_model.setPlaceholderText("Guitar weights (.npz) — none")
        tip = "A local file; weights are not part of the repository."
        self.activity_model.setToolTip(tip)
        self.guitar_model.setToolTip(tip)
        browse_activity, browse_guitar = QPushButton("Choose…"), QPushButton("Choose…")
        browse_activity.clicked.connect(lambda: self.choose_weights(self.activity_model))
        browse_guitar.clicked.connect(lambda: self.choose_weights(self.guitar_model))
        self.beat_tracker = QCheckBox("Beat tracker (experimental, not validated)")
        self.capabilities = QLabel(objectName="value")
        self.capabilities.setWordWrap(True)
        self.form.addRow("Instruments", row(self.activity_model, browse_activity))
        self.form.addRow("Guitar", row(self.guitar_model, browse_guitar))
        self.form.addRow(self.beat_tracker)
        self.form.addRow(self.capabilities)
        for signal in (
            self.activity_model.textChanged,
            self.guitar_model.textChanged,
            self.beat_tracker.toggled,
        ):
            signal.connect(self.engine_changed)
            signal.connect(self.show_capabilities)
        self.show_capabilities()

    def build_visualizer(self):
        self.heading("Visualizer")
        self.mode = QComboBox()
        self.mode.addItem("Spectrum bars", "bars")
        self.mode.addItem("Radial", "radial")
        self.mode.currentIndexChanged.connect(lambda: self.view.set_mode(self.mode.currentData()))
        self.lanes = QToolButton(text="Choose lanes and timeline rows  ▾")
        self.lanes.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        menu = QMenu(self.lanes)
        menu.aboutToShow.connect(self.fill_lane_menu)
        self.lanes.setMenu(menu)
        self.show_beats = QCheckBox("Show beats when the beat tracker is on")
        self.show_beats.setChecked(True)
        self.show_beats.toggled.connect(self.view.set_beats)
        self.smoothing = QSlider(Qt.Orientation.Horizontal)
        self.smoothing.setRange(5, 60)
        self.smoothing.setValue(22)
        self.smoothing.valueChanged.connect(lambda value: self.view.set_smoothing(value / 100))
        self.gain = QSlider(Qt.Orientation.Horizontal)
        self.gain.setRange(-12, 18)
        self.gain.valueChanged.connect(lambda value: self.view.set_gain(float(value)))
        self.window = QComboBox()
        for seconds in (10, 15, 20, 30):
            self.window.addItem(f"{seconds} seconds", seconds)
        self.window.setCurrentIndex(1)
        self.window.currentIndexChanged.connect(
            lambda: self.view.set_window(self.window.currentData())
        )
        self.reset = QPushButton("Reset history")
        self.reset.clicked.connect(self.view.state.reset)
        self.fullscreen = QPushButton("Full screen (F11)")
        self.form.addRow("Layout", self.mode)
        self.form.addRow("Show", self.lanes)
        self.form.addRow(self.show_beats)
        self.form.addRow("Smoothing", self.smoothing)
        self.form.addRow("Intensity", self.gain)
        self.form.addRow("Timeline", self.window)
        self.form.addRow(row(self.reset, self.fullscreen))

    def build_sessions(self):
        self.heading("Sessions")
        self.logging = QCheckBox("Save analysis logs for each session (never audio)")
        self.logging.setChecked(True)
        self.logging.toggled.connect(self.engine_changed)
        self.session = QLabel("No session yet", objectName="value")
        self.session.setWordWrap(True)
        self.open_session = QPushButton("Open folder")
        self.export_session = QPushButton("Export logs…")
        self.replay = QPushButton("Replay a session…")
        self.form.addRow(self.logging)
        self.form.addRow(self.session)
        self.form.addRow(row(self.open_session, self.export_session, self.replay))

    def build_diagnostics(self):
        self.heading("Diagnostics")
        self.diagnostics = QLabel("Not started", objectName="value")
        self.diagnostics.setWordWrap(True)
        self.diagnostics.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.form.addRow(self.diagnostics)

    def build_advanced(self):
        self.heading("Advanced", "Rarely needed. Changes apply at the next start.")
        self.window_ms, self.hop_ms = QDoubleSpinBox(), QDoubleSpinBox()
        self.window_ms.setRange(20, 200)
        self.window_ms.setValue(46.44)
        self.hop_ms.setRange(5, 50)
        self.hop_ms.setValue(20)
        self.band_edges = QLineEdit(", ".join(f"{edge:g}" for edge in DEFAULT_EDGES))
        self.delayed = QCheckBox("Delayed harmonic/percussive analysis (experimental)")
        self.delayed.setToolTip("Two seconds late; does not identify instruments")
        self.debug_audio = QSpinBox()
        self.debug_audio.setRange(0, 60)
        self.debug_audio.setSpecialValueText("Off — no audio saved")
        self.debug_audio.setSuffix(" seconds")
        self.debug_audio.setToolTip("Explicit opt-in: saves at most this much audio per session")
        self.form.addRow("Window / hop (ms)", row(self.window_ms, self.hop_ms))
        self.form.addRow("Band edges (Hz)", self.band_edges)
        self.form.addRow(self.delayed)
        self.form.addRow("Debug recording", self.debug_audio)
        for signal in (
            self.window_ms.valueChanged,
            self.hop_ms.valueChanged,
            self.band_edges.textChanged,
            self.delayed.toggled,
            self.debug_audio.valueChanged,
        ):
            signal.connect(self.engine_changed)

    # -- behaviour ---------------------------------------------------------------------------

    def source_changed(self):
        kind = self.source.currentData()
        self.devices.setEnabled(kind == "device")
        self.applications.setEnabled(kind == "application")
        self.file.setEnabled(kind == "file")
        self.browse_file.setEnabled(kind == "file")
        self.source_note.setText(
            {
                "system": "Everything playing on the default output. Volume and routing are not changed.",
                "device": "Everything playing on the chosen output device.",
                "application": "Only the chosen application and its child processes. It never "
                "falls back to capturing everything.",
                "file": "The file is analysed at playback speed; it is not played aloud.",
            }[kind]
        )

    def show_capabilities(self):
        def state(path, what):
            if not path.strip():
                return f"{what}: no weights chosen, lane shows no detector output"
            return f"{what}: weights chosen" if Path(path).is_file() else f"{what}: file not found"

        self.capabilities.setText(
            "\n".join(
                (
                    "Spectrum, level, onsets, changes: always on",
                    state(self.activity_model.text(), "Drums, bass, vocals"),
                    state(self.guitar_model.text(), "Guitar"),
                    "Beats: "
                    + ("experimental tracker on" if self.beat_tracker.isChecked() else "off"),
                )
            )
        )

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose audio file", "", "Audio (*.wav *.flac *.ogg *.mp3 *.aiff);;All files (*)"
        )
        if path:
            self.file.setText(path)

    def choose_weights(self, target):
        path, _ = QFileDialog.getOpenFileName(
            self, "Choose weights", str(ROOT / ".model-cache"), "Weights (*.npz)"
        )
        if path:
            target.setText(path)

    def fill_lane_menu(self):
        menu = self.lanes.menu()
        menu.clear()
        entries = list(ROWS) + [(lane.key, lane.title) for lane, _ in self.view.state.board.lanes()]
        for position, (key, title) in enumerate(entries):
            if position == len(ROWS):
                menu.addSeparator()
            action = QAction(title, menu, checkable=True)
            action.setChecked(key not in self.view.canvas.hidden)
            action.toggled.connect(lambda shown, key=key: self.view.set_shown(key, shown))
            menu.addAction(action)

    def set_sources(self, devices, applications):
        for box, items in (
            (self.devices, [(d["name"], d["name"]) for d in devices]),
            (self.applications, [(f"{a['name']} — PID {a['pid']}", a) for a in applications]),
        ):
            previous = box.currentText()
            box.blockSignals(True)
            box.clear()
            for text, data in items:
                box.addItem(text, data)
            box.setCurrentIndex(max(0, box.findText(previous)))
            box.blockSignals(False)

    def settings(self):
        try:
            edges = tuple(float(v) for v in self.band_edges.text().split(","))
        except ValueError:
            edges = ()
        return Settings(
            source=self.source.currentData(),
            device=self.devices.currentData(),
            application=self.applications.currentData(),
            application_label=self.applications.currentText(),
            file=self.file.text().strip(),
            activity_model=self.activity_model.text().strip(),
            guitar_model=self.guitar_model.text().strip(),
            beat_tracker=self.beat_tracker.isChecked(),
            logging=self.logging.isChecked(),
            delayed_analysis=self.delayed.isChecked(),
            debug_audio_seconds=self.debug_audio.value(),
            window_ms=self.window_ms.value(),
            hop_ms=self.hop_ms.value(),
            band_edges=edges,
        )

    def remembered(self):
        """What is worth keeping between runs (no process ids, nothing from Advanced)."""
        chosen = asdict(self.settings())
        keep = (
            "source",
            "device",
            "file",
            "activity_model",
            "guitar_model",
            "beat_tracker",
            "logging",
        )
        return {
            **{key: chosen[key] for key in keep},
            "mode": self.mode.currentData(),
            "smoothing": self.smoothing.value(),
            "gain": self.gain.value(),
            "window": self.window.currentData(),
            "show_beats": self.show_beats.isChecked(),
        }

    def restore(self, saved):
        self.source.setCurrentIndex(max(0, self.source.findData(saved.get("source", "system"))))
        self.wanted_device = saved.get("device")
        self.file.setText(saved.get("file", ""))
        for box, key in (
            (self.activity_model, "activity_model"),
            (self.guitar_model, "guitar_model"),
        ):
            if saved.get(key) and Path(saved[key]).is_file():
                box.setText(saved[key])
        self.beat_tracker.setChecked(bool(saved.get("beat_tracker", False)))
        self.logging.setChecked(bool(saved.get("logging", True)))
        self.mode.setCurrentIndex(max(0, self.mode.findData(saved.get("mode", "bars"))))
        self.smoothing.setValue(int(saved.get("smoothing", 22)))
        self.gain.setValue(int(saved.get("gain", 0)))
        self.window.setCurrentIndex(max(0, self.window.findData(saved.get("window", 15))))
        self.show_beats.setChecked(bool(saved.get("show_beats", True)))


class MainWindow(QWidget):
    discovered = Signal(object, object, str)

    def __init__(self, session_root=None, remember=True, discover_sources=True):
        super().__init__(objectName="app")
        self.setWindowTitle("Granular Music Visualizer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(STYLE)
        self.resize(1440, 900)
        self.controller = SessionController(session_root or ROOT / "sessions")
        self.remember = remember
        self.restart_wanted = False
        self.problem = ""
        self.view = VisualizerWindow(self)
        self.drawer = SettingsDrawer(self.view, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(self.build_top_bar())
        layout.addWidget(self.view, 1)
        self.drawer.hide()
        self.drawer.raise_()
        self.slide = QPropertyAnimation(self.drawer, b"pos", self)
        self.slide.setDuration(180)
        self.slide.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.slide.finished.connect(self.slide_finished)
        self.drawer_open = False
        self.drawer.close_button.clicked.connect(lambda: self.show_settings(False))
        self.drawer.engine_changed.connect(self.settings_changed)
        self.drawer.refresh_requested.connect(self.refresh_sources)
        self.drawer.fullscreen.clicked.connect(self.toggle_fullscreen)
        self.drawer.open_session.clicked.connect(self.open_session_folder)
        self.drawer.export_session.clicked.connect(self.export_session)
        self.drawer.replay.clicked.connect(self.choose_replay)
        self.discovered.connect(self.sources_found)
        for name, default in (
            (
                "activity_model",
                os.environ.get("GMV_ACTIVITY_MODEL") or ROOT / ".model-cache/activity_v2.npz",
            ),
            (
                "guitar_model",
                os.environ.get("GMV_GUITAR_MODEL") or ROOT / ".model-cache/guitar_v1.npz",
            ),
        ):
            if Path(default).is_file():
                getattr(self.drawer, name).setText(str(default))
        self.drawer.wanted_device = None
        if remember:
            try:
                self.drawer.restore(json.loads(QSettings().value("settings", "{}")))
            except (TypeError, ValueError):
                pass
        self.poller = QTimer(self)
        self.poller.timeout.connect(self.poll)
        self.poller.start(250)
        if discover_sources:
            self.refresh_sources()
        self.poll()

    def build_top_bar(self):
        bar = QWidget(objectName="topbar")
        line = QHBoxLayout(bar)
        line.setContentsMargins(22, 9, 16, 9)
        line.setSpacing(12)
        line.addWidget(QLabel("Granular Music Visualizer", objectName="title"))
        self.status = QLabel()
        self.status.setContentsMargins(16, 0, 0, 0)
        line.addWidget(self.status, 1)
        self.apply = QPushButton("Apply changes", objectName="apply")
        self.apply.setToolTip("Restart analysis with the settings you changed")
        self.apply.clicked.connect(self.apply_changes)
        self.apply.hide()
        self.toggle = QPushButton("Start", objectName="primary")
        self.toggle.clicked.connect(self.start_or_stop)
        self.settings_button = QPushButton("Settings")
        self.settings_button.clicked.connect(lambda: self.show_settings(not self.drawer_open))
        for widget in (self.apply, self.toggle, self.settings_button):
            line.addWidget(widget)
        return bar

    # -- settings drawer -----------------------------------------------------------------------

    def drawer_position(self, opened):
        top = self.view.geometry().top()
        return QPoint(self.width() - (DRAWER_WIDTH if opened else 0), top)

    def show_settings(self, opened):
        """Slide the drawer. It only covers part of the picture and never stops analysis."""
        self.drawer_open = opened
        self.drawer.setFixedHeight(self.height() - self.view.geometry().top())
        self.slide.stop()
        if opened:
            self.drawer.move(self.drawer_position(False))
            self.drawer.show()
            self.drawer.raise_()
        self.slide.setStartValue(self.drawer.pos())
        self.slide.setEndValue(self.drawer_position(opened))
        self.slide.start()

    def slide_finished(self):
        if not self.drawer_open:
            self.drawer.hide()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.drawer.setFixedHeight(max(100, self.height() - self.view.geometry().top()))
        self.drawer.move(self.drawer_position(self.drawer_open))

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_F11:
            self.toggle_fullscreen()
        elif event.key() == Qt.Key.Key_Escape and self.drawer_open:
            self.show_settings(False)
        elif event.key() == Qt.Key.Key_Escape and self.isFullScreen():
            self.showNormal()
        elif event.key() == Qt.Key.Key_D:
            self.view.canvas.show_stats = not self.view.canvas.show_stats
        else:
            super().keyPressEvent(event)

    def toggle_fullscreen(self):
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    # -- sources -------------------------------------------------------------------------------

    def refresh_sources(self):
        self.drawer.refresh.setEnabled(False)
        Thread(
            target=lambda: self.discovered.emit(*discover()), name="source-discovery", daemon=True
        ).start()

    def sources_found(self, devices, applications, error):
        self.drawer.set_sources(devices, applications)
        if self.drawer.wanted_device:
            index = self.drawer.devices.findText(self.drawer.wanted_device)
            if index >= 0:
                self.drawer.devices.setCurrentIndex(index)
            self.drawer.wanted_device = None
        self.drawer.refresh.setEnabled(True)
        if error:
            self.problem = "Source discovery: " + error

    # -- start, stop, replay -------------------------------------------------------------------

    def start(self):
        self.problem = ""
        try:
            engine = self.controller.prepare(self.drawer.settings())
            self.view.attach(engine.bus, **self.controller.describe())
            engine.start()
        except Exception as exc:
            self.problem = "Unable to start: " + str(exc)
            if not self.controller.running:
                self.view.detach()
        self.poll()

    def stop(self):
        self.restart_wanted = False
        if self.controller.replaying is not None:
            self.view.detach()
            self.controller.stop_replay()
        self.controller.stop()
        self.poll()

    def start_or_stop(self):
        if self.controller.running or self.controller.replaying is not None:
            self.stop()
        else:
            self.start()

    def apply_changes(self):
        self.restart_wanted = True
        self.controller.stop()
        self.poll()

    def settings_changed(self):
        self.poll()

    def replay(self, path, speed=1.0):
        self.problem = ""
        try:
            session = self.controller.replay(path, speed)
            self.view.attach(
                session.bus,
                source=session.title,
                beat_tracker=session.configuration.get("experimental_beat_phase"),
                activity_model=session.configuration.get("activity_model_path"),
            )
            session.start()
        except Exception as exc:
            self.problem = "Unable to replay: " + str(exc)
        self.poll()

    def choose_replay(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose a recorded session",
            str(self.controller.session_root),
            "Messages (messages.jsonl)",
        )
        if path:
            self.replay(path)

    def open_session_folder(self):
        if self.controller.session_path:
            QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(self.controller.session_path.resolve()))
            )

    def export_session(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export logs", "analysis.zip", "ZIP (*.zip)")
        if path:
            try:
                self.controller.export(path)
                self.problem = ""
            except Exception as exc:
                self.problem = "Export failed: " + str(exc)

    # -- keeping the interface current -----------------------------------------------------------

    def poll(self):
        controller, drawer = self.controller, self.drawer
        running, replaying = controller.running, controller.replaying
        if replaying is not None and replaying.error:
            self.problem, replaying.error = "Replay failed: " + replaying.error, None
        if self.restart_wanted and not running:
            self.restart_wanted = False
            self.start()
            return
        engine = controller.engine
        if engine is not None and engine.error and not running:
            self.problem = "Error: " + engine.error
        stopping = running and engine.stop_event.is_set()
        busy = running or replaying is not None
        self.toggle.setText("Stopping…" if stopping else "Stop" if busy else "Start")
        self.toggle.setEnabled(not stopping)
        self.toggle.setProperty("running", "true" if busy else "false")
        self.toggle.style().polish(self.toggle)
        pending = running and not stopping and drawer.settings() != controller.settings
        self.apply.setVisible(pending)
        if self.problem:
            text = self.problem
        elif replaying is not None:
            text = f"Replaying {replaying.title}" + ("  ·  finished" if replaying.finished else "")
        elif running:
            text = f"{controller.settings.source_label()}  ·  {engine.detail}"
        else:
            text = f"{drawer.settings().source_label()}  ·  stopped"
        self.status.setObjectName("problem" if self.problem else "")
        self.status.style().polish(self.status)
        self.status.setText(text)
        drawer.replay.setEnabled(not running)
        path = controller.session_path
        drawer.open_session.setEnabled(path is not None)
        drawer.export_session.setEnabled(path is not None and not running)
        drawer.session.setText(
            f"Last session: {path.name}" if path else "No saved session in this run"
        )
        if self.drawer_open:
            canvas, lines = self.view.canvas, controller.diagnostics()
            if canvas.paint_ms:
                ordered = sorted(canvas.paint_ms)
                lines["Display"] = (
                    f"drawing p95 {ordered[int(len(ordered) * 0.95)]:.1f} ms per frame · "
                    f"messages dropped by the display {canvas.info.get('dropped', 0)}"
                )
            drawer.diagnostics.setText("\n".join(f"{key}: {value}" for key, value in lines.items()))

    def closeEvent(self, event):
        self.poller.stop()
        if self.remember:
            QSettings().setValue("settings", json.dumps(self.drawer.remembered()))
        self.view.close()
        self.controller.shutdown()
        event.accept()


def application():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("GranularMusicVisualizer")
    app.setApplicationName("Granular Music Visualizer")
    app.setStyle("Fusion")
    return app


def main(replay=None, speed=1.0):
    app = application()
    window = MainWindow()
    window.show()
    if replay:
        window.replay(replay, speed)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
