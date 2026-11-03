"""Phase One diagnostic control center; no artistic renderer."""
import sys
import zipfile
from pathlib import Path
from threading import Thread

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .engine import Engine
from .models import AnalysisConfig, FeatureFrame, MusicalEvent
from .sources.file import FileAudioSource
from .sources.windows import SystemAudioSource, list_output_devices

ROOT = Path(__file__).resolve().parents[2]


class Discovery(QObject):
    finished = Signal(object, object, str)

    def run(self):
        devices, apps, error = [], [], ""
        try:
            devices = list_output_devices()
            try:
                from .sources.application import list_audio_applications
                apps = list_audio_applications()
            except ImportError:
                pass
        except Exception as exc:
            error = str(exc)
        self.finished.emit(devices, apps, error)


class ControlCenter(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Granular Music Visualizer — Phase One Control Center")
        self.resize(960, 780)
        self.engine, self.subscription = None, None
        self.session_path = None
        self.closing = False
        self.discovery = Discovery()
        self.discovery.finished.connect(self.discovered)
        body = QWidget()
        self.setCentralWidget(body)
        layout = QVBoxLayout(body)
        title = QLabel("Granular Music Visualizer  •  Audio analysis")
        title.setStyleSheet("font-size: 21px; font-weight: 600")
        layout.addWidget(title)
        layout.addWidget(QLabel("Live desktop audio → musical features → timestamped session logs"))
        setup = QGroupBox("Audio source and analysis settings")
        form = QFormLayout(setup)
        self.source_type = QComboBox()
        self.source_type.addItems(["Default system output", "Specific output device",
                                   "Application / process tree", "Audio file"])
        self.devices, self.apps = QComboBox(), QComboBox()
        self.file_path = QLineEdit()
        self.file_path.setPlaceholderText("Choose a WAV, FLAC, OGG, or another supported audio file")
        self.browse = QPushButton("Choose file…")
        self.browse.clicked.connect(self.choose_file)
        file_row = QHBoxLayout()
        file_row.addWidget(self.file_path)
        file_row.addWidget(self.browse)
        self.refresh = QPushButton("Refresh sources")
        self.refresh.clicked.connect(self.refresh_sources)
        self.window_ms, self.hop_ms = QDoubleSpinBox(), QDoubleSpinBox()
        self.window_ms.setRange(20, 200)
        self.window_ms.setValue(46.44)
        self.hop_ms.setRange(5, 50)
        self.hop_ms.setValue(20)
        self.band_edges = QLineEdit("20, 60, 250, 500, 2000, 4000, 6000, 24000")
        self.logging = QCheckBox("Save analysis logs (audio recording is off)")
        self.logging.setChecked(True)
        self.advanced = QCheckBox("Delayed harmonic/percussive analysis — experimental, CPU only")
        self.advanced.setEnabled(False)
        self.advanced.setToolTip("Available after the immediate pipeline is validated")
        form.addRow("Source", self.source_type)
        form.addRow("Output device", self.devices)
        form.addRow("Application", self.apps)
        form.addRow("File", file_row)
        form.addRow("Window / hop (ms)", self._row(self.window_ms, self.hop_ms, self.refresh))
        form.addRow("Frequency band edges (Hz)", self.band_edges)
        form.addRow(self.logging)
        form.addRow(self.advanced)
        layout.addWidget(setup)
        self.source_note = QLabel()
        self.source_note.setWordWrap(True)
        layout.addWidget(self.source_note)
        controls = QHBoxLayout()
        self.start_button, self.stop_button = QPushButton("Start analysis"), QPushButton("Stop analysis")
        self.start_button.clicked.connect(self.start_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.stop_button.setEnabled(False)
        self.open_button, self.export_button = QPushButton("Open session folder"), QPushButton("Export logs…")
        self.open_button.clicked.connect(self.open_session)
        self.export_button.clicked.connect(self.export_session)
        for button in (self.start_button, self.stop_button, self.open_button, self.export_button):
            controls.addWidget(button)
        layout.addLayout(controls)
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        monitor = QGroupBox("Live diagnostics")
        monitor_form = QFormLayout(monitor)
        self.level, self.transient = QProgressBar(), QProgressBar()
        self.level.setRange(0, 100)
        self.transient.setRange(0, 100)
        self.activity = QLabel("Waiting for capture")
        self.tempo = QLabel("Collecting rhythmic context")
        self.performance = QLabel("No measurements yet")
        self.performance.setWordWrap(True)
        self.advanced_status = QLabel("Disabled; stem separation is not running")
        monitor_form.addRow("Audio activity", self.activity)
        monitor_form.addRow("Input level", self.level)
        monitor_form.addRow("Transient strength", self.transient)
        monitor_form.addRow("Tempo estimate", self.tempo)
        monitor_form.addRow("Performance", self.performance)
        monitor_form.addRow("Advanced analysis", self.advanced_status)
        layout.addWidget(monitor)
        self.events = QPlainTextEdit()
        self.events.setReadOnly(True)
        self.events.setMaximumBlockCount(200)
        self.events.setPlaceholderText("Timestamped events appear here. Instrument labels are candidates.")
        layout.addWidget(self.events, 1)
        self.session_label = QLabel("Session logs are local. No raw audio is saved.")
        self.session_label.setWordWrap(True)
        layout.addWidget(self.session_label)
        self.source_type.currentIndexChanged.connect(self.source_changed)
        self.source_changed()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll)
        self.timer.start(50)
        self.refresh_sources()

    @staticmethod
    def _row(*widgets):
        row = QHBoxLayout()
        for widget in widgets:
            row.addWidget(widget)
        return row

    def source_changed(self):
        index = self.source_type.currentIndex()
        self.devices.setEnabled(index == 1)
        self.apps.setEnabled(index == 2)
        self.file_path.setEnabled(index == 3)
        self.browse.setEnabled(index == 3)
        self.source_note.setText(
            "Application capture includes the selected process tree. If unavailable, select Default system output."
            if index == 2 else "Capture uses your existing playback configuration. Volume and routing are unchanged.")

    def refresh_sources(self):
        self.refresh.setEnabled(False)
        Thread(target=self.discovery.run, name="source-discovery", daemon=True).start()

    def discovered(self, devices, apps, error):
        previous = self.devices.currentData()
        self.devices.clear()
        for device in devices:
            self.devices.addItem(device["name"], device["name"])
        if previous is not None:
            self.devices.setCurrentIndex(max(0, self.devices.findData(previous)))
        self.apps.clear()
        for app in apps:
            self.apps.addItem(f"{app['name']} — PID {app['pid']}", app)
        self.refresh.setEnabled(self.engine is None or self.engine.done.is_set())
        if error:
            self.status_label.setText("Source discovery: " + error)

    def choose_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Choose audio file", "", "Audio files (*.wav *.flac *.ogg *.mp3 *.aiff);;All files (*)")
        if path:
            self.file_path.setText(path)

    def start_analysis(self):
        if self.engine is not None and not self.engine.done.is_set():
            return
        try:
            config = AnalysisConfig(window_ms=self.window_ms.value(), hop_ms=self.hop_ms.value(),
                                    band_edges=tuple(float(v.strip()) for v in self.band_edges.text().split(",")),
                                    advanced=self.advanced.isChecked())
            index = self.source_type.currentIndex()
            if index == 3:
                if not Path(self.file_path.text()).is_file():
                    raise ValueError("Choose an existing audio file")
                source = FileAudioSource(self.file_path.text(), realtime=True)
            elif index == 2:
                from .sources.application import ApplicationAudioSource
                app = self.apps.currentData()
                if not app:
                    raise ValueError("No audio application selected. Refresh sources or choose Default system output.")
                source = ApplicationAudioSource(**app)
            else:
                if index == 1 and not self.devices.currentData():
                    raise ValueError("No output device selected")
                source = SystemAudioSource(self.devices.currentData() if index == 1 else None)
            self.engine = Engine(source, config, session_root=ROOT / "sessions", logging=self.logging.isChecked())
            self.subscription = self.engine.bus.subscribe(256)
            self.engine.start()
            self.session_path = self.engine.writer.path if self.engine.writer else None
            self.session_label.setText(str(self.session_path) if self.session_path else "Session logging disabled")
            self.events.clear()
            self.set_running(True)
        except Exception as exc:
            self.status_label.setText("Unable to start: " + str(exc))

    def set_running(self, running):
        for widget in (self.start_button, self.source_type, self.window_ms, self.hop_ms,
                       self.band_edges, self.logging, self.advanced, self.refresh):
            widget.setEnabled(not running)
        self.stop_button.setEnabled(running)
        if not running:
            self.source_changed()
        else:
            for widget in (self.devices, self.apps, self.file_path, self.browse):
                widget.setEnabled(False)

    def stop_analysis(self):
        if self.engine:
            self.engine.stop()
            self.stop_button.setEnabled(False)
            self.status_label.setText("Stopping capture and finalizing logs…")

    def poll(self):
        if not self.engine:
            return
        latest = None
        for _ in range(256):
            message = self.subscription.get(timeout=0)
            if message is None:
                break
            if isinstance(message, FeatureFrame):
                latest = message
            elif isinstance(message, MusicalEvent):
                self.events.appendPlainText(f"{message.timestamp:9.3f}s  {message.event_type}  "
                                            f"confidence {message.confidence:.2f}  [{message.analysis_tier}]")
            else:
                self.status_label.setText(message.state.capitalize() + ": " + message.detail)
        if latest:
            self.activity.setText(f"{'Silence / low activity' if latest.silence else 'Audio detected'}  |  "
                                  f"{latest.sample_rate} Hz, {latest.channels} channels  |  {latest.timestamp:.2f}s")
            self.level.setValue(round(max(0, min(100, (latest.loudness_dbfs+80)/80*100))))
            self.level.setFormat(f"{latest.loudness_dbfs:.1f} dBFS")
            self.transient.setValue(round(latest.transient_strength*100))
            self.tempo.setText(f"{latest.tempo_bpm:.1f} BPM • confidence {latest.tempo_confidence:.2f} (preliminary)"
                               if latest.tempo_bpm else "Uncertain — collecting rhythmic context")
        if self.engine.state == "waiting":
            self.level.setValue(0)
            self.activity.setText("No audio packets — waiting for playback")
        metrics = self.engine.metrics()
        latency = metrics["estimated_capture_center_to_publish_ms"]
        dsp = metrics["dsp_ms"]
        self.performance.setText(
            f"DSP p95: {dsp['p95']:.2f} ms  |  Capture-center age p95: {latency['p95']:.1f} ms (estimated)\n"
            f"CPU: {self.engine.cpu:.1f}% of one core  |  Memory: {self.engine.memory_mb:.1f} MB  |  "
            f"Dropped buffers: {metrics['dropped_blocks']}  |  UI drops: {self.subscription.dropped}"
            if dsp and latency else "Waiting for performance measurements")
        self.advanced_status.setText(metrics["advanced_status"] + "; stem separation is not running")
        self.open_button.setEnabled(self.session_path is not None)
        self.export_button.setEnabled(self.session_path is not None and self.engine.done.is_set())
        if self.engine.done.is_set():
            self.set_running(False)
            if self.engine.error:
                self.status_label.setText("Error: " + self.engine.error)
            if self.closing:
                self.close()

    def open_session(self):
        if self.session_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.session_path.resolve())))

    def export_session(self):
        if not self.session_path or not self.engine.done.is_set():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Export analysis package", "analysis.zip", "ZIP (*.zip)")
        if path:
            try:
                with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
                    for file in self.session_path.rglob("*"):
                        if file.is_file() and file.resolve() != Path(path).resolve():
                            archive.write(file, file.relative_to(self.session_path))
                self.status_label.setText("Exported " + path)
            except Exception as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def closeEvent(self, event):
        if self.engine and not self.engine.done.is_set():
            self.closing = True
            self.stop_analysis()
            event.ignore()
        else:
            event.accept()


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ControlCenter()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
