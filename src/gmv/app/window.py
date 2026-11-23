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
    QPointF,
    QPropertyAnimation,
    QSettings,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QAction, QColor, QDesktopServices, QIcon, QPainter, QPen
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QColorDialog,
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

from .. import paths
from ..visualizer.state import ROWS
from ..visualizer.themes import DEFAULT as DEFAULT_THEME
from ..visualizer.themes import THEMES
from ..visualizer.themes import radiant_palette
from ..visualizer.window import VisualizerWindow
from .controller import DEFAULT_EDGES, SOURCES, SessionController, Settings, discover

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
QComboBox { padding-right: 26px; }
QComboBox::drop-down { subcontrol-origin: border; subcontrol-position: center right;
    width: 26px; border: none; }
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


class Combo(QComboBox):
    """Drop-down with its arrow drawn inside the box. The style sheet's border and padding
    leave the platform arrow cut off at the right edge, so it is switched off there."""

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor("#c9cfdd" if self.isEnabled() else "#5b6377"), 1.6)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        x, y = self.width() - 16.0, self.height() / 2 + 0.5
        painter.drawPolyline([QPointF(x - 4, y - 2), QPointF(x, y + 2), QPointF(x + 4, y - 2)])
        painter.end()


def row(*widgets):
    box = QHBoxLayout()
    box.setSpacing(8)
    for widget in widgets:
        box.addWidget(widget, 1 if isinstance(widget, (QLineEdit, QComboBox, QSlider)) else 0)
    return box


class SettingsDrawer(QFrame):
    """Every control that is not Start/Stop. Reading ``settings()`` never has side effects."""

    engine_changed = Signal()  # something that needs a restart to take effect
    sensitivity_changed = Signal(str, float)  # applies to the running analysis at once
    refresh_requested = Signal()
    colours_changed = Signal()  # the colours instruments are drawn in may have changed

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
        self.build_instruments()
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
        self.source = Combo()
        for key, title in SOURCES:
            self.source.addItem(title, key)
        self.devices, self.applications = Combo(), Combo()
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
        self.guitar_model.setPlaceholderText("Guitar weights (.npz) or model (.json) — off")
        tip = "A local file; weights are not part of the repository."
        self.activity_model.setToolTip(tip)
        self.guitar_model.setToolTip(
            tip + " A .json file selects a context recognition model: it reports about a"
            " second late and marks guitar as detected, candidate or absent."
        )
        browse_activity, browse_guitar = QPushButton("Choose…"), QPushButton("Choose…")
        browse_activity.clicked.connect(lambda: self.choose_weights(self.activity_model))
        browse_guitar.clicked.connect(lambda: self.choose_weights(self.guitar_model))
        self.beat_tracker = QCheckBox("Beat tracker (experimental, not validated)")
        self.pitch = QCheckBox("Note candidates (experimental; any pitched instrument)")
        self.pitch.setChecked(True)
        self.pitch.setToolTip(
            "Pitches found in the guitar register. They are not attributed to an instrument "
            "unless a guitar detector is loaded and active."
        )
        self.capabilities = QLabel(objectName="value")
        self.capabilities.setWordWrap(True)
        self.form.addRow("Instruments", row(self.activity_model, browse_activity))
        self.form.addRow("Guitar", row(self.guitar_model, browse_guitar))
        self.form.addRow(self.beat_tracker)
        self.form.addRow(self.pitch)
        self.form.addRow(self.capabilities)
        for signal in (
            self.activity_model.textChanged,
            self.guitar_model.textChanged,
            self.beat_tracker.toggled,
            self.pitch.toggled,
        ):
            signal.connect(self.engine_changed)
            signal.connect(self.show_capabilities)
        self.show_capabilities()

    def build_visualizer(self):
        self.heading("Visualizer")
        self.mode = Combo()
        self.mode.addItem("Spectrum bars", "bars")
        self.mode.addItem("Radial", "radial")
        self.mode.currentIndexChanged.connect(lambda: self.view.set_mode(self.mode.currentData()))
        self.mode.currentIndexChanged.connect(lambda: self.colours_changed.emit())
        self.radial_theme = Combo()
        for key, title in THEMES:
            self.radial_theme.addItem(title, key)
        self.radial_theme.setToolTip(
            "How the Radial layout is drawn. Classic is the radial view as it always was; "
            "Radiant is a luminous concentric picture with a palette of its own. Only the "
            "drawing: nothing about the analysis, and Spectrum bars is not affected."
        )
        self.radial_theme.currentIndexChanged.connect(self.theme_changed)
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
        self.window = Combo()
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
        self.form.addRow("Radial theme", self.radial_theme)
        self.build_radiant_palette()
        self.form.addRow("Show", self.lanes)
        self.form.addRow(self.show_beats)
        self.form.addRow("Smoothing", self.smoothing)
        self.form.addRow("Intensity", self.gain)
        self.form.addRow("Timeline", self.window)
        self.frame_rate = Combo()
        self.frame_rate.addItem("Automatic (a whole fraction of the display's rate)", "auto")
        self.frame_rate.addItem("Every refresh of the display (more CPU)", "display")
        self.frame_rate.setToolTip(
            "Automatic draws at the display's refresh rate divided by a whole number, near "
            "60 to 72 frames a second, so every frame is on screen equally long."
        )
        self.frame_rate.currentIndexChanged.connect(self.apply_frame_rate)
        self.form.addRow("Frame rate", self.frame_rate)
        self.intensity_colour = QCheckBox("Colour the spectrum by musical intensity")
        self.intensity_colour.setChecked(True)
        self.intensity_colour.setToolTip(
            "The spectrum keeps its indigo in quiet passages and turns towards a dark red as "
            "the music as a whole gets louder and fuller. Instrument colours never change. "
            "Only the drawing: nothing about the analysis."
        )
        self.intensity_colour.toggled.connect(self.view.set_intensity_colour)
        self.form.addRow(self.intensity_colour)
        self.reactive_background = QCheckBox("Instrument-reactive background")
        self.reactive_background.setChecked(True)
        self.reactive_background.setToolTip(
            "Match the radial background illumination to the currently dominant instrument. "
            "Off: the neutral glow. Only the drawing: nothing about the analysis."
        )
        self.reactive_background.toggled.connect(self.view.set_reactive_background)
        self.form.addRow(self.reactive_background)
        self.expressive = QCheckBox("Expressive Mode")
        self.expressive.setToolTip(
            "Amplify the visual impact of instrument animations, spectrum movement, and "
            "illumination without changing audio analysis."
        )
        self.expressive.toggled.connect(self.view.set_expressive)
        self.form.addRow(self.expressive)
        self.form.addRow(row(self.reset, self.fullscreen))

    # -- radial theme and the Radiant palette ---------------------------------------------------

    def build_radiant_palette(self):
        """The Radiant theme's colours: a preset to start from and one control per role,
        in groups that fold away. Shown only while Radiant is the radial theme; the
        colours are kept whatever is shown."""
        self.radiant_box = QWidget()
        box = QVBoxLayout(self.radiant_box)
        box.setContentsMargins(0, 0, 0, 0)
        box.setSpacing(6)
        self.radiant_effects = Combo()
        self.radiant_effects.addItem("Plasma effects on the graphics card (automatic)", "auto")
        self.radiant_effects.addItem("Off: geometry only (phase 1 drawing)", "off")
        self.radiant_effects.setToolTip(
            "The plasma, energy rings, filaments, core and bloom are drawn by shaders on the "
            "graphics card when it has OpenGL 3.3. Without it, or switched off here, Radiant "
            "is drawn as its geometry alone. Diagnostics says which is in use."
        )
        self.radiant_effects.currentIndexChanged.connect(
            lambda: self.view.set_radiant_effects(self.radiant_effects.currentData())
        )
        box.addLayout(row(QLabel("Effects"), self.radiant_effects))
        self.radiant_fold = QToolButton(text="Radiant palette  \u25b8")
        self.radiant_fold.setCheckable(True)
        self.radiant_fold.setToolTip("Choose the colour of every part of the Radiant theme.")
        box.addWidget(self.radiant_fold)
        self.radiant_body = QWidget()
        body = QVBoxLayout(self.radiant_body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(6)
        box.addWidget(self.radiant_body)
        self.radiant_body.hide()
        self.radiant_fold.toggled.connect(
            lambda on: (
                self.radiant_body.setVisible(on),
                self.radiant_fold.setText("Radiant palette  " + ("\u25be" if on else "\u25b8")),
            )
        )
        self.radiant_preset = Combo()
        for key, (title, _) in radiant_palette.PRESETS.items():
            self.radiant_preset.addItem(title, key)
        self.radiant_preset.addItem("Custom colours", radiant_palette.CUSTOM)
        self.radiant_preset.setToolTip(
            "A preset is a starting point: choose one, then change any colour you like."
        )
        self.radiant_preset.activated.connect(self.choose_radiant_preset)
        self.radiant_restore = QPushButton("Restore Radiant defaults")
        self.radiant_restore.setToolTip("Put back the original Radiant colours. Nothing else changes.")
        self.radiant_restore.clicked.connect(
            lambda: self.choose_radiant_preset(preset=radiant_palette.FACTORY)
        )
        body.addLayout(row(QLabel("Preset"), self.radiant_preset))
        body.addWidget(self.radiant_restore)
        self.radiant_controls = {}  # role -> (swatch button, hex field)
        for group, group_title in radiant_palette.GROUPS:
            fold = QToolButton(text=group_title + "  \u25b8")
            fold.setCheckable(True)
            panel = QWidget()
            form = QFormLayout(panel)
            form.setContentsMargins(8, 0, 0, 4)
            form.setVerticalSpacing(5)
            panel.hide()
            fold.toggled.connect(
                lambda on, panel=panel, fold=fold, title=group_title: (
                    panel.setVisible(on),
                    fold.setText(title + "  " + ("\u25be" if on else "\u25b8")),
                )
            )
            body.addWidget(fold)
            body.addWidget(panel)
            for role, title, role_group, about in radiant_palette.ROLES:
                if role_group != group:
                    continue
                swatch = QPushButton()
                swatch.setFixedSize(46, 24)
                swatch.setCursor(Qt.CursorShape.PointingHandCursor)
                swatch.setToolTip(about + " Click to choose a colour.")
                swatch.setAccessibleName(f"{title} colour")
                swatch.clicked.connect(lambda _=False, role=role: self.pick_radiant_colour(role))
                code = QLineEdit()
                code.setMaxLength(7)
                code.setFixedWidth(84)
                code.setToolTip("The colour as #rrggbb. Type one and press Enter.")
                code.setAccessibleName(f"{title} colour as hexadecimal")
                code.editingFinished.connect(
                    lambda role=role, code=code: self.set_radiant_colour(role, code.text())
                )
                line = QHBoxLayout()
                line.setSpacing(8)
                line.addWidget(swatch)
                line.addWidget(code)
                line.addStretch(1)
                form.addRow(title, line)
                self.radiant_controls[role] = (swatch, code)
        self.form.addRow(self.radiant_box)
        self.radiant_box.hide()
        self.sync_radiant_controls()

    def theme_changed(self):
        theme = self.radial_theme.currentData()
        self.view.set_radial_theme(theme)
        self.radiant_box.setVisible(theme == "radiant")
        self.colours_changed.emit()

    def sync_radiant_controls(self):
        """Show the palette as it is: every swatch, every code, and the preset's name."""
        palette = self.view.radiant_palette
        for role, (swatch, code) in self.radiant_controls.items():
            colour = palette[role]
            swatch.setStyleSheet(
                f"QPushButton {{ background: {colour}; border: 1px solid #5b6377; border-radius: 5px; }}"
                "QPushButton:hover { border-color: #e8ebf4; }"
            )
            if code.text() != colour:
                code.setText(colour)
        self.radiant_preset.setCurrentIndex(max(0, self.radiant_preset.findData(palette.preset)))
        self.colours_changed.emit()

    def set_radiant_colour(self, role, colour):
        """One role, one colour: drawn from the next frame on. Something that is not a
        colour is refused and the field shows the colour in use again."""
        taken = self.view.set_radiant_colour(role, colour)
        self.sync_radiant_controls()
        return taken

    def choose_radiant_preset(self, _index=None, preset=None):
        preset = preset or self.radiant_preset.currentData()
        if preset != radiant_palette.CUSTOM:  # "Custom colours" only names what is there
            self.view.set_radiant_preset(preset)
        self.sync_radiant_controls()

    def pick_radiant_colour(self, role):
        """Qt's colour dialog for one role (any colour; RGB, HSV or #rrggbb). The picture
        follows the dialog while it is open; cancelling puts the colours back."""
        palette = self.view.radiant_palette
        before = palette.saved()
        dialog = QColorDialog(QColor(palette[role]), self)
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog)  # has a #rrggbb field
        dialog.setWindowTitle(dict((r[0], r[1]) for r in radiant_palette.ROLES)[role])
        dialog.currentColorChanged.connect(
            lambda colour: self.set_radiant_colour(role, colour.name())
        )
        if dialog.exec() != QColorDialog.DialogCode.Accepted:
            palette.load(before)
            self.view.canvas.update()
            self.sync_radiant_controls()

    def apply_frame_rate(self):
        if self.frame_rate.currentData() == "display":
            screen = self.view.screen()
            self.view.set_frame_rate(screen.refreshRate() if screen is not None else 60.0)
        else:
            self.view.set_frame_rate(None)

    def build_instruments(self):
        """One block of controls per instrument layer, built from the layer registry: a new
        instrument gets its controls without any change here."""
        self.heading(
            "Instrument layers",
            "Each instrument is drawn from its own analysis. Sensitivity changes what the "
            "analysis reports; intensity only changes how strongly it is drawn.",
        )
        self.spectrum_place = Combo()
        for key, title in (
            ("behind", "Dimmed behind the instruments"),
            ("beside", "Full brightness"),
            ("hidden", "Hidden"),
        ):
            self.spectrum_place.addItem(title, key)
        self.spectrum_place.currentIndexChanged.connect(
            lambda: self.view.set_spectrum_place(self.spectrum_place.currentData())
        )
        self.form.addRow("Whole-mix spectrum", self.spectrum_place)
        self.radial_rings = Combo()
        self.radial_rings.addItem("Live: each instrument as it plays now", "live")
        self.radial_rings.addItem("History: the last six seconds, swept by a hand", "history")
        self.radial_rings.setToolTip(
            "How the instrument rings of the Radial view are drawn. The timeline and the "
            "detail lanes show the history either way."
        )
        self.radial_rings.currentIndexChanged.connect(
            lambda: self.view.set_radial_rings(self.radial_rings.currentData())
        )
        self.form.addRow("Radial rings", self.radial_rings)
        self.reset_layout = QPushButton("Reset layout")
        self.reset_layout.setToolTip(
            "Show the timeline, hide the details and notes, and size everything automatically."
        )
        self.form.addRow(self.reset_layout)
        self.layer_controls = {}
        for key, layer in self.view.canvas.layers.items():
            shown = QCheckBox(f"Show the {layer.profile.title.lower()} layer")
            shown.setChecked(True)
            shown.setToolTip("Hiding a layer does not stop its analysis.")
            shown.toggled.connect(lambda on, key=key: self.view.set_layer(key, enabled=on))
            sensitivity = QSlider(Qt.Orientation.Horizontal)
            sensitivity.setRange(0, 100)
            sensitivity.setValue(50)
            sensitivity.setToolTip(
                "How readily the analysis reports this instrument's articulation. "
                "It does not change the recognition model or its thresholds."
            )
            sensitivity.valueChanged.connect(
                lambda value, key=key: self.sensitivity_changed.emit(key, value / 100)
            )
            intensity = QSlider(Qt.Orientation.Horizontal)
            intensity.setRange(20, 160)
            intensity.setValue(100)
            intensity.setToolTip("How strongly the layer is drawn. The analysis is not affected.")
            intensity.valueChanged.connect(
                lambda value, key=key: self.view.set_layer(key, intensity=value / 100)
            )
            style = Combo()
            for name, title in layer.profile.styles:
                style.addItem(title, name)
            style.currentIndexChanged.connect(
                lambda _=0, key=key, box=style: self.view.set_layer(key, style=box.currentData())
            )
            self.form.addRow(shown)
            self.form.addRow(f"{layer.profile.title} sensitivity", sensitivity)
            self.form.addRow(f"{layer.profile.title} intensity", intensity)
            if layer.profile.styles:
                self.form.addRow(f"{layer.profile.title} style", style)
            self.layer_controls[key] = {
                "shown": shown,
                "sensitivity": sensitivity,
                "intensity": intensity,
                "style": style,
            }

    def layer_settings(self):
        return {
            key: {
                "shown": controls["shown"].isChecked(),
                "sensitivity": controls["sensitivity"].value(),
                "intensity": controls["intensity"].value(),
                "style": controls["style"].currentData(),
            }
            for key, controls in self.layer_controls.items()
        }

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
        self.own_process = QCheckBox("Run capture and analysis in their own process")
        self.own_process.setChecked(True)
        self.own_process.setToolTip(
            "Smoother animation: drawing and analysis no longer wait for each other. "
            "Uses about 110 MiB more memory. Turn off to run both in one process."
        )
        self.form.addRow(self.own_process)
        self.form.addRow("Debug recording", self.debug_audio)
        for signal in (
            self.window_ms.valueChanged,
            self.hop_ms.valueChanged,
            self.band_edges.textChanged,
            self.delayed.toggled,
            self.own_process.toggled,
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
            if not path.strip() and what == "Guitar":
                return "Guitar: no recognition model chosen, the guitar layer stays unconfirmed"
            if not path.strip():
                return f"{what}: no weights chosen, lane shows no detector output"
            if not Path(path).is_file():
                return f"{what}: file not found"
            if path.strip().lower().endswith(".json"):
                return f"{what}: context recognition model chosen (experimental, reports late)"
            return f"{what}: weights chosen"

        self.capabilities.setText(
            "\n".join(
                (
                    "Spectrum, level, onsets, changes: always on",
                    state(self.activity_model.text(), "Drums, bass, vocals"),
                    state(self.guitar_model.text(), "Guitar"),
                    "Beats: "
                    + ("experimental tracker on" if self.beat_tracker.isChecked() else "off"),
                    "Notes: "
                    + (
                        "candidates on, not attributed to an instrument"
                        if self.pitch.isChecked()
                        else "off"
                    ),
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
            self,
            "Choose weights",
            str(paths.models_folder()),
            "Weights and models (*.npz *.json)"
            if target is self.guitar_model
            else "Weights (*.npz)",
        )
        if path:
            target.setText(path)

    def fill_lane_menu(self):
        menu = self.lanes.menu()
        menu.clear()
        entries = list(ROWS) + [(lane.key, lane.title) for lane, _ in self.view.state.board.lanes()]
        entries.append(("notes", "Note candidates"))
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
            pitch_tracking=self.pitch.isChecked(),
            logging=self.logging.isChecked(),
            instrument_sensitivity=tuple(
                (key, controls["sensitivity"].value() / 100)
                for key, controls in self.layer_controls.items()
            ),
            delayed_analysis=self.delayed.isChecked(),
            analysis_process=self.own_process.isChecked(),
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
            "spectrum_place": self.spectrum_place.currentData(),
            "layers": self.layer_settings(),
            "sections": self.view.layout_state(),
            "analysis_process": self.own_process.isChecked(),
            "frame_rate": self.frame_rate.currentData(),
            "intensity_colour": self.intensity_colour.isChecked(),
            "radial_rings": self.radial_rings.currentData(),
            "reactive_background": self.reactive_background.isChecked(),
            "expressive": self.expressive.isChecked(),
            "radial_theme": self.radial_theme.currentData(),
            "radiant_palette": self.view.radiant_palette.saved(),
            "radiant_effects": self.radiant_effects.currentData(),
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
        self.spectrum_place.setCurrentIndex(
            max(0, self.spectrum_place.findData(saved.get("spectrum_place", "behind")))
        )
        self.view.restore_layout(saved.get("sections"))
        self.own_process.setChecked(bool(saved.get("analysis_process", True)))
        self.intensity_colour.setChecked(bool(saved.get("intensity_colour", True)))
        self.reactive_background.setChecked(bool(saved.get("reactive_background", True)))
        self.expressive.setChecked(bool(saved.get("expressive", False)))
        # Settings saved before the themes existed: Classic and the factory Radiant palette.
        self.view.radiant_palette.load(saved.get("radiant_palette"))
        self.sync_radiant_controls()
        self.radiant_effects.setCurrentIndex(
            max(0, self.radiant_effects.findData(saved.get("radiant_effects", "auto")))
        )
        self.radial_theme.setCurrentIndex(
            max(0, self.radial_theme.findData(saved.get("radial_theme", DEFAULT_THEME)))
        )
        self.theme_changed()
        self.radial_rings.setCurrentIndex(
            max(0, self.radial_rings.findData(saved.get("radial_rings", "live")))
        )
        self.frame_rate.setCurrentIndex(
            max(0, self.frame_rate.findData(saved.get("frame_rate", "auto")))
        )
        for key, values in (saved.get("layers") or {}).items():
            controls = self.layer_controls.get(key)
            if controls is None or not isinstance(values, dict):
                continue
            controls["shown"].setChecked(bool(values.get("shown", True)))
            controls["sensitivity"].setValue(int(values.get("sensitivity", 50)))
            controls["intensity"].setValue(int(values.get("intensity", 100)))
            controls["style"].setCurrentIndex(
                max(0, controls["style"].findData(values.get("style")))
            )


class MainWindow(QWidget):
    discovered = Signal(object, object, str)

    def __init__(self, session_root=None, remember=True, discover_sources=True):
        super().__init__(objectName="app")
        self.setWindowTitle("Granular Music Visualizer")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground)
        self.setStyleSheet(STYLE)
        self.resize(1440, 900)
        self.controller = SessionController(session_root or paths.sessions_folder())
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
        self.drawer.reset_layout.clicked.connect(self.reset_layout)
        # A layer's button in the top bar and its checkbox in Settings are one switch.
        for key, chip in self.chips.items():
            box = self.drawer.layer_controls[key]["shown"]
            chip.toggled.connect(box.setChecked)
            box.toggled.connect(chip.setChecked)
        self.drawer.sensitivity_changed.connect(self.controller.set_sensitivity)
        self.drawer.refresh_requested.connect(self.refresh_sources)
        self.drawer.fullscreen.clicked.connect(self.toggle_fullscreen)
        self.drawer.open_session.clicked.connect(self.open_session_folder)
        self.drawer.export_session.clicked.connect(self.export_session)
        self.drawer.replay.clicked.connect(self.choose_replay)
        self.discovered.connect(self.sources_found)
        for name, default in (
            (
                "activity_model",
                os.environ.get("GMV_ACTIVITY_MODEL") or paths.models_folder() / "activity_v2.npz",
            ),
            (
                "guitar_model",
                os.environ.get("GMV_GUITAR_MODEL") or paths.models_folder() / "guitar_v1.npz",
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
            self.apply_build_defaults()
        self.sync_layout_controls()
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
        # One button per instrument, in its colour: a click shows or hides its layer. Then
        # one per section under the visualizer. All of it works while analysis runs.
        self.chips, self.section_buttons = {}, {}
        for key, layer in self.view.canvas.layers.items():
            chip = QPushButton(layer.profile.title, objectName="chip")
            chip.setCheckable(True)
            chip.setChecked(True)
            chip.setCursor(Qt.CursorShape.PointingHandCursor)
            chip.setToolTip(
                f"Show or hide the {layer.profile.title.lower()} layer. Its analysis keeps running."
            )
            line.addWidget(chip)
            self.chips[key] = chip
        self.colour_chips()
        self.drawer.colours_changed.connect(self.colour_chips)
        line.addSpacing(10)
        for name, title in (("timeline", "Timeline"), ("details", "Details"), ("notes", "Notes")):
            button = QPushButton(title, objectName="section")
            button.setCheckable(True)
            button.setToolTip(
                f"Show or hide the {title.lower()} section. Drag its upper edge to resize it."
            )
            button.toggled.connect(lambda on, name=name: self.view.set_section(name, shown=on))
            line.addWidget(button)
            self.section_buttons[name] = button
        line.addSpacing(10)
        # Expressive Mode at hand: the same switch as in Settings, kept in step with it.
        self.expressive_button = QPushButton("Expressive", objectName="section")
        self.expressive_button.setCheckable(True)
        self.expressive_button.setToolTip(
            "Expressive Mode: amplify the visual impact of instrument animations, spectrum "
            "movement, and illumination without changing audio analysis."
        )
        self.expressive_button.toggled.connect(self.drawer.expressive.setChecked)
        self.drawer.expressive.toggled.connect(self.expressive_button.setChecked)
        line.addWidget(self.expressive_button)
        line.addSpacing(10)
        for widget in (self.apply, self.toggle, self.settings_button):
            line.addWidget(widget)
        return bar

    def colour_chips(self):
        """Each instrument's button in the colour the instrument is drawn in now: its own,
        or the one the Radiant palette gives it while Radiant is on screen."""
        for key, chip in self.chips.items():
            colour = self.view.instrument_colour(key)
            if chip.property("colour") == colour:
                continue
            chip.setProperty("colour", colour)
            chip.setStyleSheet(
                f"QPushButton#chip {{ border: 1px solid {colour}; color: {colour}; border-radius: 11px;"
                " padding: 3px 12px; background: transparent; }"
                f"QPushButton#chip:checked {{ background: {colour}; color: #0a0c13; font-weight: 600; }}"
                "QPushButton#chip:!checked { border-color: #3a4055; color: #8089a0; }"
            )

    def show_readings(self):
        """Each instrument button's tooltip says how its layer reads the music right now."""
        for key, chip in self.chips.items():
            layer = self.view.canvas.layers[key]
            reading = layer.reading(self.view.state)
            chip.setToolTip(
                f"{layer.profile.title}: {reading or 'no activity yet'}. "
                "Click to show or hide this layer; its analysis keeps running."
            )

    def sync_layout_controls(self):
        """Top-bar buttons follow the view (after restoring settings or a reset)."""
        for name, button in self.section_buttons.items():
            button.blockSignals(True)
            button.setChecked(self.view.canvas.sections[name]["shown"])
            button.blockSignals(False)

    def reset_layout(self):
        self.view.reset_layout()
        self.sync_layout_controls()

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
                instruments=session.configuration.get("instrument_analyzers") or (),
                # As recorded; a recording without its configuration has none.
                band_edges=session.configuration.get("band_edges"),
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
        self.show_readings()
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
            if canvas.themed() is not None:
                report = self.view.renderer_report()
                text = f"{report['backend'].upper()}: {report['detail']}"
                if report.get("gpu_ms") is not None:
                    text += (
                        f" · card {report['gpu_ms']:.1f} ms, submission {report['submit_ms']:.1f} ms "
                        f"per frame · buffers {report['buffers_mib']} MiB"
                    )
                lines["Radiant renderer"] = text
            drawer.diagnostics.setText("\n".join(f"{key}: {value}" for key, value in lines.items()))

    def apply_build_defaults(self):
        """A packaged build starts with the weights it was made with. Each set of weights is
        applied once, so switching one off afterwards is remembered."""
        store = QSettings()
        for name, path in paths.build_defaults().items():
            key = f"build_default/{name}"
            if store.value(key, "") != Path(path).name:
                getattr(self.drawer, name).setText(path)
                store.setValue(key, Path(path).name)

    def closeEvent(self, event):
        self.poller.stop()
        if self.remember:
            QSettings().setValue("settings", json.dumps(self.drawer.remembered()))
        self.view.close()
        self.controller.shutdown()
        event.accept()


def application():
    # Analysis and drawing are separate threads of one interpreter. Letting them hand over
    # every millisecond (the default is five) keeps a busy analysis frame from holding up
    # a paint, and a paint from holding up analysis.
    sys.setswitchinterval(0.001)
    app = QApplication.instance() or QApplication(sys.argv)
    app.setOrganizationName("GranularMusicVisualizer")
    app.setApplicationName("Granular Music Visualizer")
    app.setStyle("Fusion")
    icon = paths.icon()
    if icon is not None:  # title bar, taskbar and Alt+Tab; the executable has its own copy
        app.setWindowIcon(QIcon(str(icon)))
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
