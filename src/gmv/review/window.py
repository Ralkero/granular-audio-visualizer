"""The review window: listen to excerpts and record what you hear.

Every annotation control starts empty for an excerpt nobody has labelled; nothing is filled
in from the model. The model's output sits in its own box, hidden until a presence answer
has been saved for the excerpt (unless "always show" is ticked), and it never writes to the
annotation store. Changes are saved automatically shortly after each edit, when moving to
another excerpt and on closing; every save is a new revision in ``annotations.jsonl``.
"""

import hashlib
import json
import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .annotations import GUITAR_TYPES, INSTRUMENTS, PRESENCE, ROLES, AnnotationStore, status_of
from .pipeline import load_tables

PRESENCE_TEXT = {
    "present_throughout": ("Present throughout", "T"),
    "present_intermittent": ("Present intermittently", "I"),
    "absent": ("Absent", "A"),
    "uncertain": ("Uncertain", "U"),
}
TYPE_TEXT = {
    "distorted_electric": ("Distorted electric", "D"),
    "clean_electric": ("Clean electric", "C"),
    "acoustic": ("Acoustic", "K"),
    "multiple": ("Multiple types", "M"),
    "unknown": ("Unknown", None),
}
ROLE_TEXT = {
    "lead": "Lead",
    "rhythm": "Rhythm",
    "both": "Both",
    "background": "Background / atmospheric",
    "unknown": "Unknown",
}
INSTRUMENT_TEXT = {
    "bass": "Bass",
    "drums": "Drums",
    "piano": "Piano",
    "synthesizer": "Synthesizer",
    "strings_orchestral": "Strings / orchestral",
    "vocals": "Vocals",
    "other": "Other",
    "unknown": "Unknown",
}
# the window offers exactly the answers the annotation store accepts
assert tuple(PRESENCE_TEXT) == PRESENCE and tuple(TYPE_TEXT) == GUITAR_TYPES
assert tuple(ROLE_TEXT) == ROLES and tuple(INSTRUMENT_TEXT) == INSTRUMENTS
FILTERS = (
    ("queue", "Review queue: unreviewed (core first)"),
    ("uncertain", "Uncertain queue: uncertain or flagged to revisit"),
    ("all", "All excerpts"),
    ("core", "Core excerpts"),
    ("feature", "Feature-selected excerpts"),
    ("model_flagged", "Model-flagged excerpts (diagnostic)"),
    ("labelled", "Labelled excerpts"),
    ("evidence", "Evidence queue: all"),
    ("ev:model_disagreement", "Evidence: the audio models disagree"),
    ("ev:possible_false_positive", "Evidence: possible false guitar detection"),
    ("ev:possible_false_negative", "Evidence: possible missed guitar"),
    ("ev:suspected_clean", "Evidence: suspected clean guitar"),
    ("ev:suspected_distorted", "Evidence: suspected distorted guitar"),
    ("ev:suspected_acoustic", "Evidence: suspected acoustic guitar"),
    ("ev:metadata_mismatch", "Evidence: credits and audio disagree"),
    ("ev:low_quality_source", "Evidence: low-quality source audio"),
    ("ev:random_audit", "Evidence: automatically classified, audit sample"),
)
CATEGORY_ORDER = {"core": 0, "evidence": 0, "feature": 1, "model_flagged": 2}


def clock(seconds):
    return f"{int(seconds // 60)}:{seconds % 60:04.1f}"


class ReviewWindow(QMainWindow):
    def __init__(self, workdir, annotator="owner", player=None, autosave_ms=800):
        super().__init__()
        self.workdir = Path(workdir)
        self.store = AnnotationStore(self.workdir, annotator)
        self.batch, self.excerpts = load_tables(self.workdir)
        self.recordings = {r["recording_id"]: r for r in self.batch}
        # A fixed, shuffled recording order, so expected-positive and expected-negative
        # recordings are interleaved instead of reviewed in blocks.
        order = {
            r["recording_id"]: hashlib.sha256(
                f"review-order:{r['recording_id']}".encode()
            ).hexdigest()
            for r in self.batch
        }
        self.excerpts.sort(
            key=lambda e: (
                CATEGORY_ORDER[e["excerpt_category"]],
                order.get(e["recording_id"], 0),
                e["excerpt_id"],
            )
        )
        self.by_id = {e["excerpt_id"]: e for e in self.excerpts}
        self.settings_path = self.workdir / "review_settings.json"
        self.settings = self._load_settings()
        if player is None:
            from .player import Player

            player = Player()
        self.player = player
        self.current = None
        self.loading = False
        self.setWindowTitle(f"Guitar review — {self.workdir.name}")
        self._build_ui()
        self.autosave = QTimer(self, singleShot=True, interval=autosave_ms)
        self.autosave.timeout.connect(self.save_current)
        self.ticker = QTimer(self, interval=50)
        self.ticker.timeout.connect(self._tick)
        self.ticker.start()
        self.refresh_list(select=self.settings.get("last_excerpt"))

    # ----------------------------------------------------------------- settings
    def _load_settings(self):
        try:
            return json.loads(self.settings_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def _save_settings(self):
        try:
            self.settings_path.write_text(json.dumps(self.settings, indent=2), encoding="utf-8")
        except OSError:
            pass

    # ----------------------------------------------------------------------- UI
    def _build_ui(self):
        splitter = QSplitter()
        left = QWidget()
        left_layout = QVBoxLayout(left)
        self.filter = QComboBox()
        for key, text in FILTERS:
            self.filter.addItem(text, key)
        self.filter.currentIndexChanged.connect(lambda: self.refresh_list())
        self.progress = QLabel()
        self.list = QListWidget()
        self.list.currentItemChanged.connect(self._selected)
        left_layout.addWidget(self.filter)
        left_layout.addWidget(self.progress)
        left_layout.addWidget(self.list)
        splitter.addWidget(left)

        right = QWidget()
        layout = QVBoxLayout(right)
        self.title = QLabel()
        self.title.setStyleSheet("font-size: 16px; font-weight: 600;")
        self.title.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.where = QLabel()
        self.where.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(self.title)
        layout.addWidget(self.where)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play (Space)")
        self.play_button.clicked.connect(self.toggle_play)
        replay = QPushButton("Replay (R)")
        replay.clicked.connect(self.replay)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 1000)
        self.slider.sliderReleased.connect(self._seek)
        self.time_label = QLabel("0:00.0 / 0:00.0")
        for w in (self.play_button, replay):
            transport.addWidget(w)
        transport.addWidget(self.slider, 1)
        transport.addWidget(self.time_label)
        layout.addLayout(transport)
        self.player_note = QLabel()
        self.player_note.setStyleSheet("color: #b45309;")
        layout.addWidget(self.player_note)

        box = QGroupBox("Your annotation (human label)")
        grid = QGridLayout(box)
        self.presence = self._radio_row(
            grid, 0, "Guitar presence", {k: f"{t} ({s})" for k, (t, s) in PRESENCE_TEXT.items()}
        )
        self.spans = QLineEdit()
        self.spans.setPlaceholderText(
            "When guitar is audible, approximately: e.g. 0:00–0:04 and 0:09–0:12 (optional)"
        )
        grid.addWidget(QLabel("Intermittent spans"), 1, 0)
        grid.addWidget(self.spans, 1, 1)
        self.gtype = self._radio_row(
            grid,
            2,
            "Guitar type",
            {k: t + (f" ({s})" if s else "") for k, (t, s) in TYPE_TEXT.items()},
        )
        self.role = self._radio_row(grid, 3, "Guitar role", ROLE_TEXT)
        instruments = QWidget()
        row = QHBoxLayout(instruments)
        row.setContentsMargins(0, 0, 0, 0)
        self.instruments = {}
        for key, text in INSTRUMENT_TEXT.items():
            box_ = QCheckBox(text)
            box_.toggled.connect(self._changed)
            self.instruments[key] = box_
            row.addWidget(box_)
        row.addStretch(1)
        grid.addWidget(QLabel("Other prominent instruments"), 4, 0)
        grid.addWidget(instruments, 4, 1)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("Optional notes")
        self.notes.setFixedHeight(54)
        grid.addWidget(QLabel("Notes"), 5, 0)
        grid.addWidget(self.notes, 5, 1)
        self.difficulty = QLineEdit()
        self.difficulty.setPlaceholderText("What makes this excerpt difficult? (optional)")
        grid.addWidget(QLabel("Difficulty note"), 6, 0)
        grid.addWidget(self.difficulty, 6, 1)
        self.revisit = QCheckBox("Flag to revisit later")
        grid.addWidget(self.revisit, 7, 1)
        buttons = QHBoxLayout()
        back = QPushButton("Previous (B)")
        back.clicked.connect(lambda: self.step(-1))
        nxt = QPushButton("Save and next (N)")
        nxt.clicked.connect(lambda: self.step(1))
        buttons.addWidget(back)
        buttons.addWidget(nxt)
        buttons.addStretch(1)
        self.saved = QLabel()
        buttons.addWidget(self.saved)
        grid.addLayout(buttons, 8, 1)
        layout.addWidget(box)
        for widget in (self.spans, self.difficulty):
            widget.textEdited.connect(self._changed)
        self.notes.textChanged.connect(self._changed)
        self.revisit.toggled.connect(self._changed)

        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        layout.addWidget(line)
        self.model_box = QGroupBox("Diagnostics — model output and selection prior, not labels")
        self.model_box.setStyleSheet("QGroupBox { color: #6b7280; } QLabel { color: #6b7280; }")
        model_layout = QVBoxLayout(self.model_box)
        self.model_text = QLabel()
        self.model_text.setWordWrap(True)
        self.model_hidden = QLabel(
            "Hidden until you save a guitar-presence answer for this excerpt."
        )
        model_layout.addWidget(self.model_hidden)
        model_layout.addWidget(self.model_text)
        self.always_model = QCheckBox("Always show model output")
        self.always_model.setChecked(bool(self.settings.get("always_show_model")))
        self.always_model.toggled.connect(self._always_model)
        model_layout.addWidget(self.always_model)
        layout.addWidget(self.model_box)

        self.history = QListWidget()
        self.history.setMaximumHeight(90)
        layout.addWidget(QLabel("Revisions of this excerpt"))
        layout.addWidget(self.history)
        layout.addStretch(1)
        splitter.addWidget(right)
        splitter.setSizes([380, 900])
        self.setCentralWidget(splitter)
        self.resize(1300, 860)

        keys = {
            "Space": self.toggle_play,
            "R": self.replay,
            "N": lambda: self.step(1),
            "B": lambda: self.step(-1),
        }
        for key, (_, letter) in PRESENCE_TEXT.items():
            keys[letter] = lambda k=key: self._press(self.presence, k)
        for key, (_, letter) in TYPE_TEXT.items():
            if letter:
                keys[letter] = lambda k=key: self._press(self.gtype, k)
        for key, action in keys.items():
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(action)

    def _radio_row(self, grid, row, title, options):
        widget = QWidget()
        line = QHBoxLayout(widget)
        line.setContentsMargins(0, 0, 0, 0)
        group = QButtonGroup(self)
        group.setExclusive(True)
        group.buttons_by_key = {}
        for key, text in options.items():
            button = QRadioButton(text)
            group.addButton(button)
            group.buttons_by_key[key] = button
            button.toggled.connect(self._changed)
            line.addWidget(button)
        line.addStretch(1)
        grid.addWidget(QLabel(title), row, 0)
        grid.addWidget(widget, row, 1)
        return group

    def _press(self, group, key):
        group.buttons_by_key[key].setChecked(True)

    @staticmethod
    def _value(group):
        for key, button in group.buttons_by_key.items():
            if button.isChecked():
                return key
        return None

    @staticmethod
    def _set(group, key):
        group.setExclusive(False)  # allows clearing every button
        for k, button in group.buttons_by_key.items():
            button.setChecked(k == key)
        group.setExclusive(True)

    # -------------------------------------------------------------------- list
    def _matches(self, excerpt, mode):
        values = self.store.values(excerpt["excerpt_id"])
        status = status_of(values)
        if mode == "queue":
            return status == "unreviewed"
        if mode == "uncertain":
            return values["guitar_presence"] == "uncertain" or values["needs_revisit"]
        if mode == "labelled":
            return status != "unreviewed"
        if mode.startswith("ev:"):
            return (
                excerpt["excerpt_category"] == "evidence"
                and excerpt["excerpt_position"] == mode[3:]
            )
        if mode in CATEGORY_ORDER:
            return excerpt["excerpt_category"] == mode
        return True

    def refresh_list(self, select=None):
        mode = self.filter.currentData()
        keep = select or (self.current["excerpt_id"] if self.current else None)
        self.list.blockSignals(True)
        self.list.clear()
        chosen = None
        for excerpt in self.excerpts:
            if not self._matches(excerpt, mode) and excerpt["excerpt_id"] != keep:
                continue
            item = QListWidgetItem(self._item_text(excerpt))
            item.setData(Qt.UserRole, excerpt["excerpt_id"])
            self.list.addItem(item)
            if excerpt["excerpt_id"] == keep:
                chosen = item
        self.list.blockSignals(False)
        done = sum(
            1 for e in self.excerpts if self.store.values(e["excerpt_id"])["guitar_presence"]
        )
        core = [e for e in self.excerpts if e["excerpt_category"] == "core"]
        core_done = sum(1 for e in core if self.store.values(e["excerpt_id"])["guitar_presence"])
        self.progress.setText(
            f"Labelled {done} of {len(self.excerpts)} (core {core_done} of {len(core)}); showing {self.list.count()}"
        )
        target = chosen or (self.list.item(0) if self.list.count() else None)
        if target:
            self.list.setCurrentItem(target)
            self._selected(target, None)

    def _item_text(self, excerpt):
        values = self.store.values(excerpt["excerpt_id"])
        mark = {
            "present_throughout": "T",
            "present_intermittent": "I",
            "absent": "A",
            "uncertain": "?",
        }.get(values["guitar_presence"], "·")
        if values["needs_revisit"]:
            mark += "!"
        return (
            f"{mark:<3}{excerpt['artist']} – {excerpt['title']}  "
            f"[{excerpt['excerpt_category']} {excerpt['excerpt_position']} {excerpt['source_start_clock'][:-2]}]"
        )

    def _update_item(self):
        item = self.list.currentItem()
        if item and self.current and item.data(Qt.UserRole) == self.current["excerpt_id"]:
            item.setText(self._item_text(self.current))

    # ---------------------------------------------------------------- excerpt
    def _selected(self, item, _previous):
        if item is None:
            return
        excerpt = self.by_id[item.data(Qt.UserRole)]
        if self.current and excerpt["excerpt_id"] == self.current["excerpt_id"]:
            return
        self.save_current()
        self.show_excerpt(excerpt)

    def show_excerpt(self, excerpt):
        self.current = excerpt
        self.settings["last_excerpt"] = excerpt["excerpt_id"]
        self._save_settings()
        self.title.setText(f"{excerpt['artist']} — {excerpt['title']}   ({excerpt['album']})")
        self.where.setText(
            f"Excerpt {excerpt['excerpt_id']} · {excerpt['excerpt_category']} / {excerpt['excerpt_position']} · "
            f"original {excerpt['source_start_clock']}–{excerpt['source_end_clock']} "
            f"(samples {excerpt['source_start_sample']}–{excerpt['source_end_sample']} at {excerpt['source_sample_rate_hz']} Hz)"
        )
        self.loading = True
        values = self.store.values(excerpt["excerpt_id"])
        self._set(self.presence, values["guitar_presence"])
        self._set(self.gtype, values["guitar_type"])
        self._set(self.role, values["guitar_role"])
        self.spans.setText(values["intermittent_spans_note"])
        for key, box in self.instruments.items():
            box.setChecked(key in values["other_instruments"])
        self.notes.setPlainText(values["notes"])
        self.difficulty.setText(values["difficulty_note"])
        self.revisit.setChecked(values["needs_revisit"])
        self.loading = False
        self._spans_enabled()
        self._show_history()
        self._show_model()
        try:
            self.player.load(excerpt["excerpt_path"])
            self.player_note.setText(self.player.error or self.player.note)
        except Exception as exc:
            self.player_note.setText(f"Cannot load the excerpt: {exc}")

    def _spans_enabled(self):
        self.spans.setEnabled(self._value(self.presence) == "present_intermittent")

    def _show_history(self):
        self.history.clear()
        for entry in reversed(self.store.history(self.current["excerpt_id"])):
            self.history.addItem(
                f"rev {entry['revision']} · {entry['annotated_at']} · {entry['guitar_presence'] or '—'}"
                f" · {entry['guitar_type'] or '—'} · {entry['guitar_role'] or '—'}"
                + (f" · {entry['notes'][:40]}" if entry["notes"] else "")
            )

    def _show_model(self):
        excerpt = self.current
        answered = self.store.values(excerpt["excerpt_id"])["guitar_presence"] is not None
        visible = answered or self.always_model.isChecked()
        self.model_hidden.setVisible(not visible)
        self.model_text.setVisible(visible)
        if not visible:
            self.model_text.setText("")
            return
        thresholds = json.loads(excerpt.get("model_thresholds") or "{}")
        lines = [
            f"Guitar: mean score {excerpt.get('model_guitar_mean') or '–'}, max {excerpt.get('model_guitar_max') or '–'}; "
            f"detected in {self._pct(excerpt.get('model_guitar_share_detected'))} of hops, detected or candidate in "
            f"{self._pct(excerpt.get('model_guitar_share_candidate_or_detected'))}; majority state "
            f"{excerpt.get('model_guitar_majority_state') or '–'} (thresholds: detected {thresholds.get('detected')}, "
            f"candidate {thresholds.get('candidate')}).",
            f"Sub-labels (weak, unvalidated): distorted {excerpt.get('model_distorted_mean') or '–'}, "
            f"clean {excerpt.get('model_clean_mean') or '–'}, acoustic {excerpt.get('model_acoustic_mean') or '–'}.",
        ]
        if excerpt.get("model_context_complete") == "false":
            lines.append("The model had less than 10 s of audio before this excerpt.")
        if excerpt["excerpt_category"] == "model_flagged":
            lines.append(f"Why flagged: {excerpt['excerpt_selection_reason']}")
        recording = self.recordings.get(excerpt["recording_id"], {})
        lines.append(
            "Selection prior (metadata, not a label): "
            f"{recording.get('selection_category', '?').replace('_', ' ')}; "
            f"{recording.get('selection_reason', '')}"
        )
        self.model_text.setText("\n".join(lines))

    @staticmethod
    def _pct(value):
        try:
            return f"{float(value):.0%}"
        except (TypeError, ValueError):
            return "–"

    def _always_model(self, checked):
        self.settings["always_show_model"] = bool(checked)
        self._save_settings()
        if self.current:
            self._show_model()

    # ------------------------------------------------------------------ saving
    def _changed(self, *_):
        if self.loading or self.current is None:
            return
        self._spans_enabled()
        self.autosave.start()

    def collect(self):
        return {
            "guitar_presence": self._value(self.presence),
            "intermittent_spans_note": self.spans.text().strip(),
            "guitar_type": self._value(self.gtype),
            "guitar_role": self._value(self.role),
            "other_instruments": [k for k, b in self.instruments.items() if b.isChecked()],
            "notes": self.notes.toPlainText().strip(),
            "difficulty_note": self.difficulty.text().strip(),
            "needs_revisit": self.revisit.isChecked(),
        }

    def save_current(self):
        self.autosave.stop()
        if self.current is None:
            return None
        entry = self.store.save(
            self.current["excerpt_id"], self.current["recording_id"], self.collect()
        )
        if entry:
            self.saved.setText(
                f"Saved revision {entry['revision']} at {entry['annotated_at'][11:19]} UTC"
            )
            self.store.write_csv(self.excerpts, self.workdir / "Human_Annotations.csv")
            self._show_history()
            self._show_model()
            self._update_item()
            done = sum(
                1 for e in self.excerpts if self.store.values(e["excerpt_id"])["guitar_presence"]
            )
            self.progress.setText(
                f"Labelled {done} of {len(self.excerpts)}; showing {self.list.count()}"
            )
        return entry

    # ------------------------------------------------------------- navigation
    def step(self, direction):
        self.save_current()
        row = self.list.currentRow() + direction
        if 0 <= row < self.list.count():
            self.list.setCurrentRow(row)

    def toggle_play(self):
        self.player.toggle()

    def replay(self):
        self.player.replay()

    def _seek(self):
        if self.player.duration_s:
            self.player.seek(self.slider.value() / 1000 * self.player.duration_s)

    def _tick(self):
        duration = self.player.duration_s
        position = self.player.position_s
        self.play_button.setText("Pause (Space)" if self.player.playing else "Play (Space)")
        if duration and not self.slider.isSliderDown():
            self.slider.setValue(int(1000 * position / duration))
        offset = float(self.current["source_start_s"]) if self.current else 0.0
        self.time_label.setText(
            f"{clock(position)} / {clock(duration)}   (original {clock(offset + position)})"
        )

    def closeEvent(self, event):
        self.save_current()
        self.player.close()
        super().closeEvent(event)


def main(workdir, annotator="owner"):
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ReviewWindow(workdir, annotator)
    window.show()
    return app.exec()
