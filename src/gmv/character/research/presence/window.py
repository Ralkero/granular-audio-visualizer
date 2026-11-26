"""The presence-annotation window: listen to an excerpt and mark where a human voice is
audible.

Built on the Phase A3 labelling tool's parts: the same player (exactly the excerpt's
samples, through the sound server), the same excerpt reader and the private song titles.
The window never shows why an excerpt was chosen (no delivery answer, no "candidate
negative"), so the annotation is not steered.

Marking: ``[`` starts an interval at the playback position; ``]`` ends it as a confirmed
voice, ``U`` ends it as uncertain, ``Esc`` drops the open mark. A mark made while paused is
exact to the sample; a mark made while playing lands late by the listener's reaction time
and the output latency, so boundaries are best refined while paused: arrow keys move the
position by 0.1 s (with Shift 0.02 s, with Ctrl 1 s), the timeline can be clicked, the
selected interval's start or end can be nudged or set to the position, and "listen around"
replays a second and a half around a boundary. Every decision is saved at once to the
partition's append-only history.
"""

from pathlib import Path

import numpy as np
from PySide6.QtCore import QLockFile, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ..labeling import dataset as labelling
from . import project, schema
from .store import Store, backup, read_session, write_session

TITLE = "GMV: mark where a voice is audible"
NUDGE_S = 0.05
AROUND_BEFORE_S, AROUND_AFTER_S = 1.0, 0.5
SAVED_FOR_MS = 4000
UNAVAILABLE = ("This excerpt cannot be played right now: its recording is not available (the drive may "
               "be disconnected, or the file moved or changed). Nothing can be marked until it plays.")
HELP = ("Mark every stretch where a human voice is audible: singing, screaming, speaking, backing or "
        "quiet voices, several at once. The kind of voice does not matter here. Mark a stretch "
        "'uncertain' when you cannot tell. When the whole excerpt is done, press 'Complete': an "
        "excerpt completed with no marks counts as having no voice at all.")


def clock(value):
    return f"{value:6.3f} s"


def envelope(audio, rate, step_s=0.01):
    """Loudness outline of the excerpt: RMS every 10 ms, in dB below its peak, 0..1."""
    mono = np.asarray(audio, dtype=np.float32)
    mono = mono.mean(axis=1) if mono.ndim > 1 else mono
    step = max(1, int(rate * step_s))
    usable = len(mono) // step * step
    if not usable:
        return np.zeros(1)
    rms = np.sqrt(np.mean(mono[:usable].reshape(-1, step) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms / max(float(rms.max()), 1e-9))
    return np.clip((db + 60.0) / 60.0, 0.0, 1.0)


class Timeline(QWidget):
    """The excerpt as a strip: loudness outline, confirmed (green) and uncertain (amber)
    intervals, an open mark (dashed) and the playback position (red). Clicking seeks."""

    clicked = Signal(float)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(120)
        self.duration, self.outline, self.position = 0.0, np.zeros(1), 0.0
        self.intervals, self.pending, self.selected = [], None, None

    def show_state(self, duration, outline, intervals, pending, position, selected):
        self.duration, self.outline, self.intervals = duration, outline, intervals
        self.pending, self.position, self.selected = pending, position, selected
        self.update()

    def x_of(self, seconds):
        return 8 + (self.width() - 16) * (seconds / self.duration if self.duration else 0.0)

    def mousePressEvent(self, event):  # noqa: N802 (Qt name)
        if self.duration:
            fraction = (event.position().x() - 8) / max(1.0, self.width() - 16)
            self.clicked.emit(float(min(self.duration, max(0.0, fraction * self.duration))))

    def paintEvent(self, _event):  # noqa: N802 (Qt name)
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 26, 30))
        height = self.height()
        middle = height / 2
        if self.duration:
            for second in range(int(self.duration) + 1):
                painter.setPen(QPen(QColor(70, 74, 82), 1))
                painter.drawLine(QPointF(self.x_of(second), 0), QPointF(self.x_of(second), height))
                painter.drawText(QPointF(self.x_of(second) + 2, height - 4), f"{second}")
            for identity, kind, start, end in self.intervals:
                colour = QColor(70, 170, 90, 110) if kind == "confirmed" else QColor(220, 160, 40, 110)
                rect = QRectF(self.x_of(start), 6, self.x_of(end) - self.x_of(start), height - 22)
                painter.fillRect(rect, colour)
                if identity == self.selected:
                    painter.setPen(QPen(QColor(255, 255, 255), 2))
                    painter.drawRect(rect)
            painter.setPen(QPen(QColor(150, 170, 200), 1))
            count = len(self.outline)
            for i in range(0, count, max(1, count // max(1, self.width()))):
                x = self.x_of(i / count * self.duration)
                half = self.outline[i] * (height / 2 - 14)
                painter.drawLine(QPointF(x, middle - half), QPointF(x, middle + half))
            if self.pending is not None:
                painter.setPen(QPen(QColor(255, 255, 255), 2, Qt.PenStyle.DashLine))
                painter.drawLine(QPointF(self.x_of(self.pending), 0), QPointF(self.x_of(self.pending), height))
            painter.setPen(QPen(QColor(235, 70, 70), 2))
            painter.drawLine(QPointF(self.x_of(self.position), 0), QPointF(self.x_of(self.position), height))
        painter.end()


class PresenceWindow(QMainWindow):
    def __init__(self, root, partition="development", player=None, autoplay=False):
        super().__init__()
        self.root, self.partition = project.layout(root), partition
        self.folder = project.partition_folder(self.root, partition)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.setWindowTitle(f"{TITLE} ({partition.replace('_', ' ')})")
        self.problem, self.current, self.available, self.pending = "", None, False, None
        self.audio, self.rate, self.stop_at, self.selected = None, 0, None, None
        self.outline = np.zeros(1)
        self.lock = QLockFile(str(self.folder / "annotating.lock"))
        self.lock.setStaleLockTime(0)
        if player is None:
            from ..labeling.playback import default_player

            player = default_player()
        self.player = player
        self.manifest = None
        try:
            self.manifest = project.read_manifest(self.root, partition)
        except project.IntegrityError as exc:
            self.problem = f"The excerpt list failed its integrity check; nothing was changed. ({exc})"
        if self.manifest is None and not self.problem:
            self.problem = ("No final-test cohort has been frozen yet." if partition == "final_test"
                            else "No excerpts have been prepared for this partition yet.")
        if self.manifest is not None and not self.lock.tryLock(200):
            self.problem, self.manifest = "This partition is already open in another window. Use that one.", None
        self.excerpts = {e["id"]: e for e in (self.manifest or {}).get("excerpts", [])}
        self.order = [e["id"] for e in (self.manifest or {}).get("excerpts", [])]
        self.store = Store(self.folder, project.durations(self.manifest)) if self.manifest else None
        if self.store is not None and not self.store.intact:
            self.problem = (f"Warning: the saved history fails its integrity check at event {self.store.broken_at}. "
                            "Your marks are shown, but development tools will refuse this partition until it is repaired.")
        import json

        try:
            self.titles = json.loads((self.root / "identity" / "titles.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            self.titles = {}
        self._build()
        self.timer = QTimer(self, interval=40, timeout=self._tick)
        self.timer.start()
        self.saved_timer = QTimer(self, singleShot=True, interval=SAVED_FOR_MS, timeout=lambda: self.saved.setText(""))
        if self.problem:
            self.problem_label.setText(self.problem)
            self.problem_label.show()
        if self.order:
            last = read_session(self.folder).get("current")
            self.open(last if last in self.excerpts else self._first_unreviewed() or self.order[0], play=autoplay)
        else:
            for widget in self.findChildren(QPushButton):
                widget.setEnabled(False)

    # -- layout --------------------------------------------------------------------------------

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        outer = QHBoxLayout(central)
        main, side = QVBoxLayout(), QVBoxLayout()
        outer.addLayout(main, 3)
        outer.addLayout(side, 1)
        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.problem_label.hide()
        main.addWidget(self.problem_label)
        self.identifier = QLabel("")
        self.song = QLabel("")
        font = self.song.font()
        font.setPointSize(font.pointSize() + 3)
        self.song.setFont(font)
        self.status = QLabel("")
        self.counts = QLabel("")
        self.saved = QLabel("")
        for label in (self.identifier, self.song, self.status, self.counts, self.saved):
            label.setWordWrap(True)
            main.addWidget(label)
        self.audio_problem = QLabel("")
        self.audio_problem.setWordWrap(True)
        main.addWidget(self.audio_problem)
        self.timeline = Timeline()
        self.timeline.clicked.connect(self.seek)
        main.addWidget(self.timeline)
        self.time = QLabel("")
        main.addWidget(self.time)

        transport = QHBoxLayout()
        self.play_button = QPushButton("▶  Play / pause  (Space)")
        self.play_button.clicked.connect(self.toggle_play)
        self.restart_button = QPushButton("⏮  From the start  (Home)")
        self.restart_button.clicked.connect(lambda: self.seek(0.0))
        self.back_button = QPushButton("−1 s  (Ctrl+←)")
        self.back_button.clicked.connect(lambda: self.move(-1.0))
        self.forward_button = QPushButton("+1 s  (Ctrl+→)")
        self.forward_button.clicked.connect(lambda: self.move(1.0))
        for button in (self.play_button, self.restart_button, self.back_button, self.forward_button):
            transport.addWidget(button)
        main.addLayout(transport)

        marking = QHBoxLayout()
        self.start_button = QPushButton("Voice starts here  [")
        self.start_button.clicked.connect(self.mark_start)
        self.end_button = QPushButton("Voice ends here  ]")
        self.end_button.clicked.connect(lambda: self.mark_end("confirmed"))
        self.uncertain_button = QPushButton("End as uncertain  U")
        self.uncertain_button.clicked.connect(lambda: self.mark_end("uncertain"))
        self.cancel_button = QPushButton("Drop open mark  Esc")
        self.cancel_button.clicked.connect(self.cancel_mark)
        for button in (self.start_button, self.end_button, self.uncertain_button, self.cancel_button):
            marking.addWidget(button)
        main.addLayout(marking)

        box = QGroupBox("Marked stretches (select one to adjust it)")
        grid = QGridLayout(box)
        self.listing = QListWidget()
        self.listing.currentItemChanged.connect(self._select)
        grid.addWidget(self.listing, 0, 0, 1, 6)
        self.edit_buttons = {}
        for column, (key, text) in enumerate((("start-", "Start −0.05"), ("start+", "Start +0.05"), ("start=", "Start = position"),
                                               ("end-", "End −0.05"), ("end+", "End +0.05"), ("end=", "End = position"))):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, k=key: self.adjust(k))
            grid.addWidget(button, 1, column)
            self.edit_buttons[key] = button
        for column, (key, text) in enumerate((("hear-start", "Listen around start"), ("hear-end", "Listen around end"),
                                               ("kind", "Confirmed ⇄ uncertain"), ("remove", "Remove"))):
            button = QPushButton(text)
            button.clicked.connect(lambda _c=False, k=key: self.adjust(k))
            grid.addWidget(button, 2, column + (1 if column >= 2 else 0))
            self.edit_buttons[key] = button
        main.addWidget(box)

        review = QHBoxLayout()
        self.complete_button = QPushButton("Complete: every voice is marked  C")
        self.complete_button.clicked.connect(self.complete)
        self.reopen_button = QPushButton("Reopen")
        self.reopen_button.clicked.connect(lambda: self.set_review("in_progress"))
        self.unusable_button = QPushButton("Unusable…")
        self.unusable_button.clicked.connect(self.unusable)
        for button in (self.complete_button, self.reopen_button, self.unusable_button):
            review.addWidget(button)
        main.addLayout(review)

        tags = QGroupBox("What else is in this excerpt (any number; not a voice label)")
        tag_grid = QGridLayout(tags)
        self.tag_boxes = {}
        for index, (key, text) in enumerate(schema.CONFOUNDERS.items()):
            box_ = QCheckBox(text)
            box_.toggled.connect(lambda _on, k=key: self.save_tags())
            tag_grid.addWidget(box_, index // 3, index % 3)
            self.tag_boxes[key] = box_
        main.addWidget(tags)
        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional note")
        self.note.editingFinished.connect(self.save_note)
        main.addWidget(self.note)

        moves = QHBoxLayout()
        self.previous_button = QPushButton("←  Previous excerpt  (PgUp)")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.next_button = QPushButton("Next excerpt  →  (PgDn)")
        self.next_button.clicked.connect(lambda: self.step(1))
        self.unreviewed_button = QPushButton("Next not completed  ⏭")
        self.unreviewed_button.clicked.connect(self.next_unreviewed)
        for button in (self.previous_button, self.next_button, self.unreviewed_button):
            moves.addWidget(button)
        main.addLayout(moves)

        help_label = QLabel(HELP)
        help_label.setWordWrap(True)
        side.addWidget(help_label)
        side.addWidget(QLabel("All excerpts (click one to open it):"))
        self.excerpt_list = QListWidget()
        self.excerpt_list.itemClicked.connect(lambda item: self.open(item.data(Qt.ItemDataRole.UserRole)))
        side.addWidget(self.excerpt_list, 1)
        self.backup_button = QPushButton("Save a backup copy…")
        self.backup_button.clicked.connect(self.save_backup)
        side.addWidget(self.backup_button)

        keys = {
            "Space": self.toggle_play, "Home": lambda: self.seek(0.0), "[": self.mark_start,
            "]": lambda: self.mark_end("confirmed"), "U": lambda: self.mark_end("uncertain"), "Escape": self.cancel_mark,
            "C": self.complete, "Left": lambda: self.move(-0.1), "Right": lambda: self.move(0.1),
            "Shift+Left": lambda: self.move(-0.02), "Shift+Right": lambda: self.move(0.02),
            "Ctrl+Left": lambda: self.move(-1.0), "Ctrl+Right": lambda: self.move(1.0),
            "PgUp": lambda: self.step(-1), "PgDown": lambda: self.step(1),
        }
        for sequence, action in keys.items():
            QShortcut(QKeySequence(sequence), self, activated=action)

    # -- what is shown -------------------------------------------------------------------------

    def state(self):
        return self.store.of(self.current) if self.store and self.current else schema.empty()

    def flash(self, text):
        self.saved.setText(text)
        self.saved_timer.start()

    def refresh(self):
        if self.current is None:
            return
        state = self.state()
        status = schema.status(state)
        words = {"positive": "Completed: voice marked", "negative": "Completed: no voice at all",
                 "uncertain": "Completed: only uncertain stretches", "incomplete": "Not completed yet",
                 "unusable": f"Unusable: {state['reason']}"}[status]
        self.status.setText(f"{words}   ·   {len(state['intervals'])} stretch(es) marked"
                            + (f"   ·   open mark at {self.pending:.3f} s" if self.pending is not None else ""))
        intervals = sorted(((k, i["kind"], i["start"], i["end"]) for k, i in state["intervals"].items()), key=lambda x: (x[2], x[0]))
        self.listing.blockSignals(True)
        self.listing.clear()
        for identity, kind, start, end in intervals:
            item = QListWidgetItem(f"{'✓ voice    ' if kind == 'confirmed' else '? uncertain'}   {start:7.3f} – {end:7.3f} s")
            item.setData(Qt.ItemDataRole.UserRole, identity)
            self.listing.addItem(item)
            if identity == self.selected:
                self.listing.setCurrentItem(item)
        self.listing.blockSignals(False)
        if self.selected not in state["intervals"]:
            self.selected = None
        for box_key, box_ in self.tag_boxes.items():
            box_.blockSignals(True)
            box_.setChecked(box_key in state["tags"])
            box_.blockSignals(False)
        needs_audio = ("start=", "end=", "hear-start", "hear-end")
        for key, button in self.edit_buttons.items():
            button.setEnabled(self.selected is not None and (self.available or key not in needs_audio))
        states = self.store.state() if self.store else {}
        tally = {s: 0 for s in schema.STATUSES}
        for excerpt in self.order:
            tally[schema.status(states.get(excerpt))] += 1
        where = self.order.index(self.current) + 1
        done = tally["positive"] + tally["negative"] + tally["uncertain"]
        self.counts.setText(f"Excerpt {where} of {len(self.order)}   ·   Completed {done} (voice {tally['positive']}, "
                            f"no voice {tally['negative']}, only uncertain {tally['uncertain']})   ·   Unusable {tally['unusable']}"
                            f"   ·   Still to do {tally['incomplete']}")
        marks = {"positive": "✓", "negative": "○", "uncertain": "?", "unusable": "✗", "incomplete": " "}
        self.excerpt_list.clear()
        for excerpt in self.order:
            item = QListWidgetItem(f"{marks[schema.status(states.get(excerpt))]}  {excerpt}")
            item.setData(Qt.ItemDataRole.UserRole, excerpt)
            self.excerpt_list.addItem(item)
            if excerpt == self.current:
                self.excerpt_list.setCurrentItem(item)
        self._paint()

    def _paint(self):
        state = self.state()
        intervals = [(k, i["kind"], i["start"], i["end"]) for k, i in state["intervals"].items()]
        duration = (self.excerpts[self.current]["end_s"] - self.excerpts[self.current]["start_s"]) if self.current else 0.0
        self.timeline.show_state(duration, self.outline, intervals, self.pending, self.player.position_s if self.available else 0.0,
                                 self.selected)

    def _tick(self):
        if not self.available:
            self.time.setText("")
            return
        position = self.player.position_s
        if self.stop_at is not None and position >= self.stop_at:
            self.player.pause()
            self.stop_at = None
        self.time.setText(f"Position {clock(self.player.position_s)} of {clock(self.player.duration_s)}"
                          f"   ·   {'playing' if self.player.playing else 'paused'}")
        self._paint()

    # -- moving --------------------------------------------------------------------------------

    def _first_unreviewed(self):
        states = self.store.state() if self.store else {}
        return next((e for e in self.order if schema.status(states.get(e)) == "incomplete"), None)

    def open(self, excerpt, play=False):
        if excerpt not in self.excerpts:
            return
        self.save_note()
        self.player.unload()
        self.current, self.pending, self.selected, self.stop_at = excerpt, None, None, None
        entry = self.excerpts[excerpt]
        self.identifier.setText(f"{excerpt}   ·   {self.partition.replace('_', ' ')}")
        title = self.titles.get(entry["recording"])
        self.song.setText(f"♪  {title}" if title else "♪  Song title not known")
        self.note.setText(self.state()["note"])
        self.available = False
        try:
            path = Path(entry["path"])
            if not path.is_file() or path.stat().st_size != entry["size"]:
                raise labelling.DatasetError("unavailable")
            self.audio, self.rate = labelling.read_excerpt(path, entry["start_s"], entry["end_s"])
            self.player.load(self.audio, self.rate)
            self.outline = envelope(self.audio, self.rate)
            self.available = True
        except Exception:
            self.outline = np.zeros(1)
        self.audio_problem.setText("" if self.available else UNAVAILABLE)
        for button in (self.play_button, self.restart_button, self.back_button, self.forward_button, self.start_button,
                       self.end_button, self.uncertain_button, self.complete_button):
            button.setEnabled(self.available)
        write_session(self.folder, {"current": excerpt})
        self.refresh()
        if self.available and play:
            self.player.play()

    def step(self, direction):
        if self.current in self.order:
            index = self.order.index(self.current) + direction
            if 0 <= index < len(self.order):
                self.open(self.order[index])

    def next_unreviewed(self):
        states = self.store.state() if self.store else {}
        start = self.order.index(self.current) + 1 if self.current in self.order else 0
        ring = self.order[start:] + self.order[:start]
        found = next((e for e in ring if e != self.current and schema.status(states.get(e)) == "incomplete"), None)
        if found:
            self.open(found)
        else:
            self.flash("Every excerpt here is completed or marked unusable. Thank you!")

    # -- playback ------------------------------------------------------------------------------

    def toggle_play(self):
        if self.available:
            self.stop_at = None
            self.player.toggle()

    def seek(self, seconds):
        if self.available:
            self.stop_at = None
            self.player.seek(max(0.0, min(self.player.duration_s, seconds)))
            self._paint()

    def move(self, delta):
        if self.available:
            self.seek(self.player.position_s + delta)

    def listen_around(self, seconds):
        if self.available:
            self.player.seek(max(0.0, seconds - AROUND_BEFORE_S))
            self.stop_at = min(self.player.duration_s, seconds + AROUND_AFTER_S)
            if not self.player.playing:
                self.player.play()

    # -- marking -------------------------------------------------------------------------------

    def position(self):
        """The playback position in seconds on the excerpt's timeline, to the millisecond."""
        return schema.seconds(self.player.position_s)

    def mark_start(self):
        if self.available and self.current:
            self.pending = self.position()
            self.flash(f"Open mark at {self.pending:.3f} s: press ] where the voice ends (U if uncertain).")
            self.refresh()

    def mark_end(self, kind):
        if not (self.available and self.current) or self.pending is None:
            return
        start, end = sorted((self.pending, self.position()))
        try:
            event = self.store.add_interval(self.current, kind, start, end)
        except schema.SchemaError as exc:
            self.flash(f"Not saved: {exc}")
            return
        self.pending, self.selected = None, event["id"]
        self.flash(f"Saved ✓  {'voice' if kind == 'confirmed' else 'uncertain'} {start:.3f} – {end:.3f} s")
        self.refresh()

    def cancel_mark(self):
        if self.pending is not None:
            self.pending = None
            self.flash("Open mark dropped.")
            self.refresh()

    def _select(self, item, _previous=None):
        self.selected = item.data(Qt.ItemDataRole.UserRole) if item else None
        self.refresh()

    def adjust(self, action):
        if self.selected is None or self.current is None:
            return
        interval = self.state()["intervals"].get(self.selected)
        if interval is None:
            return
        changes = {}
        if action in ("start-", "start+"):
            changes["start"] = interval["start"] + (NUDGE_S if action == "start+" else -NUDGE_S)
        elif action in ("end-", "end+"):
            changes["end"] = interval["end"] + (NUDGE_S if action == "end+" else -NUDGE_S)
        elif action == "start=" and self.available:
            changes["start"] = self.position()
        elif action == "end=" and self.available:
            changes["end"] = self.position()
        elif action == "kind":
            changes["kind"] = "uncertain" if interval["kind"] == "confirmed" else "confirmed"
        elif action == "hear-start":
            return self.listen_around(interval["start"])
        elif action == "hear-end":
            return self.listen_around(interval["end"])
        elif action == "remove":
            self.store.remove_interval(self.current, self.selected)
            self.selected = None
            self.flash("Removed (the history keeps it).")
            return self.refresh()
        if not changes:
            return
        try:
            self.store.edit_interval(self.current, self.selected, **changes)
        except schema.SchemaError as exc:
            self.flash(f"Not saved: {exc}")
            return
        self.flash("Saved ✓")
        self.refresh()

    # -- review --------------------------------------------------------------------------------

    def set_review(self, state, reason=""):
        if self.current is None:
            return
        self.store.set_review(self.current, state, reason)
        self.refresh()

    def complete(self):
        if not (self.current and self.available):
            return
        if self.pending is not None:
            self.flash("Close or drop the open mark first (] , U or Esc).")
            return
        self.set_review("complete")
        status = schema.status(self.state())
        self.flash("Completed ✓  " + {"positive": "voice marked.", "negative": "no voice at all.",
                                       "uncertain": "only uncertain stretches."}[status])

    def unusable(self, reason=None):
        if self.current is None:
            return
        if reason is None:
            reason, ok = QInputDialog.getText(self, "Unusable", "Why can this excerpt not be used?")
            if not ok:
                return
        if not str(reason).strip():
            self.flash("An unusable excerpt needs a reason.")
            return
        self.pending = None
        self.set_review("unusable", reason)
        self.flash("Marked unusable.")

    def save_tags(self):
        if self.current is None:
            return
        tags = [k for k, box_ in self.tag_boxes.items() if box_.isChecked()]
        if tags != self.state()["tags"]:
            self.store.set_tags(self.current, tags)
            self.flash("Tags saved ✓")
            self.refresh()

    def save_note(self):
        if self.current is None or self.store is None:
            return
        text = self.note.text().strip()
        if text != self.state()["note"]:
            self.store.note(self.current, text)
            self.flash("Note saved ✓")

    def save_backup(self, destination=None):
        if destination is None:
            destination = QFileDialog.getExistingDirectory(self, "Where should the backup copy go?")
            if not destination:
                return None
        files = [self.folder / "annotations.jsonl", self.folder / "session.json",
                 *(self.root / "manifests").glob("*.json"), self.root / "fingerprints" / "manifests.json",
                 self.root / "identity" / "titles.json", self.root / "identity" / "aliases.json"]
        target = backup(files, destination, f"vocal-presence-{self.partition}-backup")
        self.flash(f"Backup saved ✓  {target.name}")
        return target

    def closeEvent(self, event):  # noqa: N802 (Qt name)
        self.save_note()
        self.player.close()
        self.lock.unlock()
        super().closeEvent(event)
