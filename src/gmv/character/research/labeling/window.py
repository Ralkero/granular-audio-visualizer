"""The labelling window: listen to a ten-second excerpt and click what the voice does.

The window shows an excerpt's anonymous identifier and, at the owner's request, the song
title of its recording (from ``titles.json``). Nothing else about it appears: no artist,
file, folder, how it was chosen, or which part of the experiment it belongs to, in any
text, tooltip, list entry, message or the window title.

Every click is saved at once (``store.Labels``); closing the window loses nothing, and
opening it again returns to the excerpt that was open.
"""

import os
import sys
from html import escape
from pathlib import Path

from PySide6.QtCore import QLockFile, Qt, QTimer
from PySide6.QtGui import QFont, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
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
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from . import CLASSES, GUIDE, RULE, SKIPS
from . import dataset as data
from . import store, titles

TITLE = "GMV: label the vocal delivery"
NEXT_AFTER_MS = 700
NOTE_AFTER_MS = 800
SAVED_FOR_MS = 4000
UNAVAILABLE = (
    "This excerpt cannot be played right now: its recording is not available (for "
    "example, the external drive is not connected, or the file was moved or changed). "
    "The answers and No vocals need the excerpt to be heard, so they are switched off. "
    "Use Skip / review later and come back to it."
)
NOT_HEARD = (
    "No vocals was not saved: this excerpt cannot be played right now, and No vocals needs "
    "it to be heard. Use Skip / review later and come back to it."
)
HELP = (
    "Listen to the excerpt, then click the one answer that describes the vocal technique "
    "you hear most. Label what is actually audible: the artist, the style or the song "
    "title never decides the answer."
)
REFERENCES = "reference_vocalists.txt"  # the owner's own examples, kept beside the answers


def clock(seconds):
    return f"{seconds:4.1f} s"


class LabelWindow(QMainWindow):
    def __init__(self, folder, player=None, autoplay=True):
        super().__init__()
        self.folder = Path(folder)
        self.setWindowTitle(TITLE)
        self.lock = QLockFile(str(self.folder / "labelling.lock"))
        self.lock.setStaleLockTime(0)
        self.dataset = None
        self.problem = ""
        self.current = None
        self.available = False
        self.labels = store.Labels(self.folder)
        self.titles = titles.load(self.folder)
        if player is None:
            from .playback import default_player

            player = default_player()
        self.player = player
        try:
            self.dataset = data.load(self.folder)
        except data.DatasetError as exc:
            self.problem = (
                "No excerpts are ready to label yet."
                if "no dataset" in str(exc)
                else "The excerpt list could not be read. Nothing has been changed. (" + str(exc).split(":")[0] + ")"
            )
        if self.dataset is not None:
            self.folder.mkdir(parents=True, exist_ok=True)
            if not self.lock.tryLock(200):
                self.problem = "The labelling window is already open. Use that one."
                self.dataset = None
        self.excerpts = {e["id"]: e for e in self.dataset["excerpts"]} if self.dataset else {}
        self.order = list(self.dataset["order"]) if self.dataset else []
        self.next_timer = QTimer(self, singleShot=True, interval=NEXT_AFTER_MS)
        self.next_timer.timeout.connect(self.next_unlabelled)
        self.note_timer = QTimer(self, singleShot=True, interval=NOTE_AFTER_MS)
        self.note_timer.timeout.connect(self.save_note)
        self.saved_timer = QTimer(self, singleShot=True, interval=SAVED_FOR_MS)
        self.saved_timer.timeout.connect(lambda: self.saved.setText(""))
        self._build(autoplay)
        self.ticker = QTimer(self, interval=50)
        self.ticker.timeout.connect(self._tick)
        self.ticker.start()
        if self.dataset is None:
            self._show_problem()
            return
        if self.labels.damaged:
            self.flash(
                f"{self.labels.damaged} unreadable line(s) in the saved answers were left out "
                "(probably a write cut off by a crash); everything else is intact.",
                warning=True,
            )
        earlier = store.progress(self.order, self.labels.state())["earlier"]
        if earlier and not self.labels.damaged:
            self.flash(
                f"The answer choices have changed. {earlier} excerpt(s) you answered with the "
                "earlier choices show that answer and need a new one. Nothing was deleted.",
                warning=True,
            )
        self.refresh_list()
        session = store.read_session(self.folder)
        self.autoplay.setChecked(session.get("autoplay", autoplay))
        self.advance.setChecked(session.get("advance", True))
        self.open(store.resume_point(self.order, self.labels.state(), session), play=False)

    # -- layout ---------------------------------------------------------------------------

    def _build(self, autoplay):
        root = QWidget()
        self.setCentralWidget(root)
        outer = QHBoxLayout(root)
        main = QVBoxLayout()
        outer.addLayout(main, 3)
        side = QVBoxLayout()
        outer.addLayout(side, 2)

        self.identifier = QLabel("")
        big = QFont()
        big.setPointSize(22)
        big.setBold(True)
        self.identifier.setFont(big)
        self.song = QLabel("")
        song_font = QFont()
        song_font.setPointSize(18)
        self.song.setFont(song_font)
        self.song.setWordWrap(True)
        self.status = QLabel("")
        status_font = QFont()
        status_font.setPointSize(13)
        self.status.setFont(status_font)
        self.status.setWordWrap(True)
        main.addWidget(self.identifier)
        main.addWidget(self.song)
        main.addWidget(self.status)

        self.counts = QLabel("")
        self.bar = QProgressBar()
        self.bar.setFormat("%v of %m done")
        main.addWidget(self.counts)
        main.addWidget(self.bar)

        listen = QGroupBox("Listen")
        grid = QGridLayout(listen)
        self.play_button = QPushButton("▶  Play")
        self.play_button.setMinimumHeight(44)
        self.play_button.clicked.connect(self.toggle_play)
        self.restart_button = QPushButton("⟲  Play again from the start")
        self.restart_button.setMinimumHeight(44)
        self.restart_button.clicked.connect(self.replay)
        self.repeat = QCheckBox("Repeat")
        self.repeat.setToolTip("Play the excerpt over and over until you pause it")
        self.position = QSlider(Qt.Orientation.Horizontal)
        self.position.setRange(0, 1000)
        self.position.sliderReleased.connect(self._seek)
        self.time = QLabel("")
        grid.addWidget(self.play_button, 0, 0)
        grid.addWidget(self.restart_button, 0, 1)
        grid.addWidget(self.repeat, 0, 2)
        grid.addWidget(self.position, 1, 0, 1, 2)
        grid.addWidget(self.time, 1, 2)
        self.audio_problem = QLabel("")
        self.audio_problem.setWordWrap(True)
        self.audio_problem.setStyleSheet("color: #b3261e;")
        grid.addWidget(self.audio_problem, 2, 0, 1, 3)
        main.addWidget(listen)

        answers = QGroupBox("Which vocal technique dominates the excerpt?")
        column = QVBoxLayout(answers)
        self.answer_buttons = {}
        for index, (key, (name, meaning)) in enumerate(CLASSES.items(), 1):
            button = QPushButton(f"{index}   {name}")
            button.setCheckable(True)
            button.setMinimumHeight(52)
            font = QFont()
            font.setPointSize(14)
            button.setFont(font)
            button.setToolTip(meaning)
            button.clicked.connect(lambda _checked, k=key: self.choose(k))
            column.addWidget(button)
            self.answer_buttons[key] = button
        rule = QLabel(RULE)
        rule.setWordWrap(True)
        column.addWidget(rule)
        main.addWidget(answers)

        self.saved = QLabel("")
        self.saved.setWordWrap(True)
        main.addWidget(self.saved)

        moves = QHBoxLayout()
        self.previous_button = QPushButton("←  Previous")
        self.previous_button.clicked.connect(lambda: self.step(-1))
        self.skip_button = QPushButton(SKIPS["later"])
        self.skip_button.setToolTip("Several techniques with none dominating, or not sure: come back to it later")
        self.skip_button.clicked.connect(lambda: self.skip("later"))
        self.no_vocals_button = QPushButton(SKIPS["no_vocals"])
        self.no_vocals_button.setToolTip("No voice in the excerpt: leave it without a vocal answer")
        self.no_vocals_button.clicked.connect(lambda: self.skip("no_vocals"))
        self.next_button = QPushButton("Next  →")
        self.next_button.clicked.connect(lambda: self.step(1))
        self.unlabelled_button = QPushButton("Next without an answer  ⏭")
        self.unlabelled_button.clicked.connect(self.next_unlabelled)
        for button in (
            self.previous_button, self.no_vocals_button, self.skip_button, self.next_button, self.unlabelled_button,
        ):
            button.setMinimumHeight(40)
            moves.addWidget(button)
        main.addLayout(moves)

        self.note = QLineEdit()
        self.note.setPlaceholderText("Optional note, for example: “scream under a clean chorus”")
        self.note.textEdited.connect(lambda _text: self.note_timer.start())
        note_row = QHBoxLayout()
        note_row.addWidget(QLabel("Note:"))
        note_row.addWidget(self.note)
        main.addLayout(note_row)

        options = QHBoxLayout()
        self.autoplay = QCheckBox("Play each excerpt as soon as it opens")
        self.autoplay.setChecked(autoplay)
        self.advance = QCheckBox("Go to the next excerpt after answering")
        self.advance.setChecked(True)
        options.addWidget(self.autoplay)
        options.addWidget(self.advance)
        options.addStretch(1)
        main.addLayout(options)
        main.addStretch(1)

        meanings = QGroupBox("What the answers mean")
        meaning_layout = QVBoxLayout(meanings)
        self.guide = QTextBrowser()
        self.guide.setOpenLinks(False)
        self.guide.setHtml(self._guide_html())
        meaning_layout.addWidget(self.guide)
        side.addWidget(meanings, 3)

        side.addWidget(QLabel("All excerpts (click one to open it):"))
        self.listing = QListWidget()
        self.listing.itemClicked.connect(lambda item: self.open(item.data(Qt.ItemDataRole.UserRole)))
        side.addWidget(self.listing, 2)
        tools = QHBoxLayout()
        self.add_button = QPushButton("Add recordings…")
        self.add_button.setToolTip("Choose more of your own audio files to take excerpts from")
        self.add_button.clicked.connect(self.add_recordings)
        self.backup_button = QPushButton("Save a backup copy…")
        self.backup_button.setToolTip("Save a copy of your answers into a folder you choose")
        self.backup_button.clicked.connect(self.save_backup)
        tools.addWidget(self.add_button)
        tools.addWidget(self.backup_button)
        side.addLayout(tools)

        self.problem_label = QLabel("")
        self.problem_label.setWordWrap(True)
        self.problem_label.setFrameShape(QFrame.Shape.StyledPanel)
        self.problem_label.hide()
        main.insertWidget(0, self.problem_label)

        shortcuts = {
            "Space": self.toggle_play,
            "R": self.replay,
            "S": lambda: self.skip("later"),
            "N": lambda: self.skip("no_vocals"),
            "Left": lambda: self.step(-1),
            "Right": lambda: self.step(1),
        }
        for index, key in enumerate(CLASSES, 1):
            shortcuts[str(index)] = lambda k=key: self.choose(k)
        for sequence, action in shortcuts.items():
            QShortcut(QKeySequence(sequence), self, activated=action)

    def _show_problem(self):
        self.problem_label.setText(self.problem)
        self.problem_label.show()
        for widget in (
            self.play_button, self.restart_button, self.previous_button, self.skip_button, self.no_vocals_button,
            self.next_button, self.unlabelled_button, self.note, self.add_button, *self.answer_buttons.values(),
        ):
            widget.setEnabled(False)
        self.backup_button.setEnabled(self.dataset is not None)

    # -- what is shown --------------------------------------------------------------------

    def _guide_html(self):
        parts = [f"<p>{escape(HELP)}</p>", f"<p><b>{escape(RULE)}</b></p>"]
        for index, (key, (name, _summary)) in enumerate(CLASSES.items(), 1):
            definition, points, note = GUIDE[key]
            items = "".join(f"<li>{escape(point)}</li>" for point in points)
            parts.append(
                f"<h3>{index}. {escape(name)}</h3><p>{escape(definition)}</p>"
                f"<ul>{items}</ul><p><i>{escape(note)}</i></p>"
            )
        references = self.folder / REFERENCES
        if references.is_file():
            lines = [line.strip() for line in references.read_text(encoding="utf-8").splitlines() if line.strip()]
            if lines:
                items = "".join(f"<li>{escape(line)}</li>" for line in lines)
                parts.append(
                    "<h3>Reference vocalists (your examples)</h3>"
                    "<p>Real-world reference points for the answers. They are not answers: a "
                    "vocalist can use several techniques, even within one song.</p>"
                    f"<ul>{items}</ul>"
                )
        parts.append(
            "<p>Keys (optional): 1–5 answer, Space play/pause, R from the start, "
            "S review later, N no vocals, ← → previous/next.</p>"
        )
        return "".join(parts)

    def title_of(self, excerpt):
        entry = self.titles.get(self.excerpts[excerpt]["recording"]) or {}
        return entry.get("title")

    def describe(self, excerpt, state):
        current = state.get(excerpt, {})
        if current.get("label"):
            return f"Your answer: {CLASSES[current['label']][0]}"
        if current.get("skipped"):
            text = "Done: marked No vocals" if current.get("skip_reason") == "no_vocals" else "Skipped: review later"
        else:
            text = "Not labelled yet"
        earlier = current.get("earlier")
        if earlier:
            text += f"   (earlier answer, with the old choices: {earlier['name']})"
        return text

    def refresh_list(self):
        state = self.labels.state()
        selected = self.current
        self.listing.blockSignals(True)
        self.listing.clear()
        for position, excerpt in enumerate(self.order, 1):
            current = state.get(excerpt, {})
            if current.get("label"):
                mark = "✓ " + CLASSES[current["label"]][0]
            elif current.get("skipped"):
                mark = "✓ No vocals" if current.get("skip_reason") == "no_vocals" else "↷ review later"
            elif current.get("earlier"):
                mark = "• earlier: " + current["earlier"]["name"]
            else:
                mark = ""
            item = QListWidgetItem(f"{position:3d}.  {excerpt}   {mark}")
            item.setData(Qt.ItemDataRole.UserRole, excerpt)
            self.listing.addItem(item)
            if excerpt == selected:
                self.listing.setCurrentItem(item)
        self.listing.blockSignals(False)
        counts = store.progress(self.order, state)
        self.bar.setRange(0, max(1, counts["total"]))
        self.bar.setValue(counts["done"])
        where = self.order.index(self.current) + 1 if self.current in self.order else 0
        text = (
            f"Excerpt {where} of {counts['total']}   ·   Answered {counts['labelled']}   ·   "
            f"No vocals {counts['skipped_no_vocals']}   ·   Review later {counts['skipped_later']}   ·   "
            f"Still to do {counts['remaining']}"
        )
        if counts["earlier"]:
            text += f"   ·   Earlier answers to redo: {counts['earlier']}"
        self.counts.setText(text)
        if self.current:
            self.status.setText(self.describe(self.current, state))
            label = state.get(self.current, {}).get("label")
            for key, button in self.answer_buttons.items():
                button.setChecked(key == label)

    def flash(self, text, warning=False):
        colour = "#b3261e" if warning else "#1b6e20"
        self.saved.setStyleSheet(f"color: {colour}; font-weight: bold;")
        self.saved.setText(text)
        self.saved_timer.start()

    # -- moving ---------------------------------------------------------------------------

    def open(self, excerpt, play=None):
        if excerpt not in self.excerpts:
            return
        self.next_timer.stop()
        self.save_note()
        self.player.unload()
        self.current = excerpt
        self.note.setText(self.labels.state().get(excerpt, {}).get("note", ""))
        self.identifier.setText(excerpt)
        title = self.title_of(excerpt)
        self.song.setText(f"♪  {title}" if title else "♪  Song title not known")
        self.audio_problem.setText("")
        self.available = False
        entry = self.excerpts[excerpt]
        recording = self.dataset["recordings"][entry["recording"]]
        try:
            path = Path(recording["path"])
            if not path.is_file() or path.stat().st_size != recording["size"]:
                raise data.DatasetError("unavailable")
            audio, rate = data.read_excerpt(path, entry["start_s"], entry["end_s"])
            self.player.load(audio, rate)
            self.available = True
        except Exception:
            self.audio_problem.setText(UNAVAILABLE)
        # Every decision about what is heard needs the excerpt to be heard: the five answers
        # and No vocals. Skip / review later stays available.
        for button in (*self.answer_buttons.values(), self.no_vocals_button, self.play_button, self.restart_button):
            button.setEnabled(self.available)
        store.write_session(
            self.folder,
            {"current": excerpt, "autoplay": self.autoplay.isChecked(), "advance": self.advance.isChecked()},
        )
        self.refresh_list()
        if self.available and (self.autoplay.isChecked() if play is None else play):
            self.player.play()

    def step(self, direction):
        if self.current in self.order:
            index = self.order.index(self.current) + direction
            if 0 <= index < len(self.order):
                self.open(self.order[index])

    def next_unlabelled(self):
        """The next excerpt after this one, in order, that still needs the owner: one never
        answered or skipped if there is any, otherwise one put aside for review. Excerpts
        marked No vocals are done and are not offered again."""
        self.next_timer.stop()
        state = self.labels.state()
        if not self.order:
            return
        start = self.order.index(self.current) + 1 if self.current in self.order else 0
        ring = self.order[start:] + self.order[:start]
        for wanted in (
            lambda s: not s.get("label") and not s.get("skipped"),
            store.pending,
        ):
            found = next((e for e in ring if e != self.current and wanted(state.get(e, {}))), None)
            if found:
                self.open(found)
                return
        self.flash(
            "All done: every excerpt has an answer or is marked No vocals. Thank you! You can "
            "still go back and change any of them."
        )

    # -- answering ------------------------------------------------------------------------

    def choose(self, answer):
        if self.current is None or not self.available:
            return
        self.save_note()
        self.labels.label(self.current, answer)
        self.flash(f"Saved ✓  {self.current}: {CLASSES[answer][0]}")
        self.refresh_list()
        if self.advance.isChecked():
            self.next_timer.start()

    def skip(self, reason="later"):
        """Leave the current excerpt without an answer, for a reason: "no_vocals" (heard, and
        there is no voice) or "later" (come back to it). While the recording cannot be played,
        No vocals is refused here as well as on its button and key, since nobody can have
        heard the excerpt, and review later does not replace a No vocals decision already
        made, so a disconnected drive never undoes what was heard."""
        if self.current is None:
            return
        if reason == "no_vocals" and not self.available:
            self.flash(NOT_HEARD)
            return
        self.save_note()
        current = self.labels.state().get(self.current, {})
        decided = current.get("skipped") and current.get("skip_reason") == "no_vocals"
        if not self.available and decided:
            self.flash(f"{self.current} stays marked No vocals: it cannot be played right now to check it again.")
        elif not current.get("label") and not (current.get("skipped") and current.get("skip_reason") == reason):
            self.labels.skip(self.current, reason)
            if reason == "no_vocals":
                self.flash(f"Saved ✓  {self.current}: no vocals (left without a vocal answer).")
            else:
                self.flash(f"Skipped {self.current}. It stays in the list to come back to.")
        self.next_unlabelled()

    def save_note(self):
        self.note_timer.stop()
        if self.current is None:
            return
        text = self.note.text().strip()
        if text != self.labels.state().get(self.current, {}).get("note", ""):
            self.labels.note(self.current, text)
            self.flash(f"Note saved ✓  {self.current}")

    # -- playback -------------------------------------------------------------------------

    def toggle_play(self):
        if self.available:
            self.player.toggle()
            self._tick()

    def replay(self):
        if self.available:
            self.player.replay()
            self._tick()

    def _seek(self):
        if self.available and self.player.duration_s:
            self.player.seek(self.position.value() / 1000 * self.player.duration_s)

    def _tick(self):
        if not self.available:
            self.time.setText("")
            self.play_button.setText("▶  Play")
            return
        if self.repeat.isChecked() and self.player.finished:
            self.player.replay()
        duration = self.player.duration_s or 1.0
        if not self.position.isSliderDown():
            self.position.setValue(int(1000 * self.player.position_s / duration))
        self.time.setText(f"{clock(self.player.position_s)} / {clock(duration)}")
        self.play_button.setText("❚❚  Pause" if self.player.playing else "▶  Play")
        if self.player.error:
            self.audio_problem.setText(self.player.error)

    # -- tools ----------------------------------------------------------------------------

    def add_recordings(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Choose recordings to add", str(Path.home()),
            "Audio files (" + " ".join("*" + s for s in data.AUDIO_SUFFIXES) + ");;All files (*)",
        )
        if not files:
            return
        names = {data.artist_name(key): key for key in self.dataset["artists"]}
        artist, ok = QInputDialog.getItem(
            self, "Who is the artist?",
            "All the chosen recordings must be by one artist.\nChoose a name from the list, or type a new one:",
            sorted(names), -1, True,
        )
        if not ok or not artist.strip():
            return
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            added, problems = data.add_recordings(self.folder, self.dataset, files, names.get(artist, artist))
        except data.DatasetError as exc:
            added, problems = [], {"": str(exc)}
        finally:
            QApplication.restoreOverrideCursor()
        self.excerpts = {e["id"]: e for e in self.dataset["excerpts"]}
        self.order = list(self.dataset["order"])
        if added:
            self.titles = titles.add_from_tags(self.folder, self.dataset)
        self.refresh_list()
        text = f"{len(added)} new excerpts were added and mixed into the list."
        if problems:
            text += "\n\nNot added:\n" + "\n".join(
                f"• {Path(path).name or 'all'}: {why}" for path, why in problems.items()
            )
        QMessageBox.information(self, "Add recordings", text)

    def save_backup(self):
        target = QFileDialog.getExistingDirectory(self, "Choose a folder for the backup copy", str(Path.home()))
        if not target:
            return
        self.save_note()
        path = store.backup(self.folder, target)
        QMessageBox.information(self, "Backup saved", f"Your answers were copied to:\n{path}")

    def closeEvent(self, event):
        self.save_note()
        self.player.close()
        if self.current:
            store.write_session(
                self.folder,
                {"current": self.current, "autoplay": self.autoplay.isChecked(), "advance": self.advance.isChecked()},
            )
        if self.lock.isLocked():
            self.lock.unlock()
        super().closeEvent(event)


def place_on_screen(window, app):
    """Open on the display named by ``GMV_SCREEN`` (part of its name), when it is set."""
    wanted = os.environ.get("GMV_SCREEN", "").lower()
    screen = next((s for s in app.screens() if wanted and wanted in s.name().lower()), None)
    if screen is None:
        window.resize(1200, 820)
        return
    window.setScreen(screen)
    area = screen.availableGeometry()
    window.setGeometry(area.x() + 40, area.y() + 40, min(1300, area.width() - 80), min(860, area.height() - 80))


def main(folder):
    app = QApplication.instance() or QApplication(sys.argv[:1])
    app.setApplicationName("GMV vocal labelling")
    window = LabelWindow(folder)
    place_on_screen(window, app)
    window.show()
    return app.exec()
