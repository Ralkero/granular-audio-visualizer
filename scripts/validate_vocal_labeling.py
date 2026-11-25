"""Check the vocal labelling window on this machine, with real sound: it opens, plays
excerpts through the real sound server, responds to clicks and keys, saves answers and
resumes after being closed.

It works on a copy of the excerpt list in a temporary folder, so the owner's own answers
are never touched, and deletes that copy at the end. Each excerpt played is heard on the
default output (ordinary playback; nothing about the sound system is changed).

    uv run --no-sync python scripts/validate_vocal_labeling.py [--folder DIR] [--json OUT] [--screenshot PNG]

Set ``GMV_SCREEN`` to part of a display's name to open the window there.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gmv.character.research.labeling import store
from gmv.character.research.labeling.cli import default_folder
from gmv.character.research.labeling.window import LabelWindow, place_on_screen

CLIENT = "GMV vocal labelling"


def streams():
    """Our playback streams on the sound server: (sink, sample spec, corked) each."""
    try:
        listed = json.loads(
            subprocess.run(
                ["pactl", "--format=json", "list", "sink-inputs"], capture_output=True, text=True, timeout=5
            ).stdout
            or "[]"
        )
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return [
        {"sink": s.get("sink"), "sample_spec": s.get("sample_specification"), "corked": s.get("corked")}
        for s in listed
        if s.get("properties", {}).get("application.name") == CLIENT
    ]


def default_sink():
    try:
        return subprocess.run(["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return None


def pump(app, seconds, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until() if until is not None else True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--folder", default=str(default_folder()))
    parser.add_argument("--json")
    parser.add_argument("--screenshot")
    args = parser.parse_args(argv)
    checks, facts = {}, {}
    work = Path(tempfile.mkdtemp(prefix="gmv-vocal-check-"))
    try:
        shutil.copy(Path(args.folder) / "dataset.json", work / "dataset.json")
        app = QApplication.instance() or QApplication(sys.argv[:1])
        facts["platform"] = app.platformName()
        facts["default_sink"] = default_sink()
        window = LabelWindow(work, autoplay=False)
        place_on_screen(window, app)
        window.show()
        checks["window_opened"] = QTest.qWaitForWindowExposed(window, 5000) and window.dataset is not None
        facts["screen"] = window.screen().name()
        window.activateWindow()
        pump(app, 0.5)
        first = window.current
        facts["first_excerpt"] = first
        facts["excerpt_rate"] = window.player.rate
        facts["excerpt_channels"] = int(window.player.audio.shape[1])
        facts["excerpt_seconds"] = round(window.player.duration_s, 3)
        checks["only_the_excerpt_is_loaded"] = abs(window.player.duration_s - 10.0) < 0.01

        # Play: a stream of ours appears on the default output, at the file's own rate.
        began = time.monotonic()
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        checks["playing_after_click"] = pump(app, 2.0, lambda: window.player.playing and bool(streams()))
        found = streams() or []
        facts["streams_while_playing"] = found
        checks["stream_on_default_output"] = bool(found) and all(s["corked"] is False for s in found)
        checks["stream_at_excerpt_rate"] = bool(found) and all(
            str(window.player.rate) in str(s["sample_spec"]) for s in found
        )
        checks["play_button_says_pause"] = "Pause" in window.play_button.text()
        pump(app, 2.5)
        # Pause: the stream ends at once and the position holds.
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        held = window.player.position_s
        facts["paused_at_s"] = round(held, 2)
        checks["pause_stops_stream"] = pump(app, 1.0, lambda: not streams())
        pump(app, 1.0)
        checks["position_holds_while_paused"] = abs(window.player.position_s - held) < 1e-6 and 2.0 < held < 4.5
        # Resume and play to the end: real time, then finished at the excerpt's end.
        resumed = time.monotonic()
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        checks["finished_at_end"] = pump(app, 15.0, lambda: window.player.finished)
        facts["seconds_to_finish_after_resume"] = round(time.monotonic() - resumed, 2)
        facts["seconds_left_at_resume"] = round(10.0 - held, 2)
        checks["plays_in_real_time"] = abs((time.monotonic() - resumed) - (10.0 - held)) < 0.8
        checks["position_at_end"] = abs(window.player.position_s - window.player.duration_s) < 0.05
        checks["no_stream_after_end"] = pump(app, 1.0, lambda: not streams())
        facts["total_seconds_first_excerpt"] = round(time.monotonic() - began, 2)

        # Play again from the start, and the position bar.
        QTest.mouseClick(window.restart_button, Qt.MouseButton.LeftButton)
        pump(app, 1.2)
        checks["from_the_start"] = window.player.playing and 0.5 < window.player.position_s < 2.0
        window.position.setValue(700)
        window._seek()
        pump(app, 0.4)
        checks["seek_with_the_bar"] = 6.5 < window.player.position_s < 8.2
        window.repeat.setChecked(True)
        checks["repeat_plays_again"] = pump(app, 5.0, lambda: window.player.position_s < 1.0 and window.player.playing)
        window.repeat.setChecked(False)
        QTest.mouseClick(window.play_button, Qt.MouseButton.LeftButton)
        pump(app, 0.5)

        # Answer by clicking: saved at once, then the next excerpt opens.
        QTest.mouseClick(window.answer_buttons["scream"], Qt.MouseButton.LeftButton)
        checks["answer_saved_on_click"] = store.Labels(work).state().get(first, {}).get("label") == "scream"
        checks["saved_message_shown"] = "Saved" in window.saved.text()
        checks["moves_to_next"] = pump(app, 2.0, lambda: window.current != first)
        second = window.current
        # Change an answer by clicking another one.
        window.advance.setChecked(False)
        QTest.mouseClick(window.answer_buttons["none"], Qt.MouseButton.LeftButton)
        QTest.mouseClick(window.answer_buttons["mixed"], Qt.MouseButton.LeftButton)
        state = store.Labels(work).state().get(second, {})
        checks["answer_changed"] = state.get("label") == "mixed" and state.get("changes") == 1
        # Answer by key. Keys reach the window only while it is the active window; Wayland
        # does not let a program make itself active, so this is reported, not required.
        window.activateWindow()
        pump(app, 0.3)
        facts["window_active_for_keys"] = window.isActiveWindow()
        QTest.keyClick(window, Qt.Key.Key_3)
        facts["answer_by_key"] = store.Labels(work).state().get(second, {}).get("label") == "shout"
        QTest.mouseClick(window.answer_buttons["mixed"], Qt.MouseButton.LeftButton)
        window.note.setFocus()
        QTest.keyClicks(window.note, "test note")
        window.save_note()
        checks["note_saved"] = store.Labels(work).state()[second]["note"] == "test note"
        window.advance.setChecked(True)
        QTest.mouseClick(window.skip_button, Qt.MouseButton.LeftButton)
        third = window.current
        QTest.mouseClick(window.skip_button, Qt.MouseButton.LeftButton)
        checks["skip_recorded"] = store.Labels(work).state().get(third, {}).get("skipped") is True
        checks["counts_shown"] = "Labelled 2" in window.counts.text() and "Skipped 1" in window.counts.text()
        checks["status_shown_in_full"] = window.status.width() >= window.status.sizeHint().width() or window.status.wordWrap()
        QTest.mouseClick(window.previous_button, Qt.MouseButton.LeftButton)
        checks["previous_goes_back"] = window.current == third
        if args.screenshot:
            window.grab().save(args.screenshot)
        left_open = window.current
        window.close()
        pump(app, 0.5)
        checks["no_stream_after_close"] = pump(app, 1.0, lambda: not streams())

        # Open again: same excerpt, same answers.
        again = LabelWindow(work, autoplay=False)
        place_on_screen(again, app)
        again.show()
        QTest.qWaitForWindowExposed(again, 5000)
        state = store.Labels(work).state()
        checks["resumes_where_left"] = again.current == left_open
        checks["answers_kept"] = state[first]["label"] == "scream" and state[second]["label"] == "mixed"
        checks["counts_after_reopen"] = "Labelled 2" in again.counts.text()
        again.close()
        pump(app, 0.3)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    passed = sum(bool(v) for v in checks.values())
    result = {"checks": checks, "facts": facts, "passed": passed, "total": len(checks)}
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1))
    return 0 if passed == len(checks) else 1


if __name__ == "__main__":
    sys.exit(main())
