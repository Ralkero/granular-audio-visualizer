"""Real-machine check of the vocal-presence annotation window (Phase A5.1).

Builds a temporary project around a generated recording (no private data), opens the
window with the real player (the sound server's default output; nothing is changed on the
system), and checks playback, pausing, seeking, exact marks while paused, keyboard marking,
"listen around", persistence after reopening, the history's integrity and the backup.
Prints a JSON report; exit status 0 when every check passes.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from gmv.character.research.labeling.window import place_on_screen
from gmv.character.research.presence import project, schema, split
from gmv.character.research.presence.store import Store
from gmv.character.research.presence.window import PresenceWindow

CLIENT = "GMV vocal labelling"  # the shared player's client name
RATE = 48000


def streams():
    try:
        listed = json.loads(subprocess.run(["pactl", "--format=json", "list", "sink-inputs"], capture_output=True,
                                           text=True, timeout=5).stdout or "[]")
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return [s for s in listed if s.get("properties", {}).get("application.name") == CLIENT]


def pump(app, seconds, until=None):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        app.processEvents()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until() if until is not None else True


def generated_project(work):
    """One 14 s generated recording: quiet noise, a voice-like tone from 3 to 7 s, noise."""
    t = np.arange(14 * RATE) / RATE
    mono = 0.02 * np.random.default_rng(0).standard_normal(len(t))
    mono += np.where((t >= 3) & (t < 7), 0.25 * np.sin(2 * np.pi * 220 * t), 0.0)
    path = work / "generated.wav"
    sf.write(path, np.stack([mono, mono], axis=1).astype(np.float32), RATE)
    from gmv.character.research.labeling import dataset as labelling

    sha = labelling.sha256(path)
    root = project.layout(work / "presence")
    excerpt = {"id": project.excerpt_id(sha, 1.0), "source": "new", "recording": sha, "path": str(path),
               "size": path.stat().st_size, "duration_s": 14.0, "sample_rate": RATE, "channels": 2,
               "start_s": 1.0, "end_s": 11.0, "group": "generated-group", "selection": "real-machine check"}
    registry = split.assign({"generated-group": {"nodes": [f"rec:{sha}", "a3:generated"]}}, {}, set(), {"generated"})
    registry["entries"]["generated-group"]["partition"] = "development"
    project.write_registry(root, registry)
    project.write_manifest(root, "development", [excerpt])
    (root / "identity" / "titles.json").write_text(json.dumps({sha: "Generated test tune"}))
    return root, excerpt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    checks, facts = {}, {}
    work = Path(tempfile.mkdtemp(prefix="gmv-presence-check-"))
    try:
        root, excerpt = generated_project(work)
        app = QApplication.instance() or QApplication(sys.argv[:1])
        facts["platform"] = app.platformName()
        window = PresenceWindow(root, "development")
        place_on_screen(window, app)
        window.show()
        checks["window_opened"] = QTest.qWaitForWindowExposed(window, 5000) and window.current == excerpt["id"]
        facts["screen"] = window.screen().name()
        window.activateWindow()
        pump(app, 0.4)
        checks["title_from_private_metadata"] = window.song.text() == "♪  Generated test tune"
        checks["envelope_drawn"] = float(np.max(window.outline)) > 0.5

        window.toggle_play()
        checks["stream_starts"] = pump(app, 2.0, lambda: bool(streams()))
        pump(app, 0.3)
        p0, t0 = window.player.position_s, time.monotonic()
        pump(app, 1.5)
        moved, waited = window.player.position_s - p0, time.monotonic() - t0
        checks["position_runs_in_real_time"] = abs(moved - waited) < 0.15
        facts["position_moved_s"], facts["wall_s"] = round(moved, 3), round(waited, 3)
        window.toggle_play()
        checks["pause_ends_stream"] = pump(app, 1.0, lambda: not streams())
        held = window.player.position_s
        pump(app, 0.5)
        checks["position_holds_when_paused"] = abs(window.player.position_s - held) < 1e-6

        window.seek(2.0)
        checks["seek_is_exact_when_paused"] = abs(window.player.position_s - 2.0) < 1e-3
        window.mark_start()
        window.seek(6.25)
        window.mark_end("confirmed")
        state = window.state()["intervals"]
        checks["paused_marks_exact"] = sorted((i["start"], i["end"]) for i in state.values()) == [(2.0, 6.25)]
        window.seek(7.5)
        QTest.keyClick(window, Qt.Key.Key_BracketLeft)
        window.seek(8.0)
        QTest.keyClick(window, Qt.Key.Key_U)
        kinds = sorted((i["kind"], i["start"], i["end"]) for i in window.state()["intervals"].values())
        checks["keyboard_marking"] = ("uncertain", 7.5, 8.0) in kinds
        window.listen_around(2.0)
        checks["listen_around_plays"] = pump(app, 1.5, lambda: bool(streams()))
        checks["listen_around_stops"] = pump(app, 3.0, lambda: not window.player.playing)
        facts["stopped_at_s"] = round(window.player.position_s, 3)
        checks["listen_around_window"] = 2.4 <= window.player.position_s <= 2.8
        window.complete()
        checks["completed_positive"] = schema.status(window.state()) == "positive"
        target = window.save_backup(work / "backups")
        checks["backup_written"] = target.is_file() and "annotations.jsonl" in zipfile.ZipFile(target).namelist()
        window.close()
        pump(app, 0.5)
        checks["no_stream_after_close"] = pump(app, 1.0, lambda: not streams())

        again = PresenceWindow(root, "development")
        place_on_screen(again, app)
        again.show()
        QTest.qWaitForWindowExposed(again, 5000)
        history = Store(project.partition_folder(root, "development"), project.durations(project.read_manifest(root, "development")))
        checks["resumes_same_excerpt"] = again.current == excerpt["id"]
        checks["marks_kept_after_reopen"] = len(again.state()["intervals"]) == 2 and schema.status(again.state()) == "positive"
        checks["history_intact"] = history.intact and history.damaged == 0
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
