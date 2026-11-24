"""Drive the real application window end to end on whatever system this is, and write what
happened as one JSON report: open the window, analyse a generated WAV and a FLAC, record
each session, replay the last one without its audio, show Radiant, and, where the system
has live capture, capture the default output for a few seconds (nothing is played).

    python scripts/platform_smoke.py OUT.json
    python scripts/platform_smoke.py OUT.json --remember      # also save and restore settings
    python scripts/platform_smoke.py OUT.json --no-live       # never start live capture

Without ``--remember`` no setting is read or written. With it the application's own
settings store is used, so point ``XDG_CONFIG_HOME`` at a scratch folder first unless the
real one is meant (``scripts/validate_linux.py`` does). Sessions go where the application
puts them (``XDG_DATA_HOME`` on Linux) unless ``--sessions FOLDER`` names another place.
``GMV_SCREEN`` holds part of the name of the display to open on.
"""

import argparse
import json
import os
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import soundfile as sf


def clicks(path, seconds=4, rate=48000, **how):
    """Clicks twice a second over a quiet tone: onsets to find and a level to draw."""
    audio = np.zeros((rate * seconds, 2), dtype=np.float32)
    rng = np.random.default_rng(3)
    for start in range(rate // 2, len(audio), rate // 2):
        burst = rng.standard_normal((2400, 1)) * np.exp(-np.arange(2400) / 300)[:, None]
        audio[start : start + 2400] += 0.5 * burst.astype(np.float32)
    tone = 0.1 * np.sin(2 * np.pi * 196 * np.arange(len(audio)) / rate)
    sf.write(path, audio + tone[:, None].astype(np.float32), rate, **how)
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--remember", action="store_true", help="use the settings store")
    parser.add_argument("--sessions", default="", help="folder for sessions (default: the application's)")
    parser.add_argument("--no-live", action="store_true", help="do not start live capture")
    args = parser.parse_args()

    from PySide6.QtCore import QSettings, qVersion

    from gmv import host, paths
    from gmv.app.window import MainWindow, application

    app = application()
    result = {
        "system": host.system(),
        "qt": qVersion(),
        "qt_platform": app.platformName(),
        "qt_platform_reason": app.property("gmv_qt_platform"),
        "audio_sources": list(host.audio_sources()),
        "sessions_folder": str(paths.sessions_folder()),
        "model_folders": [str(folder) for folder in paths.model_folders()],
        "settings_store": QSettings().fileName(),
    }

    def pump(until, seconds):
        deadline = perf_counter() + seconds
        while perf_counter() < deadline:
            app.processEvents()
            if until():
                return True
            sleep(0.005)
        return False

    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        window = MainWindow(
            session_root=args.sessions or None, remember=args.remember, discover_sources=False
        )
        wanted = os.environ.get("GMV_SCREEN", "").lower()
        screen = next((s for s in app.screens() if wanted and wanted in s.name().lower()), None)
        if screen is not None:
            area = screen.geometry()
            window.setGeometry(
                area.x() + 40, area.y() + 40, min(1500, area.width() - 80), min(900, area.height() - 80)
            )
        window.show()
        drawer = window.drawer
        result["screen"] = window.screen().name()
        result["problem_on_opening"] = window.problem
        result["source_choices"] = [drawer.source.itemText(i) for i in range(drawer.source.count())]
        result["source_note"] = drawer.source_note.text()
        result["status_at_start"] = window.status.text()
        result["restored"] = {"mode": drawer.mode.currentData(), "expressive": drawer.expressive.isChecked()}

        result["files"], last = {}, None
        for name, how in (("clicks.wav", {"subtype": "PCM_16"}), ("clicks.flac", {"subtype": "PCM_24"})):
            drawer.source.setCurrentIndex(drawer.source.findData("file"))
            drawer.file.setText(str(clicks(folder / name, **how)))
            drawer.logging.setChecked(True)
            window.start()
            engine, entry = window.controller.engine, {"problem_at_start": window.problem}
            if engine is not None and window.controller.running:
                entry["finished"] = pump(engine.done.is_set, 30)
                metrics, last = engine.metrics(), window.controller.session_path
                window.view.tick()
                entry |= {
                    "frames_analysed": metrics.get("frames"),
                    "events": metrics.get("events"),
                    "engine_error": str(engine.error or ""),
                    "analysis_process": getattr(engine, "process_id", None),
                    "onsets_shown": sum(m.kind == "onset" for m in window.view.state.markers),
                    "frames_drawn": len(window.view.canvas.paint_ms),
                    "session": str(last),
                    "session_files": sorted(p.name for p in last.iterdir()) if last else [],
                }
                if last is not None and (last / "manifest.json").is_file():
                    manifest = json.loads((last / "manifest.json").read_text(encoding="utf-8"))
                    entry["session_complete"] = manifest.get("complete")
                    entry["raw_audio_saved"] = manifest.get("raw_audio_saved")
            result["files"][name] = entry

        if last is not None:
            live = [m.time for m in window.view.state.markers if m.kind == "onset"]
            (folder / "clicks.flac").unlink()  # replay must not need the audio
            window.replay(last)
            replaying = window.controller.replaying
            pump(lambda: replaying is None or replaying.finished, 30)
            window.view.tick()
            again = [m.time for m in window.view.state.markers if m.kind == "onset"]
            result["replay"] = {
                "problem": window.problem,
                "error": str(getattr(replaying, "error", "") or ""),
                "marked_replayed": bool(window.view.state.replayed),
                "onsets_live": len(live),
                "onsets_replayed": len(again),
                "same_times": bool(len(live) == len(again) and np.allclose(live, again)),
            }
            view = window.view  # Radiant, as the application draws it here
            view.set_mode("radial")
            view.set_radial_theme("radiant")
            pump(lambda: False, 2.0)
            result["radiant_renderer"] = view.renderer_report()
            view.set_radial_theme("classic")
            view.set_mode("bars")
            window.stop()

        window.show_settings(True)
        pump(lambda: False, 0.6)
        result["diagnostics"] = drawer.diagnostics.text().splitlines()

        if host.live_capture() and not args.no_live:
            drawer.source.setCurrentIndex(drawer.source.findData("system"))
            drawer.logging.setChecked(False)
            window.start()
            states = []

            def watch():
                engine = window.controller.engine
                state = f"{engine.state}: {engine.detail}" if engine else "no engine"
                if not states or states[-1] != state:
                    states.append(state)
                return False

            pump(watch, 6)
            engine = window.controller.engine
            result["live"] = {
                "problem_at_start": window.problem,
                "states": states[-6:],
                "frames_analysed": engine.metrics().get("frames") if engine else None,
                "backend": engine.source.describe().get("backend") if engine else None,
                "engine_error": str(engine.error or "") if engine else "",
            }
            window.stop()
            pump(lambda: not window.controller.running, 10)
        else:
            result["live"] = host.capture_note() or "not started (--no-live)"

        if args.remember:  # something to find again on the next run
            drawer.mode.setCurrentIndex(drawer.mode.findData("radial"))
            drawer.expressive.setChecked(True)
        began = perf_counter()
        window.close()
        result["closed_in_s"] = round(perf_counter() - began, 2)
        result["threads_left"] = [
            t.name for t in threading.enumerate() if t.is_alive() and not t.daemon and t is not threading.main_thread()
        ]
        store = Path(QSettings().fileName())
        result["settings_file_exists"] = store.is_file() if store.is_absolute() else None
    Path(args.report).write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
