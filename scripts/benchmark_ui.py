"""Measure the visualizer's own cost by replaying a recorded session through the real
window, on screen, at playing speed.

    python scripts/benchmark_ui.py --session PATH --mode radial --size 1440x900
    python scripts/benchmark_ui.py --session PATH --mode bars --sections timeline,details,notes

A session's ``messages.jsonl`` holds analysis messages only (never audio), so the same
workload can be measured before and after a change: message consumption, every part of the
paint, frame pacing. No capture and no analysis run here; for those see ``benchmark.py``
and ``benchmark_visualizer.py``. Prints one JSON object. The window opens on screen for the
duration of the run.
"""

import argparse
import json
import sys
import time

import psutil
from PySide6.QtCore import Qt, QTimer

from gmv.app.window import application
from gmv.visualizer.replay import ReplaySession
from gmv.visualizer.window import VisualizerWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--size", default="1440x900")
    parser.add_argument("--mode", default="radial", choices=("bars", "radial"))
    parser.add_argument("--seconds", type=float, default=36)
    parser.add_argument(
        "--sections", default="timeline,details,notes", help="comma list; '' for none"
    )
    parser.add_argument("--layers", default="vocals,guitar,bass,drums")
    parser.add_argument("--label", default="")
    parser.add_argument("--screen", default="", help="part of the name of the display to open on")
    parser.add_argument("--fps", type=float, default=0, help="0: the window's own choice")
    parser.add_argument(
        "--rings", default="live", choices=("live", "history"), help="radial instrument rings"
    )
    parser.add_argument(
        "--plain-spectrum", action="store_true", help="do not colour the spectrum by intensity"
    )
    args = parser.parse_args()
    app = application()
    session = ReplaySession(args.session)
    width, height = map(int, args.size.split("x"))
    window = VisualizerWindow()
    window.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    place(app, window, args.screen, width, height)
    window.set_mode(args.mode)
    window.set_intensity_colour(not args.plain_spectrum)
    window.set_radial_rings(args.rings)
    if args.fps and hasattr(window, "set_frame_rate"):
        window.set_frame_rate(args.fps)
    shown = {name for name in args.sections.split(",") if name}
    layers = {name for name in args.layers.split(",") if name}
    configure(window, shown, layers)
    window.attach(
        session.bus,
        source="Benchmark replay",
        activity_model=session.configuration.get("activity_model_path"),
        recognition_model=session.configuration.get("recognition_model_path"),
        instruments=session.configuration.get("instrument_analyzers") or (),
    )
    window.show()
    session.start()
    process = psutil.Process()
    process.cpu_percent()
    began = time.perf_counter()
    warm = {"done": False}

    def settle():  # leave the first seconds (window creation, caches) out of the figures
        window.canvas.perf.clear()
        process.cpu_percent()
        warm["done"], warm["at"], warm["painted"] = True, time.perf_counter(), window.canvas.painted

    def finish():
        elapsed = time.perf_counter() - warm.get("at", began)
        perf = window.canvas.perf.stats()
        intervals = perf.get("frame interval") or {}
        result = {
            "label": args.label,
            "window": f"{window.width()}x{window.height()}",
            "device_pixel_ratio": window.devicePixelRatioF(),
            "mode": args.mode,
            "sections": sorted(shown),
            "layers": sorted(layers),
            "measured_seconds": round(elapsed, 1),
            "target_fps": round(1 / window.frame_s, 2) if hasattr(window, "frame_s") else 60.0,
            "screen": f"{window.screen().size().width()}x{window.screen().size().height()} at {window.screen().refreshRate():.0f} Hz",
            "frames_over": {
                limit: int(
                    sum(
                        v > float(limit) for v in window.canvas.perf.series["frame interval"].values
                    )
                )
                for limit in ("16.67", "20", "25", "33.33")
            }
            if intervals
            else None,
            "frames_per_second": round(
                (window.canvas.painted - warm.get("painted", 0)) / elapsed, 1
            ),
            "frames_over_25ms": int(
                sum(v > 25 for v in window.canvas.perf.series["frame interval"].values)
            )
            if intervals
            else None,
            "timings_ms": {
                k: v
                for k, v in perf.items()
                if k != "counts" and k not in ("backlog", "messages per tick")
            },
            "backlog_messages": perf.get("backlog"),
            "messages_per_tick": perf.get("messages per tick"),
            "counts": perf["counts"] | strips(window),
            "messages_dropped_by_window": window.subscription.dropped
            if window.subscription
            else None,
            "process_cpu_percent_of_one_core": round(process.cpu_percent()),
            "process_memory_mib": round(process.memory_info().rss / 2**20),
            "replay_error": session.error,
        }
        window.close()
        session.stop()
        print(json.dumps(result, indent=1))
        app.quit()

    QTimer.singleShot(4000, settle)
    QTimer.singleShot(int(args.seconds * 1000), finish)
    return app.exec()


def place(app, window, name, width, height):
    """Put the window on the display whose name contains ``name`` (the primary one when
    empty), no larger than that display."""
    screen = next((s for s in app.screens() if name and name.lower() in s.name().lower()), None)
    if name and screen is None:
        raise SystemExit(f"No display named like {name!r}: {[s.name() for s in app.screens()]}")
    area = (screen or app.primaryScreen()).geometry()
    window.setGeometry(area.x(), area.y(), min(width, area.width()), min(height, area.height()))


def configure(window, shown, layers):
    """Sections and layers, through whichever interface this version of the window has."""
    canvas = window.canvas
    for key in canvas.layers:
        window.set_layer(key, enabled=key in layers)
    if hasattr(window, "set_section"):
        for name in ("timeline", "details", "notes"):
            window.set_section(name, shown=name in shown)
        return
    # Before sections existed: rows and lanes were hidden one by one.
    from gmv.visualizer.lanes import KNOWN
    from gmv.visualizer.state import ROWS

    for key, _title in ROWS:
        window.set_shown(key, "timeline" in shown)
    for lane in KNOWN:
        window.set_shown(lane.key, "details" in shown)
    window.set_shown("notes", "notes" in shown)


def strips(window):
    canvas = window.canvas
    found = {}
    for name in ("lane_strip", "level_strip", "mark_strip", "note_strip"):
        strip = getattr(canvas, name, None)
        if strip is not None:
            found[f"{name} shifts"], found[f"{name} redraws"] = strip.shifts, strip.redraws
    for key, layer in canvas.layers.items():
        for attribute in ("strip", "sweep"):
            strip = getattr(layer, attribute, None)
            if strip is not None and hasattr(strip, "shifts"):
                found[f"layer {key} {attribute} shifts"] = strip.shifts
                found[f"layer {key} {attribute} redraws"] = strip.redraws
    return found


if __name__ == "__main__":
    sys.exit(main())
