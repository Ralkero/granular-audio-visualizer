"""Measure the visualizer next to a running engine in a real on-screen window.

    python scripts/benchmark_visualizer.py --size 2560x1440 --file track.flac
    python scripts/benchmark_visualizer.py --size 1440x900 --live

Reports frames per second, drawing time, DSP time per analysis frame, drops, CPU and memory.
A file source wakes on the same Windows timer tick as the display, so its analysis frames
start together with a repaint far more often than with live capture; ``--live`` (default
system output, something must be playing) is the figure that matters for normal use.
The window opens on screen for the duration of the run. Nothing is recorded.
"""

import argparse
import json
import sys
import time

import numpy as np
import psutil
from PySide6.QtCore import Qt, QTimer

from gmv.app.window import application
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.file import FileAudioSource
from gmv.sources.system_native import system_audio_source
from gmv.visualizer.window import VisualizerWindow


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", default="1440x900")
    parser.add_argument("--file")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--seconds", type=float, default=12)
    parser.add_argument("--mode", default="bars", choices=("bars", "radial"))
    parser.add_argument("--activity-model")
    parser.add_argument("--guitar-model")
    parser.add_argument("--no-window", action="store_true", help="Engine only, for a baseline")
    args = parser.parse_args()
    if bool(args.file) == args.live:
        parser.error("give either --file PATH or --live")
    app = application()
    source = system_audio_source() if args.live else FileAudioSource(args.file, realtime=True)
    config = AnalysisConfig(
        experimental_beat_phase=True,
        activity_model_path=args.activity_model,
        guitar_model_path=args.guitar_model,
    )
    engine = Engine(source, config, logging=False)
    width, height = map(int, args.size.split("x"))
    window = None
    if not args.no_window:
        window = VisualizerWindow()
        window.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        window.setGeometry(0, 0, width, height)
        window.set_mode(args.mode)
        window.attach(engine.bus, source="Benchmark", beat_tracker=True)
        window.show()
    engine.start()
    process = psutil.Process()
    process.cpu_percent()
    began = time.perf_counter()

    def finish():
        elapsed = time.perf_counter() - began
        metrics = engine.metrics()
        result = {
            "source": "live system output" if args.live else "file",
            "seconds": round(elapsed, 1),
            "analysis_frames": metrics["frames"],
            "dsp_ms": {k: round(v, 2) for k, v in (metrics["dsp_ms"] or {}).items()},
            "capture_dropped_blocks": metrics["dropped_blocks"],
            "process_cpu_percent_of_one_core": round(process.cpu_percent()),
            "process_memory_mib": round(process.memory_info().rss / 2**20),
        }
        if window is not None:
            paint = np.array(window.canvas.paint_ms)
            result.update(
                window=f"{window.width()}x{window.height()}",
                mode=args.mode,
                frames_per_second=round(window.canvas.painted / elapsed, 1),
                paint_ms={
                    "p50": round(float(np.median(paint)), 2),
                    "p95": round(float(np.percentile(paint, 95)), 2),
                    "max": round(float(paint.max()), 2),
                    "scope": "last 240 frames",
                },
                messages_dropped_by_window=window.subscription.dropped,
            )
            window.close()
        engine.stop()
        engine.done.wait(10)
        print(json.dumps(result, indent=2))
        app.quit()

    QTimer.singleShot(int(args.seconds * 1000), finish)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
