"""What the live Sonic Character costs in the running engine (Phase B).

Generated audio (no recording, no music file) is played through the real engine at real-time
pace, in the application's process (``GMV_ANALYSIS=thread`` style, the case where the live
analysis shares the interpreter with the analysis pass), alternately without and with
``LiveCharacter`` attached to the bus. For each run it reports the engine's own figures
(analysis time per frame, queue delay, dropped blocks, discontinuities), the process's
processor time and memory, and the live analysis's own counters. Then an extended run (by
default 20 minutes of audio, as fast as it goes) checks that memory stays flat, and the
monitor window is refreshed and drawn offscreen to time a UI update.

Usage::

    python scripts/benchmark_character_live.py [--seconds 60] [--rounds 2] [--extended 1200]
        [--json OUT]
"""

import argparse
import json
import os
import platform
import resource
import statistics
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter, sleep

import numpy as np
import soundfile as sf

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from gmv.character.live import LiveCharacter  # noqa: E402
from gmv.engine import Engine  # noqa: E402
from gmv.sources.file import FileAudioSource  # noqa: E402

RATE = 48000


def music(seconds, seed=1):
    """Noise bursts over a bass tone with a slow swell and a silent gap every 40 s."""
    rng = np.random.default_rng(seed)
    time = np.arange(round(seconds * RATE)) / RATE
    burst = (np.modf(time * 4.0)[0] < 0.2) * rng.standard_normal(len(time))
    tone = np.sin(2 * np.pi * 82.4 * time) + 0.4 * np.sin(2 * np.pi * 660 * time)
    swell = 0.3 + 0.7 * np.sin(2 * np.pi * time / 23.0) ** 2
    gap = (np.mod(time, 40.0) <= 37.0).astype(np.float64)  # three silent seconds in every forty
    audio = 0.25 * swell * gap * (0.5 * burst + tone)
    return np.column_stack((audio, audio)).astype(np.float32)


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def rss_mb():
    try:
        import psutil

        return psutil.Process().memory_info().rss / 2**20
    except ImportError:  # pragma: no cover
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def run(path, realtime, with_live, folder):
    engine = Engine(FileAudioSource(path, realtime=realtime), session_root=folder, logging=False)
    live = LiveCharacter() if with_live else None
    if live is not None:
        live.attach(engine.bus)
    samples, stop = [], threading.Event()

    def watch():
        while not stop.wait(1.0):
            samples.append(rss_mb())

    watcher = threading.Thread(target=watch, daemon=True)
    cpu0, wall0 = cpu_seconds(), perf_counter()
    watcher.start()
    engine.start()
    engine.done.wait()
    if live is not None:
        deadline = perf_counter() + 10
        while live.backlog and perf_counter() < deadline:
            sleep(0.01)
    wall, cpu = perf_counter() - wall0, cpu_seconds() - cpu0
    stop.set()
    metrics = engine.metrics()
    out = {
        "with_live": with_live,
        "realtime": realtime,
        "wall_s": round(wall, 2),
        "cpu_s": round(cpu, 2),
        "cpu_percent_of_one_core": round(100 * cpu / wall, 1),
        "frames": metrics["frames"],
        "dsp_ms_p50": round(metrics["dsp_ms"]["p50"], 3),
        "dsp_ms_p95": round(metrics["dsp_ms"]["p95"], 3),
        "dsp_ms_max": round(metrics["dsp_ms"]["max"], 3),
        "queue_ms_p95": round(metrics["queue_ms"]["p95"], 3) if metrics["queue_ms"] else None,
        "capture_to_publish_ms_p95": round(metrics["estimated_capture_center_to_publish_ms"]["p95"], 3)
        if metrics["estimated_capture_center_to_publish_ms"]
        else None,
        "dropped_blocks": metrics["dropped_blocks"],
        "discontinuities": metrics["discontinuities"],
        "rss_mb_start": round(samples[0], 1) if samples else None,
        "rss_mb_end": round(samples[-1], 1) if samples else None,
        "rss_mb_peak": round(max(samples), 1) if samples else None,
        "error": metrics["error"],
    }
    if live is not None:
        stats = live.stats
        out.update(
            live_frames=stats.frames,
            live_messages=stats.messages,
            live_update_us_per_frame=round(stats.update_s / max(1, stats.frames) * 1e6, 1),
            live_worst_update_us=round(stats.worst_update_s * 1e6, 1),
            live_snapshot_ms=round(stats.profile_s / max(1, stats.snapshots) * 1e3, 3),
            live_snapshots_per_audio_s=round(stats.snapshots / max(1e-9, metrics["frames"] * 0.02), 2),
            live_backlog_high=stats.backlog_high,
            live_dropped=live.dropped,
            live_errors=stats.errors,
            live_history_points=len(live.history()),
        )
        live.detach()
    return out


def monitor_cost(path, folder):
    from PySide6.QtGui import QImage
    from PySide6.QtWidgets import QApplication

    from gmv.app.character_monitor import CharacterMonitor

    app = QApplication.instance() or QApplication([])
    engine = Engine(FileAudioSource(path, realtime=False), session_root=folder, logging=False)
    live = LiveCharacter(capacity=1 << 17)
    live.attach(engine.bus)
    engine.start()
    engine.done.wait()
    while live.backlog:
        sleep(0.01)
    monitor = CharacterMonitor(live)
    monitor.resize(640, 640)
    monitor.show()
    app.processEvents()
    image = QImage(monitor.size(), QImage.Format.Format_ARGB32)
    times = []
    for window in ("short", "medium", "long") * 20:
        monitor.window_choice.setCurrentIndex(monitor.window_choice.findData(window))
        start = perf_counter()
        monitor.force_refresh()
        monitor.render(image)
        times.append((perf_counter() - start) * 1e3)
    unchanged = []
    for _ in range(60):
        start = perf_counter()
        monitor.refresh()
        unchanged.append((perf_counter() - start) * 1e3)
    monitor.close()
    live.detach()
    return {
        "history_points": len(live.history()),
        "refresh_and_draw_ms_mean": round(statistics.mean(times), 2),
        "refresh_and_draw_ms_p95": round(sorted(times)[int(len(times) * 0.95)], 2),
        "refresh_without_new_data_ms_mean": round(statistics.mean(unchanged), 3),
        "refresh_rate_hz": 4,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--extended", type=float, default=1200.0)
    parser.add_argument("--json")
    args = parser.parse_args(argv)
    report = {
        "machine": f"{platform.system()} {platform.release()} · {platform.processor() or platform.machine()} · "
        f"Python {platform.python_version()} · {os.cpu_count()} logical processors",
        "audio": "generated (noise bursts, bass tone, swell, silent gap every 40 s), 48 kHz stereo",
        "runs": [],
    }
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        short = folder / "music.wav"
        sf.write(short, music(args.seconds), RATE, subtype="FLOAT")
        for round_ in range(args.rounds):
            for with_live in (False, True):
                result = run(short, True, with_live, folder / "sessions")
                result["round"] = round_ + 1
                report["runs"].append(result)
                print(json.dumps(result), flush=True)
        long = folder / "long.wav"
        sf.write(long, music(args.extended, seed=2), RATE, subtype="FLOAT")
        extended = run(long, False, True, folder / "sessions")
        extended["audio_s"] = args.extended
        report["extended"] = extended
        print("extended", json.dumps(extended), flush=True)
        report["monitor"] = monitor_cost(short, folder / "sessions")
        print("monitor", json.dumps(report["monitor"]), flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
