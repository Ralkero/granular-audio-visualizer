"""Capture what this Linux system plays for a while and measure it.

    uv run python scripts/capture_probe_linux.py --seconds 30
    uv run python scripts/capture_probe_linux.py --seconds 600 --every 30 --report out.json

The live-capture part of ``scripts/validate_linux.py --audio-capture``, and on its own the
tool for a longer run. It records the default output's monitor through the PulseAudio service
(PipeWire's ``pipewire-pulse``) with the application's own source and engine, in this
process; logs the session into a scratch folder; replays that session; and removes the
folder. It plays nothing and changes no volume, device or routing; no audio is saved. What
is playing decides what is measured: with nothing playing, the frames are silent.

Progress is printed every ``--every`` seconds. The report (JSON, to standard output or
``--report``) names the server and the output by their descriptions, not by their device
names, which carry serial numbers. Exit status: 0 when every check passed, 1 when one
failed, 2 when the machine cannot capture (no server, no default output, no monitor,
missing client programs): an environment problem, not the application's.
"""

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import psutil

from gmv.engine import Engine
from gmv.models import AnalysisConfig, FeatureFrame, Status
from gmv.sources import system_pulse
from gmv.visualizer.replay import ReplaySession


def version(program):
    found = shutil.which(program)
    if not found:
        return None
    try:
        done = subprocess.run([found, "--version"], capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (done.stdout or done.stderr).strip().splitlines()[0] if done.returncode == 0 else None


def environment():
    """What the machine has, read-only. ``problem`` says why capture cannot run."""
    found = {"pactl": version("pactl"), "parec": version("parec"), "problem": ""}
    try:
        output = system_pulse.check()
        info = system_pulse.server_info()
        found |= {
            "server": output.server,
            "server_default_format": info.get("default_sample_specification"),
            "output": output.description,
            "monitor_found": True,
            "monitor_rate": output.rate,
        }
    except system_pulse.PulseProblem as problem:
        found["problem"] = str(problem)
    return found


def percentiles(values):
    if not values:
        return None
    values = np.asarray(values, dtype=float)
    return {
        "p50": round(float(np.percentile(values, 50)), 2),
        "p95": round(float(np.percentile(values, 95)), 2),
        "max": round(float(values.max()), 2),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=20)
    parser.add_argument("--every", type=float, default=5, help="seconds between progress lines")
    parser.add_argument("--report", default="", help="write the JSON report here")
    args = parser.parse_args()

    report = {"environment": environment(), "checks": []}
    if report["environment"]["problem"]:
        print("Cannot capture on this machine: " + report["environment"]["problem"], flush=True)
        write(report, args.report)
        return 2
    env = report["environment"]
    print(f"Capturing {env['output']} through {env['server']} for {args.seconds:g} s", flush=True)

    with tempfile.TemporaryDirectory(prefix="gmv-capture-probe-") as scratch:
        source = system_pulse.PulseSystemAudioSource()
        engine = Engine(source, AnalysisConfig(), session_root=Path(scratch) / "sessions")
        received = engine.bus.subscribe(1 << 16)
        frames, states, progress = [], [], []
        began = perf_counter()
        engine.start()
        next_line = began + args.every
        try:
            while perf_counter() - began < args.seconds and not engine.done.is_set():
                message = received.get(timeout=0.1)
                while message is not None:
                    if isinstance(message, FeatureFrame):
                        frames.append(
                            (
                                message.timestamp,
                                message.segment,
                                message.rms,
                                message.peak,
                                message.silence,
                                message.capture_to_publish_ms,
                                message.queue_latency_ms,
                                perf_counter(),
                            )
                        )
                    elif isinstance(message, Status):
                        states.append((round(perf_counter() - began, 2), message.state, message.detail))
                    message = received.get(timeout=0)
                if perf_counter() >= next_line:
                    next_line += args.every
                    progress.append(line(began, frames, engine, source))
                    print(progress_text(progress[-1]), flush=True)
        except KeyboardInterrupt:
            print("Interrupted; stopping", flush=True)
        stopping = perf_counter()
        engine.stop()
        stopped = engine.done.wait(15)
        stop_seconds = perf_counter() - stopping
        leftovers = [p.name() for p in psutil.Process().children(recursive=True)]
        threads = [t.name for t in threading.enumerate() if t.name.startswith("pulse-capture")]
        metrics = engine.metrics()
        clock = source.describe()["clock"] or {}
        folder = engine.writer.path if engine.writer else None
        replayed = replay(folder) if folder else None
        manifest = json.loads((folder / "manifest.json").read_text()) if folder else {}
        audio_files = [p.name for p in folder.rglob("*") if p.suffix.lower() in (".wav", ".flac")]

    elapsed = (frames[-1][7] - frames[0][7]) if len(frames) > 1 else 0.0
    audio_span = (frames[-1][0] - frames[0][0]) if len(frames) > 1 else 0.0
    timestamps = [f[0] for f in frames]
    rms = np.array([f[2] for f in frames]) if frames else np.zeros(0)
    report |= {
        "seconds": round(perf_counter() - began, 1),
        "frames_analysed": len(frames),
        "frames_expected": int(args.seconds / (AnalysisConfig().hop_ms / 1000)),
        "segments": sorted({f[1] for f in frames}),
        "silent_frames": int(sum(f[4] for f in frames)),
        "level": {
            "rms_dbfs_median": round(float(20 * np.log10(np.median(rms) + 1e-9)), 1) if len(rms) else None,
            "peak_max": round(float(max((f[3] for f in frames), default=0.0)), 4),
        },
        "audio_time_per_wall_second": round(audio_span / elapsed, 6) if elapsed else None,
        "capture_to_publish_ms": percentiles([f[5] for f in frames if f[5] is not None]),
        "queue_latency_ms": percentiles([f[6] for f in frames]),
        "dsp_ms": metrics.get("dsp_ms"),
        "dropped_blocks": metrics.get("dropped_blocks"),
        "discontinuities": metrics.get("discontinuities"),
        "queue_high_water": metrics.get("queue_high_water"),
        "clock": clock,
        "recoveries": source.recoveries,
        "output_switches": source.output_switches,
        "streams_opened": source.streams_opened,
        "process": {
            "cpu_percent_of_one_core": round(metrics.get("process_cpu_percent", 0.0), 1),
            "rss_mb": round(metrics.get("rss_mb", 0.0), 1),
            "peak_rss_mb": round(metrics.get("peak_sampled_rss_mb", 0.0), 1),
            # this process and parec, sampled once a second by the engine
            "with_parec": {
                key: (round(value, 1) if isinstance(value, float) else value)
                for key, value in (metrics.get("owned_process_resources") or {}).items()
                if key in ("cpu_one_core_percent", "rss_mb", "process_count")
            },
        },
        "states": states[-20:],
        "progress": progress,
        "stopped_in_s": round(stop_seconds, 2),
        "left_running": leftovers + threads,
        "session": {
            "complete": manifest.get("complete"),
            "raw_audio_saved": manifest.get("raw_audio_saved"),
            "audio_files": audio_files,
            "replayed_frames": replayed,
        },
    }
    checks = report["checks"]

    def check(name, passed, detail=""):
        checks.append({"check": name, "passed": bool(passed), "detail": str(detail)})
        print(f"  {'ok  ' if passed else 'FAIL'}  {name}" + (f": {detail}" if detail else ""), flush=True)

    check("capture started and frames arrived", len(frames) > 0.8 * report["frames_expected"],
          f"{len(frames)} of about {report['frames_expected']} frames")
    check("audio time advanced with the clock", report["audio_time_per_wall_second"] is not None
          and abs(report["audio_time_per_wall_second"] - 1) < 0.01, report["audio_time_per_wall_second"])
    check("timestamps never went back", all(b >= a for a, b in zip(timestamps, timestamps[1:], strict=False)))
    check("no analysis blocks dropped", report["dropped_blocks"] == 0, report["dropped_blocks"])
    check("stopped cleanly", stopped and not report["left_running"],
          f"in {report['stopped_in_s']} s; left: {report['left_running']}")
    check("session logged without audio", report["session"]["complete"]
          and report["session"]["raw_audio_saved"] is False and not audio_files)
    check("session replays without the live source", replayed == len(frames) and replayed > 0,
          f"{replayed} of {len(frames)} frames")
    report["passed"] = all(c["passed"] for c in checks)
    write(report, args.report)
    return 0 if report["passed"] else 1


def line(began, frames, engine, source):
    now = perf_counter()
    recent = [f for f in frames if f[7] > now - 5]
    rms = np.median([f[2] for f in recent]) if recent else 0.0
    clock = source.describe()["clock"] or {}
    return {
        "t": round(now - began, 1),
        "frames": len(frames),
        "rms_dbfs": round(float(20 * np.log10(rms + 1e-9)), 1),
        "dropped_blocks": engine.audio.dropped,
        "discontinuities": engine.discontinuities,
        "lateness_floor_ms": None if clock.get("lateness_floor_ms") is None else round(clock["lateness_floor_ms"], 2),
        "resyncs": (clock.get("behind_resyncs") or 0) + (clock.get("ahead_resyncs") or 0),
        "recoveries": source.recoveries,
        "cpu": round(engine.cpu, 1),
        "rss_mb": round(engine.memory_mb, 1),
    }


def progress_text(entry):
    return (
        f"  {entry['t']:7.1f} s  frames {entry['frames']:6d}  level {entry['rms_dbfs']:6.1f} dBFS  "
        f"dropped {entry['dropped_blocks']}  discontinuities {entry['discontinuities']}  "
        f"clock floor {entry['lateness_floor_ms']} ms  resyncs {entry['resyncs']}  "
        f"recoveries {entry['recoveries']}  CPU {entry['cpu']}%  {entry['rss_mb']} MiB"
    )


def replay(folder):
    session = ReplaySession(folder, speed=1000)
    received = session.bus.subscribe(1 << 20)
    session.start()
    session.thread.join(120)
    if not session.finished or session.error:
        return None
    return sum(isinstance(m, FeatureFrame) and m.replayed for m in (received.get() for _ in range(len(received))))


def write(report, path):
    text = json.dumps(report, indent=1, default=str)
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(text, encoding="utf-8")
        print(f"Report: {path}", flush=True)
    else:
        print(text)


if __name__ == "__main__":
    sys.exit(main())
