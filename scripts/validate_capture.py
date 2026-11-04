"""Read-only capture soak and manual observation probe. Never controls playback."""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from time import perf_counter
from threading import enumerate as threads
from gmv.diagnostics import ResourceMonitor, RunningStats
from gmv.engine import Engine
from gmv.models import AnalysisConfig
from gmv.sources.windows import SystemAudioSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--pid", type=int)
    parser.add_argument("--device")
    parser.add_argument("--pair-system", action="store_true")
    parser.add_argument("--advanced", action="store_true")
    parser.add_argument("--output", default="local-evidence/phase1_1/capture.json")
    parser.add_argument("--mark", help="Append a user-observed stage marker instead of capturing")
    args = parser.parse_args()
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    if args.mark:
        with path.with_suffix(".markers.jsonl").open("a", encoding="utf-8") as file:
            file.write(
                json.dumps({"monotonic": perf_counter(), "user_observation": args.mark}) + "\n"
            )
        return
    if not 1 <= args.seconds <= 86400:
        parser.error("Duration must be 1–86400 seconds")
    if args.pid:
        from gmv.sources.application import ApplicationAudioSource

        source = ApplicationAudioSource(args.pid)
    else:
        source = SystemAudioSource(args.device)
    sources = [source] + ([SystemAudioSource()] if args.pid and args.pair_system else [])
    engines = [Engine(s, AnalysisConfig(advanced=args.advanced)) for s in sources]
    queues = [e.bus.subscribe(512) for e in engines]
    stats = [
        {k: RunningStats() for k in ("signed_center_age_ms", "dsp_ms", "peak", "rms")}
        for _ in engines
    ]
    monitor = ResourceMonitor()
    initial_threads = {t.ident for t in threads()}
    started = perf_counter()
    resource_stats = {k: RunningStats() for k in ("rss_mb", "cpu_one_core_percent")}
    metadata = {
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "ui": False,
        "logging": True,
        "advanced": args.advanced,
        "raw_audio_saved": False,
    }
    for engine in engines:
        engine.start()
    next_sample = started
    try:
        with path.with_suffix(".samples.jsonl").open("w", encoding="utf-8") as samples:
            while perf_counter() - started < args.seconds and not all(
                e.done.is_set() for e in engines
            ):
                for engine, queue, metrics in zip(engines, queues, stats, strict=True):
                    for _ in range(128):
                        message = queue.get(timeout=0.005)
                        if message is None:
                            break
                        if message.type == "feature_frame":
                            metrics["signed_center_age_ms"].add(
                                (message.published_monotonic - engine.started - message.timestamp)
                                * 1000
                            )
                            for key in ("dsp_ms", "peak", "rms"):
                                metrics[key].add(getattr(message, key))
                now = perf_counter()
                if now >= next_sample:
                    resource = monitor.sample()
                    for key in resource_stats:
                        resource_stats[key].add(resource[key])
                    snapshot = {
                        "elapsed": now - started,
                        "monotonic": now,
                        "resources": resource,
                        "sources": [
                            {
                                "kind": e.source.kind,
                                "state": e.state,
                                "frames": e.frame_count,
                                "queue_blocks": len(e.audio),
                                "peak": e.latest.peak if e.latest else None,
                                "rms": e.latest.rms if e.latest else None,
                                "segment": e.latest.segment if e.latest else None,
                                "audio_timestamp": e.latest.timestamp if e.latest else None,
                                "clock": e.latest.timestamp_quality if e.latest else None,
                            }
                            for e in engines
                        ],
                    }
                    samples.write(json.dumps(snapshot) + "\n")
                    samples.flush()
                    path.with_suffix(".latest.json").write_text(
                        json.dumps(snapshot, indent=2), encoding="utf-8"
                    )
                    next_sample = now + 1
    finally:
        for engine in engines:
            engine.stop()
        for engine in engines:
            engine.done.wait(20)
    result = {
        **metadata,
        "elapsed_seconds": perf_counter() - started,
        "resources": {k: v.report() for k, v in resource_stats.items()},
        "sources": [
            {
                "source": e.source.describe(),
                "configuration": e.config.model_dump(),
                "metrics": e.metrics(),
                "whole_session": {k: v.report() for k, v in m.items()},
                "probe_subscriber_drops": q.dropped,
                "logger_drops": e.writer.queue.dropped,
                "clean_shutdown": e.done.is_set() and not any(t.is_alive() for t in e.threads),
            }
            for e, q, m in zip(engines, queues, stats, strict=True)
        ],
        "remaining_owned_child_processes": monitor.sample()["process_count"] - 1,
        "remaining_new_threads": [t.name for t in threads() if t.ident not in initial_threads],
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(path),
                "elapsed_seconds": result["elapsed_seconds"],
                "frames": [e.frame_count for e in engines],
                "errors": [e.error for e in engines],
            }
        )
    )


if __name__ == "__main__":
    main()
