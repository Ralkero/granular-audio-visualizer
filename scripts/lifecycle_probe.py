"""Repeated read-only real capture, explicitly invoked; no target process control."""

import argparse
import json
from pathlib import Path
from time import sleep
from threading import enumerate as threads
import psutil
from gmv.engine import Engine
from gmv.sources.application import ApplicationAudioSource
from gmv.sources.windows import SystemAudioSource


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int)
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--output", default="local-evidence/phase1_1/lifecycle.json")
    args = parser.parse_args()
    if not 1 <= args.cycles <= 20 or not 0.2 <= args.seconds <= 30:
        parser.error("Use 1–20 cycles and 0.2–30 seconds per cycle")
    original_threads = {t.ident for t in threads()}
    original_children = {p.pid for p in psutil.Process().children(recursive=True)}
    rows = []
    for _ in range(args.cycles):
        source = ApplicationAudioSource(args.pid) if args.pid else SystemAudioSource()
        engine = Engine(source)
        engine.start()
        sleep(args.seconds)
        engine.stop()
        done = engine.done.wait(15)
        rows.append(
            {
                "finished": done,
                "error": engine.error,
                "frames": engine.frame_count,
                "drops": engine.audio.dropped,
                "remaining_capture_threads": sum(t.is_alive() for t in engine.threads),
            }
        )
    result = {
        "cycles": rows,
        "new_live_threads": [t.name for t in threads() if t.ident not in original_threads],
        "new_live_children": len(
            {p.pid for p in psutil.Process().children(recursive=True)} - original_children
        ),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
