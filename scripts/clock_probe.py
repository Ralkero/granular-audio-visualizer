"""Read-only system-loopback clock probe. Never controls playback; saves no audio.

Taps the blocks a production system source emits and stores, per block, arrival time, frame
count, session timestamp, segment and discontinuity flag. Two analyses are reported:

* ``arrival``: delivered sample time against arrival time, separating lost-audio steps from
  a smooth rate mismatch (meaningful for the PortAudio callback backend, where timestamps
  are counted from delivered samples);
* ``timeline``: session timestamp against arrival time, i.e. whether published timestamps
  stay aligned with the wall clock (the property live synchronization needs).

``--engine`` adds the full analysis engine as load; ``--pair-app`` adds a second engine
capturing one application, the condition under which the PortAudio backend lost audio.
"""

import argparse
import json
import subprocess
from pathlib import Path
from threading import Event, Thread
from time import perf_counter

import numpy as np

from gmv.buffer import BoundedQueue
from gmv.clock_analysis import analyze_clock, analyze_timeline
from gmv.sources.system_native import system_audio_source


class TapQueue(BoundedQueue):
    def __init__(self, capacity=16):
        super().__init__(capacity)
        self.rows = []

    def put(self, item, **kwargs):
        self.rows.append(
            (
                item.received_monotonic,
                item.frames,
                item.sample_rate,
                item.timestamp,
                item.segment,
                item.discontinuity,
            )
        )
        return super().put(item, **kwargs)


def drain(queue, stop):
    while not stop.is_set() or len(queue):
        queue.get()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument("--device")
    parser.add_argument("--backend", choices=("native", "portaudio"), default="native")
    parser.add_argument("--output", default="local-evidence/clock_probe/probe.json")
    parser.add_argument("--engine", action="store_true", help="Capture under full engine load")
    parser.add_argument("--advanced", action="store_true", help="With --engine: delayed HPSS on")
    parser.add_argument(
        "--pair-app", help="With --engine: also capture this process name in a second engine"
    )
    args = parser.parse_args()
    if not 5 <= args.seconds <= 86400:
        parser.error("Duration must be 5–86400 seconds")
    if args.pair_app and not args.engine:
        parser.error("--pair-app requires --engine")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)

    source, stop, statuses = system_audio_source(args.device, args.backend), Event(), []
    started = perf_counter()
    engine = paired = None
    if args.engine:
        from gmv.engine import Engine
        from gmv.models import AnalysisConfig

        engine = Engine(source, AnalysisConfig(advanced=args.advanced))
        tap = engine.audio = TapQueue(engine.config.queue_blocks)
        if args.pair_app:
            from gmv.sources.application import ApplicationAudioSource, list_audio_applications

            apps = [
                a
                for a in list_audio_applications()
                if a["name"].casefold() == args.pair_app.casefold()
            ]
            if len(apps) != 1:
                raise SystemExit(f"Expected one audio application named {args.pair_app}: {apps}")
            paired = Engine(
                ApplicationAudioSource(apps[0]["pid"]), AnalysisConfig(advanced=args.advanced)
            )
            paired.start()
        engine.start()
        origin = engine.started
        threads = [engine.threads[0]]
    else:
        tap = TapQueue()
        origin = source.timeline_origin = started
        threads = [
            Thread(
                target=source.run,
                args=(tap, stop, lambda *s: statuses.append((perf_counter() - started, *s))),
                daemon=True,
            ),
            Thread(target=drain, args=(tap, stop), daemon=True),
        ]
        for thread in threads:
            thread.start()
    try:
        while perf_counter() - started < args.seconds and threads[0].is_alive():
            stop.wait(10)
            print(f"{perf_counter() - started:7.1f}s blocks={len(tap.rows)}", flush=True)
    except KeyboardInterrupt:
        pass
    finally:
        stop.set()
        for running in (engine, paired):
            if running:
                running.stop()
                running.done.wait(20)
        for thread in threads:
            thread.join(10)

    if len(tap.rows) < 3:
        raise SystemExit(f"Too few blocks ({len(tap.rows)}); statuses: {statuses}")
    rows = np.array(tap.rows, dtype=np.float64)
    np.savez_compressed(path.with_suffix(".npz"), rows=rows, origin=origin)
    result = {
        "starting_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "raw_audio_saved": False,
        "backend": args.backend,
        "source": source.describe(),
        "statuses": statuses,
        "arrival": analyze_clock(rows[:, 0], rows[:, 1], rows[:, 2]),
        "timeline": analyze_timeline(rows[:, 0] - origin, rows[:, 3]),
        "segments": int(rows[-1, 4]),
        "discontinuity_blocks": int(rows[:, 5].sum()),
        "probe_queue_drops": tap.dropped,
        "engine": {"advanced": args.advanced, "metrics": engine.metrics()} if engine else None,
        "paired_application_metrics": paired.metrics() if paired else None,
    }
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key in ("arrival", "timeline"):
        result[key] = {k: v for k, v in result[key].items() if k != "envelope"}
    print(json.dumps({**result, "output": str(path)}, indent=2))


if __name__ == "__main__":
    main()
