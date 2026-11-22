"""Wall-clock side of instrument responsiveness: from capture to publication, and from
publication to the picture.

    python scripts/trace_responsiveness.py log SESSION_DIR [...]
    python scripts/trace_responsiveness.py replay SESSION_DIR --mode radial --seconds 40 \
        [--screen OMEN] [--size 1920x1080]

``log`` reads sessions the application recorded while capturing (nothing is replayed) and
reports, per instrument, what the engine itself stamped on every message:

    rate, interval       activity messages per second, spacing of their audio timestamps
    capture to publish   ``age_at_publication_ms``: wall time from the capture of the moment
                         a message describes to its publication. It contains the analyzer's
                         own back-dating (drums 60 ms, bass 80 ms), the analysis window, the
                         capture block and the processing. It does not contain the time the
                         sound takes to reach the loudspeakers, which is not known here.
    attribution          how much of the reported activity the presence judgement lets
                         through at the end of the session (1, 0.65, 0.4 tentative, 0)

``replay`` plays a recorded session into a real visualizer window and hooks, in this
process only, the two places a row passes on its way to the screen:

    publish to state     from publication on the bus to the visual state taking the row in
    publish to drawn     from publication to the paint call that put the row on the picture
    behind now           at each paint, how far behind the hand (the display's "now") the
                         newest drawn row is, in audio time
    rows / attacks per paint, rows refused by the state, messages the window dropped

The moment a paint call returns is not the moment light leaves the display: composition and
scan-out (up to a refresh or two) are not measured. The hooks are installed by this script;
the application has none of them.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def spread(values, scale=1.0, digits=1):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float) * scale
    if not len(values):
        return None
    return {
        "p50": round(float(np.percentile(values, 50)), digits),
        "p95": round(float(np.percentile(values, 95)), digits),
        "max": round(float(values.max()), digits),
        "n": int(len(values)),
    }


def read_log(folder):
    from gmv.models import FeatureFrame, InstrumentActivity, InstrumentEstimate
    from gmv.visualizer.state import COLUMN, VisualState

    from gmv.contract import parse_message

    state = VisualState(capacity=1 << 16)
    kept, frames, estimates = {}, [], []
    with open(Path(folder) / "messages.jsonl", encoding="utf-8") as file:
        for line in file:
            message = parse_message(json.loads(line))
            state.consume(message, arrival=message.published_monotonic)
            if isinstance(message, InstrumentActivity):
                kept.setdefault(message.instrument, []).append(
                    (
                        message.timestamp,
                        message.age_at_publication_ms,
                        (message.published_monotonic - message.detected_monotonic) * 1000,
                        message.attack,
                        message.published_monotonic,
                    )
                )
            elif isinstance(message, FeatureFrame):
                frames.append((message.timestamp, message.age_at_publication_ms, message.dsp_ms))
            elif isinstance(message, InstrumentEstimate) and message.available:
                estimates.append(
                    (message.interval_end, message.published_monotonic, message.processing_ms)
                )
    report = {"frames": {}, "instruments": {}}
    frames = np.asarray(frames, dtype=float)
    seconds = float(frames[-1, 0] - frames[0, 0]) if len(frames) > 1 else 0.0
    report["seconds"] = round(seconds, 1)
    report["frames"] = {
        "per_second": round((len(frames) - 1) / seconds, 2) if seconds else None,
        "interval_ms": spread(np.diff(frames[:, 0]), 1000),
        "capture_to_publish_ms": spread(frames[:, 1]),
        "analysis_ms": spread(frames[:, 2], digits=2),
    }
    for name, rows in kept.items():
        rows = np.asarray(rows, dtype=float)
        gaps = np.diff(rows[:, 0])
        track = state.activity[name].since()
        weights, tentative = state.attribution(name, track[:, 0])
        struck = track[:, COLUMN["attack"]] > 0
        lively = track[:, COLUMN["level"]] > 0.1

        def shares(mask, weights=weights):
            if not mask.any():
                return None
            return {
                "full": round(float(np.mean(weights[mask] == 1.0)), 3),
                "weakened": round(float(np.mean(weights[mask] == 0.65)), 3),
                "tentative": round(float(np.mean(weights[mask] == 0.4)), 3),
                "hidden": round(float(np.mean(weights[mask] == 0.0)), 3),
            }

        report["instruments"][name] = {
            "messages": len(rows),
            "per_second": round((len(rows) - 1) / (rows[-1, 0] - rows[0, 0]), 2),
            "interval_ms": spread(gaps, 1000),
            "gaps_over_30_ms": int((gaps > 0.03).sum()),
            "capture_to_publish_ms": spread(rows[:, 1]),
            "detected_to_published_ms": spread(rows[:, 2], digits=3),
            "attacks": int((rows[:, 3] > 0).sum()),
            "attacks_per_second": round(float((rows[:, 3] > 0).sum() / (rows[-1, 0] - rows[0, 0])), 2),
            "attack_capture_to_publish_ms": spread(rows[rows[:, 3] > 0, 1]),
            "attacks_shown": shares(struck),
            "active_frames_shown": shares(lively),
        }
    if len(estimates) > 2:
        estimates = np.asarray(estimates, dtype=float)
        report["recognition"] = {
            "estimates": len(estimates),
            "interval_s": spread(np.diff(estimates[:, 0]), digits=2),
            "processing_ms": spread(estimates[:, 2]),
            "context_s": state.board.context_s,
        }
    return report


def replay(args):
    import psutil
    from PySide6.QtCore import Qt, QTimer

    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_ui import configure, place

    from gmv.app.window import application
    from gmv.models import InstrumentActivity
    from gmv.visualizer import layers as layer_registry
    from gmv.visualizer.replay import ReplaySession
    from gmv.visualizer.state import VisualState
    from gmv.visualizer.window import VisualizerWindow

    app = application()
    session = ReplaySession(args.session)
    width, height = map(int, args.size.split("x"))
    window = VisualizerWindow()
    window.setWindowFlag(Qt.WindowType.FramelessWindowHint)
    window.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
    place(app, window, args.screen, width, height)
    window.set_mode(args.mode)
    window.set_radial_rings("history")  # the hooks follow rows into the history pictures
    configure(window, {"timeline"}, {"vocals", "guitar", "bass", "drums"})

    published = {}  # (instrument, timestamp) -> (published, taken in, attack)
    record = {"state": {}, "drawn": {}, "behind": {}, "rows": {}, "attacks": {}, "refused": {}}
    measuring = {"on": False}
    take_in = VisualState._activity
    slice_ = layer_registry.Layer._slice
    consume = VisualState.consume

    def consume_hook(self, message, arrival=None):
        if isinstance(message, InstrumentActivity):
            self._now_published = message.published_monotonic
        return consume(self, message, arrival)

    def activity_hook(self, message):
        track = self.activity.get(message.instrument)
        last = None if track is None or not len(track) else float(track.last[0])
        take_in(self, message)
        if not measuring["on"]:
            return
        clock = time.perf_counter()
        if last is not None and message.timestamp <= last:
            record["refused"][message.instrument] = record["refused"].get(message.instrument, 0) + 1
            return
        published[(message.instrument, round(message.timestamp, 5))] = (
            message.published_monotonic,
            message.attack,
        )
        record["state"].setdefault(message.instrument, []).append(
            (clock - message.published_monotonic) * 1000
        )

    def slice_hook(self, artist, to_x, track, since, oldest, state, stage):
        before = self.drawn_until
        slice_(self, artist, to_x, track, since, oldest, state, stage)
        if not measuring["on"] or self.drawn_until is None or since is None:
            return
        clock = time.perf_counter()
        rows = track.since(before if before is not None else self.drawn_until)
        new = rows[rows[:, 0] > (before if before is not None else -1.0)]
        struck = 0
        for row in new:
            found = published.pop((self.key, round(float(row[0]), 5)), None)
            if found is not None:
                record["drawn"].setdefault(self.key, []).append((clock - found[0]) * 1000)
                struck += found[1] > 0
        record["rows"].setdefault(self.key, []).append(len(new))
        record["attacks"].setdefault(self.key, []).append(struck)
        if stage.now is not None:
            record["behind"].setdefault(self.key, []).append((stage.now - self.drawn_until) * 1000)

    VisualState.consume = consume_hook
    VisualState._activity = activity_hook
    layer_registry.Layer._slice = slice_hook
    window.attach(
        session.bus,
        source="Responsiveness trace",
        activity_model=session.configuration.get("activity_model_path"),
        recognition_model=session.configuration.get("recognition_model_path"),
        instruments=session.configuration.get("instrument_analyzers") or (),
    )
    window.show()
    session.start()
    process = psutil.Process()

    def settle():
        window.canvas.perf.clear()
        process.cpu_percent()
        measuring["on"] = True

    def finish():
        perf = window.canvas.perf.stats()
        result = {
            "mode": args.mode,
            "window": f"{window.width()}x{window.height()}",
            "screen": f"{window.screen().size().width()}x{window.screen().size().height()} at {window.screen().refreshRate():.0f} Hz",
            "target_fps": round(1 / window.frame_s, 1),
            "frame_interval_ms": perf.get("frame interval"),
            "paint_ms": perf.get("paint"),
            "messages_per_frame": perf.get("messages per tick"),
            "messages_dropped_by_window": window.subscription.dropped if window.subscription else None,
            "never_drawn": len(published),
            "cpu_percent_of_one_core": round(process.cpu_percent()),
            "instruments": {},
        }
        for key in record["drawn"]:
            attacks = np.asarray(record["attacks"][key])
            result["instruments"][key] = {
                "publish_to_state_ms": spread(record["state"].get(key, [])),
                "publish_to_drawn_ms": spread(record["drawn"][key]),
                "behind_now_ms": spread(record["behind"].get(key, [])),
                "rows_per_paint": spread(record["rows"][key]),
                "paints_with_two_or_more_attacks": int((attacks >= 2).sum()),
                "most_attacks_in_one_paint": int(attacks.max()) if len(attacks) else 0,
                "rows_refused_by_state": record["refused"].get(key, 0),
            }
        window.close()
        session.stop()
        print(json.dumps(result, indent=1))
        if args.out:
            Path(args.out).write_text(json.dumps(result, indent=1), encoding="utf-8")
        app.quit()

    QTimer.singleShot(4000, settle)
    QTimer.singleShot(int(args.seconds * 1000), finish)
    return app.exec()


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    log = commands.add_parser("log")
    log.add_argument("sessions", nargs="+")
    log.add_argument("--out")
    again = commands.add_parser("replay")
    again.add_argument("session")
    again.add_argument("--mode", default="radial", choices=("radial", "bars"))
    again.add_argument("--seconds", type=float, default=40)
    again.add_argument("--size", default="1920x1080")
    again.add_argument("--screen", default="")
    again.add_argument("--out")
    args = parser.parse_args()
    if args.command == "replay":
        return replay(args)
    reports = {}
    for number, folder in enumerate(args.sessions, 1):
        reports[f"session {number}"] = read_log(folder)
    print(json.dumps(reports, indent=1))
    if args.out:
        Path(args.out).write_text(json.dumps(reports, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
