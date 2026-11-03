"""Asynchronous analysis logging. No audio samples are persisted by this writer."""
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from threading import Thread
from time import perf_counter
from .models import BAND_NAMES, FeatureFrame, MusicalEvent, Status


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


class SessionWriter:
    def __init__(self, root, session_id, bus, config, source):
        self.path = Path(root) / ("session_" + datetime.now().strftime("%Y-%m-%d_%H%M%S")
                                  + "_" + session_id[:8])
        (self.path / "debug").mkdir(parents=True)
        (self.path / "advanced_analysis").mkdir()
        self.bus, self.queue = bus, bus.subscribe(4096)
        self.error = None
        self.count = 0
        self.manifest = {"schema_version": "1.0", "session_id": session_id,
                         "started_utc": utc_now(), "stopped_utc": None,
                         "raw_audio_saved": False, "configuration": config.model_dump(),
                         "source": source.describe(), "complete": False}
        write_json(self.path / "manifest.json", self.manifest)
        self.thread = Thread(target=self._run, name="session-writer", daemon=True)
        self.thread.start()

    def _run(self):
        try:
            with (self.path / "messages.jsonl").open("w", encoding="utf-8") as messages, \
                 (self.path / "realtime_features.csv").open("w", newline="", encoding="utf-8") as features, \
                 (self.path / "realtime_events.jsonl").open("w", encoding="utf-8") as events, \
                 (self.path / "advanced_analysis" / "mixed_hpss.jsonl").open("w", encoding="utf-8") as advanced, \
                 (self.path / "debug" / "errors.log").open("w", encoding="utf-8") as errors:
                writer = None
                flushed = perf_counter()
                while True:
                    message = self.queue.get()
                    if message is None:
                        if self.queue.closed:
                            break
                        continue
                    line = message.model_dump_json() + "\n"
                    messages.write(line)
                    self.count += 1
                    if isinstance(message, FeatureFrame):
                        row = message.model_dump()
                        row.update(dict(zip((n + "_energy" for n in BAND_NAMES), row.pop("band_energy"))))
                        row["spectral_contrast_db"] = json.dumps(row["spectral_contrast_db"])
                        if writer is None:
                            writer = csv.DictWriter(features, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow(row)
                    elif isinstance(message, MusicalEvent):
                        (advanced if message.analysis_tier == "advanced" else events).write(line)
                    elif message.state in {"error", "recovering", "discontinuity"}:
                        errors.write(line)
                    if perf_counter() - flushed >= 1:
                        for file in (messages, features, events, errors, advanced):
                            file.flush()
                        flushed = perf_counter()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.bus.unsubscribe(self.queue)

    def close(self, performance, source):
        self.bus.unsubscribe(self.queue)
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            self.error = "Log writer did not finish within 10 seconds"
        self.manifest.update(stopped_utc=utc_now(), complete=not bool(self.error),
                             log_messages=self.count, log_dropped_messages=self.queue.dropped,
                             error=self.error)
        write_json(self.path / "manifest.json", self.manifest)
        write_json(self.path / "capture_info.json", source.describe())
        write_json(self.path / "performance.json", performance)
        (self.path / "summary.md").write_text(
            f"# Analysis session\n\nSource: {source.kind}\n\n"
            f"Messages: {self.count}; logger drops: {self.queue.dropped}.\n\n"
            f"Capture blocks dropped: {performance['dropped_blocks']}.\n\n"
            f"Raw audio saved: no. Advanced analysis: {performance['advanced_status']}.\n\n"
            "Timing estimates refer to the capture timeline. Physical playback latency was not measured.\n\n"
            f"Writer error: {self.error or 'none'}.\n", encoding="utf-8")


def replay(path, bus, stop, *, speed=1.0):
    if speed <= 0:
        raise ValueError("Replay speed must be positive")
    models = {"feature_frame": FeatureFrame, "musical_event": MusicalEvent, "status": Status}
    first, started = None, perf_counter()
    with Path(path).open(encoding="utf-8") as file:
        while not stop.is_set():
            line = file.readline(1_000_001)
            if not line:
                break
            if len(line) > 1_000_000:
                raise ValueError("Replay message exceeds size limit")
            data = json.loads(line)
            message = models[data["type"]].model_validate(data)
            # Replay publication order, preserving original evidence timestamps (Tier B may be late).
            first = message.published_monotonic if first is None else first
            delay = (message.published_monotonic - first) / speed
            if stop.wait(max(0, started + delay - perf_counter())):
                break
            bus.publish(message)
