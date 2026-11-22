"""The engine in a process of its own.

Capture, analysis, recognition and session logging are one ``Engine``. In the window's
process they share one Python interpreter with the drawing: whichever holds it makes the
other wait, and measurements show both sides paying (late frames; analysis frames several
times slower when the window draws often). ``RemoteEngine`` runs that same ``Engine``,
unchanged, in a worker process and stands in for it here:

    worker:  source -> Engine (capture, analysis, logging) -> its EventBus -> pipe
    here:    pipe -> this object's EventBus -> the window, as before

* One persistent worker per session, started with the session and gone with it.
* The audio path is entirely inside the worker: nothing here can hold up capture.
* Messages cross in small batches, in publication order, exactly as published (same
  timestamps, sequence numbers and ages): nothing is restamped, dropped or merged. The
  pipe is drained by a thread here whatever the window is doing; the window's own bounded
  queue is, as before, the only place a slow window loses messages.
* ``stop``, sensitivity changes and a few status fields are the whole control surface.
* If the worker dies, the session ends with an error that says so; the application carries
  on and the next Start makes a new worker.

The monotonic clock (``time.perf_counter``) is system-wide on Windows, so the worker's
timestamps mean the same here; the offset is measured anyway and corrected if it is not.
"""

import multiprocessing
from threading import Event, Thread
from time import perf_counter, sleep
from types import SimpleNamespace
from uuid import uuid4

from .bus import EventBus
from .models import AnalysisConfig, FeatureFrame, Status

BATCH_S = 0.004  # how long the worker gathers messages before sending them
SNAPSHOT_S = 0.25  # how often it reports its state and measurements
CLOCK_TOLERANCE_S = 0.05


class Blank(dict):
    """Measurements before the worker has reported any."""

    def __missing__(self, key):
        return None


def blank_metrics():
    return Blank(
        frames=0,
        events=0,
        dropped_blocks=0,
        dropped_audio_frames=0,
        queue_high_water=0,
        discontinuities=0,
        native_dropped_packets=0,
        logger_dropped_messages=0,
        owned_process_resources={},
        advanced_status="starting",
        recognition_status="starting",
        recognition_jobs=0,
        recognition_skipped_hops=0,
        recognition_drops=0,
        error=None,
    )


def _serve(conn, factory, arguments, keywords, config, session_root, logging):
    """Worker process: build the source and the engine, run it, forward what it publishes."""
    from .engine import Engine

    try:
        source = factory(*arguments, **keywords)
        engine = Engine(source, config, session_root=session_root, logging=logging)
        queue = engine.bus.subscribe(16384)
        engine.start()
    except Exception as exc:
        conn.send(("failed", f"{type(exc).__name__}: {exc}"))
        conn.close()
        return

    def snapshot():
        return (
            "state",
            {
                "clock": perf_counter(),
                "session_id": engine.session_id,
                "state": engine.state,
                "detail": engine.detail,
                "error": engine.error,
                "metrics": engine.metrics(),
                "session_path": str(engine.writer.path) if engine.writer else None,
                "forwarder_dropped": queue.dropped,
            },
        )

    try:
        conn.send(snapshot())
        reported = perf_counter()
        while True:
            while conn.poll():  # commands from the window's process
                command = conn.recv()
                if command[0] == "stop":
                    engine.stop()
                elif command[0] == "sensitivity":
                    engine.instrument_sensitivity[command[1]] = command[2]
            finished = engine.done.is_set()
            batch, deadline = [], perf_counter() + BATCH_S
            first = queue.get(timeout=BATCH_S)
            while first is not None:
                batch.append(first)
                if len(batch) >= 256 or perf_counter() >= deadline:
                    break
                first = queue.get(timeout=0) if len(queue) else None
            if batch:
                conn.send(("messages", batch))
            if perf_counter() - reported >= SNAPSHOT_S or (finished and not batch):
                conn.send(snapshot())
                reported = perf_counter()
            if finished and not batch and not len(queue):
                break
        conn.send(("done",))
    except (BrokenPipeError, EOFError, OSError):
        engine.stop()  # the window's process is gone: stop capturing and finish the log
        engine.done.wait(10)
    finally:
        conn.close()


class Sensitivity(dict):
    """The engine's live sensitivity mapping: a change is sent to the worker."""

    def __init__(self, values, send):
        super().__init__(values)
        self._send = send

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self._send(("sensitivity", key, value))


class RemoteEngine:
    """Has the parts of ``Engine`` the application uses, backed by a worker process.

    ``factory(*arguments, **keywords)`` must build the audio source and be importable in
    the worker (a module-level function or a class), with picklable arguments."""

    def __init__(
        self,
        factory,
        arguments=(),
        keywords=None,
        config=None,
        *,
        session_root="sessions",
        logging=True,
        source=None,
    ):
        self.source = source  # a source built here from the same arguments, for description
        self.config = config or AnalysisConfig()
        self._target = (factory, tuple(arguments), dict(keywords or {}))
        self._session_root, self._logging = str(session_root), logging
        self.bus = EventBus()
        self.stop_event, self.done = Event(), Event()
        self.session_id = uuid4().hex  # replaced by the worker's as soon as it reports
        self.state, self.detail, self.error = "ready", "Ready", None
        self.latest = None
        self.writer = None
        self.frame_count = 0
        self.clock_offset = 0.0
        self.forwarder_dropped = 0
        self._metrics = blank_metrics()
        self._conn = self._process = self._reader = None
        self.threads = []  # this side's threads, as ``Engine.threads`` lists that side's
        self.instrument_sensitivity = Sensitivity(self.config.instrument_sensitivity, self._send)

    # -- the Engine interface ------------------------------------------------------------

    def start(self):
        """Start the worker and return at once; a failure to start arrives as an error."""
        if self._process is not None:
            raise RuntimeError("Create a new engine for each session")
        context = multiprocessing.get_context("spawn")
        self._conn, theirs = context.Pipe()
        self._process = context.Process(
            target=_serve,
            args=(theirs, *self._target, self.config, self._session_root, self._logging),
            name="gmv-analysis",
            daemon=True,  # never outlives this process
        )
        self.state, self.detail = "starting", "Starting the analysis process"
        self._process.start()
        theirs.close()
        self._reader = Thread(target=self._read, name="analysis-pipe", daemon=True)
        self.threads = [self._reader]
        self._reader.start()

    def stop(self):
        self.stop_event.set()
        self._send(("stop",))

    def metrics(self):
        return self._metrics

    @property
    def process_id(self):
        return self._process.pid if self._process is not None else None

    # -- the pipe --------------------------------------------------------------------------

    def _send(self, command):
        try:
            if self._conn is not None and not self.done.is_set():
                self._conn.send(command)
        except (BrokenPipeError, OSError, ValueError):
            pass  # the worker has gone; the reader reports it

    def _read(self):
        ended = None
        try:
            while True:
                kind, *rest = self._conn.recv()
                if kind == "messages":
                    for message in rest[0]:
                        if isinstance(message, FeatureFrame):
                            self.latest = message
                            self.frame_count += 1
                        if abs(self.clock_offset) > CLOCK_TOLERANCE_S:
                            message = self._on_this_clock(message)
                        self.bus.forward(message)
                elif kind == "state":
                    self._take(rest[0])
                elif kind == "failed":
                    ended = rest[0]
                    break
                elif kind == "done":
                    break
        except (EOFError, OSError):
            self._process.join(timeout=2)
            ended = f"The analysis process ended unexpectedly (exit code {self._process.exitcode})"
        if ended:
            self.error = self.error or ended
            self.state, self.detail = "error", self.error
            now = perf_counter()
            self.bus.publish(
                Status(
                    session_id=self.session_id,
                    source="analysis_process",
                    timestamp=self.latest.timestamp if self.latest else 0.0,
                    detected_monotonic=now,
                    state="error",
                    detail=self.error,
                )
            )
        try:
            self._conn.close()
        except OSError:
            pass
        self._process.join(timeout=5)
        if self._process.is_alive():  # it should have left by itself
            self._process.terminate()
            self._process.join(timeout=2)
        self.done.set()  # only now: when the session is over, so is the worker

    def _take(self, report):
        self.session_id = report["session_id"]
        self.state, self.detail = report["state"], report["detail"]
        self.error = report["error"]
        self._metrics = report["metrics"]
        self.forwarder_dropped = report["forwarder_dropped"]
        if report["session_path"] and self.writer is None:
            from pathlib import Path

            self.writer = SimpleNamespace(path=Path(report["session_path"]), error=None)
        # perf_counter here less perf_counter there, when the report was sent: the pipe's
        # delay if the clocks agree, something large if they do not.
        offset = perf_counter() - report["clock"]
        self.clock_offset = offset if abs(offset) > CLOCK_TOLERANCE_S else 0.0

    def _on_this_clock(self, message):
        shift = self.clock_offset
        update = {
            name: getattr(message, name) + shift
            for name in ("detected_monotonic", "published_monotonic", "evidence_monotonic")
            if getattr(message, name, None)
        }
        return message.model_copy(update=update)

    def close(self, timeout=10):
        """Stop and wait for the worker to finish; True when it has."""
        self.stop()
        finished = self.done.wait(timeout)
        if not finished and self._process is not None:
            self._process.terminate()
            sleep(0.1)
        return finished


__all__ = ["RemoteEngine", "blank_metrics"]
