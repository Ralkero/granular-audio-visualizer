"""Runs a ``Recognizer`` beside the analysis without ever holding it up.

The analysis thread hands over raw blocks with ``offer`` and returns at once. This thread
takes whatever has arrived, and when a hop is due asks the model about the newest audio. If
the model is slower than the hop, older audio is simply taken in without asking: an estimate
is always about the latest audio and the queue cannot grow.

Nothing here can stop a session. A model that fails to load, a missing package or an error
during inference ends recognition only; one ``InstrumentEstimate`` with ``available`` false
says why, and the worker then discards what it is offered.
"""

from collections import deque
from threading import Event, Lock, Thread
from time import perf_counter

import numpy as np

from ..buffer import BoundedQueue
from ..models import InstrumentEstimate
from . import Recognizer, downmix, load_backend


class RecognitionWorker:
    def __init__(self, bus, session_id, source, model_path, *, live=True, backend=None):
        self.bus, self.session_id, self.source = bus, session_id, source
        self.model_path, self.live = model_path, live
        self.backend = backend  # tests and offline runs pass one in
        self.queue = BoundedQueue(512)  # about five seconds of 10 ms blocks
        self.state = "starting"
        self.available = False
        self.ended = False
        self.times = deque(maxlen=256)
        self.jobs = 0
        self.skipped_hops = 0
        self.discarded_results = 0
        self.latest_age_ms = None
        self._lock, self._stop = Lock(), Event()
        self._generation = 0
        self._format, self._expected = None, None
        self.thread = Thread(target=self._run, name="recognition", daemon=True)

    def start(self):
        self.thread.start()

    def offer(self, block):
        if self.ended:
            return  # recognition has ended; nothing is kept
        with self._lock:
            current = (block.sample_rate, block.channels, block.segment)
            if (
                current != self._format
                or block.discontinuity
                or self._expected is None
                or abs(block.timestamp - self._expected) > 2 / block.sample_rate
                or len(self.queue) >= self.queue.capacity
            ):
                self._generation += 1
            self._format = current
            self._expected = block.timestamp + block.frames / block.sample_rate
            self.queue.put((self._generation, block))

    def _publish(self, segment, **fields):
        backend = self.backend
        message = InstrumentEstimate(
            session_id=self.session_id,
            source=self.source,
            segment=segment,
            detected_monotonic=perf_counter(),
            backend=backend.name if backend else "none",
            backend_version=backend.version if backend else "",
            context_s=backend.context_s if backend else 0,
            **fields,
        )
        return self.bus.publish(message)

    def _fail(self, reason, segment=None, time=0.0):
        self.available, self.ended = False, True
        self.state = f"unavailable: {reason}"
        try:
            self._publish(
                segment, timestamp=time, interval_end=time, available=False, detail=reason
            )
        except Exception:
            pass  # the status text above still says what happened

    def _run(self):
        try:
            if self.backend is None:
                self.backend = load_backend(self.model_path)
            recognizer = Recognizer(self.backend)
        except Exception as exc:
            self._fail(str(exc))
            self.queue.close()
            return
        self.available = True
        self.state = f"{self.backend.name}: waiting for audio"
        generation, evidence = None, None
        segment, time = None, 0.0
        try:
            while not self._stop.is_set():
                item = self.queue.get()
                if item is None:
                    if self.queue.closed:
                        break
                    continue
                # Take in everything that is waiting; only the newest audio is asked about.
                while item is not None:
                    job_generation, block = item
                    if job_generation != generation:
                        recognizer.reset()
                        generation = job_generation
                    samples = np.frombuffer(block.data, dtype="<f4").reshape(
                        block.frames, block.channels
                    )
                    recognizer.push(downmix(samples), block.sample_rate, block.timestamp)
                    evidence = block.capture_monotonic if self.live else None
                    segment, time = block.segment, recognizer.end
                    item = self.queue.get(timeout=0) if len(self.queue) else None
                if not recognizer.due:
                    continue
                self.skipped_hops += recognizer.hops_waiting - 1
                began = perf_counter()
                estimate = recognizer.estimate()
                elapsed = (perf_counter() - began) * 1000
                self.times.append(elapsed)
                self.jobs += 1
                with self._lock:
                    if generation != self._generation or self._stop.is_set():
                        self.discarded_results += 1
                        continue
                    published = self._publish(
                        segment,
                        timestamp=estimate.start,
                        interval_end=estimate.end,
                        evidence_monotonic=evidence,
                        scores=estimate.scores,
                        states=estimate.states,
                        processing_ms=elapsed,
                    )
                    self.latest_age_ms = published.age_at_publication_ms
                self.state = f"{self.backend.name}: {elapsed:.0f} ms per estimate"
        except Exception as exc:
            self._fail(str(exc), segment, time)
        finally:
            self.queue.close()

    def close(self):
        with self._lock:
            self._stop.set()
            self._generation += 1
        self.queue.close()
        self.thread.join(timeout=5)
        self.ended = True
        if self.thread.is_alive():
            self.state = "recognition worker exceeded shutdown deadline"
        elif self.available:
            self.state = "stopped"
