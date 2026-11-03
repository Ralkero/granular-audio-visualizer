from collections import deque
from threading import Event, Thread
from time import perf_counter
from uuid import uuid4
import numpy as np
import psutil
from .buffer import BoundedQueue
from .bus import EventBus
from .dsp import Analyzer, WindowAssembler
from .models import AnalysisConfig, Status
from .session import SessionWriter


class Engine:
    def __init__(self, source, config=None, *, session_root="sessions", logging=True):
        self.source, self.config = source, config or AnalysisConfig()
        self.session_root, self.logging = session_root, logging
        self.session_id = uuid4().hex
        self.bus = EventBus()
        self.audio = BoundedQueue(self.config.queue_blocks)
        self.stop_event, self.done = Event(), Event()
        self.writer = None
        self.advanced = None
        self.frame_count, self.event_count, self.discontinuities = 0, 0, 0
        self.dsp_times, self.ages, self.queue_times = deque(maxlen=4096), deque(maxlen=4096), deque(maxlen=4096)
        self.state, self.detail, self.error = "ready", "Ready", None
        self.latest = None
        self.cpu, self.memory_mb, self.peak_memory_mb = 0.0, 0.0, 0.0
        self.started = perf_counter()
        self.threads = []
        self._log_error_reported = False

    def status(self, state, detail):
        self.state, self.detail = state, detail
        if state == "error":
            self.error = detail
        now = perf_counter()
        self.bus.publish(Status(session_id=self.session_id, source=self.source.kind,
                                timestamp=max(0, now-self.started), detected_monotonic=now,
                                state=state, detail=detail))

    def start(self):
        if self.threads:
            raise RuntimeError("Create a new Engine for each session")
        self.started = perf_counter()
        if self.logging:
            self.writer = SessionWriter(self.session_root, self.session_id, self.bus,
                                        self.config, self.source)
        if self.config.advanced:
            from .advanced import AdvancedWorker
            self.advanced = AdvancedWorker(self.bus, self.session_id, self.source.kind)
            self.advanced.start()
        self.status("starting", "Opening audio source")
        self.threads = [Thread(target=self._capture, name="capture", daemon=True),
                        Thread(target=self._analyze, name="analysis", daemon=True)]
        for thread in self.threads:
            thread.start()

    def _capture(self):
        try:
            self.source.run(self.audio, self.stop_event, self.status)
        except Exception as exc:
            self.status("error", f"Capture: {exc}")
        finally:
            self.audio.close()

    def _analyze(self):
        assembler = WindowAssembler(self.config)
        analyzer = None
        process = psutil.Process()
        process.cpu_percent()
        last_metrics = perf_counter()
        try:
            while True:
                block = self.audio.get()
                if block is None and self.audio.closed:
                    break
                if block is not None:
                    queue_ms = (perf_counter() - block.received_monotonic) * 1000
                    for window, start, mono, reset in assembler.push(block):
                        if analyzer is None or reset:
                            if analyzer is not None:
                                self.discontinuities += 1
                                self.status("discontinuity", "Audio gap/format change; analysis context reset")
                            analyzer = Analyzer(block.sample_rate, block.channels, self.config,
                                                self.session_id, self.source.kind)
                        frame, events = analyzer.process(window, start, mono, block,
                                                         queue_ms, self.source.live)
                        if self.source.live:
                            frame = frame.model_copy(update={"capture_to_publish_ms": max(0,
                                (perf_counter()-mono-len(window)/block.sample_rate/2)*1000)})
                        self.latest = self.bus.publish(frame)
                        for event in events:
                            self.bus.publish(event)
                        self.frame_count += 1
                        self.event_count += len(events)
                        self.dsp_times.append(frame.dsp_ms)
                        self.queue_times.append(frame.queue_latency_ms)
                        if frame.capture_to_publish_ms is not None:
                            self.ages.append(frame.capture_to_publish_ms)
                    if self.advanced:
                        self.advanced.offer(block)
                if perf_counter() - last_metrics > 1:
                    self.cpu = process.cpu_percent()
                    self.memory_mb = process.memory_info().rss / 1024**2
                    self.peak_memory_mb = max(self.peak_memory_mb, self.memory_mb)
                    last_metrics = perf_counter()
                    if self.writer and self.writer.error and not self._log_error_reported:
                        self._log_error_reported = True
                        self.status("error", f"Logging failed: {self.writer.error}")
        except Exception as exc:
            self.status("error", f"Analysis: {exc}")
            self.stop_event.set()
        finally:
            self.stop_event.set()
            self.audio.close()
            self.threads[0].join(timeout=5)
            if self.threads[0].is_alive():
                self.status("error", "Capture worker has not stopped; restart application before recapturing")
            if self.advanced:
                self.advanced.close()
            self.status("error" if self.error else "stopped", self.error or "Session complete")
            if self.writer:
                try:
                    self.writer.close(self.metrics(), self.source)
                    if self.writer.error:
                        self.error = self.writer.error
                        self.state, self.detail = "error", self.error
                except Exception as exc:
                    self.error = f"Session finalization: {exc}"
                    self.state, self.detail = "error", self.error
            self.done.set()

    def stop(self):
        self.stop_event.set()

    def metrics(self):
        def stats(values):
            values = list(values)
            return {"samples": len(values), "p50": float(np.percentile(values, 50)),
                    "p95": float(np.percentile(values, 95)), "max": max(values)} if values else None
        return {"frames": self.frame_count, "events": self.event_count,
                "dropped_blocks": self.audio.dropped, "dropped_audio_frames": self.audio.dropped_frames,
                "queue_high_water": self.audio.high_water, "discontinuities": self.discontinuities,
                "dsp_ms": stats(self.dsp_times), "queue_ms": stats(self.queue_times),
                "estimated_capture_center_to_publish_ms": stats(self.ages),
                "timing_scope": "latest 4096 frames; callback clock estimate, not physical output latency",
                "process_cpu_percent": self.cpu, "rss_mb": self.memory_mb,
                "peak_sampled_rss_mb": self.peak_memory_mb,
                "advanced_status": self.advanced.state if self.advanced else "disabled",
                "advanced_drops": self.advanced.queue.dropped if self.advanced else 0,
                "error": self.error}
