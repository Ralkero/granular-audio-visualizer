"""Optional delayed spectral HPSS proxy. This does NOT identify vocals, drums or bass."""

from collections import deque
from math import gcd
from threading import Thread, Lock, Event
from time import perf_counter
from typing import Protocol
from uuid import uuid4
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import resample_poly
from .buffer import BoundedQueue
from .models import MusicalEvent


class BackgroundAnalyzer(Protocol):
    """Future model plugins receive bounded audio and return evidence, never control capture."""

    def __call__(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]: ...


def harmonic_percussive_balance(audio, sample_rate):
    # Limit compute cost at high device rates. Keep channels independent until spectra aggregate.
    if sample_rate != 22050:
        divisor = gcd(sample_rate, 22050)
        audio = resample_poly(audio, 22050 // divisor, sample_rate // divisor, axis=0)
    size, hop = 1024, 256
    windows = np.lib.stride_tricks.sliding_window_view(audio, size, axis=0)[::hop]
    spectrum = np.fft.rfft(windows * np.hanning(size), axis=-1)
    power = np.mean(np.abs(spectrum) ** 2, axis=1).T
    magnitude = np.sqrt(power)
    harmonic = median_filter(magnitude, size=(1, 17), mode="nearest")
    percussive = median_filter(magnitude, size=(17, 1), mode="nearest")
    harmonic_mask = harmonic**2 / (harmonic**2 + percussive**2 + 1e-20)
    total = float(power.sum())
    ratio = float(np.sum(power * harmonic_mask) / max(total, 1e-20))
    return {
        "harmonic_power_fraction": ratio,
        "percussive_power_fraction": 1 - ratio if total > 1e-12 else 0,
        "rms": float(np.sqrt(np.mean(audio**2))),
    }


class AdvancedWorker:
    def __init__(
        self,
        bus,
        session_id,
        source,
        analyzer: BackgroundAnalyzer = harmonic_percussive_balance,
        *,
        live=True,
    ):
        self.bus, self.session_id, self.source = bus, session_id, source
        self.analyzer = analyzer
        self.live = live
        self.queue = BoundedQueue(64)
        self.state = "waiting for two seconds of contiguous audio"
        self.times = deque(maxlen=256)
        self.jobs = 0
        self.error = None
        self._lock, self._stop = Lock(), Event()
        self._generation = 0
        self._offered_format, self._expected = None, None
        self.discarded_results = 0
        self.latest_age_ms = None
        self.thread = Thread(target=self._run, name="advanced-analysis", daemon=True)

    def start(self):
        self.thread.start()

    def offer(self, block):
        with self._lock:
            current = (block.sample_rate, block.channels, block.segment)
            if (
                current != self._offered_format
                or block.discontinuity
                or self._expected is None
                or abs(block.timestamp - self._expected) > 2 / block.sample_rate
                or len(self.queue) >= self.queue.capacity
            ):
                self._generation += 1
            self._offered_format = current
            self._expected = block.timestamp + block.frames / block.sample_rate
            self.queue.put((self._generation, block))

    def _run(self):
        parts, frames, start, expected, fmt = [], 0, 0, None, None
        generation, evidence_mono = None, None
        try:
            while not self._stop.is_set():
                item = self.queue.get()
                if item is None:
                    if self.queue.closed:
                        break
                    continue
                job_generation, block = item
                if job_generation != generation:
                    parts, frames = [], 0
                    generation = job_generation
                current = (block.sample_rate, block.channels, block.segment)
                if (
                    current != fmt
                    or block.discontinuity
                    or expected is None
                    or abs(block.timestamp - expected) > 2 / block.sample_rate
                ):
                    parts, frames = [], 0
                if not parts:
                    start = block.timestamp
                    evidence_mono = block.capture_monotonic if self.live else None
                fmt = current
                parts.append(
                    np.frombuffer(block.data, dtype="<f4").reshape(block.frames, block.channels)
                )
                frames += block.frames
                expected = block.timestamp + block.frames / block.sample_rate
                if frames >= 2 * block.sample_rate:
                    began = perf_counter()
                    self.state = "running delayed harmonic/percussive spectral analysis"
                    values = self.analyzer(np.concatenate(parts), block.sample_rate)
                    elapsed = (perf_counter() - began) * 1000
                    self.times.append(elapsed)
                    self.jobs += 1
                    result = MusicalEvent(
                        session_id=self.session_id,
                        source=self.source,
                        segment=block.segment,
                        evidence_monotonic=evidence_mono,
                        provenance="harmonic_percussive_proxy",
                        intended_use="context",
                        evidence_kind="context",
                        timestamp=start,
                        detected_monotonic=perf_counter(),
                        event_id=uuid4().hex,
                        event_type="harmonic_percussive_balance",
                        analysis_tier="advanced",
                        intensity=values["percussive_power_fraction"],
                        confidence=0.35,
                        interval_end=expected,
                        method="median_hpss_spectral_proxy_v1",
                        measurements=tuple(values.items()) + (("processing_ms", elapsed),),
                    )
                    with self._lock:
                        if generation == self._generation and not self._stop.is_set():
                            published = self.bus.publish(result)
                            self.latest_age_ms = published.age_at_publication_ms
                        else:
                            self.discarded_results += 1
                    parts, frames = [], 0
                    self.state = (
                        f"delayed HPSS: {elapsed:.1f} ms per two-second chunk; no instrument stems"
                    )
        except Exception as exc:
            self.error = str(exc)
            self.state = "advanced analysis failed: " + str(exc)
        finally:
            self.queue.close()

    def close(self):
        with self._lock:
            self._stop.set()
            self._generation += 1
        self.queue.close()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            self.state = "advanced worker exceeded shutdown deadline"
        else:
            self.state = "stopped; delayed HPSS proxy"
