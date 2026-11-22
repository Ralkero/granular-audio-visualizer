"""Rolling timing measurements for the visualizer: cheap enough to stay on all the time."""

from collections import deque
from contextlib import contextmanager
from time import perf_counter

import numpy as np


class Rolling:
    """The last ``size`` values of one measurement."""

    def __init__(self, size=1800):
        self.values = deque(maxlen=size)
        self.total = 0

    def add(self, value):
        self.values.append(value)
        self.total += 1

    def stats(self, digits=2):
        if not self.values:
            return None
        values = np.fromiter(self.values, dtype=float)
        p50, p95, p99 = np.percentile(values, [50, 95, 99])
        return {
            "samples": len(values),
            "p50": round(float(p50), digits),
            "p95": round(float(p95), digits),
            "p99": round(float(p99), digits),
            "max": round(float(values.max()), digits),
        }


class Timings:
    """Named ``Rolling`` series, created on first use. ``measure(name)`` times a block in
    milliseconds; ``count(name)`` counts occurrences."""

    def __init__(self, size=1800):
        self.size = size
        self.series = {}
        self.counts = {}

    def add(self, name, value):
        series = self.series.get(name)
        if series is None:
            series = self.series[name] = Rolling(self.size)
        series.add(value)

    def count(self, name, amount=1):
        self.counts[name] = self.counts.get(name, 0) + amount

    @contextmanager
    def measure(self, name):
        began = perf_counter()
        try:
            yield
        finally:
            self.add(name, (perf_counter() - began) * 1000)

    def stats(self):
        report = {name: series.stats() for name, series in sorted(self.series.items())}
        report["counts"] = dict(sorted(self.counts.items()))
        return report

    def clear(self):
        self.series.clear()
        self.counts.clear()
