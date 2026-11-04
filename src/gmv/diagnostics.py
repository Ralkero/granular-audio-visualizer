"""Bounded whole-session sampling and resource scope, independent of capture."""

import random
import numpy as np
import psutil


class RunningStats:
    def __init__(self, capacity=4096):
        self.capacity, self.values, self.count = capacity, [], 0
        self.minimum, self.maximum, self.total = float("inf"), -float("inf"), 0.0
        self.random = random.Random(1701)

    def add(self, value):
        value = float(value)
        if not np.isfinite(value):
            return
        self.count += 1
        self.minimum, self.maximum = min(self.minimum, value), max(self.maximum, value)
        self.total += value
        if len(self.values) < self.capacity:
            self.values.append(value)
        else:
            index = self.random.randrange(self.count)
            if index < self.capacity:
                self.values[index] = value

    def report(self):
        return (
            {
                "count": self.count,
                "scope": "whole-session uniform reservoir; approximate percentiles",
                "reservoir_size": len(self.values),
                "mean": self.total / self.count,
                "min": self.minimum,
                "max": self.maximum,
                "p50": float(np.percentile(self.values, 50)),
                "p95": float(np.percentile(self.values, 95)),
            }
            if self.count
            else None
        )


class ResourceMonitor:
    def __init__(self):
        self.parent = psutil.Process()
        self.processes = {self.parent.pid: self.parent}
        self.parent.cpu_percent()

    def sample(self):
        alive = {self.parent.pid: self.parent}
        for child in self.parent.children(recursive=True):
            alive[child.pid] = self.processes.get(child.pid, child)
        rows = []
        for pid, process in alive.items():
            try:
                rows.append(
                    {
                        "role": "python_parent" if pid == self.parent.pid else "owned_child",
                        "cpu_one_core_percent": process.cpu_percent(),
                        "rss_mb": process.memory_info().rss / 1024**2,
                        "threads": process.num_threads(),
                    }
                )
            except psutil.Error:
                continue
        self.processes = alive
        return {
            "scope": "Python parent plus all its owned descendant processes; excludes captured apps",
            "processes": rows,
            "cpu_one_core_percent": sum(r["cpu_one_core_percent"] for r in rows),
            "rss_mb": sum(r["rss_mb"] for r in rows),
            "process_count": len(rows),
        }
