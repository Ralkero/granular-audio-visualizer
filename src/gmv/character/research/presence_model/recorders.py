"""Step 4, S4.7: the host-activity record and the memory record (protocol Amendment 3, A3.8, C1
and V6) *[tested before use]*. Linux only (``/proc``). Built and tested on generated jobs only.

**Host activity**, about once a second, as aggregate counters only: total processor use, the
job's own use, other programs' use (the difference), available memory, swap activity (pages
swapped in plus out since the previous sample, machine-wide) and the mean processor frequency.
**Never process names, command lines, window titles or file names:** the recorder reads only
``/proc/stat``, ``/proc/meminfo``, ``/proc/vmstat``, each traced process's ``stat`` (numbers
after the name field; the name is discarded unread), ``smaps_rollup`` and ``task/*/children``
(process numbers), and ``/sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq`` (read only; the
power settings are never changed).

**The job's own use** sums, over the live processes of its tree, user + system time plus the
time of the children each has already waited for, so every process counts once while it is
reaped by a live ancestor.

**Memory (V6):** the experiment's whole process tree (the launcher's job, every process it
starts, found through ``task/*/children``), measured once a second as the sum over processes of
the proportional set size plus the proportional swap share (``Pss`` + ``SwapPss`` in
``smaps_rollup``). A page shared by four processes contributes a quarter to each, so it is counted
once; copy-on-write children are counted the same way; swapped-out memory still counts. The peak
is recorded. Above the limit (8 GiB; tests lower it) the recorder calls its ``on_limit`` action.

**Disturbance** (C1): a run is disturbed if other programs average more than one logical
processor, use more than four for over a tenth of the samples, or any swap activity occurs. Swap
activity is reported separately and never counted towards memory.
"""

import glob
import os
import threading
import time
from dataclasses import dataclass

GIB = 1024**3
MEMORY_LIMIT = 8 * GIB
OTHERS_AVERAGE_LIMIT = 1.0
OTHERS_BURST_LIMIT, OTHERS_BURST_SHARE = 4.0, 0.10
TICKS = os.sysconf("SC_CLK_TCK")


def _read(path):
    with open(path, encoding="ascii", errors="replace") as file:
        return file.read()


def tree(root):
    """Every live process in ``root``'s tree, found through each thread's children list."""
    seen, stack = set(), [int(root)]
    while stack:
        pid = stack.pop()
        if pid in seen:
            continue
        try:
            tasks = os.listdir(f"/proc/{pid}/task")
        except OSError:
            continue
        seen.add(pid)
        for task in tasks:
            try:
                stack.extend(int(c) for c in _read(f"/proc/{pid}/task/{task}/children").split())
            except OSError:
                continue
    return seen


def _stat_numbers(pid):
    """The numeric fields of /proc/<pid>/stat after the name (which is discarded)."""
    text = _read(f"/proc/{pid}/stat")
    return text[text.rindex(")") + 2 :].split()


def tree_cpu_ticks(pids):
    total = 0
    for pid in pids:
        try:
            fields = _stat_numbers(pid)
        except (OSError, ValueError):
            continue
        # fields[11:15] are utime, stime, cutime, cstime (stat fields 14-17)
        total += sum(int(x) for x in fields[11:15])
    return total


def memory_of(pid):
    """(Pss, SwapPss) in bytes from smaps_rollup; (0, 0) when the process has gone."""
    try:
        text = _read(f"/proc/{pid}/smaps_rollup")
    except OSError:
        return 0, 0
    values = {}
    for line in text.splitlines():
        name, _, rest = line.partition(":")
        if name in ("Pss", "SwapPss"):
            values[name] = int(rest.split()[0]) * 1024
    return values.get("Pss", 0), values.get("SwapPss", 0)


def tree_memory(root):
    """(bytes, processes): Σ Pss + SwapPss over the live tree."""
    pids = tree(root)
    return sum(sum(memory_of(p)) for p in pids), len(pids)


def cpu_ticks():
    """(busy, total) jiffies over every logical processor."""
    fields = [int(x) for x in _read("/proc/stat").splitlines()[0].split()[1:9]]
    idle = fields[3] + fields[4]
    return sum(fields) - idle, sum(fields)


def swap_pages():
    values = dict(line.split() for line in _read("/proc/vmstat").splitlines() if line.startswith(("pswpin", "pswpout")))
    return int(values.get("pswpin", 0)) + int(values.get("pswpout", 0))


def available_memory():
    for line in _read("/proc/meminfo").splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def frequency_mhz():
    values = []
    for path in glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq"):
        try:
            values.append(int(_read(path)) / 1000)
        except (OSError, ValueError):
            continue
    return sum(values) / len(values) if values else None


@dataclass(frozen=True)
class Sample:
    """One aggregate sample. Processor use is in logical processors over the interval."""

    elapsed_s: float
    cpu_all: float
    cpu_job: float
    cpu_other: float
    available_bytes: int | None
    swap_pages: int  # swapped in + out since the previous sample, machine-wide
    frequency_mhz: float | None
    tree_bytes: int
    processes: int


class Recorder(threading.Thread):
    """Samples the job's tree and the host about every ``interval`` seconds until ``stop()``."""

    def __init__(self, root, interval=1.0, memory_limit=MEMORY_LIMIT, on_limit=None):
        super().__init__(daemon=True)
        self.root, self.interval, self.memory_limit, self.on_limit = int(root), interval, memory_limit, on_limit
        self.samples, self.peak_bytes, self.limit_hit = [], 0, False
        self._halt = threading.Event()
        self._cpus = os.cpu_count() or 1

    def stop(self):
        self._halt.set()
        self.join()

    def sample_memory(self):
        used, count = tree_memory(self.root)
        self.peak_bytes = max(self.peak_bytes, used)
        if used > self.memory_limit and not self.limit_hit:
            self.limit_hit = True
            if self.on_limit is not None:
                self.on_limit()
        return used, count

    def run(self):
        start = time.monotonic()
        busy0, total0 = cpu_ticks()
        job0 = tree_cpu_ticks(tree(self.root))
        swap0, t0 = swap_pages(), start
        self.sample_memory()
        while not self._halt.wait(self.interval):
            now = time.monotonic()
            busy, total = cpu_ticks()
            pids = tree(self.root)
            job = tree_cpu_ticks(pids)
            swap = swap_pages()
            used, count = self.sample_memory()
            dt = max(now - t0, 1e-9)
            share = (busy - busy0) / max(total - total0, 1)
            cpu_all = share * self._cpus
            cpu_job = max(job - job0, 0) / TICKS / dt
            self.samples.append(Sample(now - start, cpu_all, cpu_job, max(cpu_all - cpu_job, 0.0), available_memory(),
                                       swap - swap0, frequency_mhz(), used, count))
            busy0, total0, job0, swap0, t0 = busy, total, job, swap, now


def disturbed(samples):
    """(disturbed, reasons) by the C1 thresholds; swap activity is its own reason."""
    reasons = []
    if not samples:
        return True, ("no host-activity samples were recorded",)
    others = [s.cpu_other for s in samples]
    if sum(others) / len(others) > OTHERS_AVERAGE_LIMIT:
        reasons.append("other programs averaged more than one logical processor")
    if sum(1 for x in others if x > OTHERS_BURST_LIMIT) > OTHERS_BURST_SHARE * len(others):
        reasons.append("other programs used more than four logical processors for over a tenth of the run")
    if any(s.swap_pages > 0 for s in samples):
        reasons.append("swap activity occurred during the run")
    return bool(reasons), tuple(reasons)
