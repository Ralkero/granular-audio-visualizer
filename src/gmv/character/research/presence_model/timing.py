"""The timing harness for the later, authorised measurements (readiness proposal, revision
4.4, section 15). Step 1 exercises it only on synthetic records and tiny sleep-based jobs; no
empirical value is produced here.

**Process-time** is the elapsed duration, on the system-wide monotonic clock, from exec to
exit of a job's process when it runs alone. A job records the phases it knows about
(``Marks``); the launcher records exec and exit. Every second between exec and exit is
accounted for: startup (exec to the first mark), the marked phases, unmarked gaps ("other")
and teardown (the last mark to exit). ``accounting`` reports the unmarked share, so an
incomplete set of marks is visible instead of silently lost.

**Projection.** A job's single-process time is S + d1 + (E - 1)·k·U + p·P: setup S (startup,
setup, teardown and unmarked time), the first epoch d1 separately, k per training unit per
epoch from epochs 2 ... E, and p per predicted excerpt; the gate-only evaluation is excluded.
Measured values are per size class.

**Scheduling.** ``list_schedule`` simulates four workers with fixed job durations, for which
the classical bound makespan <= (sum + 3·max)/4 holds; ``contention_schedule`` simulates
durations that depend on how many jobs run at once. The planning estimate built from
estimated durations is not a proven bound for contention-dependent durations.
"""

import json
import os
import subprocess
import time
from dataclasses import dataclass, field

import numpy as np

WORKERS = 4
PHASES = ("setup", "epoch", "gate_eval", "prediction")


# -- recording -----------------------------------------------------------------------------------


class Marks:
    """Phase intervals recorded inside a job: ``with marks.phase("epoch"): ...``."""

    def __init__(self, path):
        self.path = path
        self.intervals = []

    def phase(self, label):
        if label not in PHASES:
            raise ValueError("unknown phase")
        marks = self

        class _Phase:
            def __enter__(self):
                self.start = time.monotonic_ns()

            def __exit__(self, *exc):
                marks.intervals.append((label, self.start, time.monotonic_ns()))
                return False

        return _Phase()

    def write(self):
        with open(self.path, "w", encoding="utf-8") as file:
            json.dump(self.intervals, file)


@dataclass(frozen=True)
class JobRecord:
    name: str
    size_class: str
    units: int
    predictions: int
    exec_ns: int
    exit_ns: int
    intervals: tuple  # (label, start_ns, end_ns)

    @property
    def elapsed_s(self):
        return (self.exit_ns - self.exec_ns) / 1e9


def run_job(name, command, marks_path, size_class, units, predictions, env=None, timeout=None):
    """Run one job alone, timing exec to exit on the monotonic clock."""
    exec_ns = time.monotonic_ns()
    subprocess.run(command, env=env, check=True, timeout=timeout)
    exit_ns = time.monotonic_ns()
    with open(marks_path, encoding="utf-8") as file:
        intervals = tuple(tuple(i) for i in json.load(file))
    return JobRecord(name, size_class, units, predictions, exec_ns, exit_ns, intervals)


def accounting(record):
    """Seconds per component; they add up to exec-to-exit exactly."""
    intervals = sorted(record.intervals, key=lambda i: i[1])
    if not intervals:
        raise ValueError("a job recorded no phases")
    for (_a, _s, end), (_b, start, _e) in zip(intervals, intervals[1:], strict=False):
        if start < end:
            raise ValueError("recorded phases overlap")
    out = {"startup": (intervals[0][1] - record.exec_ns) / 1e9,
           "teardown": (record.exit_ns - intervals[-1][2]) / 1e9, "other": 0.0,
           "setup": 0.0, "gate_eval": 0.0, "prediction": 0.0, "epochs": []}
    for label, start, end in intervals:
        if label == "epoch":
            out["epochs"].append((end - start) / 1e9)
        else:
            out[label] += (end - start) / 1e9
    for (_a, _s, end), (_b, start, _e) in zip(intervals, intervals[1:], strict=False):
        out["other"] += (start - end) / 1e9
    return out


def complete(record, max_other_s):
    """True when the unmarked time stays within ``max_other_s`` (a stated tolerance)."""
    return accounting(record)["other"] <= max_other_s


def components(record):
    """(s, d1, rest, q): setup-like time, epoch 1, epochs 2 ... E, prediction."""
    a = accounting(record)
    if not a["epochs"]:
        raise ValueError("a training job recorded no epoch")
    s = a["startup"] + a["setup"] + a["teardown"] + a["other"]
    return s, a["epochs"][0], a["epochs"][1:], a["prediction"]


# -- measurement summaries ------------------------------------------------------------------------


@dataclass(frozen=True)
class Measured:
    """Per size class: setup S, first epoch d1, per-unit epoch cost k; p per prediction."""

    S: dict
    d1: dict
    k: dict
    p: float


def summarise(records):
    S, d1, k, p = {}, {}, {}, 0.0
    for record in records:
        s, first, rest, q = components(record)
        z = record.size_class
        S[z] = max(S.get(z, 0.0), s)
        d1[z] = max(d1.get(z, 0.0), first)
        if rest:
            k[z] = max(k.get(z, 0.0), float(np.mean(rest)) / record.units)
        if record.predictions:
            p = max(p, q / record.predictions)
    return Measured(S, d1, k, p)


@dataclass(frozen=True)
class Job:
    size_class: str
    units: int
    epochs: int
    predictions: int
    seed_index: int = 0
    label: str = field(default="", compare=False)


def process_time(job, m):
    """Projected single-process seconds: S + d1 + (E - 1)·k·U + p·P."""
    z = job.size_class
    return m.S[z] + m.d1[z] + (job.epochs - 1) * m.k[z] * job.units + m.p * job.predictions


def split(job, m):
    """(s, t, q) of the projection, for contention factors per component."""
    z = job.size_class
    return m.S[z], m.d1[z] + (job.epochs - 1) * m.k[z] * job.units, m.p * job.predictions


def h_proc(jobs, m, preprocessing_s=0.0):
    return preprocessing_s + sum(process_time(j, m) for j in jobs)


def plan_jobs(sizes, policy_seeds, epochs):
    """Every model job of a phase. ``sizes``: {"inner": [(U, P), ...], "outer": [...],
    "final": [...]} (one entry per fold); ``policy_seeds``: {"inner": n, "outer": n, "final": n}."""
    jobs = []
    for kind in ("inner", "outer", "final"):
        for units, predictions in sizes.get(kind, ()):
            jobs += [Job(kind, units, epochs, predictions, s) for s in range(policy_seeds[kind])]
    return jobs


def estimate(job, m, factors):
    """D̂ = c_S·s + c_T·t + c_P·q with factors {"S": {z: c}, "T": {z: c}, "P": c}."""
    s, t, q = split(job, m)
    return factors["S"][job.size_class] * s + factors["T"][job.size_class] * t + factors["P"] * q


def h_plan(durations, workers=WORKERS):
    durations = list(durations)
    return (sum(durations) + (workers - 1) * max(durations)) / workers


# -- scheduling simulations -----------------------------------------------------------------------


def list_schedule(durations, workers=WORKERS, longest_first=True):
    """Makespan of greedy list scheduling with fixed job durations."""
    order = sorted(durations, reverse=True) if longest_first else list(durations)
    free = [0.0] * workers
    for d in order:
        i = int(np.argmin(free))
        free[i] += d
    return max(free)


def contention_schedule(work, slowdown, workers=WORKERS):
    """Greedy list scheduling where each running job progresses at 1/slowdown(n) of its
    single-process speed while n jobs run. Returns (realised durations, makespan)."""
    queue = list(enumerate(work))
    running, remaining, started, finished, now = [], {}, {}, {}, 0.0
    while queue or running:
        while queue and len(running) < workers:
            i, w = queue.pop(0)
            running.append(i)
            remaining[i], started[i] = w, now
        rate = 1.0 / slowdown(len(running))
        step = min(remaining[i] for i in running) / rate
        now += step
        for i in list(running):
            remaining[i] -= step * rate
            if remaining[i] <= 1e-12:
                running.remove(i)
                finished[i] = now
    durations = [finished[i] - started[i] for i in range(len(work))]
    return durations, now


# -- throughput diagnostics and stops -------------------------------------------------------------


def window_mean(epochs, first, last):
    """Mean of 1-based epochs first ... last."""
    return float(np.mean(epochs[first - 1 : last]))


def throughput(single_epochs, concurrent_epochs):
    """tau_1, tau_4 (slowest process, epochs 11-30), phi = 4·tau_1/tau_4 and the drift per
    process, (mean of 21-30 - mean of 11-20)/mean of 11-20 in percent."""
    tau1 = window_mean(single_epochs, 11, 30)
    tau4 = max(window_mean(q, 11, 30) for q in concurrent_epochs)
    drift = [100 * (window_mean(q, 21, 30) - window_mean(q, 11, 20)) / window_mean(q, 11, 20)
             for q in concurrent_epochs]
    return {"tau1": tau1, "tau4": tau4, "phi": WORKERS * tau1 / tau4, "drift_percent": drift}


def cv(epochs):
    """Coefficient of variation of epochs 2 ... E (population standard deviation)."""
    rest = np.asarray(epochs[1:], dtype=float)
    return float(rest.std() / rest.mean())


def stops(ratios=(), cvs=(), drifts=(), realised=(), estimated=(), makespan=None, plan=None,
          cv_threshold=None, drift_threshold=None, per_count_limit=False):
    """Reasons (fixed texts) that make a measurement unresolved. ``ratios``: (slowdown,
    co-runners) pairs. Pending thresholds have no defaults: a check is applied only when its
    threshold is given."""
    out = []
    if any(r > WORKERS for r, _m in ratios):
        out.append("a slowdown exceeds 4: four at once would be slower than one at a time")
    if per_count_limit and any(r > m + 1 for r, m in ratios):
        out.append("a slowdown exceeds its co-runner count plus one")
    if cv_threshold is not None and any(c > cv_threshold for c in cvs):
        out.append("epoch times vary more than the coefficient-of-variation threshold")
    if drift_threshold is not None and any(d > drift_threshold for d in drifts):
        out.append("throughput is still falling (drift above threshold)")
    if any(d > e * (1 + 1e-9) for d, e in zip(realised, estimated, strict=True)):  # float tolerance only
        out.append("a job took longer than its estimate")
    if makespan is not None and plan is not None and makespan > plan * (1 + 1e-9):
        out.append("the schedule took longer than the planning estimate")
    return out


def thread_environment(threads=4):
    """The registered thread settings, for job environments."""
    env = dict(os.environ)
    env.update({k: str(threads) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")})
    return env
