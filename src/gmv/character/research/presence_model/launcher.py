"""Step 4, S4.7: the experiment launcher for step 5's jobs and the guard that keeps synthetic
loads on generated data (protocol Amendment 3, A3.8, B9 and C1) *[tested before use]*. Linux only.
Built and tested on generated jobs only.

* **The lock:** the launcher takes the one exclusive experiment lock (``runlock``) outside any
  sandbox and holds it for the whole experiment; a second runner refuses at once. The lock's
  descriptor is close-on-exec, so no job inherits it.
* **Jobs die with the launcher:** each job is started in its own process group with the parent
  death signal set to SIGKILL and crash dumps disabled. A cap hit or the memory stop kills the
  job's whole group.
* **Process-time** is measured exec to exit on the monotonic clock (``timing``'s definition);
  CPU time is kept for information only.
* **Recorders:** ``recorders.Recorder`` samples the job's tree and the host once a second; the
  memory stop fires above 8 GiB (tests lower it).
* **Synthetic data only** (B9): a synthetic load's inputs must lie inside a root that
  ``make_synthetic_root`` created and marked, with no symbolic link leading out, and no part of
  any path may be one of the project's private folders. Anything else is refused before the job
  starts.
"""

import ctypes
import os
import resource
import secrets
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import recorders, runlock

PR_SET_PDEATHSIG = 1
MARKER = ".gmv-synthetic-data"
PRIVATE_FOLDERS = ("local-evidence", "corpus-data", "sonic-character")  # never part of a synthetic input


class SyntheticDataError(RuntimeError):
    """A synthetic load named an input that is not generated data; nothing was started."""


def make_synthetic_root(path):
    """Create a new, marked folder for generated inputs and return it."""
    root = Path(path)
    root.mkdir(parents=True)
    (root / MARKER).write_text(secrets.token_hex(16) + "\n", encoding="ascii")
    return root


def check_synthetic(inputs, root):
    """Refuse unless every input resolves inside the marked root and no path names a private folder."""
    root = Path(root)
    if not (root / MARKER).is_file():
        raise SyntheticDataError("the synthetic root is not a marked generated-data folder")
    real_root = root.resolve(strict=True)
    for part in (*real_root.parts, *root.absolute().parts):
        if part in PRIVATE_FOLDERS:
            raise SyntheticDataError("the synthetic root lies inside a private project folder")
    for item in inputs:
        path = Path(item)
        for part in path.absolute().parts:
            if part in PRIVATE_FOLDERS:
                raise SyntheticDataError("a synthetic input names a private project folder")
        try:
            real = path.resolve(strict=True)
        except OSError:
            raise SyntheticDataError("a synthetic input does not exist") from None
        if real_root not in (real, *real.parents):
            raise SyntheticDataError("a synthetic input lies outside the generated-data folder")


def _child_setup():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    ctypes.CDLL(None, use_errno=True).prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


@dataclass
class Outcome:
    status: str  # "completed", "failed", "cap", "memory"
    returncode: int | None
    elapsed_s: float
    cpu_s: float | None
    peak_bytes: int
    disturbed: bool
    reasons: tuple
    samples: list = field(default_factory=list)  # private: aggregate counters only


def _kill_group(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_job(command, cap_s, env=None, memory_limit=recorders.MEMORY_LIMIT, interval=1.0, synthetic_root=None,
            inputs=(), lock_path=None, hold_lock=True):
    """Run one job under the lock with the recorders. ``synthetic_root`` and ``inputs``: a synthetic
    load's declared inputs, checked before anything starts. ``hold_lock=False`` is for a job that
    runs inside an experiment whose launcher already holds the lock."""
    if synthetic_root is not None:
        check_synthetic(inputs, synthetic_root)
    if hold_lock:
        with runlock.exclusive(lock_path):
            return _run(command, cap_s, env, memory_limit, interval)
    return _run(command, cap_s, env, memory_limit, interval)


def _run(command, cap_s, env, memory_limit, interval):
    exec_ns = time.monotonic_ns()
    process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, close_fds=True, preexec_fn=_child_setup,
                               start_new_session=True)
    state = {"memory": False}

    def on_limit():
        state["memory"] = True
        _kill_group(process)

    recorder = recorders.Recorder(process.pid, interval, memory_limit, on_limit)
    recorder.start()
    status, cpu_s = None, None
    try:
        _pid, wait_status, usage = _wait(process, cap_s)
        if wait_status is None:
            status = "cap"
            _kill_group(process)
            _pid, wait_status, usage = os.wait4(process.pid, 0)
        process.returncode = os.waitstatus_to_exitcode(wait_status)
        cpu_s = usage.ru_utime + usage.ru_stime
    finally:
        exit_ns = time.monotonic_ns()
        recorder.stop()
        _kill_group(process)  # nothing of the job outlives it
    if state["memory"]:
        status = "memory"
    elif status is None:
        status = "completed" if process.returncode == 0 else "failed"
    disturbed, reasons = recorders.disturbed(recorder.samples)
    return Outcome(status, process.returncode, (exit_ns - exec_ns) / 1e9, cpu_s, recorder.peak_bytes, disturbed,
                   reasons, recorder.samples)


def _wait(process, cap_s):
    """wait4 with a deadline: (pid, status, usage), or (None, None, None) at the cap."""
    deadline = time.monotonic() + cap_s
    while True:
        pid, status, usage = os.wait4(process.pid, os.WNOHANG)
        if pid:
            return pid, status, usage
        if time.monotonic() >= deadline:
            return None, None, None
        time.sleep(0.02)
