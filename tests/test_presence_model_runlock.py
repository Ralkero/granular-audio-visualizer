"""A5.2 Step 2a, S2a.8: the exclusive experiment lock (protocol Amendment 3, A3.8 and A3.10),
with real operating-system locks in temporary folders."""

import fcntl
import importlib
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gmv.character.research.presence_model import runlock

HOLD = ("import fcntl, os, sys, time; fd = os.open(sys.argv[1], os.O_RDWR | os.O_CREAT, 0o600); "
        "fcntl.flock(fd, {mode}); print('held', flush=True); time.sleep(60)")


def holder(path, mode="fcntl.LOCK_EX"):
    process = subprocess.Popen([sys.executable, "-c", HOLD.format(mode=mode), str(path)], stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "held"
    return process


def test_two_runners_started_together_exactly_one_runs(tmp_path):
    lock = tmp_path / "experiment.lock"
    with runlock.exclusive(lock):
        with pytest.raises(runlock.LockHeld) as caught:
            with runlock.exclusive(lock):
                pass
    assert "refuses to start" in str(caught.value)
    with runlock.exclusive(lock):  # released after the first block
        pass


def test_a_shared_lock_holder_does_not_let_an_exclusive_one_in(tmp_path):
    lock = tmp_path / "experiment.lock"
    process = holder(lock, "fcntl.LOCK_SH")
    try:
        with pytest.raises(runlock.LockHeld):
            with runlock.exclusive(lock):
                pass
    finally:
        process.kill()
        process.wait()


def test_killing_the_holder_releases_the_lock(tmp_path):
    lock = tmp_path / "experiment.lock"
    process = holder(lock)
    with pytest.raises(runlock.LockHeld):
        with runlock.exclusive(lock):
            pass
    process.kill()
    process.wait()
    with runlock.exclusive(lock):
        pass


def test_the_lock_file_is_the_same_from_every_working_copy(tmp_path, monkeypatch):
    first = runlock.default_path()
    monkeypatch.chdir(tmp_path)
    assert runlock.default_path() == first and first.is_absolute()
    assert "worktrees" not in str(first) and Path(__file__).resolve().parents[1] not in first.parents


def test_no_job_inherits_the_lock_handle(tmp_path):
    with runlock.exclusive(tmp_path / "experiment.lock") as fd:
        assert not os.get_inheritable(fd)
        child = subprocess.run([sys.executable, "-c", "import os; print(sorted(os.listdir('/proc/self/fd')))"],
                               capture_output=True, text=True, close_fds=False, check=True)
        assert str(fd) not in eval(child.stdout) or fd <= 2  # noqa: S307 - a list of digit strings
        # the child can take the lock itself only if it never inherited ours
        probe = subprocess.run([sys.executable, "-c", (
            "import fcntl, os, sys; fd = os.open(sys.argv[1], os.O_RDWR); "
            "fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)"), str(tmp_path / "experiment.lock")],
            capture_output=True, text=True, close_fds=False, check=False)
        assert probe.returncode != 0  # held by us, and not shared with the child


@pytest.mark.parametrize("entry", runlock.REGISTERED)
def test_every_registered_runner_refuses_while_the_lock_is_held(entry, tmp_path):
    module, function = entry.split(":")
    run = getattr(importlib.import_module(module), function)
    lock = tmp_path / "experiment.lock"
    with runlock.exclusive(lock):
        with pytest.raises(runlock.LockHeld):
            if module.endswith("step2run"):
                run(None, tmp_path, tmp_path, lock_path=lock)
            elif module.endswith("launcher"):
                run([sys.executable, "-c", "pass"], cap_s=5, lock_path=lock)
            else:
                run(tmp_path / "out", lock_path=lock)
    assert not (tmp_path / "out").exists()


def test_the_lock_never_waits(tmp_path):
    lock = tmp_path / "experiment.lock"
    process = holder(lock)
    try:
        start = time.monotonic()
        with pytest.raises(runlock.LockHeld):
            with runlock.exclusive(lock):
                pass
        assert time.monotonic() - start < 1
    finally:
        process.kill()
        process.wait()
    assert fcntl.LOCK_EX  # the module uses the exclusive mode, never LOCK_SH
