"""Step 2a, S2a.7: probes that run **inside** the sandbox for the Step 2a checks of Amendment 3,
A3.4 and A3.10 (a full walk, writes, connections, ``/proc``, a shell, a crash, io_uring, handles,
being killed). They are used only by the synthetic tests and the RG1c review, never by a real
step. Each prints one JSON line on standard output and touches no project data.
"""

import ctypes
import json
import os
import socket
import subprocess
import sys
import time

from . import sandbox


def _emit(value):
    sys.stdout.write(json.dumps(value, sort_keys=True) + "\n")
    sys.stdout.flush()


def walk():
    """Everything reachable outside ``/proc``: the top level, and each small folder in full."""
    top = sorted(os.listdir("/"))
    listing = {}
    for folder in ("/etc", "/usr", "/usr/bin", "/gmv", "/gmv/site-packages", "/dev", "/out"):
        listing[folder] = sorted(os.listdir(folder)) if os.path.isdir(folder) else None
    inputs = sorted(os.path.join(d, f) for d, _sub, files in os.walk("/in") for f in files) if os.path.isdir("/in") else []
    _emit({"top": top, "listing": listing, "inputs": inputs})


def writes():
    results = {}
    for target in ("/x", "/etc/x", "/usr/lib/x", "/gmv/src/x", "/in/x", "/dev/x", "/tmp/x", "/proc/x", "/out/ok"):
        try:
            with open(target, "w", encoding="utf-8") as file:
                file.write("x")
            results[target] = "written"
        except OSError as error:
            results[target] = type(error).__name__
    _emit(results)


def connections():
    results = {}
    attempts = {
        "loopback": (socket.AF_INET, ("127.0.0.1", 9)),
        "outside": (socket.AF_INET, ("1.1.1.1", 53)),
        "host-socket-file": (socket.AF_UNIX, "/run/user/1000/bus"),
        "abstract-socket": (socket.AF_UNIX, "\0gmv-step2a-probe"),
    }
    for name, (family, address) in attempts.items():
        s = socket.socket(family, socket.SOCK_STREAM)
        s.settimeout(2)
        try:
            s.connect(address)
            results[name] = "connected"
        except OSError as error:
            results[name] = type(error).__name__
        finally:
            s.close()
    _emit(results)


def processes():
    _emit({"pids": sorted(int(p) for p in os.listdir("/proc") if p.isdigit())})


def shell():
    try:
        subprocess.run(["/bin/sh", "-c", "true"], check=False)
        _emit({"shell": "started"})
    except OSError as error:
        _emit({"shell": type(error).__name__})


def crash():
    sandbox.not_dumpable()
    os.abort()


def io_uring():
    params = ctypes.create_string_buffer(120)
    fd = ctypes.CDLL(None, use_errno=True).syscall(425, 1, params)  # io_uring_setup(entries, params)
    _emit({"io_uring_setup": "returned" if fd >= 0 else "failed"})


def handles():
    try:
        _emit({"handles": sandbox.check_handles()})
    except sandbox.SandboxError as error:
        _emit({"handles": "refused", "reason": str(error)})
        sys.exit(3)


def dumpable():
    sandbox.not_dumpable()
    _emit({"dumpable": ctypes.CDLL(None).prctl(sandbox.PR_GET_DUMPABLE, 0, 0, 0, 0)})


def heartbeat(seconds="60"):
    """Write a growing counter into /out every 0.1 s, so a test can see when the step stops."""
    _emit({"beating": True})
    end = time.monotonic() + float(seconds)
    count = 0
    while time.monotonic() < end:
        count += 1
        with open("/out/beat", "w", encoding="utf-8") as file:
            file.write(str(count))
        time.sleep(0.1)
    _emit({"finished": True})
