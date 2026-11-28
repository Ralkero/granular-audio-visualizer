"""Step 2a, S2a.7: the bubblewrap sandbox and the traced launch of a real-data step (protocol
Amendment 3, A3.4), **the proposed configuration for synthetic Step 2a testing, not a proven
security boundary.** Host paths are given by the caller (the run declaration) and appear only in
private run records; the sandbox sees fixed paths.

Inside: an empty read-only root; the inputs read-only under ``/in``; GMV's ``src`` at ``/gmv/src``
and seven packages at ``/gmv/site-packages``; the interpreter, ``/usr/lib`` (with ``/lib`` and
``/lib64`` links) and the loader cache; ``/out`` the only writable path; a new ``/proc`` and
bubblewrap's minimal ``/dev``; no ``/sys``, ``/tmp``, ``/home``, ``/root``, ``/run``, ``/var``,
``/usr/share`` or the rest of ``/etc`` and ``/usr/bin``. Separate user, process, IPC, network,
hostname and cgroup namespaces, loopback only, no nested namespaces, no capabilities, the
environment cleared except ``OPENBLAS_NUM_THREADS=1``.

Outside: the launcher passes exactly three handles (input from ``/dev/null``; output and errors
to private pipes), sets the crash-dump limit to zero, makes strace and the sandbox die with it,
and runs strace outside the sandbox. ``check_handles`` and ``not_dumpable`` run first inside.
"""

import ctypes
import os
import resource
import signal
import subprocess
from dataclasses import dataclass
from pathlib import Path

PACKAGES = ("numpy", "numpy.libs", "pydantic", "pydantic_core", "annotated_types", "typing_inspection",
            "typing_extensions.py")
INTERPRETER = "/usr/bin/python3.14"
STRACE_CALLS = "%file,%process,%network,getdents64,fchdir,io_uring_setup"
PR_SET_PDEATHSIG, PR_SET_DUMPABLE, PR_GET_DUMPABLE = 1, 4, 3


class SandboxError(RuntimeError):
    """The sandbox could not be set up or a check inside it failed; the step stops."""


@dataclass(frozen=True)
class Layout:
    """Host paths for one sandboxed run (private). ``inputs``: sandbox path under ``/in`` ->
    host file; ``out``: the host staging folder (created empty here); ``src``: GMV's source
    folder; ``site_packages``: the environment's site-packages folder."""

    inputs: dict
    out: Path
    src: Path
    site_packages: Path


def bwrap_argv(layout, startup):
    for sandbox_path in layout.inputs:
        if not (sandbox_path.startswith("/in/") and ".." not in sandbox_path.split("/")):
            raise SandboxError("an input's sandbox path must lie under /in")
    argv = ["bwrap", "--unshare-user", "--unshare-pid", "--unshare-ipc", "--unshare-net", "--unshare-uts",
            "--unshare-cgroup", "--disable-userns", "--die-with-parent", "--new-session", "--cap-drop", "ALL",
            "--clearenv", "--setenv", "OPENBLAS_NUM_THREADS", "1",
            "--tmpfs", "/", "--proc", "/proc", "--dev", "/dev",
            "--ro-bind", "/usr/lib", "/usr/lib", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib", "/lib64",
            "--ro-bind", INTERPRETER, INTERPRETER, "--ro-bind", "/etc/ld.so.cache", "/etc/ld.so.cache",
            "--ro-bind", str(layout.src), "/gmv/src"]
    for name in PACKAGES:
        argv += ["--ro-bind", str(Path(layout.site_packages) / name), f"/gmv/site-packages/{name}"]
    for sandbox_path, host in sorted(layout.inputs.items()):
        argv += ["--ro-bind", str(host), sandbox_path]
    argv += ["--bind", str(layout.out), "/out", "--remount-ro", "/", "--remount-ro", "/dev", "--chdir", "/",
             "--", INTERPRETER, "-I", "-S", "-B", "-X", "utf8", "-c", startup]
    return argv


def startup(module, function, *arguments):
    """The fixed start-up line: no automatic import paths, then one registered entry point."""
    for part in (module, function, *arguments):
        if not all(c.isalnum() or c in "._-/" for c in part):
            raise SandboxError("a start-up name holds characters outside the registered set")
    args = ", ".join(repr(a) for a in arguments)
    return (f"import sys; sys.path[:0] = ['/gmv/src', '/gmv/site-packages']; "
            f"import {module} as m; m.{function}({args})")


def _child_setup():
    """In the launcher's child before exec (strace): no crash dumps; die with the launcher."""
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    libc = ctypes.CDLL(None, use_errno=True)
    libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)


@dataclass(frozen=True)
class Launch:
    strace_exit: int
    stdout: bytes  # private: never printed or published
    stderr: bytes  # private
    trace: Path


def prepare(layout, entry, trace_path):
    """Create the staging folder (refusing an existing one) and return the full traced command."""
    out = Path(layout.out)
    if out.exists():
        raise SandboxError("the staging folder already exists")
    out.mkdir(parents=True)
    os.chmod(out, 0o700)
    trace_path = Path(trace_path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    return ["strace", "-f", "--kill-on-exit", "--seccomp-bpf", "-s", "0", "-e", f"trace={STRACE_CALLS}",
            "-o", str(trace_path), "--", *bwrap_argv(layout, entry)]


def popen(argv, extra_fds=()):
    """Start a prepared command with exactly three handles, no crash dumps, dying with us."""
    return subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            close_fds=True, pass_fds=tuple(extra_fds), preexec_fn=_child_setup,
                            env={"PATH": "/usr/bin"})


def launch(layout, entry, trace_path, timeout, extra_fds=()):
    """Run ``entry`` (a start-up line) traced and sandboxed and return the private ``Launch``.
    ``extra_fds`` exists only for the test that shows a planted handle stops the step."""
    argv = prepare(layout, entry, trace_path)
    process = popen(argv, extra_fds)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
    if Path(trace_path).exists():
        os.chmod(trace_path, 0o600)
    return Launch(process.returncode, stdout, stderr, Path(trace_path))


# -- inside the sandbox -------------------------------------------------------------------------------------


def check_handles(require_pipes=True):
    """The step's first action: exactly descriptors 0, 1 and 2 may be open, input from
    ``/dev/null`` and, under the launcher, output and errors to pipes. The listing's own
    descriptor (closed by the time it is examined) is recognised and ignored. Anything else stops
    the step."""
    found = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            found[int(name)] = os.readlink(f"/proc/self/fd/{name}")
        except FileNotFoundError:
            continue  # the listing's own descriptor, already closed
    extra = set(found) - {0, 1, 2}
    if extra:
        raise SandboxError(f"{len(extra)} unexpected handle(s) are open")
    if found.get(0) != "/dev/null":
        raise SandboxError("standard input is not /dev/null")
    if require_pipes and not all(str(found.get(fd, "")).startswith("pipe:") for fd in (1, 2)):
        raise SandboxError("standard output and errors are not the launcher's pipes")
    return sorted(found)


def not_dumpable():
    """Mark the step never to be crash-dumped (the dump limit alone may not stop a piped
    core_pattern such as systemd-coredump)."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0 or libc.prctl(PR_GET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise SandboxError("the step could not be marked non-dumpable")
