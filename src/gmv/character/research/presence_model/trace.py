"""Step 2a, S2a.7: the live-trace reader for sandboxed real-data steps (protocol Amendment 3,
A3.4). It reads one strace recording of ``strace -f --kill-on-exit --seccomp-bpf -s 0 -e
trace=%file,%process,%network,getdents64,fchdir,io_uring_setup -o TRACE -- bwrap …`` and decides
whether the run may be accepted.

* **Set-up is kept apart.** Every call before the step's interpreter starts (bubblewrap building
  the sandbox from host paths) is counted as set-up and not checked against the step's
  allow-list. The step is the process that executes the sandbox's interpreter, and any process
  it starts.
* **Deny by default.** Each path the step touches, successfully or not, must fall in an allowed
  class: the runtime (``/gmv/src``, ``/gmv/site-packages``, ``/usr/lib``, the interpreter, the
  loader cache), the step's own inputs (exact files under ``/in``), the staging folder ``/out``,
  the step's own ``/proc`` entries and ``/dev/null``, or the registered **baseline**: the
  interpreter's start-up probes of files the sandbox does not contain, recorded from synthetic
  runs and reviewed. Anything else is a violation, including listing an input folder.
* **No escape routes.** A new program (``execve``), a child process (a clone without
  ``CLONE_THREAD``), any network call, a change of directory, and any ``io_uring_setup`` are
  violations.
* **Complete or rejected.** strace must exit 0; every traced process must end with an exit line;
  every line must parse; every path must resolve. An unresolved path is a violation.

Public: counts by class and kind, the number of unresolved records and the verdict. The raw trace
holds paths and stays private.
"""

import re
from collections import Counter
from dataclasses import dataclass, field

RUNTIME = ("/gmv/src", "/gmv/site-packages", "/usr/lib")
RUNTIME_FILES = ("/usr/bin/python3.14", "/etc/ld.so.cache")
BASELINE = frozenset({
    "/", "/usr", "/usr/bin", "/usr/pyvenv.cfg", "/usr/bin/pyvenv.cfg", "/usr/bin/python3.14._pth",
    "/usr/bin/pybuilddir.txt", "/usr/bin/lib/python314.zip", "/usr/bin/lib/python3.14/os.py",
    "/usr/bin/lib/python3.14/os.pyc", "/usr/bin/Modules/Setup", "/usr/bin/Modules/Setup.local",
    "/usr/share/locale/locale.alias", "/etc/ld.so.preload", "/etc/localtime", "/etc/nsswitch.conf",
    "/etc/passwd", "/etc/ssl/openssl.cnf", "/selinux", "/sys/devices/system/cpu/possible",
    "/sys/fs/cgroup/cgroup.controllers", "/proc/stat", "/proc/mounts", "/proc/sys/vm/overcommit_memory",
})
NETWORK = {"socket", "socketpair", "connect", "bind", "listen", "accept", "accept4", "sendto", "sendmsg",
           "sendmmsg", "recvfrom", "recvmsg", "recvmmsg", "getsockname", "getpeername", "setsockopt",
           "getsockopt", "shutdown"}
LAUNCH = {"execve", "execveat"}
CLONES = {"clone", "clone3", "fork", "vfork"}
MOVES = {"chdir", "fchdir", "chroot", "pivot_root"}
LISTING = {"getdents64", "getdents"}
DIRFD_FIRST = {"openat", "openat2", "newfstatat", "fstatat64", "statx", "faccessat", "faccessat2", "readlinkat",
               "mkdirat", "unlinkat", "renameat", "renameat2", "linkat", "symlinkat", "fchmodat", "fchownat",
               "utimensat", "mknodat", "name_to_handle_at", "futimesat"}
PATH_FIRST = {"open", "creat", "stat", "lstat", "stat64", "lstat64", "access", "readlink", "statfs", "truncate",
              "mkdir", "rmdir", "unlink", "rename", "link", "symlink", "chmod", "chown", "lchown", "utime",
              "utimes", "getxattr", "lgetxattr", "listxattr", "llistxattr", "setxattr", "lsetxattr",
              "removexattr", "lremovexattr", "mknod", "statmount", "uselib", "swapon", "swapoff", "acct"}
WRITES = re.compile(r"O_WRONLY|O_RDWR|O_CREAT|O_TRUNC")

_LINE = re.compile(r"^(\d+) +(.*)$", re.S)
_RESUMED = re.compile(r"^<\.\.\. ([A-Za-z0-9_]+) resumed>(.*)$", re.S)
_CALL = re.compile(r"^([A-Za-z0-9_]+)\((.*)\) += (.*)$", re.S)
_EXIT = re.compile(r"^\+\+\+ (exited with -?\d+|killed by SIG[A-Z0-9]+( \(core dumped\))?) \+\+\+$")
_SIGNAL = re.compile(r"^--- SIG[A-Z0-9]+ .* ---$|^--- stopped by SIG[A-Z0-9]+ ---$", re.S)
_QUOTED = re.compile(r'"((?:[^"\\]|\\.)*)"')


@dataclass
class Verdict:
    """The reader's decision. ``violations`` and ``counts`` hold classes and kinds only."""

    accepted: bool
    complete: bool
    violations: Counter = field(default_factory=Counter)
    counts: Counter = field(default_factory=Counter)
    unresolved: int = 0
    core_dumped: bool = False
    problems: list = field(default_factory=list)

    def public(self):
        return {"verdict": "accepted" if self.accepted else "rejected", "complete": self.complete,
                "violations": dict(sorted(self.violations.items())), "counts": dict(sorted(self.counts.items())),
                "unresolved": self.unresolved}


def _unescape(text):
    return bytes(text, "utf-8").decode("unicode_escape").encode("latin-1").decode("utf-8", "replace")


def _split_args(args):
    """Top-level comma-separated arguments (quotes, brackets and braces respected)."""
    out, depth, quote, start, i = [], 0, False, 0, 0
    while i < len(args):
        c = args[i]
        if quote:
            if c == "\\":
                i += 1
            elif c == '"':
                quote = False
        elif c == '"':
            quote = True
        elif c in "([{":
            depth += 1
        elif c in ")]}":
            depth -= 1
        elif c == "," and depth == 0:
            out.append(args[start:i].strip())
            start = i + 1
        i += 1
    out.append(args[start:].strip())
    return out


def _path(token):
    m = re.fullmatch(r'"((?:[^"\\]|\\.)*)"(\.\.\.)?', token or "")
    if not m or m.group(2):
        return None  # not a path string, or truncated
    return _unescape(m.group(1))


def _join(base, path):
    if path.startswith("/"):
        full = path
    else:
        full = base.rstrip("/") + "/" + path
    parts = []
    for part in full.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
        else:
            parts.append(part)
    return "/" + "/".join(parts)


def _under(path, roots):
    return any(path == r or path.startswith(r + "/") for r in roots)


class _Reader:
    def __init__(self, inputs, initial_cwd="/"):
        self.inputs = frozenset(inputs)
        self.cwd = initial_cwd
        self.v = Verdict(False, False)
        self.step = set()  # pids belonging to the step
        self.fds = {}  # (pid, fd) -> path
        self.pending = {}  # pid -> unfinished call text
        self.exited = set()
        self.seen = set()

    def classify(self, path, kind, flags=""):
        """The class of a path the step touched, or a violation."""
        if _under(path, RUNTIME) or path in RUNTIME_FILES:
            return "runtime"
        if path in self.inputs:
            return "input" if kind != "list" else None
        if _under(path, ("/in",)):
            return None  # any other path under /in, or a listing of it
        if _under(path, ("/out",)):
            return "out"
        if path == "/dev/null":
            return "dev"
        if re.fullmatch(r"/proc/(self|\d+)(/.*)?", path):
            return "proc-self"
        if path in BASELINE and kind != "write":
            return "baseline"
        return None

    def touch(self, path, kind, flags=""):
        if path is None:
            self.v.unresolved += 1
            self.v.violations["unresolved path"] += 1
            return
        if kind == "open" and WRITES.search(flags):
            kind = "write"
        cls = self.classify(path, kind, flags)
        if cls is None or (kind == "write" and cls not in ("out",)):
            self.v.violations[f"{kind} outside the allowed classes"] += 1
        else:
            self.v.counts[f"{cls}:{kind}"] += 1

    def call(self, pid, name, args, result):
        if pid not in self.step:
            if name in LAUNCH and result.startswith("0") and "python3.14" in args:
                self.step.add(pid)  # the sandbox's interpreter: the step starts here
                self.v.counts["step:start"] += 1
            else:
                self.v.counts["set-up"] += 1
            return
        argv = _split_args(args)
        ok = not result.startswith("-1")
        if name in LAUNCH:
            self.v.violations["a new program"] += 1
        elif name in CLONES:
            if "CLONE_THREAD" in args:
                self.v.counts["thread"] += 1
                child = re.match(r"(\d+)", result)
                if child:
                    self.step.add(int(child.group(1)))
            else:
                self.v.violations["a child process"] += 1
                child = re.match(r"(\d+)", result)
                if child:
                    self.step.add(int(child.group(1)))
        elif name in NETWORK:
            self.v.violations["a network call"] += 1
        elif name in MOVES:
            self.v.violations["a change of directory or root"] += 1
        elif name == "io_uring_setup":
            self.v.violations["io_uring"] += 1
        elif name in LISTING:
            fd = re.match(r"(\d+)", argv[0] or "")
            path = self.fds.get((pid, int(fd.group(1)))) if fd else None
            self.touch(path, "list")
        elif name in DIRFD_FIRST:
            base_token, path_token = argv[0], argv[1] if len(argv) > 1 else None
            path = _path(path_token)
            if base_token == "AT_FDCWD":
                base = self.cwd
            elif re.fullmatch(r"\d+", base_token or ""):
                base = self.fds.get((pid, int(base_token)))
            else:
                base = None
            if path == "" and base is not None:  # AT_EMPTY_PATH: the descriptor itself
                full = base
            elif path is not None and (path.startswith("/") or base is not None):
                full = _join(base or "/", path)
            else:
                full = None
            kind = "open" if name.startswith("open") else ("stat" if "stat" in name or "access" in name else "other")
            self.touch(full, kind, argv[2] if len(argv) > 2 else "")
            if name.startswith("open") and ok and full is not None:
                fd = re.match(r"(\d+)", result)
                if fd:
                    self.fds[(pid, int(fd.group(1)))] = full
        elif name in PATH_FIRST:
            path = _path(argv[0])
            full = _join(self.cwd, path) if path is not None else None
            kind = "open" if name in ("open", "creat") else ("stat" if "stat" in name or name == "access" else "other")
            self.touch(full, kind if name != "creat" else "write", argv[1] if len(argv) > 1 else "")
            if name in ("open", "creat") and ok and full is not None:
                fd = re.match(r"(\d+)", result)
                if fd:
                    self.fds[(pid, int(fd.group(1)))] = full
        elif name in ("exit", "exit_group", "wait4", "waitid", "kill", "tkill", "tgkill", "arch_prctl", "prctl"):
            self.v.counts[f"process:{name}"] += 1
        else:
            self.v.counts[f"other:{name}"] += 1

    def line(self, text):
        m = _LINE.match(text)
        if not m:
            self.v.problems.append("an unparsable line")
            return
        pid, body = int(m.group(1)), m.group(2).rstrip("\n")
        self.seen.add(pid)
        if _EXIT.match(body):
            self.exited.add(pid)
            if "core dumped" in body:
                self.v.core_dumped = True
            return
        if _SIGNAL.match(body):
            return
        if body.endswith(" <unfinished ...>"):
            self.pending[pid] = body[: -len(" <unfinished ...>")]
            return
        resumed = _RESUMED.match(body)
        if resumed:
            start = self.pending.pop(pid, None)
            if start is None or not start.startswith(resumed.group(1) + "("):
                self.v.problems.append("a resumed call without its start")
                return
            body = start + resumed.group(2)
        call = _CALL.match(body)
        if not call:
            self.v.problems.append("an unparsable call")
            return
        self.call(pid, call.group(1), call.group(2), call.group(3))


def read(text, inputs, strace_exit, initial_cwd="/", allowed_exits=(0,)):
    """The verdict on one trace. ``inputs``: the exact sandbox paths of the step's inputs;
    ``strace_exit``: strace's own exit status (the step's, under ``strace``); ``allowed_exits``:
    the step's registered exit codes (0, or a registered stop's code)."""
    reader = _Reader(inputs, initial_cwd)
    for line in text.splitlines():
        if line.strip():
            reader.line(line)
    v = reader.v
    if strace_exit not in allowed_exits:
        v.problems.append("strace did not exit with a registered code")
    if reader.seen - reader.exited:
        v.problems.append("a traced process has no exit line")
    if reader.pending:
        v.problems.append("a call never resumed")
    if not reader.step:
        v.problems.append("the step's interpreter never started")
    v.complete = not v.problems
    v.accepted = v.complete and not v.violations and v.unresolved == 0
    return v
