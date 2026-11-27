"""The A5.2 spy-test harness: run a target under three recorders, turn every record into
accesses, classify them against the synthetic world, and produce a sanitised report (counts
and evidence hashes only).

**Recorders, kept apart.** ``strace`` (layer 1) records the system calls of the target and
of every process it starts (``strace -f -y``); ``audit-hook`` (layer 2) and ``stat-probes``
(layer 3) are the in-process recorders of ``privacy_bootstrap``. Each writes its own evidence
file, and every detection carries the name of the recorder that made it. ``require`` checks
one recorder at a time: a detection by one recorder never satisfies a requirement placed on
another, and a non-empty log proves nothing unless the required access was classified in it.

**Accesses.** Each record becomes (recorder, origin, kind, path). ``origin`` is ``main`` for
the target process and its threads and ``child`` for processes it starts. ``kind`` is one of
``KINDS``. Relative paths are resolved against the working directory of the process that made
the call, at the time of the call: recorded with each event (layers 2 and 3), or followed
through ``chdir``, ``fchdir`` and process creation in the trace (layer 1), where a new process
copies its parent's directory and a thread shares it. Paths are compared after
``os.path.realpath`` (``..``, ``.``, repeated separators and symbolic links).

**Fail closed.** A record whose target cannot be established becomes an ``unresolved``
access, never dropped. Examples: a bare file descriptor, a ``dir_fd``, a process whose
directory is unknown, a truncated string, a shell command line, an unparsable trace line or an
unrecordable event. ``unresolved`` is not an allowed category, so it is a violation.

A listing of a world folder is ``directory-listing``, also a violation: it reveals the names of
Tier 3 items. Paths outside the world are not project data and are not counted. The target
runs with ``python -B``, so writing bytecode caches (which reopens files by descriptor) does not
appear as an unresolved access. Raw evidence stays in a private folder inside pytest's
temporary directory (mode 0700) and is never printed.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

BOOTSTRAP = Path(__file__).with_name("privacy_bootstrap.py")
RECORDERS = ("strace", "audit-hook", "stat-probes")
KINDS = ("open", "stat", "list", "chdir", "launch", "exec", "load", "copy", "other")
ALLOWED_STEP3 = {"development-manifest", "development-history", "tier1:validation-manifest",
                 "tier1:registry", "tier1:final-test-manifest", "tier1:fingerprints", "directory",
                 "paired-audio"}
STRACE_SYSCALLS = "%file,%process,getdents64,fchdir"
WILDCARDS = re.compile(r"[*?\[]")


class RecorderRequirement(AssertionError):
    """A required recorder did not run, or did not detect the required access (fixed texts)."""


@dataclass(frozen=True)
class Access:
    recorder: str
    origin: str
    kind: str
    path: str | None  # None: unresolved
    pattern: bool = False


def strace_available():
    return shutil.which("strace") is not None


def _join(base, path):
    if os.path.isabs(path):
        return path
    return None if base is None else os.path.join(base, path)


# -- layers 2 and 3: the in-process recorders ---------------------------------------------------


def _text(value):
    return value["s"] if isinstance(value, dict) and isinstance(value.get("s"), str) else None


def _items(value):
    return value["l"] if isinstance(value, dict) and isinstance(value.get("l"), list) else None


def _one(recorder, kind, value, base, dir_fd=False, pattern=False):
    text = _text(value)
    if text is None:  # a descriptor, an unknown type or a missing value: the target is unknown
        return Access(recorder, "main", kind, None)
    if dir_fd and not os.path.isabs(text):
        return Access(recorder, "main", kind, None)
    return Access(recorder, "main", kind, _join(base, text), pattern)


def _argv(recorder, value, base):
    """Program arguments: each is treated as a possible path. A command line that would need a
    shell to split it cannot be classified."""
    items = _items(value)
    if items is None:
        items = [value]
    out = []
    for item in items:
        text = _text(item)
        if text is None or re.search(r"\s", text):
            out.append(Access(recorder, "main", "launch", None))
        else:
            out.append(Access(recorder, "main", "launch", _join(base, text)))
    return out


def _python_event(recorder, event, args, cwd, dir_fd):
    def arg(i):
        return args[i] if i < len(args) else None

    if event == "open":
        return [_one(recorder, "open", arg(0), cwd)]
    if event in ("os.stat", "os.lstat", "os.access"):
        return [_one(recorder, "stat", arg(0), cwd, dir_fd=dir_fd)]
    if event in ("os.listdir", "os.scandir"):
        return [_one(recorder, "list", {"s": "."} if arg(0) is None else arg(0), cwd)]
    if event == "glob.glob":
        return [_one(recorder, "list", arg(0), cwd, pattern=True)]
    if event == "glob.glob/2":
        if arg(3) is not None:
            return [Access(recorder, "main", "list", None)]
        root = _text(arg(2))
        return [_one(recorder, "list", arg(0), cwd if root is None else _join(cwd, root), pattern=True)]
    if event == "os.chdir":
        return [_one(recorder, "chdir", arg(0), cwd)]
    if event == "subprocess.Popen":
        popen_cwd = _text(arg(2))
        base = cwd if popen_cwd is None else _join(cwd, popen_cwd)
        out = [] if arg(0) is None else [_one(recorder, "launch", arg(0), base)]
        return out + _argv(recorder, arg(1), base)
    if event == "os.system":
        return [Access(recorder, "main", "launch", None)]
    if event in ("os.posix_spawn",) or event.startswith("os.exec"):
        return [_one(recorder, "launch", arg(0), cwd), *_argv(recorder, arg(1), cwd)]
    if event.startswith("os.spawn"):
        return [_one(recorder, "launch", arg(1), cwd), *_argv(recorder, arg(2), cwd)]
    if event == "ctypes.dlopen":
        name = _text(arg(0))
        if arg(0) is None or (name is not None and "/" not in name):
            return []  # the main program, or a library found by the loader's own search
        return [_one(recorder, "load", arg(0), cwd)]
    if event == "shutil.copyfile":
        return [_one(recorder, "copy", arg(0), cwd), _one(recorder, "copy", arg(1), cwd)]
    return [Access(recorder, "main", "other", None)]


def python_accesses(lines, recorder):
    out = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            event, args = record["event"], record.get("args", [])
        except (ValueError, KeyError, TypeError):
            out.append(Access(recorder, "main", "other", None))
            continue
        if event == "unrecordable" or not isinstance(args, list):
            out.append(Access(recorder, "main", "other", None))
            continue
        out += _python_event(recorder, event, args, record.get("cwd"), bool(record.get("dir_fd")))
    return out


# -- layer 1: strace ----------------------------------------------------------------------------

_PID_LINE = re.compile(r"^(\d+) +(.*)$", re.S)
_RESUMED = re.compile(r"^<\.\.\. ([A-Za-z0-9_]+) resumed>(.*)$", re.S)
_UNFINISHED = " <unfinished ...>"
_NOTICE = re.compile(r"^(---|\+\+\+) .* (---|\+\+\+)$", re.S)
_QUOTED = re.compile(r'^"((?:[^"\\]|\\.)*)"(\.\.\.)?$', re.S)
_DESCRIPTOR = re.compile(r"^(AT_FDCWD|-?\d+)(?:<(.*)>)?$", re.S)
_RETURN = re.compile(r"^(-?\d+|\?)(?:<(.*)>)?(?:\s.*)?$", re.S)
_OPEN = {"open", "openat", "openat2", "creat", "open_by_handle_at"}
_STAT = {"stat", "lstat", "stat64", "lstat64", "newfstatat", "fstatat64", "statx", "access", "faccessat",
         "faccessat2", "readlink", "readlinkat", "statfs", "statfs64", "getxattr", "lgetxattr", "listxattr",
         "llistxattr"}
_LIST = {"getdents64", "getdents"}
_EXEC = {"execve": 0, "execveat": 1}
_CLONE = {"clone", "clone3", "fork", "vfork"}
_DIRFD = {"openat", "openat2", "newfstatat", "fstatat64", "statx", "faccessat", "faccessat2", "readlinkat",
          "execveat", "mkdirat", "mknodat", "unlinkat", "renameat", "renameat2", "linkat", "symlinkat",
          "fchmodat", "fchmodat2", "fchownat", "futimesat", "utimensat", "name_to_handle_at", "open_tree"}
_ESCAPES = {"n": 10, "t": 9, "r": 13, "v": 11, "f": 12, "a": 7, "b": 8, "e": 27, "\\": 92, '"': 34}


def unescape(body):
    """strace's C-style string escapes to text (file-system encoding); raises ValueError."""
    out, i = bytearray(), 0
    while i < len(body):
        c = body[i]
        if c != "\\":
            out += c.encode("utf-8", "surrogateescape")
            i += 1
            continue
        n = body[i + 1] if i + 1 < len(body) else ""
        if n in _ESCAPES:
            out.append(_ESCAPES[n])
            i += 2
        elif n == "x":
            out.append(int(body[i + 2 : i + 4], 16))
            i += 4
        elif n in "01234567" and n:
            j = i + 1
            while j < len(body) and j < i + 4 and body[j] in "01234567":
                j += 1
            out.append(int(body[i + 1 : j], 8) & 0xFF)
            i = j
        else:
            raise ValueError("unknown escape")
    return os.fsdecode(bytes(out))


def split_call(text):
    """``name(arg, ...) = result`` -> (name, [args], result), aware of quotes, brackets and
    descriptor decorations; None when the text is not one complete call."""
    m = re.match(r"^([A-Za-z0-9_]+)\(", text)
    if not m:
        return None
    name, i, depth, args, current = m.group(1), m.end(), 0, [], []
    while i < len(text):
        ch = text[i]
        if ch == '"':
            j = i + 1
            while j < len(text) and text[j] != '"':
                j += 2 if text[j] == "\\" else 1
            if j >= len(text):
                return None
            current.append(text[i : j + 1])
            i = j + 1
            if text.startswith("...", i):
                current.append("...")
                i += 3
            continue
        if ch == "<":
            j = text.find(">", i)
            if j < 0:
                return None
            current.append(text[i : j + 1])
            i = j + 1
            continue
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            if depth == 0:
                if ch != ")":
                    return None
                args.append("".join(current).strip())
                rest = text[i + 1 :].strip()
                if not rest.startswith("="):
                    return None
                return name, ([] if args == [""] else args), rest[1:].strip()
            depth -= 1
        elif ch == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
            i += 1
            continue
        current.append(ch)
        i += 1
    return None


def _calls(text):
    """(start line, pid, name, args, result) per system call, in start order. ``name`` is None
    for a line that could not be read; ``result`` is ``?`` for a call that never returned."""
    pending, calls = {}, []
    for index, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        m = _PID_LINE.match(line)
        if not m:
            calls.append((index, None, None, None, None))
            continue
        pid, body = int(m.group(1)), m.group(2)
        if _NOTICE.match(body):
            continue
        start = index
        resumed = _RESUMED.match(body)
        if resumed:
            if pid not in pending:
                calls.append((index, pid, None, None, None))
                continue
            start, head = pending.pop(pid)
            body = head + resumed.group(2)
        if body.endswith(_UNFINISHED):
            pending[pid] = (start, body[: -len(_UNFINISHED)])
            continue
        parsed = split_call(body)
        calls.append((start, pid, *parsed) if parsed else (start, pid, None, None, None))
    for pid, (start, head) in pending.items():
        parsed = split_call(head.rstrip().rstrip(",") + ") = ?")
        calls.append((start, pid, *parsed) if parsed else (start, pid, None, None, None))
    return sorted(calls, key=lambda c: c[0])


def _decorated(token):
    """(descriptor, path or None) for ``AT_FDCWD``, ``AT_FDCWD</p>``, ``3`` or ``3</p>``; None
    when the token is not a descriptor."""
    m = _DESCRIPTOR.match(token)
    if not m:
        return None
    path = None
    if m.group(2) is not None:
        try:
            path = unescape(m.group(2)).removesuffix(" (deleted)")
        except ValueError:
            path = None
        if path is not None and not path.startswith("/"):
            path = None  # a socket, pipe or other non-file descriptor
    return m.group(1), path


def _quoted(token):
    """(text, truncated) for a quoted string argument, else None."""
    m = _QUOTED.match(token)
    if not m:
        return None
    try:
        return unescape(m.group(1)), m.group(2) is not None
    except ValueError:
        return None, True


def _kind(name):
    if name in _OPEN:
        return "open"
    if name in _STAT:
        return "stat"
    if name in _LIST:
        return "list"
    if name in ("chdir", "fchdir"):
        return "chdir"
    if name in _EXEC:
        return "exec"
    return "other"


def _path_arguments(name, args, cwd):
    """Resolved paths (None when unresolved) of a call's path arguments."""
    positions = [_EXEC[name]] if name in _EXEC else range(len(args))
    out = []
    for i in positions:
        if i >= len(args):
            continue
        quoted = _quoted(args[i])
        if quoted is None:
            continue
        text, truncated = quoted
        if text is None or truncated:
            out.append(None)
            continue
        base = cwd
        if name in _DIRFD and i > 0:
            descriptor = _decorated(args[i - 1])
            if descriptor is not None:
                fd, path = descriptor
                base = path if path is not None else (cwd if fd == "AT_FDCWD" else None)
        out.append(base if text == "" else _join(base, text))
    return out


def strace_accesses(text, initial_cwd):
    calls = _calls(text)
    births = {}
    for start, pid, name, args, result in calls:
        if name in _CLONE and result:
            m = re.match(r"^(\d+)", result)
            if m and int(m.group(1)) > 0:
                flags = ",".join(args)
                births.setdefault(start, []).append((int(m.group(1)), pid, "CLONE_FS" in flags, "CLONE_THREAD" in flags))
    root = next((pid for _s, pid, *_rest in calls if pid is not None), None)
    cells, origin = {}, {}
    if root is not None:
        cells[root], origin[root] = [initial_cwd], "main"
    out = []
    for start, pid, name, args, result in calls:
        if pid is not None and pid not in cells:  # a process whose creation the trace does not show
            cells[pid], origin[pid] = [None], "child"
        who = origin.get(pid, "child")
        if name is None:
            out.append(Access("strace", who, "other", None))
            continue
        kind, cwd = _kind(name), cells[pid][0]
        ret = _RETURN.match(result or "")
        succeeded = ret is not None and ret.group(1) not in ("?",) and not ret.group(1).startswith("-")
        if name == "fchdir" or name in _LIST:
            descriptor = _decorated(args[0]) if args else None
            target = descriptor[1] if descriptor else None
            out.append(Access("strace", who, kind, target))
            if name == "fchdir" and succeeded:
                cells[pid][0] = target
        elif name in _OPEN and ret is not None and ret.group(2) is not None and succeeded:
            decorated = _decorated(f"{ret.group(1)}<{ret.group(2)}>")
            target = decorated[1] if decorated else None
            out.append(Access("strace", who, kind, target if target is not None else None))
        else:
            paths = _path_arguments(name, args, cwd)
            out += [Access("strace", who, kind, p) for p in paths]
            if name == "chdir" and succeeded:
                cells[pid][0] = paths[0] if paths else None
        for child, parent, shares, thread in births.get(start, ()):
            cells[child] = cells[parent] if shares else [cells[parent][0]]
            origin[child] = origin[parent] if thread else "child"
    return out


# -- classification and the report --------------------------------------------------------------


def categorise(path, world, paired_audio, pattern=False):
    world = os.path.realpath(world)
    if pattern and WILDCARDS.search(path):
        parts = path.split(os.sep)
        literal = []
        for part in parts:
            if WILDCARDS.search(part):
                break
            literal.append(part)
        prefix = os.path.realpath(os.sep.join(literal) or os.sep)
        if prefix == world or prefix.startswith(world + os.sep):
            category = categorise(prefix, world, paired_audio)
            return category if category.startswith("tier3") else "unclassified"
        return "unclassified" if world.startswith(prefix.rstrip(os.sep) + os.sep) else None
    p = os.path.realpath(path)
    if p != world and not p.startswith(world + os.sep):
        return None
    rel = Path(p).relative_to(world).as_posix()
    files = {
        "presence/manifests/development.json": "development-manifest",
        "presence/manifests/validation.json": "tier1:validation-manifest",
        "presence/manifests/split-registry.json": "tier1:registry",
        "presence/manifests/final-test.json": "tier1:final-test-manifest",
        "presence/fingerprints/manifests.json": "tier1:fingerprints",
        "presence/development/annotations.jsonl": "development-history",
    }
    if rel in files:
        return files[rel]
    if rel in (".", "presence", "presence/manifests", "presence/fingerprints", "presence/development", "audio"):
        return "directory"
    for prefix, category in (("presence/validation", "tier3:validation"), ("presence/final-test", "tier3:final-test"),
                             ("presence/identity", "tier3:identity"), ("presence/backups", "tier3:backups"),
                             ("presence/selection", "tier3:selection")):
        if rel == prefix or rel.startswith(prefix + "/"):
            return category
    if rel.endswith("session.json"):
        return "tier3:session"
    if rel.startswith("audio/"):
        return "paired-audio" if p in paired_audio else "tier3:audio"
    return "unclassified"


def analyse(evidence, world, paired_audio, initial_cwd):
    """Classify every recorder's evidence in ``evidence``; counts only."""
    evidence = Path(evidence)
    paired = {os.path.realpath(p) for p in paired_audio}
    recorders, accesses = [], []
    trace = evidence / "strace.log"
    if trace.is_file():
        recorders.append("strace")
        accesses += strace_accesses(trace.read_text(encoding="utf-8", errors="surrogateescape"), str(initial_cwd))
    for name, recorder in (("audit.jsonl", "audit-hook"), ("probes.jsonl", "stat-probes")):
        path = evidence / name
        if path.is_file():
            recorders.append(recorder)
            accesses += python_accesses(path.read_text(encoding="utf-8", errors="replace").splitlines(), recorder)
    detections, counts, unresolved = {}, {}, {r: 0 for r in recorders}
    for a in accesses:
        if a.path is None:
            category = "unresolved"
            unresolved[a.recorder] += 1
        else:
            category = categorise(a.path, world, paired, pattern=a.pattern)
            if category is None:
                continue
            if category == "directory" and a.kind == "list":
                category = "directory-listing"  # a listing reveals the names of Tier 3 items
        key = f"{a.recorder}|{a.origin}|{category}|{a.kind}"
        detections[key] = detections.get(key, 0) + 1
        counts[category] = counts.get(category, 0) + 1
    return {"recorders": recorders, "detections": dict(sorted(detections.items())),
            "counts": dict(sorted(counts.items())), "unresolved": unresolved}


def run_target(world, evidence, target, arguments, paired_audio, timeout=600):
    """Run ``privacy_targets.<target>`` under every available recorder; returns the sanitised
    report."""
    evidence.mkdir(parents=True, mode=0o700)
    # -B: no bytecode-cache writes, which reopen files by descriptor (an unresolved access)
    command = [sys.executable, "-B", str(BOOTSTRAP), str(evidence), f"privacy_targets:{target}", json.dumps(arguments)]
    if strace_available():
        command = ["strace", "-f", "-qq", "-y", "-e", f"trace={STRACE_SYSCALLS}",
                   "-o", str(evidence / "strace.log"), *command]
    done = subprocess.run(command, cwd=evidence, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=timeout, check=False)
    exception = evidence / "exception.txt"
    report = {
        "exit": done.returncode,
        "exception_type": exception.read_text(encoding="utf-8").splitlines()[0] if exception.is_file() else None,
        **analyse(evidence, world, paired_audio, evidence),
        "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(evidence.iterdir()) if p.is_file()},
    }
    (evidence.parent / f"{evidence.name}.report.json").write_text(json.dumps(report, indent=1))
    return report


def detected(report, recorder, category, kind=None, origin=None):
    """How many accesses ``recorder`` itself classified in ``category`` (of ``kind``, from
    ``origin``)."""
    total = 0
    for key, n in report["detections"].items():
        r, o, c, k = key.split("|")
        if r == recorder and c == category and kind in (None, k) and origin in (None, o):
            total += n
    return total


def require(report, recorder, category, kind=None, origin=None):
    """Fail unless ``recorder`` ran and itself detected the access. Messages are fixed texts
    built from recorder, category and kind names only."""
    if recorder not in report["recorders"]:
        missing = " (strace is not installed)" if recorder == "strace" and not strace_available() else ""
        raise RecorderRequirement(f"the {recorder} recorder did not run{missing}")
    if not detected(report, recorder, category, kind, origin):
        raise RecorderRequirement(f"the {recorder} recorder did not detect the required {kind or 'any'} access "
                                  f"by the {origin or 'any'} process in {category}")


def violations(report, allowed=ALLOWED_STEP3):
    """Categories outside ``allowed`` seen by any recorder, including ``unresolved``."""
    return {c: n for c, n in report["counts"].items() if c not in allowed}


def canary_hits(canaries, folder_or_files):
    """How many of ``canaries`` occur in the files (read as bytes); never returns which."""
    files = []
    for item in folder_or_files:
        item = Path(item)
        files += [p for p in item.rglob("*") if p.is_file()] if item.is_dir() else [item]
    found = set()
    for path in files:
        data = path.read_bytes()
        found.update(c for c in canaries if c.encode() in data)
    return len(found)
