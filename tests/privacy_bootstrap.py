"""Run one target function with file-access recording (A5.2 Step 1 spy test).

    python privacy_bootstrap.py <evidence-dir> <module:function> <json-arguments>

Two in-process recorders are installed before the target module is imported. Each writes its
own evidence file, so their detections stay distinguishable:

* the audit hook (``audit.jsonl``, layer 2): file opens, directory listings, globbing,
  directory changes, process launches, library loads and file copies, from Python's audit
  events;
* the stat probes (``probes.jsonl``, layer 3): ``os.stat``, ``os.lstat`` and ``os.access``,
  wrapped, because the audit hook has no stat event.

Each record keeps the call's arguments with their types (strings, file descriptors, lists),
the process's working directory at the time of the call and, for probes, whether a
``dir_fd`` was given, so the harness can resolve relative paths later or report them as
unresolved. A record that cannot be written in full is replaced by an ``unrecordable``
marker, never dropped. Interpretation happens in the harness, not here.

Standard output and error are redirected to files in the evidence folder, so nothing the
target prints reaches pytest. strace (layer 1) is applied from outside by the harness.
"""

import json
import os
import sys

evidence = sys.argv[1]
AUDIT = open(os.path.join(evidence, "audit.jsonl"), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
PROBES = open(os.path.join(evidence, "probes.jsonl"), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
WATCHED = {"open", "os.listdir", "os.scandir", "glob.glob", "glob.glob/2", "os.chdir", "subprocess.Popen",
           "os.system", "os.posix_spawn", "ctypes.dlopen", "shutil.copyfile"}


def _encode(value):
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, str):
        return {"s": value}
    if isinstance(value, (bytes, os.PathLike)):
        return {"s": os.fsdecode(value)}
    if isinstance(value, int):
        return {"fd": value}
    if isinstance(value, (list, tuple)):
        return {"l": [_encode(v) for v in value]}
    return {"t": type(value).__name__}


def _cwd():
    try:
        return os.getcwd()
    except OSError:
        return None


def _write(stream, record):
    try:
        stream.write(json.dumps(record) + "\n")
    except Exception:  # noqa: BLE001 - recording must never break the target
        try:
            stream.write('{"event": "unrecordable"}\n')
        except Exception:  # noqa: BLE001, S110
            pass


def hook(event, args):
    if event in WATCHED or event.startswith(("os.exec", "os.spawn")):
        _write(AUDIT, {"event": event, "args": [_encode(a) for a in args], "cwd": _cwd()})


sys.addaudithook(hook)
for _name in ("stat", "lstat", "access"):
    _original = getattr(os, _name)

    def _probe(path, *args, _original=_original, _name=_name, **kwargs):
        _write(PROBES, {"event": f"os.{_name}", "args": [_encode(path)], "cwd": _cwd(),
                        "dir_fd": kwargs.get("dir_fd") is not None})
        return _original(path, *args, **kwargs)

    setattr(os, _name, _probe)

for fd, name in ((1, "stdout.txt"), (2, "stderr.txt")):
    os.dup2(os.open(os.path.join(evidence, name), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600), fd)

module, function = sys.argv[2].split(":")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    getattr(__import__(module), function)(**json.loads(sys.argv[3]))
except BaseException as exc:  # noqa: BLE001 - the harness inspects every failure
    with open(os.path.join(evidence, "exception.txt"), "w", encoding="utf-8") as file:
        file.write(type(exc).__name__ + "\n" + str(exc))
    sys.stdout.flush()
    os._exit(3)
sys.stdout.flush()
os._exit(0)
