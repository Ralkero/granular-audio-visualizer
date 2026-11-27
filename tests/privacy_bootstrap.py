"""Run one target function with file-access recording (A5.2 Step 1 spy test).

    python privacy_bootstrap.py <evidence-dir> <module:function> <json-arguments>

Layer 2: a Python audit hook records file opens, directory listings and process starts.
Layer 3: ``os.stat``, ``os.lstat`` and ``os.access`` are wrapped to record metadata probes.
Both are installed before the target module is imported. Standard output and error are
redirected to files in the evidence folder, so nothing the target prints reaches pytest.
Layer 1 (strace) is applied from outside by the harness when it is installed.
"""

import json
import os
import sys

evidence = sys.argv[1]
log = open(os.path.join(evidence, "python_access.log"), "a", buffering=1, encoding="utf-8")  # noqa: SIM115
WATCHED = ("open", "os.listdir", "os.scandir", "glob.glob", "subprocess.Popen", "os.system",
           "os.posix_spawn", "ctypes.dlopen", "shutil.copyfile")


def _flat(values):
    for value in values:
        if isinstance(value, (list, tuple)):
            yield from _flat(value)
        elif isinstance(value, (str, bytes, os.PathLike)):
            yield os.fsdecode(value)


def hook(event, args):
    if event in WATCHED or event.startswith(("os.exec", "os.spawn")):
        try:
            log.write(event + "\t" + json.dumps(list(_flat(args))) + "\n")
        except Exception:  # noqa: BLE001 - recording must never break the target
            pass


sys.addaudithook(hook)
for _name in ("stat", "lstat", "access"):
    _original = getattr(os, _name)

    def _probe(path, *args, _original=_original, _name=_name, **kwargs):
        log.write(f"os.{_name}\t" + json.dumps([str(path)]) + "\n")
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
