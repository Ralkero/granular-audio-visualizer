"""A stand-in for ``pactl`` and ``parec``, so the PulseAudio backend can be tested without a
sound server and without recording anything real.

    python fake_pulse.py pactl --format=json info
    python fake_pulse.py parec --device=NAME --rate=48000 --channels=2 ...

The server it pretends to be is a JSON file, ``state.json`` in the folder named by
``FAKE_PULSE_DIR``; only the test writes it. Each ``parec`` writes what it was asked into
``streams/<pid>.json`` there. The audio is a known signal: frame ``n`` holds
``((n % 1000) - 500) / 2000`` on the left channel and its negative on the right, as
little-endian float32, delivered in real time in blocks of ``period`` frames.

State keys: ``server`` (false: nothing answers), ``json`` (false: an old pactl),
``server_name``, ``default_sink``, ``sinks`` and ``sources`` (as pactl's JSON lists them),
``move_to`` (a source name the server reports new streams on), ``die_after`` (seconds until
a stream ends by itself), ``period`` (frames per delivery).
"""

import json
import os
import sys
import time
from array import array
from pathlib import Path

FOLDER = Path(os.environ.get("FAKE_PULSE_DIR", "."))


def state():
    try:
        return json.loads((FOLDER / "state.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def streams():
    """(pid, record) of every parec that is still running."""
    import psutil

    found = []
    for path in (FOLDER / "streams").glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        pid = int(path.stem)
        try:
            alive = psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
        except psutil.Error:
            alive = False
        if alive:
            found.append((pid, record))
    return found


def refused(now):
    if not now.get("server", True):
        print("Connection failure: Connection refused", file=sys.stderr)
        print("pa_context_connect() failed: Connection refused", file=sys.stderr)
        raise SystemExit(1)


def pactl(arguments):
    now = state()
    refused(now)
    if "--format=json" not in arguments:
        raise SystemExit("this stand-in answers JSON questions only")
    asked = [a for a in arguments if not a.startswith("--")]
    if not now.get("json", True):
        print("Server Name: an old pactl")
        return
    sources = now.get("sources", [])
    if asked == ["info"]:
        answer = {
            "server_name": now.get("server_name", "PulseAudio (on PipeWire 1.6.11)"),
            "default_sink_name": now.get("default_sink", ""),
        }
    elif asked == ["list", "sinks"]:
        answer = now.get("sinks", [])
    elif asked == ["list", "sources"]:
        answer = sources
    elif asked == ["list", "short", "sources"]:
        answer = [
            {
                "index": s["index"],
                "name": s["name"],
                "sample_specification": s.get("sample_specification", ""),
            }
            for s in sources
        ]
    elif asked == ["list", "source-outputs"]:
        index = {s["name"]: s["index"] for s in sources}
        answer = [
            {
                "source": index.get(record["source"], 999),
                "properties": {"application.process.id": str(pid)},
            }
            for pid, record in streams()
        ]
    else:
        raise SystemExit(f"this stand-in does not answer {asked}")
    print(json.dumps(answer))


def parec(arguments):
    options = dict(a[2:].split("=", 1) for a in arguments if a.startswith("--") and "=" in a)
    device, rate = options.get("device", ""), int(options.get("rate", 44100))
    channels = int(options.get("channels", 2))
    now = state()
    refused(now)
    if device not in {s["name"] for s in now.get("sources", [])}:
        print("Stream error: No such entity", file=sys.stderr)
        raise SystemExit(1)
    (FOLDER / "streams").mkdir(exist_ok=True)
    record = {"device": device, "source": now.get("move_to") or device, "arguments": arguments}
    (FOLDER / "streams" / f"{os.getpid()}.json").write_text(json.dumps(record), encoding="utf-8")
    period = int(now.get("period", rate // 100))
    began, frame = time.perf_counter(), 0
    out = sys.stdout.buffer
    while True:
        now = state()
        if not now.get("server", True) or device not in {s["name"] for s in now.get("sources", [])}:
            print("Stream error: Entity killed", file=sys.stderr)
            raise SystemExit(1)
        if now.get("die_after") is not None and time.perf_counter() - began > now["die_after"]:
            print("Stream error: Connection terminated", file=sys.stderr)
            raise SystemExit(1)
        block = array("f")
        for n in range(frame, frame + period):
            left = ((n % 1000) - 500) / 2000
            block.extend([left, -left][:channels] + [0.0] * max(0, channels - 2))
        if sys.byteorder != "little":
            block.byteswap()
        try:
            out.write(block.tobytes())
            out.flush()
        except (BrokenPipeError, OSError):
            raise SystemExit(0) from None
        frame += period
        time.sleep(max(0.0, began + frame / rate - time.perf_counter()))


if __name__ == "__main__":
    tool, *rest = sys.argv[1:]
    {"pactl": pactl, "parec": parec}[tool](rest)
