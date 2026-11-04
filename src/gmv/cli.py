import argparse
import json
from pathlib import Path
from threading import Event
from time import perf_counter

from .engine import Engine
from .models import AnalysisConfig
from .sources.file import FileAudioSource
from .sources.windows import SystemAudioSource, list_output_devices


def main():
    parser = argparse.ArgumentParser(description="Granular Music Visualizer — Phase One")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices")
    commands.add_parser("ui")
    capture = commands.add_parser("capture")
    capture.add_argument("--device")
    capture.add_argument("--pid", type=int, help="Capture this application's process tree")
    capture.add_argument("--seconds", type=float, default=30)
    file = commands.add_parser("file")
    file.add_argument("path")
    file.add_argument("--realtime", action="store_true")
    for command in (capture, file):
        command.add_argument("--sessions", default="sessions")
        command.add_argument("--no-log", action="store_true")
        command.add_argument("--advanced", action="store_true")
    replay_cmd = commands.add_parser("replay")
    replay_cmd.add_argument("path")
    replay_cmd.add_argument("--speed", type=float, default=1)
    args = parser.parse_args()
    if args.command == "ui":
        from .ui import main as ui_main

        return ui_main()
    if args.command == "devices":
        print(json.dumps(list_output_devices(), indent=2))
        return
    if args.command == "replay":
        from threading import Thread

        from .bus import EventBus
        from .session import replay

        bus = EventBus()
        q = bus.subscribe(256)
        stop = Event()
        worker = Thread(target=replay, args=(args.path, bus, stop), kwargs={"speed": args.speed})
        worker.start()
        try:
            while worker.is_alive() or len(q):
                msg = q.get()
                if msg:
                    print(msg.model_dump_json())
        except KeyboardInterrupt:
            stop.set()
        worker.join()
        return
    if args.command == "capture" and args.pid:
        from .sources.application import ApplicationAudioSource

        source = ApplicationAudioSource(args.pid)
    else:
        source = (
            SystemAudioSource(args.device)
            if args.command == "capture"
            else FileAudioSource(args.path, realtime=args.realtime)
        )
    engine = Engine(
        source,
        AnalysisConfig(advanced=args.advanced),
        session_root=Path(args.sessions),
        logging=not args.no_log,
    )
    engine.start()
    deadline = perf_counter() + args.seconds if args.command == "capture" else float("inf")
    try:
        while not engine.done.wait(0.1):
            if perf_counter() >= deadline:
                engine.stop()
    except KeyboardInterrupt:
        engine.stop()
        engine.done.wait(15)
    print(json.dumps(engine.metrics(), indent=2))
    if engine.writer:
        print(f"Session: {engine.writer.path.resolve()}")
    if engine.error:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
