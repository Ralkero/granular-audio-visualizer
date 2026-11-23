import argparse
import json
from pathlib import Path
from threading import Event
from time import perf_counter

from . import paths, sources
from .engine import Engine
from .models import AnalysisConfig
from .sources.file import FileAudioSource


def main(argv=None):
    parser = argparse.ArgumentParser(description="Granular Music Visualizer")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("devices")
    commands.add_parser("ui")
    visualizer = commands.add_parser(
        "visualizer", help="Open the application, optionally replaying a recorded session"
    )
    visualizer.add_argument("--replay", metavar="SESSION", help="Session folder or messages.jsonl")
    visualizer.add_argument("--speed", type=float, default=1)
    capture = commands.add_parser("capture")
    capture.add_argument("--device")
    capture.add_argument(
        "--system-backend", choices=("auto", "native", "portaudio"), default="auto"
    )
    capture.add_argument("--pid", type=int, help="Capture this application's process tree")
    capture.add_argument("--seconds", type=float, default=30)
    file = commands.add_parser("file")
    file.add_argument("path")
    file.add_argument("--realtime", action="store_true")
    for command in (capture, file):
        command.add_argument(
            "--sessions",
            default=None,
            help="Folder for the session's logs (default: %s)" % paths.command_line_sessions(),
        )
        command.add_argument("--no-log", action="store_true")
        command.add_argument("--advanced", action="store_true")
    replay_cmd = commands.add_parser("replay")
    replay_cmd.add_argument("path")
    replay_cmd.add_argument("--speed", type=float, default=1)
    from .corpus import cli as corpus_cli

    corpus_cli.add_parser(commands)
    from .review import cli as review_cli

    review_cli.add_parser(commands)
    from .evidence import cli as evidence_cli

    evidence_cli.add_parser(commands)
    args = parser.parse_args(argv)
    if args.command == "corpus":
        return corpus_cli.run(args)
    if args.command == "review":
        return review_cli.run(args)
    if args.command == "evidence":
        return evidence_cli.run(args)
    if args.command in ("ui", "visualizer"):
        from .app.window import main as app_main

        return app_main(getattr(args, "replay", None), getattr(args, "speed", 1.0))
    if args.command == "devices":
        print(json.dumps(sources.list_output_devices(), indent=2))
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
    try:
        if args.command == "capture" and args.pid:
            source = sources.application_audio_source(pid=args.pid)
        elif args.command == "capture":
            source = sources.system_audio_source(args.device, args.system_backend)
        else:
            source = FileAudioSource(args.path, realtime=args.realtime)
    except sources.Unavailable as exc:  # a capture source this system does not have
        raise SystemExit(str(exc)) from exc
    session_root = Path(args.sessions) if args.sessions else paths.command_line_sessions()
    if not args.no_log:
        try:
            paths.writable_folder(session_root, "session logs")
        except RuntimeError as exc:
            raise SystemExit(str(exc)) from exc
    engine = Engine(
        source,
        AnalysisConfig(advanced=args.advanced),
        session_root=session_root,
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
