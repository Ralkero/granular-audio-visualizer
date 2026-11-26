"""``gmv character presence``: prepare the presence project, annotate, show its status."""

import json
import sys
from pathlib import Path

from . import project

def add_parser(sub):
    parser = sub.add_parser("presence", help="Mark where a voice is audible in excerpts (research; Phase A5.1)",
                            description="Research tooling. Everything stays in the ignored local-evidence folder.")
    commands = parser.add_subparsers(dest="presence_command", required=True)
    prepare = commands.add_parser("prepare", help="Build or extend the presence project from the Phase A3 project")
    prepare.add_argument("--a3-folder", help="The Phase A3 labelling folder (read only)")
    prepare.add_argument("--inventory", required=True, help="Music inventory CSV (kept local)")
    prepare.add_argument("--map", action="append", default=[], metavar="FROM=TO",
                         help="Rewrite an inventory path prefix to where the drive is mounted here (repeatable)")
    prepare.add_argument("--aliases", help="Private alias and shared-vocalist file (default: identity/aliases.json)")
    prepare.add_argument("--corpus-manifest", help="Corpus manifest, to exclude MUSDB18-HQ test artists (recommended)")
    prepare.add_argument("--freeze-final-test", action="store_true",
                         help="Freeze the final-test cohort, once and for good; refused, with nothing written, "
                              "unless enough never-used artist groups exist")
    prepare.add_argument("--no-verify", action="store_true", help="Skip checking new recordings' checksums")
    annotate = commands.add_parser("annotate", help="Open the annotation window")
    annotate.add_argument("--partition", choices=("development", "validation", "final_test"), default="development")
    status = commands.add_parser("status", help="Counts only: excerpts and how many are completed")
    for command in (prepare, annotate, status):
        command.add_argument("--folder", help="The presence project folder (default: local-evidence/sonic-character/vocal-presence)")
    return parser


def run(args):
    root = Path(args.folder) if args.folder else project.default_folder()
    if args.presence_command == "annotate":
        from PySide6.QtWidgets import QApplication

        from ..labeling.window import place_on_screen
        from .window import PresenceWindow

        app = QApplication.instance() or QApplication(sys.argv[:1])
        app.setApplicationName("GMV vocal presence")
        window = PresenceWindow(root, args.partition, autoplay=False)
        place_on_screen(window, app)
        window.show()
        return app.exec()
    if args.presence_command == "status":
        try:
            return print(json.dumps(status(root), indent=1))
        except project.IntegrityError as exc:
            raise SystemExit(f"The presence project failed its integrity check; nothing was changed: {exc}") from None
    from ..labeling.cli import default_folder as a3_default
    from . import prepare as preparing
    from . import split

    a3_folder = Path(args.a3_folder) if args.a3_folder else a3_default()
    try:
        drive_map = tuple(tuple(m.split("=", 1)) for m in args.map if "=" in m)
        summary = preparing.prepare(root, a3_folder, args.inventory, args.aliases, args.corpus_manifest,
                                    drive_map=drive_map, freeze=args.freeze_final_test, verify=not args.no_verify,
                                    say=lambda text: print(text, flush=True))
    except (preparing.PrepareError, split.ConflictError, split.FreezeRefused, project.IntegrityError) as exc:
        raise SystemExit(f"Stopped, nothing more written: {exc}") from None
    print(json.dumps(summary, indent=1, default=list))


def status(root):
    """Counts only. Final-test excerpts are counted, but their annotations are not read."""
    from . import loaders, schema

    out = {"partitions": {}}
    examples = loaders.development_examples(root) if project.read_manifest(root, "development") else []
    for partition in ("development", "validation"):
        rows = [e for e in examples if e["partition"] == partition]
        counts = {s: sum(1 for e in rows if e["status"] == s) for s in schema.STATUSES}
        out["partitions"][partition] = {"excerpts": len(rows), **counts}
    final = project.read_manifest(root, "final_test")
    out["partitions"]["final_test"] = {"frozen": final is not None, "groups": len(final.get("groups", [])) if final else 0,
                                      "excerpts": len(final["excerpts"]) if final else 0,
                                      "annotations": "not read by this command"}
    return out
