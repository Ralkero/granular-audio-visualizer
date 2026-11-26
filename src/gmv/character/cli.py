"""``gmv character``: Sonic Character diagnostics of audio files and recorded sessions."""

from pathlib import Path

DEFAULT_OUT = "local-evidence/sonic-character/report"


def add_parser(commands):
    parser = commands.add_parser(
        "character",
        help="Sonic Character diagnostics: profile audio files or recorded sessions",
    )
    sub = parser.add_subparsers(dest="character_command", required=True)
    analyze = sub.add_parser(
        "analyze",
        help="Profile the given recordings and write JSON, a report and charts",
        description="Each path is an audio file (WAV, FLAC, MP3, OGG, AIFF), a recorded "
        "session folder or its messages.jsonl. Files are only read. Nothing is searched "
        "for: only the paths given here, and those listed in --list, are analysed.",
    )
    analyze.add_argument("paths", nargs="*", help="Audio files or recorded sessions")
    analyze.add_argument(
        "--list",
        dest="listing",
        metavar="FILE",
        help="A text file with one path per line (blank lines and lines starting with # "
        "are skipped), for a corpus of many recordings",
    )
    analyze.add_argument(
        "--out",
        default=DEFAULT_OUT,
        help="Folder for the outputs (default: %(default)s, under the working directory)",
    )
    analyze.add_argument(
        "--gain-db",
        type=float,
        default=None,
        metavar="DB",
        help="Also analyse each audio file with its samples scaled by this many decibels "
        "in memory, and report how far each dimension moved (it should not)",
    )
    analyze.add_argument(
        "--processes",
        type=int,
        default=1,
        help="Recordings analysed at the same time (default: one)",
    )
    from .research import cli as research_cli

    research_cli.add_parser(sub)
    from .research.labeling import cli as labeling_cli

    labeling_cli.add_parsers(sub)
    from .research.presence import cli as presence_cli

    presence_cli.add_parser(sub)
    return parser


def run(args):
    if getattr(args, "character_command", None) in ("label", "prepare-labels", "label-titles"):
        from .research.labeling import cli as labeling_cli

        return labeling_cli.run(args)
    if getattr(args, "character_command", None) == "presence":
        from .research.presence import cli as presence_cli

        return presence_cli.run(args)
    if getattr(args, "character_command", None) == "research":
        from .research import cli as research_cli

        return research_cli.run(args)
    from .diagnose import run as analyse

    paths = [Path(p) for p in args.paths]
    if args.listing:
        for line in Path(args.listing).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#"):
                paths.append(Path(line))
    if not paths:
        raise SystemExit("Give at least one audio file or recorded session, or --list FILE")
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise SystemExit("Not found:\n  " + "\n  ".join(missing))
    summary = analyse(paths, args.out, args.gain_db, max(1, args.processes))
    out = Path(args.out).resolve()
    print(f"Analysed {len(summary['recordings'])} of {len(paths)}.")
    print(f"Report:  {out / 'report.md'}")
    print(f"Summary: {out / 'summary.json'}")
    if summary["not_analysed"]:
        raise SystemExit(1)
