"""``gmv review``: build a listening batch, review it, report and export."""

import json
from pathlib import Path

DEFAULT_WORKDIR = "review-local/pilot-v1"


def add_parser(commands):
    parser = commands.add_parser(
        "review", help="Real-world excerpt review: build, listen, label, report"
    )
    parser.add_argument(
        "--workdir", default=DEFAULT_WORKDIR, help="Local folder for excerpts and labels"
    )
    sub = parser.add_subparsers(dest="review_command", required=True)
    build = sub.add_parser(
        "build", help="Select recordings, check sources, run the model timeline, cut excerpts"
    )
    build.add_argument("--inventory", required=True, help="Complete_Music_Inventory.csv")
    build.add_argument("--selection", required=True, help="Local selection JSON (picks and quotas)")
    build.add_argument("--model", default=".model-cache/guitar_mn10_v1.json")
    build.add_argument(
        "--corpus",
        default="corpus-data/owned/manifest.json",
        help="Corpus manifest the recordings are registered in (owned_local)",
    )
    build.add_argument("--processes", type=int, default=6)
    sub.add_parser("verify", help="Re-hash sources and compare every excerpt with a fresh decode")
    opened = sub.add_parser("open", help="Open the review window")
    opened.add_argument("--annotator", default="owner")
    sub.add_parser("report", help="Rebuild Human_Annotations.csv and Validation_Batch_Report.md")
    export = sub.add_parser("export", help="Write reviewed labels in the corpus annotation format")
    export.add_argument("--corpus", default="corpus-data/owned/manifest.json")
    export.add_argument(
        "--categories",
        default="core",
        help="Comma-separated excerpt categories (core, feature, model_flagged)",
    )
    return parser


def run(args):
    workdir = Path(args.workdir)
    if args.review_command == "build":
        from .pipeline import build
        from .report import write_report

        build(args.inventory, args.selection, workdir, args.model, args.corpus, args.processes)
        write_report(workdir)
        print(f"Built {workdir.resolve()}")
    elif args.review_command == "verify":
        from .pipeline import verify
        from .report import write_report

        result = verify(workdir)
        write_report(workdir)
        print(
            json.dumps(
                {k: (len(v) if isinstance(v, list) else v) for k, v in result.items()}, indent=1
            )
        )
        if not result["passed"]:
            raise SystemExit(1)
    elif args.review_command == "open":
        from .window import main

        return main(workdir, args.annotator)
    elif args.review_command == "report":
        from .report import write_report

        write_report(workdir)
        print(f"Wrote {workdir / 'Validation_Batch_Report.md'}")
    elif args.review_command == "export":
        from .annotations import AnnotationStore
        from .pipeline import TOOL_VERSION, load_tables

        categories = tuple(c.strip() for c in args.categories.split(",") if c.strip())
        unknown = set(categories) - {"core", "feature", "model_flagged"}
        if unknown:
            raise SystemExit(f"Unknown categories: {sorted(unknown)}")
        if "model_flagged" in categories:
            print(
                "Note: model-flagged excerpts were chosen from the model's output; keep this export "
                "out of unbiased evaluation."
            )
        _, excerpts = load_tables(workdir)
        from .annotations import export_corpus

        written = export_corpus(
            AnnotationStore(workdir), excerpts, args.corpus, categories, tool_version=TOOL_VERSION
        )
        print(json.dumps({"written": [str(p) for p in written]}, indent=1))
