"""``gmv evidence``: gather and combine evidence about a local collection."""

import json
from pathlib import Path


def add_parser(commands):
    parser = commands.add_parser(
        "evidence", help="Metadata, dataset and model evidence for a local collection"
    )
    parser.add_argument("--workdir", default="research-local/evidence-v1")
    sub = parser.add_subparsers(dest="evidence_command", required=True)
    enrich = sub.add_parser("enrich", help="MusicBrainz identity and credits (cached, resumable)")
    enrich.add_argument("--inventory", required=True)
    enrich.add_argument("--limit", type=int)
    enrich.add_argument("--offline", action="store_true", help="Use cached answers only")
    scan = sub.add_parser("scan", help="The project's recogniser over every recording (resumable)")
    scan.add_argument("--inventory", required=True)
    scan.add_argument("--model", default=".model-cache/guitar_mn10_v1.json")
    scan.add_argument("--processes", type=int, default=6)
    scan.add_argument("--limit", type=int)
    prints = sub.add_parser("fingerprint", help="Local Chromaprint fingerprints (no network)")
    prints.add_argument("--inventory", required=True)
    prints.add_argument("--fpcalc", default=".model-cache/tools/fpcalc.exe")
    prints.add_argument("--limit", type=int)
    fuse = sub.add_parser("fuse", help="Combine the evidence: candidates, review queue, tables")
    fuse.add_argument("--inventory", required=True)
    fuse.add_argument("--review", default="review-local/pilot-v1", help="review batch with human labels")
    check = sub.add_parser("check", help="Automatic classes against the human labels")
    check.add_argument("--review", default="review-local/pilot-v1")
    data = sub.add_parser("datasets", help="External dataset manifest and OpenMIC labels")
    data.add_argument("--openmic", default="")
    data.add_argument("--moisesdb", default="")
    data.add_argument("--moisesdb-jobs", default="local-evidence/guitar/jobs_moisesdb.json")
    data.add_argument("--medleydb", default="")
    acoustid = sub.add_parser(
        "acoustid",
        help="AcoustID lookup of unmatched or ambiguous recordings (sends fingerprints)",
    )
    acoustid.add_argument("--inventory", required=True)
    acoustid.add_argument("--fpcalc", default=".model-cache/tools/fpcalc.exe")
    acoustid.add_argument(
        "--owner-approved",
        action="store_true",
        help="The owner has approved sending fingerprints and durations to acoustid.org",
    )
    return parser


def run(args):
    workdir = Path(args.workdir)
    if args.evidence_command == "enrich":
        from .enrich import enrich

        print(json.dumps(enrich(args.inventory, workdir, args.limit, args.offline), indent=1))
    elif args.evidence_command == "scan":
        from .scan import scan

        print(json.dumps(scan(args.inventory, workdir, args.model, args.processes, args.limit), indent=1))
    elif args.evidence_command == "fingerprint":
        from . import fingerprint, write_jsonl

        result = fingerprint.compute(args.inventory, workdir, args.fpcalc, limit=args.limit)
        groups, pairs = fingerprint.same_audio_groups(workdir)
        write_jsonl(
            workdir / "same_audio_groups.jsonl",
            [{"group": f"same-audio-{i:03d}", "recordings": g} for i, g in enumerate(groups, 1)],
        )
        print(json.dumps({**result, "same_audio_groups": len(groups), "pairs": len(pairs)}, indent=1))
    elif args.evidence_command == "fuse":
        from .outputs import fuse

        print(json.dumps(fuse(args.inventory, workdir, args.review), indent=1))
    elif args.evidence_command == "check":
        from .outputs import hop_level_check

        print(json.dumps(hop_level_check(workdir, args.review), indent=1))
    elif args.evidence_command == "datasets":
        from . import datasets, openmic, write_csv

        if args.openmic:
            clips = openmic.read(args.openmic)
            write_csv(workdir / "datasets" / "openmic_labels.csv", openmic.label_rows(clips))
        rows = datasets.manifest(
            workdir, args.openmic, args.moisesdb, args.moisesdb_jobs, args.medleydb
        )
        print(json.dumps({r["dataset"]: r["local_availability"] for r in rows}, indent=1))
    elif args.evidence_command == "acoustid":
        import os

        from . import fingerprint

        result = fingerprint.lookup(
            args.inventory,
            workdir,
            args.fpcalc,
            os.environ.get("GMV_ACOUSTID_KEY", ""),
            approved=args.owner_approved,
        )
        print(json.dumps(result, indent=1))
