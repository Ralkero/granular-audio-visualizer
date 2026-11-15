"""``gmv corpus ...``: register local audio, manage splits and annotations, validate."""

import hashlib
import json
from pathlib import Path

from . import audacity, registry, stems
from .compare import compare
from .schema import LABELS, CorpusManifest, Recording

SPLITS = ("train", "validation", "test")


def add_parser(commands):
    parser = commands.add_parser("corpus", help="Real-music corpus and annotation workflow")
    parser.add_argument("--manifest", default="corpus/manifest.json")
    sub = parser.add_subparsers(dest="corpus_command", required=True)

    register = sub.add_parser("register", help="Add a local recording (audio is not copied)")
    register.add_argument("audio")
    register.add_argument(
        "--source-kind",
        required=True,
        choices=("owned_local", "open_dataset", "original", "public_domain", "licensed"),
    )
    register.add_argument("--license", required=True, help="Licence or ownership statement")
    register.add_argument("--permission-basis", required=True, help="Why local use is allowed")
    register.add_argument("--group", required=True, help="Album/session/dataset-song id")
    register.add_argument("--tag", action="append", default=[])
    register.add_argument("--stem", action="append", default=[], metavar="LABEL=PATH")

    splits = sub.add_parser("assign-splits", help="Assign unassigned groups deterministically")
    splits.add_argument("--seed", default="gmv-corpus-1")
    splits.add_argument("--ratios", default="0.6,0.2,0.2", help="train,validation,test")

    labels_in = sub.add_parser("import-labels", help="Import Audacity label files")
    labels_in.add_argument("recording_id")
    labels_in.add_argument("folder")
    labels_in.add_argument("--annotator", required=True)
    labels_in.add_argument("--uncertainty", type=float, default=0.02)
    labels_in.add_argument("--tool-version", default="")

    labels_out = sub.add_parser("export-labels", help="Write Audacity label files for editing")
    labels_out.add_argument("recording_id")
    labels_out.add_argument("--annotator", required=True)
    labels_out.add_argument("--out", required=True)

    review = sub.add_parser("review", help="Mark an annotation reviewed (usable as reference)")
    review.add_argument("recording_id")
    review.add_argument("--annotator", required=True)

    draft = sub.add_parser("draft-from-stems", help="Draft activity intervals from stems")
    draft.add_argument("recording_id")

    check = sub.add_parser("validate", help="Check manifest, annotations and local audio")
    check.add_argument("--no-audio", action="store_true", help="Skip local file hash checks")

    diff = sub.add_parser("compare", help="Agreement between two annotators")
    diff.add_argument("recording_id")
    diff.add_argument("annotator_a")
    diff.add_argument("annotator_b")

    sub.add_parser("status", help="What is registered, annotated and available locally")
    version = sub.add_parser("set-version", help="Set the corpus version (x.y.z)")
    version.add_argument("version")
    return parser


def _load(args):
    path = Path(args.manifest)
    if path.is_file():
        return registry.load_manifest(path)
    return CorpusManifest(corpus_version="0.1.0")


def _replace(manifest, recording):
    others = tuple(r for r in manifest.recordings if r.recording_id != recording.recording_id)
    return CorpusManifest(corpus_version=manifest.corpus_version, recordings=others + (recording,))


def register(args):
    manifest, paths = _load(args), registry.load_paths(args.manifest)
    audio = Path(args.audio).resolve()
    duration, rate, channels = registry.probe(audio)
    digest = registry.file_sha256(audio)
    stem_hashes, stem_paths = {}, {}
    for item in args.stem:
        label, _, raw = item.partition("=")
        if label not in LABELS or not raw:
            raise SystemExit(f"--stem must be LABEL=PATH with LABEL in {', '.join(LABELS)}")
        stem = Path(raw).resolve()
        stem_duration, _, _ = registry.probe(stem)
        if abs(stem_duration - duration) > 0.5:
            raise SystemExit(f"Stem {label} length differs from the mix by more than 0.5 s")
        stem_hashes[label], stem_paths[label] = registry.file_sha256(stem), str(stem)
    recording = Recording(
        recording_id="r-" + digest[:12],
        sha256=digest,
        duration_s=duration,
        sample_rate=rate,
        channels=channels,
        source_kind=args.source_kind,
        license=args.license,
        permission_basis=args.permission_basis,
        group_id=args.group,
        tags=tuple(sorted(set(args.tag))),
        stems=stem_hashes,
    )
    if any(r.recording_id == recording.recording_id for r in manifest.recordings):
        existing = manifest.get(recording.recording_id)
        if existing.sha256 != digest:
            raise SystemExit("A different recording already has this identifier")
        recording = existing  # Already registered: only refresh this machine's path map.
    else:
        peers = {r.split for r in manifest.recordings if r.group_id == recording.group_id}
        if peers:
            recording = recording.model_copy(update={"split": peers.pop()})
        manifest = _replace(manifest, recording)
        registry.save_manifest(args.manifest, manifest)
    paths["recordings"][recording.recording_id] = str(audio)
    if stem_paths:
        paths["stems"][recording.recording_id] = stem_paths
    registry.save_paths(args.manifest, paths)
    print(json.dumps({"recording_id": recording.recording_id, "split": recording.split}))


def assign_splits(args):
    manifest = _load(args)
    ratios = [float(x) for x in args.ratios.split(",")]
    if len(ratios) != 3 or min(ratios) < 0 or abs(sum(ratios) - 1) > 1e-6:
        raise SystemExit("--ratios needs three non-negative numbers that sum to 1")
    groups = {}
    for recording in manifest.recordings:
        groups[recording.group_id] = recording.split
    counts = {split: sum(v == split for v in groups.values()) for split in SPLITS}
    pending = sorted(
        (g for g, split in groups.items() if split == "unassigned"),
        key=lambda g: hashlib.sha256(f"{args.seed}:{g}".encode()).hexdigest(),
    )
    for group in pending:
        # Existing assignments never move; each new group goes where the deficit is largest.
        target = max(
            SPLITS,
            key=lambda s: (ratios[SPLITS.index(s)] * len(groups) - counts[s], -SPLITS.index(s)),
        )
        groups[group], counts[target] = target, counts[target] + 1
    manifest = CorpusManifest(
        corpus_version=manifest.corpus_version,
        recordings=tuple(
            r.model_copy(update={"split": groups[r.group_id]}) for r in manifest.recordings
        ),
    )
    registry.save_manifest(args.manifest, manifest)
    print(json.dumps({"assigned_groups": len(pending), "groups_per_split": counts}))


def import_labels(args):
    recording = _load(args).get(args.recording_id)
    annotation = audacity.import_folder(
        args.folder,
        recording,
        args.annotator,
        uncertainty_seconds=args.uncertainty,
        tool_version=args.tool_version,
    )
    path = registry.save_annotation(args.manifest, annotation)
    print(json.dumps({"written": str(path), "layers": list(annotation.coverage.layers)}))


def _annotation(args, annotator):
    path = registry.annotation_path(args.manifest, args.recording_id, annotator)
    if not path.is_file():
        raise SystemExit(f"No annotation by {annotator} for {args.recording_id}")
    return registry.load_annotation(path)


def export_labels(args):
    written = audacity.export_folder(_annotation(args, args.annotator), args.out)
    print(json.dumps({"written": [str(p) for p in written]}))


def review(args):
    annotation = _annotation(args, args.annotator).model_copy(update={"status": "reviewed"})
    registry.save_annotation(args.manifest, annotation)
    print(json.dumps({"recording_id": args.recording_id, "status": "reviewed"}))


def draft_from_stems(args):
    recording = _load(args).get(args.recording_id)
    found, problems = registry.resolve_stems(recording, registry.load_paths(args.manifest))
    if problems or not found:
        raise SystemExit(f"Stems unavailable: {problems or 'none registered'}")
    annotation = stems.draft(recording, {k: registry.read_audio(p) for k, p in found.items()})
    path = registry.save_annotation(args.manifest, annotation)
    print(json.dumps({"written": str(path), "status": annotation.status}))


def validate_corpus(manifest_path, *, check_audio=True):
    """Return (problems, warnings). Problems make the corpus unusable for evaluation."""
    problems, warnings = [], []
    try:
        manifest = registry.load_manifest(manifest_path)
    except Exception as exc:
        return [f"manifest: {exc}"], warnings
    paths = registry.load_paths(manifest_path)
    known = {r.recording_id for r in manifest.recordings}
    folder = Path(manifest_path).parent / "annotations"
    for stray in sorted(p.name for p in folder.iterdir() if p.is_dir()) if folder.is_dir() else []:
        if stray not in known:
            problems.append(f"{stray}: annotations for a recording that is not in the manifest")
    for recording in manifest.recordings:
        rid = recording.recording_id
        if recording.split == "unassigned":
            warnings.append(f"{rid}: no split assigned")
        if check_audio:
            _, reason = registry.resolve(recording, paths)
            if reason:
                warnings.append(f"{rid}: {reason}")
        folder = Path(manifest_path).parent / "annotations" / rid
        for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
            try:
                annotation = registry.load_annotation(path)
            except Exception as exc:
                problems.append(f"{rid}/{path.name}: {exc}")
                continue
            if annotation.recording_id != rid or annotation.annotator_id != path.stem:
                problems.append(f"{rid}/{path.name}: file location does not match its contents")
            if annotation.recording_sha256 != recording.sha256:
                problems.append(f"{rid}/{path.name}: annotated a different version of the audio")
            ends = (
                [e.end or e.time for e in annotation.events]
                + [e.time for e in annotation.instrument_events]
                + [a.end for a in annotation.activity]
                + [r.end for r in annotation.coverage.ranges]
            )
            if ends and max(ends) > recording.duration_s + 0.05:
                problems.append(f"{rid}/{path.name}: labels extend past the recording end")
            if not annotation.usable_as_reference:
                warnings.append(f"{rid}/{path.name}: status {annotation.status}, not reference")
    return problems, warnings


def validate(args):
    problems, warnings = validate_corpus(args.manifest, check_audio=not args.no_audio)
    print(json.dumps({"problems": problems, "warnings": warnings}, indent=2))
    if problems:
        raise SystemExit(1)


def compare_cmd(args):
    report = compare(_annotation(args, args.annotator_a), _annotation(args, args.annotator_b))
    print(json.dumps(report, indent=2))


def status(args):
    manifest, paths = _load(args), registry.load_paths(args.manifest)
    rows = []
    for recording in manifest.recordings:
        _, reason = registry.resolve(recording, paths, verify_hash=False)
        annotations = registry.annotations_for(args.manifest, recording.recording_id)
        rows.append(
            {
                "recording_id": recording.recording_id,
                "split": recording.split,
                "group": recording.group_id,
                "duration_s": round(recording.duration_s, 1),
                "tags": list(recording.tags),
                "stems": sorted(recording.stems),
                "audio": reason or "available",
                "annotations": {a.annotator_id: a.status for a in annotations},
            }
        )
    print(json.dumps({"corpus_version": manifest.corpus_version, "recordings": rows}, indent=2))


def set_version(args):
    manifest = _load(args)
    registry.save_manifest(
        args.manifest,
        CorpusManifest(corpus_version=args.version, recordings=manifest.recordings),
    )
    print(json.dumps({"corpus_version": args.version}))


HANDLERS = {
    "register": register,
    "assign-splits": assign_splits,
    "import-labels": import_labels,
    "export-labels": export_labels,
    "review": review,
    "draft-from-stems": draft_from_stems,
    "validate": validate,
    "compare": compare_cmd,
    "status": status,
    "set-version": set_version,
}


def run(args):
    try:
        return HANDLERS[args.corpus_command](args)
    except (ValueError, KeyError, OSError) as exc:
        raise SystemExit(f"corpus {args.corpus_command}: {exc}") from exc
