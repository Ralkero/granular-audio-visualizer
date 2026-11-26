"""``gmv character research``: the experimental measurements, offline."""

import json
from pathlib import Path

DEFAULT_OUT = "local-evidence/sonic-character/research"
DELIVERY_OUT = "local-evidence/sonic-character/vocal-delivery-classifier"
HELDOUT_OUT = "local-evidence/sonic-character/vocal-delivery-heldout"


def add_parser(sub):
    parser = sub.add_parser(
        "research",
        help="Experimental measurements (lead line, harmony, rhythm) beside the Phase A profile",
        description="Research tooling. Offline only; the visualizer does not use any of it.",
    )
    commands = parser.add_subparsers(dest="research_command", required=True)
    analyze = commands.add_parser("analyze", help="Measure the given audio files")
    analyze.add_argument("paths", nargs="+")
    corpus = commands.add_parser(
        "corpus",
        help="Evaluate over a corpus table (columns: id, path, group, optional split and label:<name>)",
    )
    corpus.add_argument("--table", required=True, help="CSV file; kept local, never committed")
    corpus.add_argument("--gain-db", type=float, nargs="*", default=[], help="Levels to re-analyse at")
    corpus.add_argument("--gain-every", type=int, default=4, help="Re-analyse every Nth recording")
    corpus.add_argument("--pair", nargs=2, action="append", default=[], metavar="ID", help="Two ids to compare in detail")
    stems = commands.add_parser("stems", help="Check the measurements against stems and annotated beats")
    stems.add_argument("--musdb", help="MUSDB18-HQ folder (with test/ and train/)")
    stems.add_argument("--mdb-drums", help="MDB Drums folder (with audio/ and annotations/)")
    stems.add_argument("--limit", type=int, help="At most this many MUSDB songs")
    stems.add_argument(
        "--scales",
        help="corpus.json of a corpus run: states differences in its calibration standard deviations",
    )
    robust = commands.add_parser(
        "robustness",
        help="Re-analyse recordings after changes that leave the music as it was (level, mono, tone, band limit, saturation, sample rate)",
    )
    robust.add_argument("--table", required=True, help="The corpus table")
    robust.add_argument("--scales", required=True, help="corpus.json of a corpus run, for the calibration scales")
    robust.add_argument("--every", type=int, default=5, help="Use every Nth evaluation recording")
    robust.add_argument("--seconds", type=float, default=90.0, help="Seconds of each recording")
    delivery = commands.add_parser(
        "delivery",
        help="Phase A4: can the audio tell Cleans, High Screams and Low Screams apart (owner-labelled excerpts, unseen artists)",
        description="Research experiment, offline. Uses development excerpts only; held-out answers are never read "
        "by answer. Reports go to the ignored local-evidence folder.",
    )
    delivery.add_argument("--folder", help="The labelling folder (default: local-evidence/sonic-character/vocal-labeling)")
    delivery.add_argument("--note", help="Private note of known vocalist overlap (default: ../expansion-audit/leakage-watch.md beside the folder)")
    delivery.add_argument("--inventory", help="Music inventory CSV, to link artist groups that share a credited person (optional)")
    delivery.add_argument("--out", default=DELIVERY_OUT)
    delivery.add_argument("--no-cache", action="store_true", help="Measure every excerpt again instead of reading the feature cache")
    delivery.add_argument("--processes", type=int, default=1)
    heldout = commands.add_parser(
        "heldout",
        help="The pre-registered held-out test of clean against harsh vocals: predict (blind), then score once",
        description="Plan: docs/VOCAL_DELIVERY_HELDOUT_PLAN.md. 'predict' reads no held-out answer and prints the "
        "SHA-256 to commit. 'score' needs that SHA-256, the same --inventory and --open-held-out-answers; it checks "
        "that the sealed predictions match the dataset before opening anything, and runs only once.",
    )
    heldout.add_argument("step", choices=("predict", "score"))
    heldout.add_argument("--folder", help="The labelling folder (default: local-evidence/sonic-character/vocal-labeling)")
    heldout.add_argument("--note", help="Private note of known vocalist overlap (default: beside the folder)")
    heldout.add_argument("--inventory", help="Music inventory CSV (needed by predict and score, for the plan's precaution)")
    heldout.add_argument("--out", default=HELDOUT_OUT)
    heldout.add_argument("--processes", type=int, default=1)
    heldout.add_argument("--seal", help="score: the committed SHA-256 of the predictions")
    heldout.add_argument("--open-held-out-answers", action="store_true",
                         help="score: confirm that the owner has authorised reading the held-out answers, once")
    for command in (analyze, corpus, stems, robust):
        command.add_argument("--out", default=DEFAULT_OUT)
        command.add_argument("--activity-model", help="Weights file of the activity model (gmv.activity), for the vocal_* values")
        command.add_argument("--processes", type=int, default=1)
    return parser


def _measure(job):
    from .profile import measure_path

    path, gain, model = job
    try:
        return measure_path(path, gain, model)
    except Exception as exc:  # one unreadable file must not lose the others
        return f"{type(exc).__name__}: {exc}"


def _map(jobs, processes, progress):
    if processes > 1 and len(jobs) > 1:
        import os
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        from ..diagnose import THREAD_LIMITS

        kept = {name: os.environ.get(name) for name in THREAD_LIMITS}
        os.environ.update(dict.fromkeys(THREAD_LIMITS, "1"))
        try:
            with ProcessPoolExecutor(min(processes, len(jobs)), mp_context=get_context("spawn")) as pool:
                for number, found in enumerate(pool.map(_measure, jobs), 1):
                    progress(number, found)
                    yield found
        finally:
            for name, value in kept.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
    else:
        for number, job in enumerate(jobs, 1):
            found = _measure(job)
            progress(number, found)
            yield found


def write(path, value):
    path.write_text(json.dumps(value, indent=1, allow_nan=False), encoding="utf-8")


def run_delivery(args):
    from ... import paths
    from .delivery import data
    from .delivery import run as experiment
    from .labeling.cli import default_folder

    folder = Path(args.folder) if args.folder else default_folder()
    out = Path(args.out)
    if not out.is_absolute() and paths.checkout():
        out = paths.REPOSITORY / out

    def progress(number, total):
        if number == total or number % 10 == 0:
            print(f"  measured {number} of {total}", flush=True)

    try:
        experiment.run(folder, out, note=args.note, inventory=args.inventory, use_cache=not args.no_cache,
                       processes=args.processes, progress=progress, say=lambda text: print(text, flush=True))
    except data.DataError as exc:
        raise SystemExit(f"Stopped: {exc}") from None


def code_version(repository):
    """The commit the code runs from, and whether ``src`` has uncommitted changes."""
    import subprocess

    def git(*words):
        return subprocess.run(["git", "-C", str(repository), *words], capture_output=True, text=True, check=False).stdout.strip()

    return {"commit": git("rev-parse", "HEAD"), "src_clean": git("status", "--porcelain", "--", "src") == ""}


def run_heldout(args):
    from ... import paths
    from .delivery import data, heldout
    from .labeling.cli import default_folder

    folder = Path(args.folder) if args.folder else default_folder()
    out = Path(args.out)
    if not out.is_absolute() and paths.checkout():
        out = paths.REPOSITORY / out
    say = lambda text: print(text, flush=True)  # noqa: E731

    def progress(number, total):
        if number == total or number % 10 == 0:
            print(f"  measured {number} of {total}", flush=True)

    try:
        if args.step == "predict":
            if not args.inventory:
                raise SystemExit("predict needs --inventory (the plan's leakage precaution uses it)")
            code = code_version(paths.REPOSITORY) if paths.checkout() else None
            if code and not code["src_clean"]:
                raise SystemExit("Commit the code first: the predictions must come from a recorded commit")
            heldout.predict(folder, out, args.inventory, note=args.note, processes=args.processes,
                            progress=progress, say=say, code=code)
            return
        if not args.seal or not args.inventory or not args.open_held_out_answers:
            raise SystemExit("score needs --seal <committed SHA-256>, --inventory and --open-held-out-answers "
                             "(only with the owner's authorisation; it runs once)")
        heldout.score(folder, out, args.seal, args.inventory, say=say)
    except (data.DataError, heldout.SealError, heldout.ConsistencyError, heldout.AlreadyOpened) as exc:
        raise SystemExit(f"Stopped: {exc}") from None


def run(args):
    if args.research_command == "delivery":
        return run_delivery(args)
    if args.research_command == "heldout":
        return run_heldout(args)
    from . import report
    from ..diagnose import label, safe

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = args.activity_model
    if model and not Path(model).is_file():
        raise SystemExit(f"No activity model at {model}")
    if args.research_command == "stems":
        from . import stems

        if not (args.musdb or args.mdb_drums):
            raise SystemExit("Name a dataset: --musdb and/or --mdb-drums")
        scales = None
        if args.scales:
            scales = json.loads(Path(args.scales).read_text(encoding="utf-8"))["scales"]
        found = stems.run(args.musdb, args.mdb_drums, model, args.processes, args.limit, scales=scales)
        found["scales_from"] = Path(args.scales).name if args.scales else None
        write(out / "stems.json", found)
        (out / "stems.md").write_text(report.stems(found), encoding="utf-8")
        print(f"Report: {(out / 'stems.md').resolve()}")
        return
    if args.research_command == "robustness":
        from concurrent.futures import ProcessPoolExecutor
        from multiprocessing import get_context

        from . import evaluate, robustness

        corpus_run = json.loads(Path(args.scales).read_text(encoding="utf-8"))
        side = corpus_run["split"]["recording_split"]
        rows = [row for row in evaluate.read_table(args.table) if side.get(row["id"]) == "evaluation"]
        rows = rows[:: max(1, args.every)]
        jobs = [(row["path"], args.seconds) for row in rows]
        with ProcessPoolExecutor(max(1, args.processes), mp_context=get_context("spawn")) as pool:
            found = []
            for number, got in enumerate(pool.map(robustness.one, jobs), 1):
                print(f"[{number}/{len(jobs)}]" + (f" not analysed: {got}" if isinstance(got, str) else ""))
                if not isinstance(got, str):
                    found.append(got)
        summary = robustness.summarise(found, corpus_run["scales"])
        write(out / "robustness.json", {"recordings": len(found), "seconds": args.seconds, "changes": summary})
        (out / "robustness.md").write_text(robustness.report(summary, len(found)), encoding="utf-8")
        print(f"Report: {(out / 'robustness.md').resolve()}")
        return
    if args.research_command == "analyze":
        paths = [Path(p) for p in args.paths]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise SystemExit("Not found:\n  " + "\n  ".join(missing))
        results, failed = {}, {}
        done = _map([(p, 0.0, model) for p in paths], args.processes, lambda n, f: print(f"[{n}/{len(paths)}]"))
        for path, found in zip(paths, done, strict=True):
            if isinstance(found, str):
                failed[str(path)] = found
                continue
            name = label(path, results)
            results[name] = found
            write(out / (safe(name) + ".research.json"), found)
            (out / (safe(name) + ".research.svg")).write_text(report.chart(found, name), encoding="utf-8")
        text = report.recordings(results)
        if failed:
            text += "\n## Not analysed\n\n" + "\n".join(f"- {p}: {why}" for p, why in failed.items()) + "\n"
        (out / "research.md").write_text(text, encoding="utf-8")
        print(f"Analysed {len(results)} of {len(paths)}. Report: {(out / 'research.md').resolve()}")
        if failed:
            raise SystemExit(1)
        return
    from . import evaluate

    rows = evaluate.read_table(args.table)
    missing = [row["id"] for row in rows if not Path(row["path"]).is_file()]
    if missing:
        raise SystemExit("No file for: " + ", ".join(missing))
    store = out / "recordings"
    store.mkdir(exist_ok=True)

    def gather(rows, gain):
        """Profiles by id at one level; a recording already measured is read back."""
        found, jobs, todo = {}, [], []
        for row in rows:
            file = store / f"{safe(row['id'])}{'' if not gain else f'.gain{gain:+g}'}.json"
            if file.is_file():
                found[row["id"]] = json.loads(file.read_text(encoding="utf-8"))
            else:
                jobs.append((row["path"], gain, model))
                todo.append((row["id"], file))
        results = _map(jobs, args.processes, lambda n, f: print(f"[{gain:+g} dB {n}/{len(jobs)}]" + (f" not analysed: {f}" if isinstance(f, str) else "")))
        for (identity, file), result in zip(todo, results, strict=True):
            if not isinstance(result, str):
                result["phase_a"]["source"]["path"] = ""  # ids only in what is kept
                write(file, result)
                found[identity] = result
        return found

    results = gather(rows, 0.0)
    levels = {gain: gather(rows[:: max(1, args.gain_every)], gain) for gain in args.gain_db}
    found = evaluate.evaluate(rows, results, levels, [tuple(pair) for pair in args.pair])
    write(out / "corpus.json", found)
    (out / "corpus.md").write_text(report.corpus(found), encoding="utf-8")
    print(f"Evaluated {len(results)} of {len(rows)}. Report: {(out / 'corpus.md').resolve()}")
