"""``gmv character research``: the experimental measurements, offline."""

import json
from pathlib import Path

DEFAULT_OUT = "local-evidence/sonic-character/research"


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


def run(args):
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
