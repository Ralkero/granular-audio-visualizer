"""``gmv character label`` (the labelling window), ``gmv character prepare-labels``
(choosing the excerpts; done once, before any label) and ``gmv character label-titles``
(the song titles shown in the window, kept beside the dataset)."""

from pathlib import Path

from .... import host, paths

FOLDER = Path("local-evidence") / "sonic-character" / "vocal-labeling"


def default_folder():
    """``local-evidence/sonic-character/vocal-labeling`` in the repository this runs from,
    whatever the working directory; the user's data folder for an installed package."""
    if paths.checkout():
        return paths.REPOSITORY / FOLDER
    return host.data_folder() / "vocal-labeling"


def add_parsers(sub):
    label = sub.add_parser(
        "label",
        help="Open the window for listening to excerpts and labelling the vocal delivery",
        description="Research tool. Your answers are saved as you click, on this computer only.",
    )
    label.add_argument("--folder", help="Where the excerpts and answers are kept (default: %s)" % FOLDER)
    prepare = sub.add_parser(
        "prepare-labels",
        help="Choose the excerpts to label and the held-out artists (once, before labelling)",
        description="Reads only the recordings named by the corpus files and --reference. "
        "Nothing is searched for, copied or changed. Refused once any answer exists.",
    )
    prepare.add_argument("--manifest", required=True, help="Corpus manifest (corpus-data/owned/manifest.json)")
    prepare.add_argument("--paths", required=True, help="Its local path map (corpus-local/paths.json)")
    prepare.add_argument(
        "--map", action="append", default=[], metavar="FROM=TO",
        help="Rewrite path prefixes written on another system, e.g. 'E:\\=/run/media/me/Drive/'",
    )
    prepare.add_argument(
        "--reference", action="append", default=[], metavar="PATH=ARTIST",
        help="A reference track and its artist group; its artist is always held out",
    )
    prepare.add_argument("--activity-model", help="Weights of the activity model, for finding likely passages")
    prepare.add_argument("--target", type=int, default=96, help="Excerpts to prepare (default: %(default)s)")
    prepare.add_argument("--processes", type=int, default=1, help="Recordings analysed at once")
    prepare.add_argument("--folder", help="Where to write (default: %s)" % FOLDER)
    named = sub.add_parser(
        "label-titles",
        help="Give the labelling window each recording's song title (the dataset is not changed)",
        description="Titles come from the music inventory, matched by checksum, then from --title, "
        "then from the file's own title tag. Never from a file name.",
    )
    named.add_argument("--inventory", help="Music inventory CSV (Enriched_Music_Inventory.csv)")
    named.add_argument(
        "--title", action="append", default=[], metavar="PATH=TITLE",
        help="The title of a recording that carries no metadata, as the owner gave it",
    )
    named.add_argument("--map", action="append", default=[], metavar="FROM=TO", help="As for prepare-labels, for checking paths")
    named.add_argument("--folder", help="Where the excerpts and answers are kept (default: %s)" % FOLDER)
    return label, prepare, named


def run(args):
    folder = Path(args.folder) if args.folder else default_folder()
    if args.character_command == "label":
        from .window import main

        return main(folder)
    from . import dataset as data

    if args.character_command == "label-titles":
        return titles_command(args, folder)

    def pair(text, what):
        if "=" not in text:
            raise SystemExit(f"{what} needs the form A=B: {text}")
        left, right = text.split("=", 1)
        return left, right

    replace = [pair(m, "--map") for m in args.map]
    references = {}
    extra = []
    for item in args.reference:
        path, artist = pair(item, "--reference")
        path = str(Path(path).expanduser())
        references[path] = Path(path).stem
        extra.append((path, artist, None, "reference track"))
    candidates = data.corpus_recordings(args.manifest, args.paths, replace)
    candidates = [c for c in candidates if c[0] not in references] + extra
    print(f"{len(candidates)} recordings named ({len(extra)} reference tracks)")
    def checking(text):
        done = int(text.split()[-1].split("/")[0])
        if done % 20 == 0 or done == len(candidates):
            print(text)

    recordings, problems = data.gather(candidates, references, progress=checking)
    print(f"{len(recordings)} usable, {len(problems)} not")
    for path, why in sorted(problems.items()):
        print(f"  not used: {path}: {why}")
    missing = [p for p in references if p not in {r['path'] for r in recordings.values()}]
    if missing:
        raise SystemExit("A reference track is not usable: " + ", ".join(missing))
    reference_artists = {recordings[k]["artist"] for k in recordings if recordings[k]["reference"]}
    model = args.activity_model
    found_scores = data.precompute_scores(
        [r["path"] for r in recordings.values()], model, args.processes, progress=print
    )

    def scores(path):
        return found_scores.get(path)

    notes = [f"{len(problems)} named recordings were not usable"] if problems else []
    found = data.prepare(folder, recordings, reference_artists, args.target, scores, notes=notes, progress=print)
    summary = data.summary(found)
    print(f"Prepared {summary['excerpts']} excerpts from {summary['recordings']} recordings in {folder}")
    for side in ("development", "evaluation"):
        print(f"  {side}: {summary[side]['excerpts']} excerpts, {summary[side]['artists']} artists")
    print(f"  methods: {summary['methods']}")


def titles_command(args, folder):
    from . import dataset as data
    from . import titles

    def pair(text):
        if "=" not in text:
            raise SystemExit(f"Needs the form A=B: {text}")
        return tuple(text.split("=", 1))

    try:
        found = data.load(folder)
    except data.DatasetError as exc:
        raise SystemExit(str(exc)) from exc
    before = (folder / "dataset.json").read_bytes()
    named = dict(pair(t) for t in args.title)
    result = titles.build(found, args.inventory, named)
    problems = titles.verify(found, result, args.inventory, [pair(m) for m in args.map])
    if problems:
        raise SystemExit("Titles not written: " + "; ".join(problems[:10]))
    titles.write(folder, result)
    assert (folder / "dataset.json").read_bytes() == before  # the dataset is never touched
    sources = {}
    for entry in result.values():
        sources[entry["from"] or "none"] = sources.get(entry["from"] or "none", 0) + 1
    print(f"Titles for {len(result)} recordings written to {folder / titles.TITLES}: {sources}")
