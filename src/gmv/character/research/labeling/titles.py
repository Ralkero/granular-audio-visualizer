"""Each recording's song title, kept beside the dataset in ``titles.json``.

The dataset (``dataset.json``) is fixed before labelling and is never rewritten to add
anything; titles live in a file of their own, keyed by the dataset's recording ids. A title
comes from, in order:

1. the owner's music inventory (the table ``gmv evidence`` builds from the files' own tags),
   matched by the recording's checksum, so a title is never taken from another file;
2. a title the owner gave for a recording that has no metadata (``named``);
3. the title tag inside the audio file itself.

A title is never made up from a file name. A recording with none of these has no title,
and the window says so.
"""

import csv
import json
from pathlib import Path

import soundfile as sf

TITLES = "titles.json"


def tag_title(path):
    """The title tag inside an audio file, or None."""
    try:
        with sf.SoundFile(str(path)) as file:
            title = (file.title or "").strip()
    except Exception:
        return None
    return title or None


def inventory_titles(path):
    """sha256 -> (title, file path as written) from a music inventory CSV."""
    found = {}
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            digest, title = (row.get("sha256") or "").strip(), (row.get("title") or "").strip()
            if digest and title:
                found[digest] = (title, row.get("file_path", ""))
    return found


def build(dataset, inventory=None, named=None, tags=True):
    """recording id -> {"title", "from"} for every recording of the dataset."""
    known = inventory_titles(inventory) if inventory else {}
    named = {str(Path(k).expanduser()): v for k, v in (named or {}).items()}
    out = {}
    for identity, recording in sorted(dataset["recordings"].items()):
        title, source = None, None
        if recording.get("sha256") in known:
            title, source = known[recording["sha256"]][0], "music inventory"
        elif recording["path"] in named:
            title, source = named[recording["path"]], "named by the owner"
        elif tags:
            title = tag_title(recording["path"])
            source = "file tag" if title else None
        out[identity] = {"title": title, "from": source}
    return out


def verify(dataset, titles, inventory=None, replace=()):
    """Problems with the titles (empty when sound): every recording of the dataset has an
    entry and no entry names a recording the dataset lacks; an inventory title belongs to a
    file with the same checksum, and (with ``replace`` turning the inventory's paths into
    this system's) at the same path."""
    problems = []
    missing = set(dataset["recordings"]) - set(titles)
    extra = set(titles) - set(dataset["recordings"])
    if missing:
        problems.append(f"{len(missing)} recordings have no title entry")
    if extra:
        problems.append(f"{len(extra)} title entries name no recording of the dataset")
    known = inventory_titles(inventory) if inventory else {}
    for identity, entry in titles.items():
        recording = dataset["recordings"].get(identity)
        if recording is None or entry.get("from") != "music inventory":
            continue
        found = known.get(recording.get("sha256"))
        if found is None:
            problems.append(f"{identity}: its checksum is not in the inventory")
            continue
        if found[0] != entry["title"]:
            problems.append(f"{identity}: the inventory gives another title")
        if replace:
            path = found[1]
            for old, new in replace:
                if path.startswith(old):
                    path = (new + path[len(old) :]).replace("\\", "/")
                    break
            if path != recording["path"]:
                problems.append(f"{identity}: the inventory's file for this checksum is elsewhere")
    return problems


def load(folder):
    try:
        return json.loads((Path(folder) / TITLES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write(folder, titles):
    path = Path(folder) / TITLES
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(titles, indent=1, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)
    return path


def add_from_tags(folder, dataset):
    """Give recordings added later (with no entry yet) the title tag of their file."""
    titles = load(folder)
    for identity, recording in dataset["recordings"].items():
        if identity not in titles:
            title = tag_title(recording["path"])
            titles[identity] = {"title": title, "from": "file tag" if title else None}
    write(folder, titles)
    return titles
