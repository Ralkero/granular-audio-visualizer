"""Leakage-safe reading of the owner's answers for the vocal-delivery experiment.

**Held-out protection.** The side of every excerpt (development or evaluation) is taken
from ``dataset.json`` before any answer is read. ``labels.jsonl`` is then read line by
line, and each line is first matched only for its excerpt identifier, by a regular
expression, without decoding the line:

* a line of a **development** excerpt is decoded in full;
* a line of a **held-out** excerpt is never decoded. Only its event kind, whether it is an
  answer of the current scheme and, for a skip, its reason are matched, so a held-out
  answer's value is never parsed, stored, logged or returned. The held-out side is reported
  as totals only: answered, No vocals, review later, neither;
* a line of an excerpt the dataset does not know is not decoded either.

No whole-dataset state or export routine of the labelling tool is called.

**Known vocalist overlap.** A private note kept with the local evidence names development
excerpts whose recording features the vocalist of a held-out reference artist. It is read
and *verified* against the dataset (every named excerpt exists, is on the development side,
and its recording carries the song title the note gives, and the note names every excerpt of
that recording). If the note is missing or does not match, the experiment stops: it never
guesses. Those excerpts are left out of training, feature selection, tuning and validation.

**Artist groups.** Validation keeps every artist group in one fold. Optionally, the music
inventory's artist credits (who is credited on a recording; the inventory has no named
vocalist credits) link two groups when one person is credited on recordings of both.
"""

import csv
import json
import re
from collections import Counter
from pathlib import Path

from ..labeling import CLASSES, SCHEME, SCHEMES, SKIPS
from ..labeling import dataset as labelling
from ..labeling import titles as labelling_titles
from . import SUPPORTED, UNSUPPORTED

NOTE = Path("local-evidence") / "sonic-character" / "expansion-audit" / "leakage-watch.md"
LABELS = "labels.jsonl"

EXCERPT = re.compile(r'"excerpt"\s*:\s*"([^"\\]+)"')
EVENT = re.compile(r'"event"\s*:\s*"([a-z_]+)"')
REASON = re.compile(r'"reason"\s*:\s*"([a-z_]+)"')
CURRENT_SCHEME = re.compile(r'"scheme"\s*:\s*' + str(SCHEME) + r"\b")
NOTE_IDS = re.compile(r"VOCAL-\d{4,}")
NOTE_TITLE = re.compile(r'"([^"]+)"')

# Why a development excerpt is not used. Public reports carry the counts only.
REASONS = (
    "no_vocals", "review_later", "unanswered", "earlier_answer_only", "gutturals", "sing_screams",
    "vocalist_overlap",
)


class DataError(Exception):
    """The data cannot be used safely; the experiment stops."""


def _decode(line):
    """The only place an answer line is decoded. Called for development lines only (tests
    replace it to prove that)."""
    return json.loads(line)


def sides(dataset):
    """excerpt -> "development" or "evaluation", from the dataset alone."""
    return {e["id"]: labelling.split_of(dataset, e) for e in dataset["excerpts"]}


def check_split(dataset):
    """The dataset's own integrity rules, then the split rule re-applied to every artist:
    a hand-edited or damaged split stops the experiment."""
    problems = labelling.validate(dataset)
    references = tuple(dataset.get("reference_artists", ()))
    for artist, entry in dataset["artists"].items():
        expected = labelling.side_of(artist, references, dataset["seed"])
        if entry["split"] != expected:
            problems.append(f"artist group {artist} is on the {entry['split']} side, the rule says {expected}")
    for artist in references:
        if dataset["artists"].get(artist, {}).get("split") != "evaluation":
            problems.append(f"reference artist {artist} is not held out")
    if problems:
        raise DataError("the dataset failed its checks: " + "; ".join(problems[:5]))


def fold_development(events):
    """The current state of each development excerpt from its decoded events, with the
    labelling tool's own rules (``store.Labels.state``): only answers of the current scheme
    count, a later answer replaces an earlier one, a skip only counts while there is no
    answer, and an answer of an earlier scheme is kept apart, never translated."""
    out = {}
    for event in events:
        current = out.setdefault(event["excerpt"], {"label": None, "skip": None, "earlier": None})
        if event["event"] == "label":
            scheme, value = event.get("scheme", 1), event.get("value")
            if scheme == SCHEME and value in CLASSES:
                current["label"], current["skip"] = value, None
            elif value in SCHEMES.get(scheme, {}):
                current["earlier"] = value
        elif event["event"] == "skip" and current["label"] is None:
            current["skip"] = event.get("reason") if event.get("reason") in SKIPS else "later"
    return out


def read_answers(folder, side):
    """(development states, held-out status per excerpt, audit counts). See the module
    docstring for what is and is not read."""
    path = Path(folder) / LABELS
    development_events, held_out = [], {}
    audit = Counter(lines=0, development_lines_decoded=0, held_out_lines_not_decoded=0,
                    unknown_excerpt_lines=0, damaged_lines=0, held_out_values_decoded=0)
    if not path.is_file():
        return {}, {}, dict(audit)
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        audit["lines"] += 1
        found = EXCERPT.search(line)
        if not found:
            audit["damaged_lines"] += 1
            continue
        excerpt = found.group(1)
        where = side.get(excerpt)
        if where == "development":
            try:
                event = _decode(line)
                if not isinstance(event, dict) or event.get("excerpt") != excerpt or "event" not in event:
                    raise ValueError
            except ValueError:
                audit["damaged_lines"] += 1
                continue
            audit["development_lines_decoded"] += 1
            development_events.append(event)
        elif where == "evaluation":
            audit["held_out_lines_not_decoded"] += 1
            kind = EVENT.search(line)
            kind = kind.group(1) if kind else None
            status = held_out.get(excerpt)
            if kind == "label" and CURRENT_SCHEME.search(line):
                held_out[excerpt] = "answered"
            elif kind == "skip" and status != "answered":
                reason = REASON.search(line)
                held_out[excerpt] = "no_vocals" if reason and reason.group(1) == "no_vocals" else "later"
        else:
            audit["unknown_excerpt_lines"] += 1
    return fold_development(development_events), held_out, dict(audit)


def read_leakage_note(path, dataset, titles):
    """excerpt -> recording of every excerpt the private note names, after verifying the
    note against the dataset. Raises ``DataError`` when it is missing or does not match."""
    path = Path(path)
    if not path.is_file():
        raise DataError(
            "the private note of known vocalist overlap was not found; the experiment does not "
            "run without it (expected at local-evidence/sonic-character/expansion-audit/leakage-watch.md)"
        )
    excerpts = {e["id"]: e for e in dataset["excerpts"]}
    side = sides(dataset)
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        ids = sorted(set(NOTE_IDS.findall(line)))
        if not ids:
            continue
        title = NOTE_TITLE.search(line)
        if not title:
            raise DataError("an entry of the leakage note names excerpts but no song title to check them by")
        entries.append((ids, title.group(1)))
    if not entries:
        raise DataError("the leakage note names no excerpt, so its exclusions cannot be verified")
    verified = {}
    for ids, title in entries:
        recordings = set()
        for excerpt in ids:
            if excerpt not in excerpts:
                raise DataError("the leakage note names an excerpt that is not in the dataset")
            if side[excerpt] != "development":
                raise DataError("the leakage note names an excerpt that is not on the development side")
            recordings.add(excerpts[excerpt]["recording"])
        for recording in recordings:
            known = (titles.get(recording) or {}).get("title")
            if not known:
                raise DataError("an excerpt in the leakage note has no known song title to verify it by")
            if title.casefold() not in known.casefold():
                raise DataError("the song title in the leakage note does not match its excerpts")
        every = {e["id"] for e in dataset["excerpts"] if e["recording"] in recordings}
        if every != set(ids):
            raise DataError("the leakage note does not name every excerpt of its recording")
        verified.update({excerpt: excerpts[excerpt]["recording"] for excerpt in ids})
    return verified


# -- artist credits ------------------------------------------------------------------------

SEPARATORS = re.compile(r"\s*(?:,|&|;|/|\bfeat\.?(?=\s)|\bft\.?(?=\s)|\bfeaturing\b|\bvs\.?(?=\s))\s*", re.I)
TITLE_FEATURE = re.compile(r"[(\[]\s*(?:feat\.?|ft\.?|featuring|with)\s+([^)\]]+)[)\]]", re.I)


def credited(row, title):
    """Name keys of everyone credited on a recording: the inventory's artist fields and
    any "(feat. …)" in the title."""
    names = []
    for field in ("artist", "mb_artist"):
        names += SEPARATORS.split(row.get(field) or "")
    for found in TITLE_FEATURE.findall(title or ""):
        names += SEPARATORS.split(found)
    return {labelling.artist_key(n) for n in names if n and n.strip() and labelling.artist_key(n) != "artist-"}


def credit_links(dataset, titles, inventory):
    """Pairs of artist groups linked because one person is credited on recordings of both.
    Returns (development pairs, development-to-held-out pairs), each a list of
    (group, group, person key); private, local only."""
    rows = {}
    with Path(inventory).open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            rows[row.get("sha256", "")] = row
    people = {}
    for identity, recording in dataset["recordings"].items():
        row = rows.get(recording["sha256"])
        if row is None:
            continue
        for person in credited(row, (titles.get(identity) or {}).get("title") or row.get("title")):
            people.setdefault(person, set()).add(recording["artist"])
    inside, across = [], []
    for person, groups in sorted(people.items()):
        groups = sorted(groups)
        for i, a in enumerate(groups):
            for b in groups[i + 1:]:
                pair = (a, b, person)
                same = dataset["artists"][a]["split"] == dataset["artists"][b]["split"]
                if same and dataset["artists"][a]["split"] == "development":
                    inside.append(pair)
                elif not same:
                    across.append(pair)
    return inside, across


def groups(artists, links=()):
    """artist -> group name: artists joined by a link share one group (union of linked
    sets), named after the first of them in sorted order."""
    parent = {a: a for a in artists}

    def root(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b, *_rest in links:
        if a in parent and b in parent:
            ra, rb = sorted((root(a), root(b)))
            parent[rb] = ra
    return {a: root(a) for a in artists}


# -- the development examples --------------------------------------------------------------


def load(folder, note=None, inventory=None):
    """Everything the experiment may use, and what was left out and why.

    Returns a dict:
    * ``examples``: usable development excerpts, sorted by identifier, each
      ``{"excerpt", "recording", "artist", "group", "label"}`` (identifiers are for grouping,
      caching and the private report only; never a model input);
    * ``excluded``: development excerpts left out, counted by reason;
    * ``private``: the left-out excerpts with their reason and recorded answer, the leakage
      exclusions and the credit links (local reports only);
    * ``held_out``: totals of the held-out side, never by answer;
    * ``audit``: how many answer lines were decoded and how many were not.
    """
    folder = Path(folder)
    try:
        dataset = labelling.load(folder)
    except labelling.DatasetError as exc:
        raise DataError(str(exc)) from exc
    check_split(dataset)
    side = sides(dataset)
    titles = labelling_titles.load(folder)
    overlap = read_leakage_note(note if note is not None else folder.parent / "expansion-audit" / "leakage-watch.md", dataset, titles)
    development, held_out_status, audit = read_answers(folder, side)
    inside, across = credit_links(dataset, titles, inventory) if inventory else ([], [])

    excerpts = sorted((e for e in dataset["excerpts"] if side[e["id"]] == "development"), key=lambda e: e["id"])
    artists = sorted({dataset["recordings"][e["recording"]]["artist"] for e in excerpts})
    group_of = groups(artists, inside)
    examples, excluded, private = [], Counter(dict.fromkeys(REASONS, 0)), []
    for excerpt in excerpts:
        state = development.get(excerpt["id"], {})
        label = state.get("label")
        if excerpt["id"] in overlap:
            reason = "vocalist_overlap"
        elif label in SUPPORTED:
            reason = None
        elif label in UNSUPPORTED:
            reason = label
        elif state.get("skip") == "no_vocals":
            reason = "no_vocals"
        elif state.get("skip") == "later":
            reason = "review_later"
        elif state.get("earlier"):
            reason = "earlier_answer_only"
        else:
            reason = "unanswered"
        artist = dataset["recordings"][excerpt["recording"]]["artist"]
        if reason is None:
            examples.append({
                "excerpt": excerpt["id"], "recording": excerpt["recording"], "artist": artist,
                "group": group_of[artist], "label": label,
            })
        else:
            excluded[reason] += 1
            private.append({
                "excerpt": excerpt["id"], "reason": reason,
                "recorded": label or (f"skip:{state['skip']}" if state.get("skip") else None),
            })
    held_excerpts = [k for k, v in side.items() if v == "evaluation"]
    held_out = Counter(held_out_status.get(e, "neither") for e in held_excerpts)
    return {
        "dataset": dataset,
        "examples": examples,
        "excluded": dict(excluded),
        "private": {
            "excluded": private,
            "vocalist_overlap": sorted(overlap),
            "credit_links_development": inside,
            "credit_links_across_sides": across,
            "inventory_checked": bool(inventory),
        },
        "held_out": {
            "excerpts": len(held_excerpts),
            "answered": held_out["answered"],
            "no_vocals": held_out["no_vocals"],
            "review_later": held_out["later"],
            "neither": held_out["neither"],
        },
        "audit": audit,
    }


def summary(found):
    """Per supported answer: examples, artists, artist groups, recordings. Counts only."""
    out = {}
    for label in SUPPORTED:
        rows = [e for e in found["examples"] if e["label"] == label]
        out[label] = {
            "examples": len(rows),
            "artists": len({e["artist"] for e in rows}),
            "groups": len({e["group"] for e in rows}),
            "recordings": len({e["recording"] for e in rows}),
            "largest_artist_share": round(max(Counter(e["artist"] for e in rows).values()) / len(rows), 3) if rows else None,
        }
    return out


def source(dataset, excerpt_id):
    """(path, start_s, end_s, expected size) of an excerpt's audio, for reading it only."""
    excerpt = next(e for e in dataset["excerpts"] if e["id"] == excerpt_id)
    recording = dataset["recordings"][excerpt["recording"]]
    return Path(recording["path"]), excerpt["start_s"], excerpt["end_s"], recording["size"]
