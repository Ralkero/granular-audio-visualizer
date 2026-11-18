"""Choose a review batch from a collection inventory, using metadata only.

The selection is driven by a local JSON file (kept out of Git because it names the owner's
recordings) with two kinds of entries:

``picks``    named recordings (path relative to the collection root), each with a category
             and a written reason
``quotas``   "take N recordings from this top-level folder": a seeded, reproducible draw
             that spreads over albums (at most ``max_per_album``, per folder override
             possible) and never takes two members of one duplicate / version group

Model scores in the inventory are copied for later comparison but never read when choosing.

Selection file::

    {"name": "pilot-v1", "seed": "gmv-pilot-v1", "max_per_album": 3,
     "min_duration_s": 75,
     "picks": [{"path": "Artist\\\\Song.mp3", "category": "positive_candidate",
                "reason": "..."}],
     "quotas": [{"folder": "Artist", "count": 6, "category": "positive_candidate",
                 "reason": "...", "max_per_album": 4}]}
"""

import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

CATEGORIES = (
    "positive_candidate",
    "mixed_candidate",
    "hard_negative_candidate",
    "negative_candidate",
)
PRIORITY = {
    "positive_candidate": 1,
    "mixed_candidate": 2,
    "hard_negative_candidate": 2,
    "negative_candidate": 1,
}


def read_inventory(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def slug(text):
    value = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return value[:48] or "unknown"


def group_of(row):
    """Leakage group: the whole artist folder, so alternate versions, duplicates and every
    excerpt of an artist's recordings always share one split."""
    return "artist-" + slug(row["top_folder"] or row.get("artist") or "unknown")


def version_key(row):
    keys = [
        row.get(k)
        for k in ("exact_duplicate_group", "potential_duplicate_group", "alternate_version_group")
    ]
    return tuple(k for k in keys if k)


def _order(seed, row):
    return hashlib.sha256(f"{seed}:{row['sha256']}".encode()).hexdigest()


def select(inventory, spec):
    """Return the chosen inventory rows, each with ``selection_*`` fields added."""
    by_relative = {os.path.normcase(r["relative_path"]): r for r in inventory}
    chosen, taken_versions, problems = [], set(), []

    def take(row, category, reason, method):
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category {category}")
        if any(c["sha256"] == row["sha256"] for c in chosen):
            return False
        versions = set(version_key(row))
        if versions & taken_versions:
            return False
        taken_versions.update(versions)
        chosen.append(
            dict(
                row,
                selection_category=category,
                selection_reason=reason,
                selection_method=method,
                selection_priority=PRIORITY[category],
            )
        )
        return True

    for pick in spec.get("picks", []):
        row = by_relative.get(os.path.normcase(pick["path"]))
        if row is None:
            problems.append(f"pick not in inventory: {pick['path']}")
            continue
        if not take(row, pick["category"], pick["reason"], "named"):
            problems.append(f"pick skipped (duplicate or version already chosen): {pick['path']}")

    minimum = float(spec.get("min_duration_s", 0))
    for quota in spec.get("quotas", []):
        folder = quota["folder"]
        already = sum(1 for c in chosen if c["top_folder"] == folder)
        need = quota["count"] - already
        cap = quota.get("max_per_album", spec.get("max_per_album", 3))
        pool = [
            r
            for r in inventory
            if r["top_folder"] == folder
            and r.get("exact_duplicate_is_first") != "false"
            and float(r["duration_s"] or 0) >= minimum
        ]
        pool.sort(key=lambda r: _order(spec.get("seed", "gmv"), r))
        per_album = Counter(c.get("album") for c in chosen if c["top_folder"] == folder)
        # round-robin over albums in draw order so no album dominates
        albums = defaultdict(list)
        for row in pool:
            albums[row.get("album")].append(row)
        order = sorted(albums, key=lambda a: _order(spec.get("seed", "gmv"), albums[a][0]))
        while need > 0 and any(albums.values()):
            progressed = False
            for album in order:
                if need <= 0:
                    break
                while albums[album]:
                    row = albums[album].pop(0)
                    if per_album[album] >= cap:
                        albums[album].clear()
                        break
                    if take(row, quota["category"], quota["reason"], "seeded_draw"):
                        per_album[album] += 1
                        need -= 1
                        progressed = True
                        break
            if not progressed:
                break
        if need > 0:
            problems.append(f"quota for {folder} short by {need}")
    return chosen, problems


def load_spec(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
