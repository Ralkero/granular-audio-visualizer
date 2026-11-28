"""Step 2a, S2a.3: the owner's private match review and the neutral public wording of step 2
(protocol Amendment 3, A3.6, C4).

* ``write_review_file`` writes the private review file the owner opens: every name-based pair
  (group key and names, MUSDB artist and title, where the credit came from) and the file
  matches for information. It is written with owner-only permissions and never printed.
* ``summary`` is all Claude sees: counts only.
* Decisions go into the review-decisions file (``readers.read_decisions``): only name-based
  pairs can be marked "not the same artist"; a file match cannot be rejected; nothing can be
  added (a missed match is fixed in ``aliases.json``).
* **Stops.** Every step 2 stop, whatever its cause, is reported publicly with ``STOP`` only;
  every re-run with ``RERUN`` only. A test-split stop writes a review file for the test pairs.
  If the owner rejects every test pair, step 2 may re-run under a new authorisation with those
  rejections as a registered input; if the owner confirms any, that is a research-safety stop:
  A5.2 halts for a separate owner decision. Nothing in the public wording distinguishes these.
"""

import json
import os
from collections import Counter
from pathlib import Path

from . import musdb

FORMAT = "a5.2-musdb-review"
STOP = "Step 2 stopped for owner review. No figures from this run are published."
RERUN = "Step 2 re-run after the owner corrected a private input."


def _names(registry, key):
    entry = (registry.get("entries") or {}).get(key) or {}
    return sorted(n[5:] for n in entry.get("nodes", []) if isinstance(n, str) and n.startswith("name:"))


def review_items(pairs, registry):
    """[(track id, split, group key, types, sources)] -> the review file's items."""
    items = []
    for track_id, split, key, types, sources in sorted(pairs):
        artist, title, _parsed = musdb.parse_track_id(track_id)
        name_based = "exact" in types
        items.append({
            "musdb_artist": artist, "musdb_title": title, "corpus_split": split,
            "group": key, "group_names": _names(registry, key), "match": list(types),
            "credit_from": list(sources),
            "decision_if_not_the_same_artist": (
                {"track_id": track_id, "group": key, "reason": "not-the-same-artist"} if name_based and "checksum" not in types
                else "a file match: the files are byte-identical and cannot be rejected"),
        })
    return items


def write_review_file(path, pairs, registry, test_split_only=False):
    """Write the private review file (owner-only permissions, created new) and return its
    path. ``pairs`` as ``musdb.review_pairs`` gives them, or a test-split stop's private pairs."""
    path = Path(path)
    body = {"format": FORMAT, "version": 1,
            "scope": "MUSDB-test pairs after a test-split stop" if test_split_only else "all name-based and file matches",
            "how_to_decide": ("Copy the decision of every pair that is not the same artist into the review-decisions "
                              "file. Leave the others. Nothing can be added here: a missed match is fixed in the "
                              "alias file and step 2 is run again."),
            "pairs": review_items(pairs, registry)}
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(body, file, indent=1, ensure_ascii=False)
    return path


def test_split_pairs(stop):
    """The private pairs a ``musdb.TestSplitOverlap`` carries, in review form."""
    private = stop.private
    return [(t, s, k, types, private.sources.get((t, k), ())) for t, s, k, types in private.matches]


def summary(pairs):
    """Counts only, for the conversation: name-based pairs, file matches, and pairs by split."""
    kinds = Counter("file" if "checksum" in types else "name" for _t, _s, _k, types, _src in pairs)
    return {"name_based_pairs": kinds.get("name", 0), "file_matches": kinds.get("file", 0),
            "pairs_by_split": dict(sorted(Counter(s for _t, s, *_x in pairs).items()))}
