"""Where the owner's answers are kept: ``labels.jsonl``, one line per action, never
rewritten.

Every click is appended as an event with the time it was made (choosing an answer,
skipping, writing a note), so a changed answer keeps the earlier one in the history, and a
crash can at worst lose the line being written. The current state of an excerpt is what its
latest events say. ``session.json`` remembers which excerpt was open.

A label is never inferred: an excerpt has an answer only if the owner chose one.

Each answer carries the number of the answer scheme it was given in (``scheme``; answers
saved before schemes were numbered belong to scheme 1). Only answers of the current scheme
count as answers. An answer of an earlier scheme is kept, reported as the excerpt's
``earlier`` answer and shown to the owner, and the excerpt is counted as still to label: it
is never translated into a current answer.
"""

import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from . import CLASSES, SCHEME, SCHEMES, SKIPS

LABELS = "labels.jsonl"
SESSION = "session.json"


def now():
    return datetime.now(UTC).isoformat(timespec="milliseconds")


class Labels:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.path = self.folder / LABELS
        self.damaged = 0  # lines that could not be read (a write cut off by a crash)
        self._events = None
        self.events()

    def events(self):
        """Every event so far, oldest first."""
        if self._events is None:
            self._events, self.damaged = [], 0
            if self.path.is_file():
                for line in self.path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        event = json.loads(line)
                        if not isinstance(event, dict) or "excerpt" not in event or "event" not in event:
                            raise ValueError
                    except ValueError:
                        self.damaged += 1
                        continue
                    self._events.append(event)
        return self._events

    def _append(self, event):
        self.folder.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        # After a write cut off by a crash the file may end mid-line: start a fresh one,
        # so the new answer is not glued to the damaged fragment.
        ends_cleanly = True
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("rb") as file:
                file.seek(-1, os.SEEK_END)
                ends_cleanly = file.read(1) == b"\n"
        with self.path.open("a", encoding="utf-8") as file:
            if not ends_cleanly:
                file.write("\n")
            file.write(line)
            file.flush()
            os.fsync(file.fileno())
        self.events().append(event)
        return event

    def label(self, excerpt, answer):
        if answer not in CLASSES:
            raise ValueError(f"unknown answer {answer!r}")
        return self._append(
            {"time": now(), "excerpt": excerpt, "event": "label", "value": answer, "scheme": SCHEME}
        )

    def skip(self, excerpt, reason="later"):
        if reason not in SKIPS:
            raise ValueError(f"unknown reason {reason!r}")
        return self._append({"time": now(), "excerpt": excerpt, "event": "skip", "reason": reason})

    def note(self, excerpt, text):
        return self._append({"time": now(), "excerpt": excerpt, "event": "note", "value": str(text)})

    def state(self):
        """excerpt -> {"label": current answer or None, "skipped": bool, "skip_reason": str
        or None, "note": str, "changes": n, "earlier": None or {"scheme", "value", "name"}}."""
        out = {}
        for event in self.events():
            current = out.setdefault(
                event["excerpt"],
                {"label": None, "skipped": False, "skip_reason": None, "note": "", "changes": 0, "earlier": None},
            )
            if event["event"] == "label":
                scheme = event.get("scheme", 1)
                value = event.get("value")
                if scheme == SCHEME and value in CLASSES:
                    if current["label"] is not None and current["label"] != value:
                        current["changes"] += 1
                    current["label"], current["skipped"], current["skip_reason"] = value, False, None
                elif value in SCHEMES.get(scheme, {}):
                    current["earlier"] = {"scheme": scheme, "value": value, "name": SCHEMES[scheme][value][0]}
            elif event["event"] == "skip" and current["label"] is None:
                current["skipped"] = True
                current["skip_reason"] = event.get("reason", "later") if event.get("reason") in SKIPS else "later"
            elif event["event"] == "note":
                current["note"] = str(event.get("value", ""))
        return out

    def history(self, excerpt):
        return [e for e in self.events() if e["excerpt"] == excerpt]


def progress(order, state):
    """Counts for the window: total; labelled (current answers); skipped, and why;
    remaining (no current answer, skipped ones included); earlier (an answer of an earlier
    scheme only, waiting for a current one)."""
    states = [state.get(e, {}) for e in order]
    labelled = sum(1 for s in states if s.get("label"))
    skipped = [s.get("skip_reason") or "later" for s in states if s.get("skipped") and not s.get("label")]
    return {
        "total": len(order),
        "labelled": labelled,
        "skipped": len(skipped),
        "skipped_no_vocals": skipped.count("no_vocals"),
        "skipped_later": skipped.count("later"),
        "remaining": len(order) - labelled,
        "earlier": sum(1 for s in states if s.get("earlier") and not s.get("label")),
    }


def resume_point(order, state, session):
    """The excerpt to open: the one open last time, else the first without an answer that
    was not skipped, else the first without an answer, else the first."""
    last = session.get("current")
    if last in order:
        return last
    for wanted in (lambda s: not s.get("label") and not s.get("skipped"), lambda s: not s.get("label")):
        found = next((e for e in order if wanted(state.get(e, {}))), None)
        if found:
            return found
    return order[0] if order else None


def read_session(folder):
    try:
        return json.loads((Path(folder) / SESSION).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_session(folder, values):
    path = Path(folder) / SESSION
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(values, indent=1), encoding="utf-8")
    temporary.replace(path)


# What a backup holds, when present: everything needed to rebuild the labelling window as
# it was (the excerpt list, every answer, where it was left, the song titles and the owner's
# reference vocalists). By name only: nothing else in the folder is copied, so no audio is.
BACKED_UP = ("dataset.json", LABELS, SESSION, "titles.json", "reference_vocalists.txt")


def backup(folder, target):
    """A zip of the files in ``BACKED_UP`` that exist in ``folder``; a missing one is left
    out, not an error. Returns its path. The audio is not included: it stays where it is."""
    folder = Path(folder)
    target = Path(target)
    stamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    path = target / f"vocal-labels-backup_{stamp}.zip" if target.is_dir() else target
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in BACKED_UP:
            if (folder / name).is_file():
                archive.write(folder / name, name)
    return path


def export(dataset, state):
    """The current answers as rows, with everything a later evaluation needs and nothing it
    should not see by accident: the split is a column, so development work can filter on it."""
    rows = []
    for excerpt in dataset["excerpts"]:
        recording = dataset["recordings"][excerpt["recording"]]
        current = state.get(excerpt["id"], {})
        rows.append(
            {
                "excerpt": excerpt["id"],
                "split": dataset["artists"][recording["artist"]]["split"],
                "artist": recording["artist"],
                "recording": excerpt["recording"],
                "start_s": excerpt["start_s"],
                "end_s": excerpt["end_s"],
                "label": current.get("label"),
                "scheme": SCHEME,
                "skipped": current.get("skipped", False),
                "skip_reason": current.get("skip_reason"),
                "earlier_label": (current.get("earlier") or {}).get("value"),
                "earlier_scheme": (current.get("earlier") or {}).get("scheme"),
                "note": current.get("note", ""),
            }
        )
    return rows
