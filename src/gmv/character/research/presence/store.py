"""The annotation history of one partition: ``annotations.jsonl``, one event per line,
appended and never rewritten.

Every decision (an interval added, edited or removed; a review state; tags; a note) is one
event, written and flushed to disk at once, so a crash can at worst lose the line being
written. Edits and removals are new events: the full history stays available for auditing.

**Integrity.** Each event carries a sequence number and a ``chain`` value: the first 16 hex
digits of SHA-256 over the previous event's chain and this event's content. Changing,
reordering or deleting a written line breaks the chain from there on; ``Store.intact``
reports it, and the protected loaders refuse a broken history. A line cut off by a crash
cannot be parsed: it is left out and counted, and the next event chains from the last good
one. Events are validated before they are written (``schema``), so an invalid annotation is
never stored.
"""

import hashlib
import json
import os
import zipfile
from datetime import UTC, datetime
from pathlib import Path

from . import schema

ANNOTATIONS = "annotations.jsonl"
SESSION = "session.json"
GENESIS = "0" * 16


def now():
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def chain_of(previous, event):
    body = json.dumps({k: v for k, v in event.items() if k != "chain"}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256((previous + body).encode("utf-8")).hexdigest()[:16]


class Store:
    """The history of one partition folder. ``durations``: excerpt -> length in seconds, from
    the partition's manifest; events are only accepted for those excerpts."""

    def __init__(self, folder, durations):
        self.folder = Path(folder)
        self.path = self.folder / ANNOTATIONS
        self.durations = dict(durations)
        self._events = None
        self.damaged = 0
        self.broken_at = None  # sequence number where the chain first fails, if it does
        self.events()

    # -- reading ----------------------------------------------------------------------------

    def events(self):
        if self._events is None:
            self._events, self.damaged, self.broken_at = [], 0, None
            previous = GENESIS
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
                    if self.broken_at is None and event.get("chain") != chain_of(previous, event):
                        self.broken_at = event.get("seq", len(self._events) + 1)
                    previous = event.get("chain", previous)
                    self._events.append(event)
        return self._events

    @property
    def intact(self):
        return self.broken_at is None

    def state(self):
        return schema.fold(self.events(), self.durations)

    def of(self, excerpt):
        return self.state().get(excerpt) or schema.empty()

    def history(self, excerpt):
        return [e for e in self.events() if e["excerpt"] == excerpt]

    # -- writing ----------------------------------------------------------------------------

    def _append(self, event):
        events = self.events()
        event = {"time": now(), "seq": (events[-1].get("seq", len(events)) + 1) if events else 1, **event}
        event["chain"] = chain_of(events[-1]["chain"] if events else GENESIS, event)
        self.folder.mkdir(parents=True, exist_ok=True)
        line = json.dumps(event, ensure_ascii=False) + "\n"
        ends_cleanly = True
        if self.path.is_file() and self.path.stat().st_size:
            with self.path.open("rb") as file:
                file.seek(-1, os.SEEK_END)
                ends_cleanly = file.read(1) == b"\n"
        with self.path.open("a", encoding="utf-8") as file:
            if not ends_cleanly:
                file.write("\n")  # after a crash: do not glue the new line to the fragment
            file.write(line)
            file.flush()
            os.fsync(file.fileno())
        events.append(event)
        return event

    def _known(self, excerpt):
        if excerpt not in self.durations:
            raise schema.SchemaError(f"{excerpt} is not in this partition")
        return self.durations[excerpt]

    def add_interval(self, excerpt, kind, start, end, note=""):
        duration = self._known(excerpt)
        if kind not in schema.KINDS:
            raise schema.SchemaError(f"unknown interval kind {kind!r}")
        start, end = schema.check_interval(start, end, duration)
        made = sum(1 for e in self.events() if e["excerpt"] == excerpt and e["event"] == "interval_add")
        identity = f"{excerpt}#{made + 1}"
        return self._append({"excerpt": excerpt, "event": "interval_add", "id": identity, "kind": kind,
                             "start": start, "end": end, "note": str(note)})

    def edit_interval(self, excerpt, identity, **changes):
        duration = self._known(excerpt)
        current = self.of(excerpt)["intervals"].get(identity)
        if current is None:
            raise schema.SchemaError(f"no interval {identity} to edit")
        allowed = {k: v for k, v in changes.items() if k in ("start", "end", "kind", "note")}
        if "kind" in allowed and allowed["kind"] not in schema.KINDS:
            raise schema.SchemaError(f"unknown interval kind {allowed['kind']!r}")
        start, end = schema.check_interval(allowed.get("start", current["start"]), allowed.get("end", current["end"]), duration)
        if "start" in allowed:
            allowed["start"] = start
        if "end" in allowed:
            allowed["end"] = end
        return self._append({"excerpt": excerpt, "event": "interval_edit", "id": identity, **allowed})

    def remove_interval(self, excerpt, identity):
        self._known(excerpt)
        if identity not in self.of(excerpt)["intervals"]:
            raise schema.SchemaError(f"no interval {identity} to remove")
        return self._append({"excerpt": excerpt, "event": "interval_remove", "id": identity})

    def set_review(self, excerpt, state, reason=""):
        self._known(excerpt)
        if state not in schema.REVIEWS:
            raise schema.SchemaError(f"unknown review state {state!r}")
        if state == "unusable" and not str(reason).strip():
            raise schema.SchemaError("an unusable excerpt needs a reason")
        return self._append({"excerpt": excerpt, "event": "review", "state": state, "reason": str(reason)})

    def set_tags(self, excerpt, tags):
        self._known(excerpt)
        return self._append({"excerpt": excerpt, "event": "tags", "tags": schema.check_tags(tags)})

    def note(self, excerpt, text):
        self._known(excerpt)
        return self._append({"excerpt": excerpt, "event": "note", "text": str(text)})


# -- session and backup ------------------------------------------------------------------------


def read_session(folder):
    try:
        return json.loads((Path(folder) / SESSION).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def write_session(folder, values):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    temporary = folder / (SESSION + ".tmp")
    temporary.write_text(json.dumps(values, indent=1), encoding="utf-8")
    temporary.replace(folder / SESSION)


def backup(files, destination, stem):
    """Zip the given existing files (by their names) into ``destination``. Returns the path.
    Only the named files are copied: never audio, never a lock file."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / f"{stem}_{datetime.now().strftime('%Y-%m-%d_%H%M%S')}.zip"
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            path = Path(path)
            if path.is_file():
                archive.write(path, path.name)
    return target
