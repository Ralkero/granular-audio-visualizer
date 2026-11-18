"""Human annotations of excerpts: an append-only log, the latest state, and export.

Only a person using the review window writes here. The store accepts the annotation fields
below and nothing else; model output has no field to go into. Every save appends a full
snapshot to ``annotations.jsonl`` with a revision number and UTC time, so earlier answers
are never overwritten. ``Human_Annotations.csv`` is rebuilt from the log.

Corpus export (``export_corpus``) writes the corpus project's own annotation format:

* present throughout  -> the excerpt is covered and guitar is active over it (certain)
* absent              -> the excerpt is covered with no guitar activity
* intermittent        -> covered, guitar activity over the excerpt marked *uncertain*, so
                         frame-level scoring leaves it out; the approximate spans written by
                         the reviewer stay in the notes, never as exact boundaries
* uncertain / unset   -> not covered: unknown

Everything outside the covered excerpts stays unknown.
"""

import csv
import json
import re
from datetime import UTC, datetime
from pathlib import Path

PRESENCE = ("present_throughout", "present_intermittent", "absent", "uncertain")
GUITAR_TYPES = ("distorted_electric", "clean_electric", "acoustic", "multiple", "unknown")
ROLES = ("lead", "rhythm", "both", "background", "unknown")
INSTRUMENTS = (
    "bass",
    "drums",
    "piano",
    "synthesizer",
    "strings_orchestral",
    "vocals",
    "other",
    "unknown",
)
FIELDS = (
    "guitar_presence",
    "intermittent_spans_note",
    "guitar_type",
    "guitar_role",
    "other_instruments",
    "notes",
    "difficulty_note",
    "needs_revisit",
)
SOURCE = "human:gmv-review-window"
CSV_COLUMNS = (
    "excerpt_id",
    "recording_id",
    "excerpt_category",
    "guitar_presence",
    "intermittent_spans_note",
    "intermittent_spans_parsed_s",
    "guitar_type",
    "guitar_role",
    "other_instruments",
    "notes",
    "difficulty_note",
    "needs_revisit",
    "annotation_status",
    "annotator",
    "annotation_source",
    "annotated_at",
    "revision",
    "revisions_total",
)


def blank():
    return {
        "guitar_presence": None,
        "intermittent_spans_note": "",
        "guitar_type": None,
        "guitar_role": None,
        "other_instruments": [],
        "notes": "",
        "difficulty_note": "",
        "needs_revisit": False,
    }


def validate(values):
    unknown = set(values) - set(FIELDS)
    if unknown:
        raise ValueError(f"Not an annotation field: {sorted(unknown)}")
    checks = (("guitar_presence", PRESENCE), ("guitar_type", GUITAR_TYPES), ("guitar_role", ROLES))
    for key, allowed in checks:
        if values.get(key) not in (None, *allowed):
            raise ValueError(f"{key} must be one of {allowed} or empty")
    instruments = values.get("other_instruments", [])
    if not isinstance(instruments, list | tuple) or set(instruments) - set(INSTRUMENTS):
        raise ValueError(f"other_instruments must be a list drawn from {INSTRUMENTS}")
    for key in ("intermittent_spans_note", "notes", "difficulty_note"):
        if not isinstance(values.get(key, ""), str):
            raise ValueError(f"{key} must be text")
    if not isinstance(values.get("needs_revisit", False), bool):
        raise ValueError("needs_revisit must be true or false")


TIME = r"\d+:\d{1,2}(?:\.\d+)?|\d+(?:\.\d+)?"
SPAN = re.compile(rf"({TIME})\s*s?\s*(?:-|–|—|to)\s*({TIME})\s*s?")


def _seconds(text):
    if ":" in text:
        minutes, seconds = text.split(":")
        return int(minutes) * 60 + float(seconds)
    return float(text)


def parse_spans(note, length):
    """Approximate (start, end) spans from a note such as "0:00-0:04 and 0:09-0:12" or
    "2-5 s", relative to the excerpt. Returns None when nothing usable is found. The
    result is approximate by nature and is labelled so wherever it is written."""
    if not note:
        return None
    spans = []
    for a_text, b_text in SPAN.findall(note):
        a, b = _seconds(a_text), _seconds(b_text)
        if 0 <= a < b <= length + 0.5:
            spans.append((round(a, 2), round(min(b, length), 2)))
    return spans or None


def status_of(values):
    if values.get("needs_revisit"):
        return "needs_revisit"
    return "labelled" if values.get("guitar_presence") else "unreviewed"


class AnnotationStore:
    def __init__(self, workdir, annotator="owner"):
        self.path = Path(workdir) / "annotations.jsonl"
        self.annotator = annotator
        self.entries = []
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    self.entries.append(json.loads(line))

    def history(self, excerpt_id):
        return [e for e in self.entries if e["excerpt_id"] == excerpt_id]

    def latest(self, excerpt_id):
        history = self.history(excerpt_id)
        return history[-1] if history else None

    def values(self, excerpt_id):
        entry = self.latest(excerpt_id)
        out = blank()
        if entry:
            out.update({k: entry[k] for k in FIELDS if k in entry})
        return out

    def save(self, excerpt_id, recording_id, values):
        """Append a revision when something changed. Returns the entry or None."""
        merged = self.values(excerpt_id)
        merged.update(values)
        validate(merged)
        if self.latest(excerpt_id) and merged == self.values(excerpt_id):
            return None
        if not self.latest(excerpt_id) and merged == blank():
            return None
        entry = {
            "excerpt_id": excerpt_id,
            "recording_id": recording_id,
            "revision": len(self.history(excerpt_id)) + 1,
            "annotated_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "annotator": self.annotator,
            "annotation_source": SOURCE,
            **{k: (list(v) if isinstance(v, tuple) else v) for k, v in merged.items()},
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            handle.flush()
        self.entries.append(entry)
        return entry

    def write_csv(self, excerpts, path):
        """One row per excerpt of the manifest (unreviewed ones left blank)."""
        rows = []
        for excerpt in excerpts:
            entry = self.latest(excerpt["excerpt_id"]) or {}
            values = self.values(excerpt["excerpt_id"])
            length = float(excerpt["excerpt_duration_s"])
            spans = (
                parse_spans(values["intermittent_spans_note"], length)
                if values["guitar_presence"] == "present_intermittent"
                else None
            )
            rows.append(
                {
                    "excerpt_id": excerpt["excerpt_id"],
                    "recording_id": excerpt["recording_id"],
                    "excerpt_category": excerpt["excerpt_category"],
                    "guitar_presence": values["guitar_presence"] or "",
                    "intermittent_spans_note": values["intermittent_spans_note"],
                    "intermittent_spans_parsed_s": json.dumps(spans) if spans else "",
                    "guitar_type": values["guitar_type"] or "",
                    "guitar_role": values["guitar_role"] or "",
                    "other_instruments": ";".join(values["other_instruments"]),
                    "notes": values["notes"],
                    "difficulty_note": values["difficulty_note"],
                    "needs_revisit": "true" if values["needs_revisit"] else "false",
                    "annotation_status": status_of(values),
                    "annotator": entry.get("annotator", ""),
                    "annotation_source": entry.get("annotation_source", ""),
                    "annotated_at": entry.get("annotated_at", ""),
                    "revision": entry.get("revision", ""),
                    "revisions_total": len(self.history(excerpt["excerpt_id"])),
                }
            )
        write_rows(path, CSV_COLUMNS, rows)
        return rows


def write_rows(path, columns, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: (
                        ""
                        if row.get(k) is None
                        else (
                            "true"
                            if row.get(k) is True
                            else "false"
                            if row.get(k) is False
                            else row.get(k)
                        )
                    )
                    for k in columns
                }
            )
    temp.replace(path)


def read_rows(path):
    with open(path, newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def export_corpus(
    store, excerpts, manifest_path, categories=("core",), annotator_id=None, tool_version=""
):
    """Write one corpus annotation per recording from the latest human answers of the
    chosen excerpt categories. Returns the written paths."""
    from ..corpus import registry
    from ..corpus.schema import Activity, AnnotationFile, Coverage, TimeRange

    manifest = registry.load_manifest(manifest_path)
    annotator_id = annotator_id or "owner-review-" + "-".join(
        c.replace("_", "") for c in categories
    )
    per_recording = {}
    for excerpt in excerpts:
        if excerpt["excerpt_category"] not in categories:
            continue
        values = store.values(excerpt["excerpt_id"])
        presence = values["guitar_presence"]
        if presence in (None, "uncertain"):
            continue
        start, end = float(excerpt["source_start_s"]), float(excerpt["source_end_s"])
        item = per_recording.setdefault(
            excerpt["recording_id"], {"ranges": [], "activity": [], "notes": []}
        )
        item["ranges"].append(TimeRange(start=start, end=end))
        if presence == "present_throughout":
            item["activity"].append(
                Activity(start=start, end=end, label="guitar", certainty="certain")
            )
        elif presence == "present_intermittent":
            item["activity"].append(
                Activity(start=start, end=end, label="guitar", certainty="uncertain")
            )
            spans = parse_spans(values["intermittent_spans_note"], end - start)
            item["notes"].append(
                {
                    "excerpt_id": excerpt["excerpt_id"],
                    "presence": "intermittent",
                    "approximate_spans_relative_s": spans,
                    "note": values["intermittent_spans_note"],
                }
            )
    written = []
    for recording_id, item in sorted(per_recording.items()):
        recording = manifest.get(recording_id)
        annotation = AnnotationFile(
            recording_id=recording_id,
            recording_sha256=recording.sha256,
            annotator_id=annotator_id,
            tool="gmv-review",
            tool_version=tool_version,
            provenance=(
                "excerpt-level human review in the gmv review window; categories: "
                + ", ".join(categories)
                + "; boundaries are excerpt edges, not onsets"
            ),
            status="reviewed",
            reference_kind="human_annotation",
            uncertainty_seconds=0.5,
            coverage=Coverage(
                ranges=tuple(sorted(item["ranges"], key=lambda r: r.start)),
                layers=("activity.guitar",),
            ),
            activity=tuple(sorted(item["activity"], key=lambda a: a.start)),
            notes=json.dumps({"intermittent": item["notes"]}, ensure_ascii=False)
            if item["notes"]
            else "",
        )
        written.append(registry.save_annotation(manifest_path, annotation))
    return written
