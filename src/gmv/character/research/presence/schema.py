"""The presence-annotation schema: what an excerpt's annotation events mean.

An excerpt's annotation is a history of events (see ``store``). Folded in order, they give
its current state:

* **intervals**, each ``confirmed`` (a human voice is audible) or ``uncertain`` (the
  annotator cannot tell), with start and end in seconds on the excerpt's own timeline,
  kept to the millisecond. Intervals may overlap: several voices at once still mean "a voice
  is present", so binary targets use the union of confirmed intervals;
* a **review** state: ``in_progress`` (the default), ``complete`` (every audible voice is
  marked, as far as the annotator can tell; a no-voice answer may rest partly on the
  annotator's knowledge of the recording, and how much was played is not recorded) or
  ``unusable`` (with a reason);
* **confounder tags**: what makes the excerpt useful for testing a detector (guitar solo,
  synth, ...). Tags are never presence labels;
* an optional free-text **note**.

Status, from the state:

* ``positive``: complete, at least one confirmed interval;
* ``negative``: complete, no interval of either kind: an explicit No-vocals example (the
  annotator's judgement, from listening and knowledge of the recording);
* ``uncertain``: complete, only uncertain intervals: its excerpt-level label is unknown;
* ``incomplete`` and ``unusable``: never used as training or evaluation data, and never a
  negative.

Frame targets of a complete excerpt: 1 inside the confirmed union, unknown inside uncertain
intervals (unless also confirmed), 0 elsewhere. An incomplete or unusable excerpt has no
known frame.
"""

import numpy as np

CONFOUNDERS = {
    "distorted_rhythm_guitar": "Distorted rhythm guitar",
    "guitar_lead_solo": "Guitar lead / solo",
    "bass_focused": "Bass-focused",
    "drums_only": "Drums only",
    "synth_keyboard": "Synthesizer / keyboard",
    "orchestral_ambient": "Orchestral / ambient",
    "dense_mix": "Dense instrumental mix",
    "quiet_buried_vocals": "Quiet or buried vocals",
    "layered_vocals": "Layered vocals",
    "other": "Other (see the note)",
}
KINDS = ("confirmed", "uncertain")
REVIEWS = ("in_progress", "complete", "unusable")
STATUSES = ("positive", "negative", "uncertain", "incomplete", "unusable")
EVENTS = ("interval_add", "interval_edit", "interval_remove", "review", "tags", "note")
PRECISION = 3  # decimal places of a second: one millisecond
MIN_LENGTH_S = 0.02  # one analysis hop; shorter marks are almost certainly slips


class SchemaError(ValueError):
    """An annotation that breaks the schema; it is never written."""


def seconds(value):
    return round(float(value), PRECISION)


def check_interval(start, end, duration):
    """(start, end) rounded to the millisecond, or ``SchemaError``: both inside
    [0, duration], start before end, at least ``MIN_LENGTH_S`` long."""
    try:
        start, end, duration = seconds(start), seconds(end), float(duration)
    except (TypeError, ValueError) as exc:
        raise SchemaError("interval times must be numbers") from exc
    if not (np.isfinite(start) and np.isfinite(end)):
        raise SchemaError("interval times must be finite")
    if start < 0 or end > seconds(duration):
        raise SchemaError(f"interval {start:.3f}-{end:.3f} s lies outside the excerpt (0-{duration:.3f} s)")
    if end - start < MIN_LENGTH_S - 1e-9:
        raise SchemaError(f"interval {start:.3f}-{end:.3f} s ends before it starts or is shorter than {MIN_LENGTH_S} s")
    return start, end


def check_tags(tags):
    tags = sorted(set(tags))
    unknown = [t for t in tags if t not in CONFOUNDERS]
    if unknown:
        raise SchemaError(f"unknown confounder tag(s): {', '.join(unknown)}")
    return tags


def empty():
    return {"intervals": {}, "review": "in_progress", "reason": "", "tags": [], "note": "", "events": 0, "rejected": 0}


def apply(state, event, duration):
    """Apply one event to one excerpt's state, in place. An event that does not fit (an edit
    of an unknown interval, a time outside the excerpt) is counted as rejected and ignored;
    the store never writes such events, so this only guards against a damaged history."""
    state["events"] += 1
    kind = event.get("event")
    try:
        if kind == "interval_add":
            if event["id"] in state["intervals"] or event.get("kind") not in KINDS:
                raise SchemaError("duplicate interval or unknown kind")
            start, end = check_interval(event["start"], event["end"], duration)
            state["intervals"][event["id"]] = {
                "kind": event["kind"], "start": start, "end": end, "note": str(event.get("note", "")), "revision": 1,
            }
        elif kind == "interval_edit":
            current = state["intervals"][event["id"]]
            start, end = check_interval(event.get("start", current["start"]), event.get("end", current["end"]), duration)
            new_kind = event.get("kind", current["kind"])
            if new_kind not in KINDS:
                raise SchemaError("unknown kind")
            current.update(start=start, end=end, kind=new_kind, note=str(event.get("note", current["note"])),
                           revision=current["revision"] + 1)
        elif kind == "interval_remove":
            del state["intervals"][event["id"]]
        elif kind == "review":
            if event.get("state") not in REVIEWS:
                raise SchemaError("unknown review state")
            state["review"], state["reason"] = event["state"], str(event.get("reason", ""))
        elif kind == "tags":
            state["tags"] = check_tags(event.get("tags", []))
        elif kind == "note":
            state["note"] = str(event.get("text", ""))
        else:
            raise SchemaError("unknown event")
    except (KeyError, SchemaError, TypeError):
        state["rejected"] += 1
    return state


def fold(events, durations):
    """excerpt -> state, from events in order. ``durations``: excerpt -> length in seconds."""
    out = {}
    for event in events:
        excerpt = event.get("excerpt")
        if excerpt not in durations:
            continue
        apply(out.setdefault(excerpt, empty()), event, durations[excerpt])
    return out


def union(spans):
    """Sorted, merged (start, end) pairs: overlapping or touching spans become one."""
    merged = []
    for start, end in sorted(spans):
        if merged and start <= merged[-1][1] + 1e-9:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def spans(state, kind):
    return union((i["start"], i["end"]) for i in state["intervals"].values() if i["kind"] == kind)


def status(state):
    """One of ``STATUSES`` (see the module docstring)."""
    if state is None or state["review"] == "in_progress":
        return "incomplete"
    if state["review"] == "unusable":
        return "unusable"
    kinds = {i["kind"] for i in state["intervals"].values()}
    if "confirmed" in kinds:
        return "positive"
    if "uncertain" in kinds:
        return "uncertain"
    return "negative"


def targets(state, times):
    """(target, known) per time in ``times`` (seconds on the excerpt's timeline): target 1.0
    inside confirmed intervals, 0.0 elsewhere; known False inside uncertain-only time and
    everywhere for an incomplete or unusable excerpt."""
    times = np.asarray(times, dtype=float)
    target = np.zeros(len(times))
    known = np.zeros(len(times), dtype=bool)
    if status(state) in ("incomplete", "unusable"):
        return target, known
    known[:] = True
    for start, end in spans(state, "uncertain"):
        known[(times >= start) & (times < end)] = False
    for start, end in spans(state, "confirmed"):
        inside = (times >= start) & (times < end)
        target[inside] = 1.0
        known[inside] = True
    return target, known
