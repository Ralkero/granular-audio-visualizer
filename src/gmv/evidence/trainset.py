"""Assemble what a training experiment may use, and what it must be scored on.

Reads the human answers of one or more review batches, the automatic label candidates and
the leakage groups; writes one table of timed segments, each with its label, source, weight,
leakage group and fold, plus an audit of that table. Nothing is trained here.

Rules:

* A person's answer always wins: any part of an automatic candidate that overlaps a
  reviewed excerpt is cut away, whatever the answer was.
* Automatic candidates enter only as "guitar present" of the two agreement classes.
  Automatic "no guitar" is not a label (it was wrong one time in five when audited), so
  every guitar-free segment is one a person confirmed.
* Folds are sets of leakage groups. A model scored on a fold must have seen nothing of it:
  neither its reviewed excerpts nor its automatic candidates.
* Only human answers are ever scored against. Excerpts that were chosen without looking at
  model output are the primary evaluation pool. Excerpts chosen *because* of model output
  (the evidence queue) are a secondary pool: the model that selected them is at its ceiling
  or floor on them by construction, so they can show a regression, not a gain.
"""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

from . import read_jsonl, write_csv
from .fusion import WEIGHTS

MIN_PIECE_S = 6.0
CONTEXT_S = 10.0
AUTOMATIC = ("strong_agreement", "moderate_agreement")
ANSWERS = {"present_throughout": "present", "absent": "absent"}
# What matters most when folds are balanced: every fold needs guitar-free excerpts to
# measure false alarms on, and those are the scarcest. Automatic time matters least.
BALANCE = {
    "unbiased_absent": 6.0,
    "selected_absent": 3.0,
    "unbiased_present": 2.0,
    "selected_present": 1.0,
    "automatic_s": 0.5,
}


def _rows(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def human_segments(review_workdir):
    """Every answered excerpt of one review batch, usable or not."""
    work = Path(review_workdir)
    manifest = {r["excerpt_id"]: r for r in _rows(work / "Excerpt_Manifest.csv")}
    out = []
    for row in _rows(work / "Human_Annotations.csv"):
        if row["annotation_status"] != "labelled":
            continue
        excerpt = manifest[row["excerpt_id"]]
        category = row["excerpt_category"]
        unbiased = category == "core"
        label = ANSWERS.get(row["guitar_presence"])
        start = float(excerpt["source_start_s"])
        out.append(
            {
                "segment_id": row["excerpt_id"],
                "recording_id": row["recording_id"],
                "start_s": start,
                "end_s": float(excerpt["source_end_s"]),
                "label": label or row["guitar_presence"],
                "guitar_type": row["guitar_type"] if label == "present" else "",
                "source": "human",
                "review_batch": work.name,
                "selection": "unbiased" if unbiased else f"model_selected:{excerpt['excerpt_position']}",
                "evaluation_pool": ("primary" if unbiased else "secondary") if label else "",
                "weight": WEIGHTS["verified_reference"] if label else 0.0,
                "usable": bool(label),
                "starts_in_first_10s": start < CONTEXT_S,
                "note": row["notes"],
            }
        )
    return out


def cut_away(start, end, taken, minimum=MIN_PIECE_S):
    """The parts of [start, end) outside every (a, b) in ``taken`` that are long enough."""
    pieces = [(start, end)]
    for a, b in sorted(taken):
        fresh = []
        for s, e in pieces:
            if b <= s or a >= e:
                fresh.append((s, e))
                continue
            if a > s:
                fresh.append((s, a))
            if b < e:
                fresh.append((b, e))
        pieces = fresh
    return [(s, e) for s, e in pieces if e - s >= minimum]


def automatic_segments(candidates, reviewed):
    """Agreement-class "present" candidates with every reviewed stretch cut away.

    ``reviewed``: {recording id: [(start_s, end_s)]} of all answered excerpts.
    """
    out = []
    for row in candidates:
        if row["label"] != "present" or row["confidence_class"] not in AUTOMATIC or row["start_s"] == "":
            continue
        rid = row["recording_id"]
        pieces = cut_away(float(row["start_s"]), float(row["end_s"]), reviewed.get(rid, []))
        for index, (start, end) in enumerate(pieces):
            out.append(
                {
                    "segment_id": f"{rid}-a{float(row['start_s']):07.1f}-{index}",
                    "recording_id": rid,
                    "start_s": start,
                    "end_s": end,
                    "label": "present",
                    "guitar_type": "",
                    "source": f"automatic:{row['confidence_class']}",
                    "review_batch": "",
                    "selection": "automatic",
                    "evaluation_pool": "",  # never scored against
                    "weight": WEIGHTS[row["confidence_class"]],
                    "usable": True,
                    "starts_in_first_10s": start < CONTEXT_S,
                    "note": "",
                }
            )
    return out


def assign_folds(loads, folds):
    """{group: fold}. ``loads``: {group: {kind: amount}}; groups are placed largest first,
    each into the fold that keeps every kind most even. Deterministic."""
    kinds = sorted({k for load in loads.values() for k in load})
    total = {k: sum(load.get(k, 0) for load in loads.values()) or 1 for k in kinds}
    held = [Counter() for _ in range(folds)]

    def size(group):
        return sum(BALANCE.get(k, 1.0) * loads[group].get(k, 0) / total[k] for k in kinds)

    def cost(index, group):
        return sum(
            BALANCE.get(k, 1.0) * ((held[index][k] + loads[group].get(k, 0)) / total[k]) ** 2
            for k in kinds
        )

    out = {}
    for group in sorted(loads, key=lambda g: (-size(g), g)):
        best = min(range(folds), key=lambda i, group=group: (cost(i, group), i))
        held[best].update(loads[group])
        out[group] = best
    return out


def build(workdir, reviews, folds=5):
    workdir = Path(workdir)
    leak = {r["recording_id"]: r["leakage_group"] for r in _rows(workdir / "leakage_groups.csv")}
    human = [s for review in reviews for s in human_segments(review)]
    reviewed = defaultdict(list)
    for segment in human:
        reviewed[segment["recording_id"]].append((segment["start_s"], segment["end_s"]))
    automatic = automatic_segments(_rows(workdir / "Automated_Label_Candidates.csv"), reviewed)
    segments = human + automatic
    missing = sorted({s["recording_id"] for s in segments} - set(leak))
    if missing:
        raise SystemExit(f"{len(missing)} recordings have no leakage group, e.g. {missing[:3]}")
    loads = defaultdict(Counter)
    for group in set(leak.values()):
        loads[group]["recordings"] += 0
    for segment in segments:
        group = leak[segment["recording_id"]]
        if not segment["usable"]:
            continue
        if segment["source"] == "human":
            pool = "unbiased" if segment["evaluation_pool"] == "primary" else "selected"
            loads[group][f"{pool}_{segment['label']}"] += 1
        else:
            loads[group]["automatic_s"] += segment["end_s"] - segment["start_s"]
    fold_of = assign_folds(loads, folds)
    for segment in segments:
        segment["leakage_group"] = leak[segment["recording_id"]]
        segment["fold"] = fold_of[segment["leakage_group"]]
        segment["duration_s"] = round(segment["end_s"] - segment["start_s"], 1)
    segments.sort(key=lambda s: (s["fold"], s["leakage_group"], s["recording_id"], s["start_s"]))
    return segments, fold_of


def audit(segments, fold_of, workdir, folds):
    """Counts and hard checks. ``problems`` must be empty for the set to be used."""
    workdir = Path(workdir)
    problems = []
    leak_rows = _rows(workdir / "leakage_groups.csv")
    leak = {r["recording_id"]: r["leakage_group"] for r in leak_rows}
    for group in read_jsonl(workdir / "same_audio_groups.jsonl"):
        where = {fold_of[leak[r]] for r in group["recordings"] if r in leak}
        if len(where) > 1:
            problems.append(f"same audio in two folds: {group['group']}")
    by_recording = defaultdict(set)
    for segment in segments:
        by_recording[segment["recording_id"]].add(segment["fold"])
    if any(len(v) > 1 for v in by_recording.values()):
        problems.append("a recording appears in two folds")
    spans = defaultdict(list)
    for segment in segments:
        spans[segment["recording_id"]].append(segment)
    disagreements = 0
    for items in spans.values():
        people = [s for s in items if s["source"] == "human"]
        for auto in (s for s in items if s["source"] != "human"):
            if any(auto["start_s"] < p["end_s"] and auto["end_s"] > p["start_s"] for p in people):
                problems.append(f"automatic segment overlaps a reviewed excerpt: {auto['segment_id']}")
        for i, a in enumerate(people):
            for b in people[i + 1 :]:
                if a["start_s"] < b["end_s"] and a["end_s"] > b["start_s"] and a["label"] != b["label"]:
                    disagreements += 1
    if disagreements:
        problems.append(f"{disagreements} overlapping reviewed excerpts carry different answers")
    if any(s["source"] != "human" and s["label"] != "present" for s in segments):
        problems.append("an automatic segment is not 'present'")
    if any(s["source"] != "human" and s["evaluation_pool"] for s in segments):
        problems.append("an automatic segment is in an evaluation pool")

    usable = [s for s in segments if s["usable"]]
    people = [s for s in usable if s["source"] == "human"]
    autos = [s for s in usable if s["source"] != "human"]

    def tally(items):
        return {
            "present": sum(s["label"] == "present" for s in items),
            "absent": sum(s["label"] == "absent" for s in items),
            "recordings": len({s["recording_id"] for s in items}),
            "groups": len({s["leakage_group"] for s in items}),
        }

    def hours(items):
        return round(sum(s["duration_s"] for s in items) / 3600, 2)

    def top_share(items, amount):
        by = Counter()
        for s in items:
            by[s["leakage_group"]] += amount(s)
        total = sum(by.values())
        return [[g, round(v / total, 3)] for g, v in by.most_common(3)] if total else []

    per_fold = []
    for fold in range(folds):
        mine = [s for s in usable if s["fold"] == fold]
        per_fold.append(
            {
                "fold": fold,
                "groups": sorted(g for g, f in fold_of.items() if f == fold),
                "primary": tally([s for s in mine if s["evaluation_pool"] == "primary"]),
                "secondary": tally([s for s in mine if s["evaluation_pool"] == "secondary"]),
                "automatic_hours": hours([s for s in mine if s["source"] != "human"]),
            }
        )
    absent = [s for s in people if s["label"] == "absent"]
    return {
        "problems": problems,
        "folds": folds,
        "leakage_groups": len(fold_of),
        "human": {
            "primary_pool": tally([s for s in people if s["evaluation_pool"] == "primary"]),
            "secondary_pool": tally([s for s in people if s["evaluation_pool"] == "secondary"]),
            "guitar_types": dict(Counter(s["guitar_type"] or "unnamed" for s in people if s["label"] == "present")),
            "start_in_first_10s": sum(s["starts_in_first_10s"] for s in people),
            "with_a_note": sum(bool(s["note"]) for s in people),
            "hours": hours(people),
        },
        "left_out": dict(Counter(s["label"] or "no answer" for s in segments if not s["usable"])),
        "automatic": {
            cls: {
                "segments": sum(s["source"] == f"automatic:{cls}" for s in autos),
                "hours": hours([s for s in autos if s["source"] == f"automatic:{cls}"]),
                "recordings": len({s["recording_id"] for s in autos if s["source"] == f"automatic:{cls}"}),
                "groups": len({s["leakage_group"] for s in autos if s["source"] == f"automatic:{cls}"}),
                "weight": WEIGHTS[cls],
            }
            for cls in AUTOMATIC
        },
        "weighted_hours": {
            "human_present": round(sum(s["duration_s"] for s in people if s["label"] == "present") / 3600, 2),
            "human_absent": round(sum(s["duration_s"] for s in absent) / 3600, 2),
            "automatic_present": round(sum(s["duration_s"] * s["weight"] for s in autos) / 3600, 2),
        },
        "concentration": {
            "automatic_time_top_groups": top_share(autos, lambda s: s["duration_s"]),
            "guitar_free_excerpts_top_groups": top_share(absent, lambda s: 1),
            "groups_with_guitar_free_excerpts": len({s["leakage_group"] for s in absent}),
            "groups_with_no_human_answer": len(set(fold_of) - {s["leakage_group"] for s in people}),
        },
        "per_fold": per_fold,
    }


def write(workdir, reviews, folds=5, name="trainset-v1"):
    workdir = Path(workdir)
    segments, fold_of = build(workdir, reviews, folds)
    report = audit(segments, fold_of, workdir, folds)
    out = workdir / name
    fields = [
        "segment_id", "recording_id", "start_s", "end_s", "duration_s", "label", "guitar_type",
        "source", "review_batch", "selection", "evaluation_pool", "weight", "usable",
        "starts_in_first_10s", "leakage_group", "fold", "note",
    ]  # fmt: skip
    write_csv(out / "Training_Segments.csv", segments, fields)
    write_csv(
        out / "fold_assignment.csv",
        [{"leakage_group": g, "fold": f} for g, f in sorted(fold_of.items())],
    )
    (out / "Trainset_Audit.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report
