"""Combine the evidence about each recording into label candidates, without pretending any
of it is ground truth.

Sources and what each can say:

``human``      reviewed excerpts (``gmv review``): the only reference on this collection
``models``     two audio models with time resolution: the project's recogniser and PANNs
               CNN14. Both were pretrained on AudioSet, so their agreement is supporting
               evidence, not independent confirmation
``separation`` the guitar output of a separation model, where it was run (a third method
               with different training data; its guitar output contains bleed)
``credits``    MusicBrainz performer credits. A recording-level guitar credit says guitar is
               on the recording somewhere, never when. Release- and artist-level credits say
               less. A missing credit says nothing at all

Two scopes are kept apart. *Interval-level* support is evidence about the stretch itself:
the two models' scores over it, and the separated stem's level over it. *Recording-level*
metadata (any credit) describes the whole recording. Only interval-level support can set or
raise the class of a stretch; a credit is written beside the candidate
(``recording_level_metadata``) and changes neither its class nor its weight.

Classes given to a stretch of time (``confidence_class``):

``verified_reference``   a person labelled it
``strong_agreement``     both models say "present" and the separated guitar stem, measured
                         over that same stretch, supports them
``moderate_agreement``   both models say "present" and no other interval-level source
                         covers the stretch
``weak_evidence``        a credit alone (a recording, no time); and every automatic
                         "absent": two models hearing no guitar measured too unreliable on
                         this collection to be a label, with or without the stem agreeing
``conflict``             the models, or the models and the stem, contradict each other
``unknown``              anything else

These are classes, not probabilities. The cut-offs in ``CUTS`` were read off material with
human labels (see docs/evidence/EVIDENCE_ENRICHMENT.md); the purity measured there is
reported with the output and is the only number that may be quoted for them.
"""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HOP_S = 0.5
CUTS = {
    "gmv_present": 0.926,  # the recogniser's own "detected" threshold
    "gmv_absent": 0.05,
    "panns_present": 0.10,  # PANNs guitar, averaged over the same ten seconds as the recogniser
    "panns_absent": 0.003,
    "gmv_clash_absent": 0.05,  # a contradiction needs both models to be sure
    "panns_clash_present": 0.2,
    "panns_window_s": 10.0,
    "silence_dbfs": -60.0,
    "min_run_s": 6.0,  # shorter agreements are not turned into intervals
    "separation_present_db": -10.0,  # guitar stem level relative to the mix
    "separation_absent_db": -40.0,
}
WEIGHTS = {  # proposals for a later training experiment, not calibrated values
    "verified_reference": 1.0,
    "strong_agreement": 0.5,
    "moderate_agreement": 0.25,
}
CATEGORY = {
    "verified_reference": "A",
    "strong_agreement": "B",
    "moderate_agreement": "B",
    "weak_evidence": "C",
    "conflict": "D",
    "unknown": "D",
}


def credit_summary(credits):
    """What the credits of one recording establish, by scope. Never a negative."""
    out = {
        "recording_guitar": [],
        "release_guitar": [],
        "artist_guitar": [],
        "recording_instruments": set(),
        "release_instruments": set(),
        "unspecified_performers": 0,
        "guitar_kinds": set(),
    }
    for credit in credits:
        scope, family = credit["scope"], credit["family"]
        if family == "unspecified":
            out["unspecified_performers"] += 1
            continue
        if scope in ("recording", "release"):
            out[f"{scope}_instruments"].add(credit["instrument"])
        if family == "guitar":
            out[f"{scope}_guitar"].append(credit["performer"])
            if scope == "recording" and credit["guitar_kind"]:
                out["guitar_kinds"].add(credit["guitar_kind"])
    if out["recording_guitar"]:
        out["scope"] = "recording_level_guitar_credit"
    elif out["release_guitar"]:
        out["scope"] = "release_level_guitar_credit"
    elif out["artist_guitar"]:
        out["scope"] = "artist_level_guitarist"
    elif out["recording_instruments"] or out["release_instruments"]:
        out["scope"] = "credits_without_guitar"  # says nothing about guitar
    else:
        out["scope"] = "no_instrument_credits"
    return out


def trailing_mean(values, size):
    """Mean of each hop and the ``size - 1`` before it (fewer at the start)."""
    values = np.asarray(values, np.float64)
    total = np.cumsum(np.concatenate([[0.0], values]))
    index = np.arange(1, len(values) + 1)
    first = np.maximum(0, index - size)
    return (total[index] - total[first]) / (index - first)


def hop_classes(gmv, panns_guitar, level_db, cuts=CUTS):
    """Per hop: 2 both say guitar, -2 both say none, 9 contradiction, 0 undecided, -9 silence.

    ``gmv`` holds the recogniser's guitar score per hop; the first ten seconds of a recording
    have an incomplete context there and are left undecided unless silent.
    """
    count = min(len(gmv), len(panns_guitar), len(level_db))
    gmv, level_db = gmv[:count], level_db[:count]
    panns_guitar = trailing_mean(panns_guitar[:count], round(cuts["panns_window_s"] / HOP_S))
    out = np.zeros(count, np.int8)
    yes = (gmv >= cuts["gmv_present"]) & (panns_guitar >= cuts["panns_present"])
    no = (gmv <= cuts["gmv_absent"]) & (panns_guitar <= cuts["panns_absent"])
    clash = ((gmv >= cuts["gmv_present"]) & (panns_guitar <= cuts["panns_absent"])) | (
        (gmv <= cuts["gmv_clash_absent"]) & (panns_guitar >= cuts["panns_clash_present"])
    )
    out[yes], out[no], out[clash] = 2, -2, 9
    out[: round(10 / HOP_S)] = 0  # incomplete context for the recogniser
    out[level_db < cuts["silence_dbfs"]] = -9
    return out


def runs(classes, value, min_hops):
    """[(first hop, hop after last)] of runs of ``value`` at least ``min_hops`` long."""
    out, start = [], None
    for index, item in enumerate(list(classes) + [None]):
        if item == value and start is None:
            start = index
        elif item != value and start is not None:
            if index - start >= min_hops:
                out.append((start, index))
            start = None
    return out


def separation_support(stem_db, start, end, cuts=CUTS):
    """"present", "absent" or "" from the separated guitar stem's level relative to the mix
    over hops [start, end); ``stem_db`` is None where separation was not run."""
    if stem_db is None or end > len(stem_db):
        return ""
    piece = np.asarray(stem_db[start:end], np.float64)
    if np.isfinite(piece).mean() < 0.8:  # separation was run on part of this stretch only
        return ""
    middle = float(np.nanmedian(piece))
    if middle >= cuts["separation_present_db"]:
        return "present"
    if middle <= cuts["separation_absent_db"]:
        return "absent"
    return ""


def recording_level_metadata(credits, match_confidence):
    """A line of context for a candidate: what the credits say about the *recording*.

    It is never evidence about a stretch of time, so nothing here may touch a class or a
    weight. Release- and artist-level credits say even less and are named as such."""
    reliable = match_confidence in ("high", "medium")
    scope = credits["scope"]
    if scope == "recording_level_guitar_credit":
        note = "guitar credited on this recording (MusicBrainz)"
    elif scope == "release_level_guitar_credit":
        note = "guitar credited on the release, not on this recording (MusicBrainz)"
    elif scope == "artist_level_guitarist":
        note = "the artist has a guitarist; no credit for this recording (MusicBrainz)"
    else:
        return ""
    return note if reliable else note + "; identity match not reliable"


def candidates_for(recording, cuts=CUTS):
    """Label candidates of one recording.

    ``recording``: {"id", "gmv": scores, "panns": scores, "level_db", "credits": summary,
    "match_confidence", "human": [(start_s, end_s, presence, type)], "separation": stem
    level per hop or None}. Returns (candidate rows, per-hop classes).
    """
    rid = recording["id"]
    rows = []
    for start, end, presence, kind in recording.get("human", []):
        label = {
            "present_throughout": "present",
            "absent": "absent",
            "present_intermittent": "intermittent",
        }.get(presence, "unknown")
        rows.append(
            {
                "recording_id": rid,
                "start_s": start,
                "end_s": end,
                "label": label,
                "guitar_type": kind if label == "present" else "",
                "evidence_category": "A",
                "confidence_class": "verified_reference",
                "supporting_sources": "human review (gmv review, core excerpt)",
                "conflicting_sources": "",
                "temporal_precision": "excerpt (12 s); an intermittent answer is not timed",
                "training_suitable": label in ("present", "absent"),
                "suggested_training_weight": WEIGHTS["verified_reference"]
                if label in ("present", "absent")
                else 0.0,
                "review_required": False,
            }
        )
    if recording.get("gmv") is None or recording.get("panns") is None:
        return rows, None
    classes = hop_classes(recording["gmv"], recording["panns"], recording["level_db"], cuts)
    min_hops = round(cuts["min_run_s"] / HOP_S)
    context = recording_level_metadata(recording["credits"], recording.get("match_confidence"))
    reviewed = [(a, b) for a, b, *_ in recording.get("human", [])]

    def overlaps_review(start_s, end_s):
        return any(start_s < b and end_s > a for a, b in reviewed)

    for value, label in ((2, "present"), (-2, "absent")):
        for first, last in runs(classes, value, min_hops):
            start_s, end_s = first * HOP_S, last * HOP_S
            if overlaps_review(start_s, end_s):
                continue  # a person already answered for this stretch
            # Interval-level sources only: what was measured over these hops.
            support = ["gmv recogniser", "PANNs CNN14"]
            against = []
            separated = separation_support(recording.get("separation"), first, last, cuts)
            if separated == label:
                support.append("separation guitar stem over this stretch")
            elif separated:
                against.append("separation guitar stem over this stretch")
            if against:
                confidence = "conflict"
            elif label == "present":
                confidence = "strong_agreement" if separated == "present" else "moderate_agreement"
            else:
                # On the owner's reviewed excerpts the two models agreeing on "no guitar"
                # was right about four times in five, on two separate samples. It is never
                # a label, whatever else agrees with it.
                confidence = "weak_evidence"
            usable = label == "present" and confidence in WEIGHTS
            rows.append(
                {
                    "recording_id": rid,
                    "start_s": start_s,
                    "end_s": end_s,
                    "label": label,
                    "guitar_type": "",
                    "evidence_category": CATEGORY[confidence],
                    "confidence_class": confidence,
                    "supporting_sources": "; ".join(support),
                    "conflicting_sources": "; ".join(against),
                    "recording_level_metadata": context if label == "present" else "",
                    "temporal_precision": "0.5 s hops; edges are model edges, not onsets",
                    "training_suitable": usable,
                    "suggested_training_weight": WEIGHTS[confidence] if usable else 0.0,
                    "review_required": confidence == "conflict",
                }
            )
    for first, last in runs(classes, 9, min_hops):
        rows.append(
            {
                "recording_id": rid,
                "start_s": first * HOP_S,
                "end_s": last * HOP_S,
                "label": "unknown",
                "guitar_type": "",
                "evidence_category": "D",
                "confidence_class": "conflict",
                "supporting_sources": "",
                "conflicting_sources": "gmv recogniser against PANNs CNN14",
                "temporal_precision": "0.5 s hops",
                "training_suitable": False,
                "suggested_training_weight": 0.0,
                "review_required": True,
            }
        )
    return rows, classes


def recording_view(classes, credits, rows):
    """A recording-level reading of the hop classes. Descriptive; not a label."""
    if classes is None:
        return {"preliminary_guitar_classification": "not_analysed"}
    audible = classes != -9
    total = max(1, int(audible.sum()))
    share = {
        "models_agree_present": float((classes == 2).sum() / total),
        "models_agree_absent": float((classes == -2).sum() / total),
        "models_conflict": float((classes == 9).sum() / total),
    }
    present = sum(r["end_s"] - r["start_s"] for r in rows if r["label"] == "present")
    absent = sum(r["end_s"] - r["start_s"] for r in rows if r["label"] == "absent")
    if present >= 20 and absent >= 20:
        view = "intermittent_candidate"
    elif share["models_agree_present"] >= 0.5 and share["models_agree_absent"] < 0.05:
        view = "guitar_through_most_candidate"
    elif share["models_agree_absent"] >= 0.7 and share["models_agree_present"] < 0.01:
        view = "guitar_absent_candidate"
    elif present >= 20:
        view = "guitar_in_parts_candidate"
    else:
        view = "uncertain"
    return {
        "preliminary_guitar_classification": view,
        **{k: round(v, 3) for k, v in share.items()},
        "candidate_present_s": round(present, 1),
        "candidate_absent_s": round(absent, 1),
        "silence_share": round(float((classes == -9).mean()), 3),
    }


def load_npz(path):
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def human_intervals(review_workdir):
    """{recording id: [(start_s, end_s, presence, type)]} of labelled core excerpts."""
    import csv

    review_workdir = Path(review_workdir)
    out = defaultdict(list)
    if not (review_workdir / "Human_Annotations.csv").is_file():
        return out
    with open(review_workdir / "Excerpt_Manifest.csv", newline="", encoding="utf-8-sig") as handle:
        spans = {
            r["excerpt_id"]: (float(r["source_start_s"]), float(r["source_end_s"]))
            for r in csv.DictReader(handle)
        }
    with open(review_workdir / "Human_Annotations.csv", newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["annotation_status"] != "labelled" or row["excerpt_category"] != "core":
                continue
            start, end = spans[row["excerpt_id"]]
            out[row["recording_id"]].append((start, end, row["guitar_presence"], row["guitar_type"]))
    return out


def summarise(candidates):
    by = Counter()
    seconds = Counter()
    for row in candidates:
        key = (row["confidence_class"], row["label"])
        by[key] += 1
        if row["start_s"] != "":  # recording-level rows carry no times
            seconds[key] += row["end_s"] - row["start_s"]
    return {
        " / ".join(k): {"intervals": by[k], "hours": round(seconds[k] / 3600, 2)} for k in sorted(by)
    }


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=1), encoding="utf-8")
