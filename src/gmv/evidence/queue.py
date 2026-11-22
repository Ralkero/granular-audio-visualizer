"""Choose the stretches a person should listen to: where listening is expected to change
something, plus a random sample of what was classified automatically (to measure how often
that is wrong). Everything else is left alone.

Reasons (``queue``), which are also the filters in the review window:

``model_disagreement``      the two audio models contradict each other
``possible_false_positive`` the recogniser reports guitar where PANNs hears keys, strings or
                            synthesizer and no guitar
``possible_false_negative`` PANNs hears guitar where the recogniser reports none
``suspected_acoustic`` / ``suspected_clean`` / ``suspected_distorted``
                            as above, split by which kind the models point to
``metadata_mismatch``       a recording-level guitar credit, but both models hear no guitar
                            for most of the recording
``low_quality_source``      a low-bitrate source among recordings with label candidates
``random_audit``            a seeded sample of automatic label candidates

Nothing is chosen because of how a person labelled it before, and stretches that already
have a human label are skipped.
"""

import hashlib
from collections import Counter

import numpy as np

from .fusion import HOP_S, runs

EXCERPT_S = 12.0
LIMITS = {
    "model_disagreement": 40,
    "possible_false_positive": 30,
    "possible_false_negative": 25,
    "suspected_acoustic": 15,
    "suspected_clean": 15,
    "suspected_distorted": 15,
    "metadata_mismatch": 20,
    "low_quality_source": 6,
    "random_audit": 90,
}
PRIORITY = {
    "model_disagreement": 1,
    "possible_false_positive": 1,
    "possible_false_negative": 1,
    "metadata_mismatch": 2,
    "suspected_acoustic": 2,
    "suspected_clean": 2,
    "suspected_distorted": 2,
    "random_audit": 2,
    "low_quality_source": 3,
}
PER_RECORDING, PER_ARTIST = 2, 8
SECONDS_PER_EXCERPT = 20  # listening once and answering, from the pilot batch


def _order(seed, *parts):
    return hashlib.sha256(":".join((seed, *map(str, parts))).encode()).hexdigest()


def _centre(first, last, duration):
    middle = (first + last) / 2 * HOP_S
    return round(max(0.0, min(middle - EXCERPT_S / 2, max(0.0, duration - EXCERPT_S))), 1)


def proposals(recording):
    """Every stretch of one recording worth a listen, as (queue, start_s, why, disagree).

    ``recording``: {"classes", "gmv": {"guitar", "distorted", "clean", "acoustic"},
    "panns": {class: scores}, "credits", "view", "candidates", "duration", "row"}.
    """
    classes = recording["classes"]
    if classes is None:
        return []
    out = []
    gmv, panns = recording["gmv"], recording["panns"]
    count = len(classes)
    duration = recording["duration"]
    guitar = panns["guitar"][:count]
    others = np.max([panns[k][:count] for k in ("piano", "synthesizer", "string_section", "violin", "organ")], axis=0)
    own = gmv["guitar"][:count]
    audible = classes != -9
    late = np.arange(count) >= round(10 / HOP_S)  # the recogniser has its full context
    min_hops = round(6 / HOP_S)

    fooled = (own >= 0.926) & (guitar <= 0.01) & (others >= 0.3) & audible & late
    for first, last in runs(fooled, True, min_hops):
        out.append(
            (
                "possible_false_positive",
                _centre(first, last, duration),
                "recogniser reports guitar; PANNs hears keys, strings or synthesizer and no guitar",
                "gmv recogniser: guitar; PANNs: no guitar, other instrument",
            )
        )
    missed = (own <= 0.2) & (guitar >= 0.3) & audible & late
    for first, last in runs(missed, True, min_hops):
        acoustic = float(panns["acoustic_guitar"][first:last].mean())
        electric = float(panns["electric_guitar"][first:last].mean())
        if acoustic >= 0.2 and acoustic >= electric:
            queue, kind = "suspected_acoustic", "acoustic guitar"
        elif electric >= 0.2 and float(gmv["distorted"][first:last].mean()) < 0.2:
            queue, kind = "suspected_clean", "electric guitar without distortion"
        else:
            queue, kind = "possible_false_negative", "guitar"
        out.append(
            (
                queue,
                _centre(first, last, duration),
                f"PANNs hears {kind}; the recogniser reports none",
                "PANNs: guitar; gmv recogniser: no guitar",
            )
        )
    heavy = (gmv["distorted"][:count] >= 0.9) & (guitar <= 0.003) & (others < 0.3) & audible & late
    for first, last in runs(heavy, True, min_hops):
        out.append(
            (
                "suspected_distorted",
                _centre(first, last, duration),
                "recogniser reports distorted guitar; PANNs hears no guitar and no keys or strings",
                "gmv recogniser: distorted guitar; PANNs: no guitar",
            )
        )
    taken = {start for _, start, *_ in out}
    for first, last in runs(classes, 9, min_hops):
        start = _centre(first, last, duration)
        if all(abs(start - t) >= EXCERPT_S for t in taken):
            out.append(
                (
                    "model_disagreement",
                    start,
                    "the two audio models contradict each other here",
                    "gmv recogniser against PANNs",
                )
            )
    view = recording["view"]
    if (
        recording["credits"]["scope"] == "recording_level_guitar_credit"
        and view.get("models_agree_absent", 0) >= 0.6
    ):
        for first, last in runs(classes, -2, min_hops)[:1]:
            out.append(
                (
                    "metadata_mismatch",
                    _centre(first, last, duration),
                    "a guitarist is credited on this recording; both models hear no guitar for most of it",
                    "MusicBrainz recording credit against both audio models",
                )
            )
    return out


def build(recordings, seed="gmv-evidence-v1", limits=LIMITS):
    """Queue rows from per-recording proposals plus the audit sample."""
    pool = []
    for recording in recordings:
        reviewed = recording.get("human", [])
        for queue, start, why, disagree in proposals(recording):
            if any(start < b and start + EXCERPT_S > a for a, b, *_ in reviewed):
                continue
            pool.append((queue, recording, start, why, disagree))
        row = recording["row"]
        try:
            low = float(row.get("bitrate_kbps") or 999) < 128
        except ValueError:
            low = False
        for candidate in recording["candidates"]:
            if candidate["confidence_class"] not in ("strong_agreement", "moderate_agreement"):
                continue
            if candidate["end_s"] - candidate["start_s"] < EXCERPT_S:
                continue
            # one audit stretch per candidate interval, at a seeded position inside it
            room = candidate["end_s"] - candidate["start_s"] - EXCERPT_S
            draw = int(_order(seed, recording["id"], candidate["start_s"])[:8], 16) / 0xFFFFFFFF
            start = round(candidate["start_s"] + draw * room, 1)
            why = (
                f"audit of an automatic candidate: {candidate['label']}, "
                f"{candidate['confidence_class'].replace('_', ' ')}"
            )
            pool.append(("random_audit", recording, start, why, ""))
            if low:
                pool.append(
                    ("low_quality_source", recording, start, f"source is {row['bitrate_kbps']} kbit/s; {why}", "")
                )
    pool.sort(key=lambda item: _order(seed, item[0], item[1]["id"], item[2]))
    chosen, per_queue, per_recording, per_artist = [], Counter(), Counter(), Counter()
    audit_classes = Counter()
    order = sorted(limits, key=lambda q: (PRIORITY[q], q))
    for queue in order:
        for name, recording, start, why, disagree in pool:
            if name != queue or per_queue[queue] >= limits[queue]:
                continue
            artist = recording["row"]["top_folder"]
            if per_recording[recording["id"]] >= PER_RECORDING or per_artist[artist] >= PER_ARTIST:
                continue
            if any(c["recording_id"] == recording["id"] and abs(c["start_s"] - start) < EXCERPT_S for c in chosen):
                continue
            if queue == "random_audit":
                # equal room for each kind of candidate, so rare kinds are measured too
                kind = why.split(": ", 1)[1]
                if audit_classes[kind] >= limits[queue] // 3:
                    continue
                audit_classes[kind] += 1
            per_queue[queue] += 1
            per_recording[recording["id"]] += 1
            per_artist[artist] += 1
            chosen.append(
                {
                    "recording_id": recording["id"],
                    "artist": recording["row"]["artist"],
                    "title": recording["row"]["title"],
                    "relative_path": recording["row"]["relative_path"],
                    "start_s": start,
                    "end_s": round(start + EXCERPT_S, 1),
                    "queue": queue,
                    "why_selected": why,
                    "systems_that_disagree": disagree,
                    "to_resolve": "is guitar audible in this stretch, and which kind",
                    "priority": PRIORITY[queue],
                    "estimated_review_s": SECONDS_PER_EXCERPT,
                    "artist_used_in_earlier_training": recording["trained_artist"],
                }
            )
    return chosen


def selection(queue_rows, name="evidence-queue-v1", leakage_groups=""):
    """A ``gmv review build`` selection file that cuts exactly the queued stretches."""
    picks = {}
    for row in queue_rows:
        pick = picks.setdefault(
            row["relative_path"],
            {
                "path": row["relative_path"],
                "category": "mixed_candidate",
                "reason": "evidence review queue",
                "excerpts": [],
            },
        )
        pick["excerpts"].append(
            {"start_s": row["start_s"], "queue": row["queue"], "reason": row["why_selected"]}
        )
    return {
        "name": name,
        "seed": name,
        "leakage_groups": leakage_groups,
        "picks": list(picks.values()),
        "quotas": [],
    }
