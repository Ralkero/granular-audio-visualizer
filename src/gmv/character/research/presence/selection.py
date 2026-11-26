"""Deterministic candidate batches. No model output is used to choose anything.

* ``development_batch``: about 100 existing Phase A3 development excerpts for interval
  annotation, spread over the delivery answers (all Gutturals and Sing-Screams; then High
  and Low Screams, a few No-vocals excerpts to re-check, and Cleans), over artist groups
  (round robin in a seeded order) and over recordings (one excerpt per recording). The A3
  answer is kept only as the reason for selection; it is not a presence label.
* ``negative_candidates``: new ten-second windows in development-side recordings, proposed
  per confounder type from the inventory's metadata only (genre, content category, the
  "instrumental or vocal" flag and title words such as "solo", "bass", "drum"). They are
  candidates: the owner must listen and confirm there is no voice at all.
* ``final_test_batch``: new ten-second windows in the recordings of never-used artist groups,
  for a final-test cohort: up to ``FINAL_PER_GROUP`` recordings per group in a seeded order,
  one window each at the first fixed position that fits (``windows``). A group with no such
  window is not eligible for the cohort.
* ``coverage``: what the development side has per harsh delivery answer, for the gap report.

Every choice uses a seeded SHA-256 order, so the same inputs give the same batches.
"""

import hashlib
import re
from collections import defaultdict

SEED = "gmv-vocal-presence-a5-selection"
QUOTAS = (("gutturals", None), ("sing_screams", None), ("high_screams", 22), ("low_screams", 22),
          ("no_vocals", 10), ("cleans", None))  # None for the last: the rest of the target
TARGET = 100
EXCERPT_S = 10.0
EDGE_S = 1.0

# (genre rule needing the "instrumental" flag, title rule, content-category rule). Only
# genre or title evidence proposes a specific confounder; instrumental soundtrack tracks and
# song-intro windows are offered as "other" for the owner to tag while listening.
NEGATIVE_RULES = {
    "orchestral_ambient": (r"ambient|orchestr|classical|cinematic|chillout|new age", r"\borchestr|\bambient\b", None),
    "synth_keyboard": (r"electronic|synth|chiptune|edm|electro|techno|house|dubstep|8.bit", r"\bsynth|\bpiano\b|\bkeys\b", None),
    "distorted_rhythm_guitar": (r"metal|core|djent|hard rock|grunge|stoner|sludge|progressive", r"\binstrumental\b", r"^band$"),
    "guitar_lead_solo": (r"progressive|djent|neoclassical|shred|instrumental rock", r"\bsolo\b|\bguitar\b", r"^band$"),
    "bass_focused": (None, r"\bbass\b", None),
    "drums_only": (None, r"\bdrum|\bpercussion\b", None),
    "other_instrumental_soundtrack": (None, None, r"^soundtrack$"),
}
INTRO = "other_song_intro"
POSITIONS = (0.2, 0.5, 0.8, 0.35, 0.65)
FINAL_PER_GROUP = 5  # recordings per artist group in a final-test cohort, so no artist dominates
FINAL_SELECTION = "final-test window at a fixed position in a never-used artist's recording"


def order(seed, *parts):
    return hashlib.sha256((seed + ":" + ":".join(map(str, parts))).encode()).hexdigest()


def spread(items, group_of, seed, limit, taken_recordings):
    """Up to ``limit`` items, one per recording, cycling over groups in a seeded order."""
    by_group = defaultdict(list)
    for item in sorted(items, key=lambda i: order(seed, i["id"])):
        by_group[group_of(item)].append(item)
    groups = sorted(by_group, key=lambda g: order(seed, "group", g))
    picked = []
    while (limit is None or len(picked) < limit) and any(by_group.values()):
        for group in groups:
            while by_group[group]:
                item = by_group[group].pop(0)
                if item["recording"] not in taken_recordings:
                    picked.append(item)
                    taken_recordings.add(item["recording"])
                    break
            if limit is not None and len(picked) >= limit:
                break
    return picked


def development_batch(pool, target=TARGET, seed=SEED):
    """``pool``: dicts with ``id``, ``recording``, ``group``, ``category`` (the A3 answer or
    "no_vocals"). Returns the chosen items, each with ``selection`` set."""
    taken, chosen = set(), []
    for category, quota in QUOTAS:
        items = [i for i in pool if i["category"] == category]
        remaining = max(0, target - len(chosen))
        limit = min(remaining, quota if quota is not None else len(items))
        for item in spread(items, lambda i: i["group"], f"{seed}:{category}", limit, taken):
            chosen.append({**item, "selection": f"phase A3 development excerpt (delivery answer: {category})"})
    return chosen


def matches(pattern, text):
    return bool(pattern) and re.search(pattern, text or "", re.I) is not None


def negative_reasons(row):
    """{category: reason} for an inventory row, from metadata only."""
    genre = row.get("genre", "")
    title = row.get("title", "")
    category = row.get("content_category", "")
    instrumental = row.get("instrumental_or_vocal") == "Instrumental"
    out = {}
    for name, (genre_rule, title_rule, category_rule) in NEGATIVE_RULES.items():
        reasons = []
        if instrumental and matches(genre_rule, genre):
            reasons.append("instrumental recording whose genre matches")
        if matches(title_rule, title):
            reasons.append("its title mentions it")
        if instrumental and matches(category_rule, category):
            reasons.append(f"instrumental {category} recording")
        if reasons:
            out[name] = " and ".join(reasons)
    if row.get("instrumental_or_vocal") == "Vocal" and category == "band":
        out[INTRO] = "song intro window of a band recording (often instrumental; must be confirmed by listening)"
    return out


def windows(duration, taken, count=1, length=EXCERPT_S):
    """Up to ``count`` window starts at fixed fractions of a recording, clear of the edges and
    of ``taken`` spans (start, end)."""
    starts = []
    for fraction in POSITIONS:
        start = round(EDGE_S + fraction * max(0.0, duration - length - 2 * EDGE_S), 3)
        if duration < length + 2 * EDGE_S:
            break
        if all(start + length <= a or start >= b for a, b in [*taken, *[(s, s + length) for s in starts]]):
            starts.append(start)
        if len(starts) >= count:
            break
    return starts


def negative_candidates(recordings, per_category=20, seed=SEED):
    """``recordings``: dicts with ``sha256``, ``group``, ``duration_s``, ``row`` (inventory row)
    and ``taken`` (spans already used in that recording). Returns {category: [candidates]},
    each candidate with ``recording``, ``group``, ``start_s``, ``selection``."""
    out, used = {}, defaultdict(list)
    for category in (*NEGATIVE_RULES, INTRO):
        items = []
        for rec in recordings:
            reason = negative_reasons(rec["row"]).get(category)
            if reason:
                items.append({"id": order(seed, category, rec["sha256"]), "recording": rec["sha256"], "rec": rec, "reason": reason})
        picked = spread(items, lambda i: i["rec"]["group"], f"{seed}:{category}", per_category, set())
        chosen = []
        for item in picked:
            rec = item["rec"]
            if category == INTRO:
                start = EDGE_S
                clear = all(start + EXCERPT_S <= a or start >= b for a, b in rec["taken"] + used[rec["sha256"]])
                starts = [start] if clear and rec["duration_s"] >= EXCERPT_S + 2 * EDGE_S else []
            else:
                starts = windows(rec["duration_s"], rec["taken"] + used[rec["sha256"]])
            if not starts:
                continue
            used[rec["sha256"]].append((starts[0], starts[0] + EXCERPT_S))
            chosen.append({"recording": rec["sha256"], "group": rec["group"], "start_s": starts[0],
                           "selection": f"negative candidate, {category}: {item['reason']}"})
        out[category] = chosen
    return out


def final_test_batch(recordings, per_group=FINAL_PER_GROUP, seed=SEED):
    """``recordings``: dicts with ``sha256``, ``group``, ``duration_s`` and optionally ``taken``
    (spans already used). Returns the chosen windows, each with ``recording``, ``group``,
    ``start_s`` and ``selection``: at most ``per_group`` recordings per group, in a seeded
    order, one window each. The same inputs, in any order, give the same batch."""
    by_group, seen = defaultdict(list), set()
    for rec in sorted(recordings, key=lambda r: order(seed, "final_test", r["sha256"])):
        if rec["sha256"] not in seen:  # one entry per recording, even if the inventory repeats it
            seen.add(rec["sha256"])
            by_group[rec["group"]].append(rec)
    chosen = []
    for group in sorted(by_group, key=lambda g: order(seed, "final_test", "group", g)):
        picked = 0
        for rec in by_group[group]:
            if picked >= per_group:
                break
            starts = windows(rec["duration_s"], list(rec.get("taken", [])))
            if starts:
                chosen.append({"recording": rec["sha256"], "group": group, "start_s": starts[0], "selection": FINAL_SELECTION})
                picked += 1
    return chosen


def coverage(pool):
    """Per delivery answer: excerpts, artist groups and recordings in the development pool."""
    out = {}
    for item in pool:
        entry = out.setdefault(item["category"], {"excerpts": 0, "groups": set(), "recordings": set()})
        entry["excerpts"] += 1
        entry["groups"].add(item["group"])
        entry["recordings"].add(item["recording"])
    return {k: {"excerpts": v["excerpts"], "groups": len(v["groups"]), "recordings": len(v["recordings"])}
            for k, v in sorted(out.items())}
