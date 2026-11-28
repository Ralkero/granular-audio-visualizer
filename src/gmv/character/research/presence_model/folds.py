"""Class-aware, capacity-limited, deterministic inner folds (protocol §9).

Input: per group, voiced time V and no-voice time N in seconds, after exclusions. Groups are
typed with the 2 s minimum: *both* (V ≥ 2 and N ≥ 2), *voice-only* or *no-voice-only*. A
group with less than 2 s of both classes cannot be typed; the protocol does not say what to
do with one, so it is refused (``ValueError``) rather than guessed.

Placement (capacity 4 groups per fold, 5 folds; targets V* = ΣV/5, N* = ΣN/5):

1. Order: single-class groups first, by type (voice-only before no-voice-only, the order the
   protocol lists them), then by descending total time; then two-class groups by descending
   total time. Ties: SHA-256 of "seed:key".
2. Candidates for every placement: folds holding fewer than 4 groups.
3. A single-class group goes to the candidate with the fewest groups of its type, then the
   least time of its class, then the seeded fold order (SHA-256 of "seed:fold:i").
4. A two-class group goes to the candidate minimising ((V_f + V)/V*)² + ((N_f + N)/N*)², then
   the seeded fold order.

Checks: every inner training partition keeps ≥ 65% of the voiced and of the no-voice time;
no fold holds more than ⌈n_type/5⌉ single-class groups of one type; no fold holds more than
4 groups. If a check fails, retry r = 1..99 orders the two-class groups by SHA-256 of
"seed:r:key" instead; if every attempt fails, ``FoldAssignmentError`` stops the run. Nothing
is relaxed automatically.

The assignment names groups and is private. ``public_summary`` holds only aggregates.
"""

import hashlib
import json
import math
from dataclasses import dataclass, field

from .seeds import FOLD_SEED

FOLDS = 5
CAPACITY = 4
MIN_CLASS_S = 2.0
MIN_TRAINING_SHARE = 0.65
RETRIES = 99
TYPES = ("both", "voice_only", "novoice_only")


class FoldAssignmentError(RuntimeError):
    """No assignment satisfies the registered checks; nothing may be relaxed silently."""


def _hash(*parts):
    return hashlib.sha256(":".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def group_type(voiced_s, novoice_s, minimum=MIN_CLASS_S):
    if voiced_s >= minimum and novoice_s >= minimum:
        return "both"
    if voiced_s >= minimum:
        return "voice_only"
    if novoice_s >= minimum:
        return "novoice_only"
    raise ValueError("a group has less than 2 s of either class; the protocol does not say how to place it")


@dataclass(frozen=True)
class Assignment:
    folds: dict  # group key -> fold index
    attempt: int
    summary: dict = field(compare=False)


def _validate(groups):
    if not groups:
        raise ValueError("no groups to assign")
    clean = {}
    for key, (voiced, novoice) in groups.items():
        voiced, novoice = float(voiced), float(novoice)
        if not (math.isfinite(voiced) and math.isfinite(novoice)) or voiced < 0 or novoice < 0:
            raise ValueError("durations must be finite and non-negative")
        clean[str(key)] = (voiced, novoice)
    return clean


def _place(groups, order, seed, folds, capacity):
    total_v = sum(v for v, _ in groups.values())
    total_n = sum(n for _, n in groups.values())
    if total_v <= 0 or total_n <= 0:
        raise FoldAssignmentError("the groups hold no voiced or no no-voice time; folds cannot be balanced")
    target_v, target_n = total_v / folds, total_n / folds
    members = [[] for _ in range(folds)]
    load_v, load_n = [0.0] * folds, [0.0] * folds
    rank = {i: _hash(seed, "fold", i) for i in range(folds)}
    for key in order:
        voiced, novoice = groups[key]
        kind = group_type(voiced, novoice)
        candidates = [i for i in range(folds) if len(members[i]) < capacity]
        if not candidates:
            raise FoldAssignmentError(f"{len(groups)} groups exceed the capacity of {folds} folds of {capacity}")
        if kind == "both":
            def cost(i, voiced=voiced, novoice=novoice):
                return (((load_v[i] + voiced) / target_v) ** 2 + ((load_n[i] + novoice) / target_n) ** 2, rank[i])
        else:
            def cost(i, kind=kind):
                same = sum(1 for k in members[i] if group_type(*groups[k]) == kind)
                return (same, load_v[i] if kind == "voice_only" else load_n[i], rank[i])
        chosen = min(candidates, key=cost)
        members[chosen].append(key)
        load_v[chosen] += voiced
        load_n[chosen] += novoice
    return members


def check(groups, members, capacity=CAPACITY, share=MIN_TRAINING_SHARE):
    """Registered checks; returns the failures found (empty when sound), with no group keys."""
    folds = len(members)
    total_v = sum(v for v, _ in groups.values())
    total_n = sum(n for _, n in groups.values())
    failures = []
    for i, fold in enumerate(members):
        if len(fold) > capacity:
            failures.append(f"fold {i} holds {len(fold)} groups, more than {capacity}")
        kept_v = total_v - sum(groups[k][0] for k in fold)
        kept_n = total_n - sum(groups[k][1] for k in fold)
        if kept_v < share * total_v - 1e-9 or kept_n < share * total_n - 1e-9:
            failures.append(f"the training partition without fold {i} keeps less than {share:.0%} of a class's time")
    for kind in ("voice_only", "novoice_only"):
        n_type = sum(1 for v, n in groups.values() if group_type(v, n) == kind)
        limit = math.ceil(n_type / folds) if n_type else 0
        for i, fold in enumerate(members):
            here = sum(1 for k in fold if group_type(*groups[k]) == kind)
            if here > limit:
                failures.append(f"fold {i} holds {here} {kind} groups, more than {limit}")
    return failures


def assign(groups, seed=FOLD_SEED, folds=FOLDS, capacity=CAPACITY, share=MIN_TRAINING_SHARE, retries=RETRIES):
    """Inner folds for ``groups`` ({key: (voiced_s, novoice_s)}); see the module docstring."""
    groups = _validate(groups)
    kinds = {k: group_type(v, n) for k, (v, n) in groups.items()}
    single = sorted((k for k in groups if kinds[k] != "both"),
                    key=lambda k: (TYPES.index(kinds[k]), -sum(groups[k]), _hash(seed, k)))
    double = [k for k in groups if kinds[k] == "both"]
    last = []
    for attempt in range(retries + 1):
        if attempt == 0:
            ordered = sorted(double, key=lambda k: (-sum(groups[k]), _hash(seed, k)))
        else:
            ordered = sorted(double, key=lambda k, r=attempt: _hash(seed, r, k))
        members = _place(groups, single + ordered, seed, folds, capacity)
        last = check(groups, members, capacity, share)
        if not last:
            assignment = {k: i for i, fold in enumerate(members) for k in fold}
            return Assignment(assignment, attempt, public_summary(groups, assignment, attempt, folds))
    raise FoldAssignmentError(f"no assignment passed the registered checks after {retries + 1} attempts; "
                              f"last failures: {'; '.join(sorted(set(last)))}")


def for_outer(groups, outer, **options):
    """Inner folds for one outer fold: the outer group is set aside completely first."""
    if outer not in groups:
        raise ValueError("the outer group is not among the groups")
    return assign({k: v for k, v in groups.items() if k != outer}, **options)


def public_summary(groups, assignment, attempt, folds=FOLDS):
    """Aggregates only: per fold, groups by type and seconds of each class; no keys."""
    total_v = sum(v for v, _ in groups.values())
    total_n = sum(n for _, n in groups.values())
    rows = []
    for i in range(folds):
        keys = [k for k, f in assignment.items() if f == i]
        voiced = sum(groups[k][0] for k in keys)
        novoice = sum(groups[k][1] for k in keys)
        rows.append({
            "fold": i,
            "groups": len(keys),
            "types": {t: sum(1 for k in keys if group_type(*groups[k]) == t) for t in TYPES},
            "voiced_s": round(voiced, 3),
            "novoice_s": round(novoice, 3),
            "training_share_voiced": round(1 - voiced / total_v, 4),
            "training_share_novoice": round(1 - novoice / total_n, 4),
        })
    return {"attempt": attempt, "folds": rows}


def serialise(assignment):
    """Canonical bytes of a (private) assignment, for its SHA-256 fingerprint."""
    return json.dumps({"attempt": assignment.attempt, "folds": dict(sorted(assignment.folds.items()))},
                      sort_keys=True, separators=(",", ":")).encode("utf-8")


def fingerprint(assignment):
    return hashlib.sha256(serialise(assignment)).hexdigest()
