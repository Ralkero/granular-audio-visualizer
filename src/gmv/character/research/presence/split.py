"""The partition registry: which artist group belongs to which partition, for good.

Partitions:

* ``development`` and ``validation``: groups already used for development in Phase A3/A4
  (never for a final test). New ones are walked in a fixed hash order of their keys; a group
  goes to validation while validation holds under 20% of the recordings and would stay at
  or below 30% with it, otherwise to development. Deterministic and independent of any
  model; weighting by recordings keeps validation from ending up a handful of tiny groups;
* ``final_test``: only groups never used in any earlier vocal experiment, and only once the
  owner freezes the cohort (``freeze_final_test``); until then they are ``reserved``;
* ``reserved``: never-used groups held back, unused by development or validation, waiting
  for a final-test cohort;
* ``excluded``: groups that touch consumed evaluation material (the Phase A4 held-out
  artist groups, MUSDB18-HQ test artists) or anything linked to them. Never used again.

**Groups never move.** The registry keeps, for each group, the identity nodes it was
assigned with. When groups are rebuilt (new recordings, a new alias), a group that shares a
node with registered entries keeps their partition and gains the new nodes; a group whose
nodes match entries of two different partitions is a conflict, and nothing is written
until the owner resolves it. Rebuilt groups that touch the same registered entry are linked
through it, so the entries they join are merged once. A merge of frozen final-test groups is
also a conflict: a frozen cohort never changes silently. The only transition is
``reserved`` to ``final_test``, made by an explicit freeze.
"""

import hashlib
from collections import defaultdict
from datetime import UTC, datetime

PARTITIONS = ("development", "validation", "final_test", "reserved", "excluded")
SEED = "gmv-vocal-presence-a5"
VALIDATION_SHARE = 0.2
VALIDATION_CAP = 0.3


class ConflictError(ValueError):
    """A rebuilt group joins groups already registered in different partitions."""


class FreezeRefused(ValueError):
    """The final-test cohort cannot be frozen (too few eligible groups)."""


def now():
    return datetime.now(UTC).isoformat(timespec="seconds")


def unit(key, seed=SEED):
    return int(hashlib.sha256(f"{seed}:{key}".encode()).hexdigest()[:8], 16) / 2**32


def consumed_nodes(a3_evaluation_keys=(), consumed_names=()):
    """Identity nodes of consumed evaluation material."""
    from .identity import normalise

    return {f"a3:{k}" for k in a3_evaluation_keys} | {f"name:{normalise(n)}" for n in consumed_names if normalise(n)}


def assign(groups, registry, consumed, development_keys, seed=SEED):
    """A new registry: every group placed, none moved. ``groups``: identity.build()["groups"];
    ``registry``: the current registry (``{"entries": {key: entry}}``) or empty;
    ``consumed``: identity nodes of consumed material; ``development_keys``: A3 development
    artist keys. Raises ``ConflictError`` before changing anything."""
    entries = {k: dict(v, nodes=list(v["nodes"])) for k, v in (registry.get("entries") or {}).items()}
    by_node = {}
    for key, entry in entries.items():
        for node in entry["nodes"]:
            by_node.setdefault(node, set()).add(key)
    matched_of = {key: set().union(*(by_node.get(n, set()) for n in group["nodes"])) if group["nodes"] else set()
                  for key, group in groups.items()}
    # Entries matched by one rebuilt group are one artist; so are entries reached through a
    # shared entry. Join them (union-find), so every rebuilt group of a component sees the
    # same entries and the merge below never uses an entry it has already merged away.
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for matched in matched_of.values():
        first, *rest = sorted(matched) or [None]
        for other in rest:
            a, b = find(first), find(other)
            if a != b:
                parent[max(a, b)] = min(a, b)
    joined = defaultdict(set)
    for entry_key in set().union(*matched_of.values()) if matched_of else ():
        joined[find(entry_key)].add(entry_key)
    conflicts, plan, candidates = [], {}, []
    for key, group in sorted(groups.items()):
        nodes = set(group["nodes"])
        matched = joined[find(min(matched_of[key]))] if matched_of[key] else set()
        partitions = {entries[m]["partition"] for m in matched}
        if (len(partitions) > 1 or (partitions and partitions != {"excluded"} and nodes & consumed)
                or (len(matched) > 1 and "final_test" in partitions)):
            # two partitions joined, a used group newly linked to consumed material, or a frozen cohort changed
            conflicts.append(key)
            continue
        if matched:
            plan[key] = ("keep", sorted(matched), partitions.pop())
        elif nodes & consumed:
            plan[key] = ("new", [], "excluded")
        elif any(n.startswith("a3:") and n[3:] in development_keys for n in nodes):
            candidates.append(key)  # development or validation, decided below by weight
        else:
            plan[key] = ("new", [], "reserved")
    if conflicts:
        raise ConflictError(f"{len(conflicts)} rebuilt group(s) join groups registered in different partitions, "
                            "link a used group to consumed material, or merge frozen final-test groups; "
                            "resolve the identity links before preparing again")

    def weight(nodes):
        return sum(1 for n in nodes if n.startswith("rec:"))

    held = [k for k, (_how, _m, p) in plan.items() if p in ("development", "validation")]
    total = sum(weight(groups[k]["nodes"]) for k in held + candidates) or 1
    validation = sum(weight(groups[k]["nodes"]) for k in held if plan[k][2] == "validation")
    for key in sorted(candidates, key=lambda k: unit(k, seed)):
        size = weight(groups[key]["nodes"])
        if validation / total < VALIDATION_SHARE and (validation + size) / total <= VALIDATION_CAP:
            plan[key] = ("new", [], "validation")
            validation += size
        else:
            plan[key] = ("new", [], "development")
    for key, (how, matched, partition) in plan.items():
        nodes = set(groups[key]["nodes"])
        if how == "keep":
            target = matched[0]  # the smallest key of the component: never merged away
            for other in matched[1:]:  # same partition: merge the entries into one
                if other in entries:  # not already merged by another group of the component
                    nodes |= set(entries.pop(other)["nodes"])
            entries[target]["nodes"] = sorted(set(entries[target]["nodes"]) | nodes)
        else:
            entries[key] = {"partition": partition, "nodes": sorted(nodes), "assigned_utc": now(),
                            "reason": {"excluded": "touches consumed evaluation material",
                                       "reserved": "never used: held back for a future final test",
                                       "development": "earlier development artist", "validation": "earlier development artist"}[partition]}
    # Keep the registry's other top-level facts (such as when the final test was frozen).
    kept = {k: v for k, v in (registry or {}).items() if k not in ("format", "version", "seed", "entries")}
    return {"format": "gmv-vocal-presence-registry", "version": 1, "seed": seed, **kept, "entries": dict(sorted(entries.items()))}


def partition_of(registry, nodes):
    """The partition of the entry sharing any of ``nodes``, or None."""
    nodes = set(nodes)
    found = {e["partition"] for e in registry.get("entries", {}).values() if nodes & set(e["nodes"])}
    if len(found) > 1:
        raise ConflictError("nodes belong to entries of different partitions")
    return found.pop() if found else None


def is_frozen(registry):
    """True once a final-test cohort has been frozen (or a freeze was partly recorded)."""
    return bool(registry.get("final_test_frozen_utc") or registry.get("final_test_cohort_sha256")) or any(
        e["partition"] == "final_test" for e in registry.get("entries", {}).values())


def freeze_final_test(registry, eligible, minimum):
    """A new registry in which every eligible ``reserved`` group is ``final_test``: the cohort is
    then fixed. ``eligible``: registry keys that have enough suitable material. Only reserved
    groups can move, so development, validation and excluded groups never join the cohort.
    Refused when fewer than ``minimum`` are eligible, and refused when a cohort is already
    frozen: a frozen cohort is never replaced or extended. The input is not changed."""
    if is_frozen(registry):
        raise FreezeRefused("a final-test cohort is already frozen; it is never replaced or extended")
    reserved = [k for k, e in registry["entries"].items() if e["partition"] == "reserved" and k in eligible]
    if len(reserved) < minimum:
        raise FreezeRefused(f"only {len(reserved)} eligible never-used artist group(s); at least {minimum} are needed")
    out = {**registry, "entries": {k: dict(v) for k, v in registry["entries"].items()}}
    stamp = now()
    for key in reserved:
        out["entries"][key].update(partition="final_test", frozen_utc=stamp, reason="final-test cohort, frozen")
    out["final_test_frozen_utc"] = stamp
    return out
