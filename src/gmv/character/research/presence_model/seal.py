"""Step 4, S4.8: the prediction seal and its verifier (protocol Amendment 3, A3.5 N2) *[tested
before use]*. Standard library only. Built and tested on generated runs only.

**The seal** is write-once canonical JSON: format and version, the run and step identity, the code
commit, the integrity-record and fold-index hashes, the seed format and enumeration hash, the
policy, the adopted epochs, the thresholds file's hash and the inventory of model and prediction
files with their hashes. It is created exclusively, then made read-only; its digest goes into the
run record (and the RG4 summary).

**The verifier**, in this order:

1. ``expectation`` builds the expected inventory **only** from the registered design (arms, roles,
   folds, units and the seed plan's identities), the published decisions (Amendment 5's fold index,
   checked against its hash, and its sensitivity sets; Amendment 6's policy and epochs) and the run
   declaration, which must agree with those decisions. The seal is not an input.
2. The expectation's digest is written to the run record (``record_expectation``) before the seal
   file is opened.
3. ``verify`` opens the seal and refuses, each with its own reason: a model identity outside the
   list; a seed outside the declared policy's set; an undeclared rerun, or a declared rerun outside
   Amendment 5's sets; an evaluation unit not in the fold index; a relabelled entry (its identity
   does not match its fields); duplicates; missing entries; extra files on disk; any header field
   that differs from the declaration; a model or prediction file whose bytes do not match the
   hash in the seal.
4. Only a verified seal (``Verified``, held in memory) releases category tokens and the low-rate
   metrics.

Inventory entry: namespace (``main``, ``sensT`` or ``sensV``), arm, role (``inner``, ``outer``,
``final``), outer position o (``final`` for final models), inner fold i (``outer`` or ``final``),
seed index, the seed identity string, the model file and its prediction file (final models have
none before step 7) with their hashes.
"""

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

from . import seedplan

FORMAT = "a5.2-seal-v1"
VERSION = 1
ARMS = ("B1", "C")
POLICIES = {"C": {"inner": (0, 1), "outer": (0, 1, 2, 3, 4)}, "D": {"inner": (0,), "outer": (0, 1, 2)}}
NAMESPACES = ("main", "sensT", "sensV")
HEADER = ("format", "version", "run", "step", "commit", "integrity_sha256", "fold_index_sha256", "seed_format",
          "seed_enumeration_sha256", "policy", "epochs", "thresholds_sha256")


class SealError(RuntimeError):
    """The seal does not match the expectation; nothing after it may run."""


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


# -- the expectation -----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Declaration:
    """The run declaration's parts the verifier uses."""

    run: str
    step: str
    commit: str
    policy: str
    epochs: dict  # {"C": n, "B1": n}
    reruns: tuple  # declared sensitivity namespaces, e.g. ("sensT",)


def _identity(namespace, arm, o, i, s):
    head = "gmv-a5.2" if namespace == "main" else f"gmv-a5.2-{namespace}"
    return f"{head}:{arm}:{o}:{i}:{s}"


def _file(namespace, arm, role, o, i, s):
    return f"models/{namespace}/{arm}/{o}-{i}-{s}.model"


def _predictions(namespace, arm, role, o, i, s):
    return None if role == "final" else f"predictions/{namespace}/{arm}/{o}-{i}-{s}.json"


def check_decisions(fold_index_bytes, amendment5, amendment6, declaration):
    """The published decisions and the declaration must agree; returns the fold index."""
    if sha256(fold_index_bytes) != amendment5["fold_index_sha256"]:
        raise SealError("the fold index does not match Amendment 5's hash")
    index = json.loads(fold_index_bytes)
    if declaration.policy != amendment6["policy"] or declaration.policy not in POLICIES:
        raise SealError("the run declaration's policy differs from Amendment 6")
    if dict(declaration.epochs) != dict(amendment6["epochs"]):
        raise SealError("the run declaration's epochs differ from Amendment 6")
    if any(ns not in amendment5.get("sensitivity_sets", ()) for ns in declaration.reruns):
        raise SealError("a declared rerun is outside Amendment 5's sensitivity sets")
    if any(ns not in index for ns in ("main", *declaration.reruns)):
        raise SealError("the fold index lacks a declared namespace")
    return index


def expectation(fold_index_bytes, amendment5, amendment6, declaration, seed_plan=None):
    """The expected inventory (without hashes), from the design, the decisions and the
    declaration only. ``fold_index``: {namespace: {outer position: [inner fold numbers]}}."""
    index = check_decisions(fold_index_bytes, amendment5, amendment6, declaration)
    final = {b: f for b, f, _n in (seed_plan or seedplan.plan(declaration.policy)).identities}
    seeds = POLICIES[declaration.policy]
    entries = []
    for namespace in ("main", *declaration.reruns):
        for arm in ARMS:
            for o, inner in sorted(index[namespace].items(), key=lambda kv: int(kv[0])):
                for i in sorted(int(x) for x in inner):
                    entries += [(namespace, arm, "inner", o, i, s) for s in seeds["inner"]]
                entries += [(namespace, arm, "outer", o, "outer", s) for s in seeds["outer"]]
            if namespace == "main":  # sensitivity reruns have no final models (A3.7)
                entries += [(namespace, arm, "final", "final", "final", s) for s in seeds["outer"]]
    inventory = []
    for namespace, arm, role, o, i, s in entries:
        base = _identity(namespace, arm, o, i, s)
        inventory.append({"namespace": namespace, "arm": arm, "role": role, "outer": str(o), "inner": str(i), "seed": s,
                          "identity": final.get(base, base), "model_file": _file(namespace, arm, role, o, i, s),
                          "predictions_file": _predictions(namespace, arm, role, o, i, s)})
    inventory.sort(key=lambda e: (e["namespace"], e["arm"], e["role"], e["outer"], e["inner"], e["seed"]))
    header = {"format": FORMAT, "version": VERSION, "run": declaration.run, "step": declaration.step,
              "commit": declaration.commit, "policy": declaration.policy, "epochs": dict(declaration.epochs)}
    return {"header": header, "inventory": inventory}


def expectation_digest(expected):
    return sha256(canonical(expected))


def record_expectation(run_record_path, expected):
    """Write the expectation's digest to the run record (created new) before the seal is opened."""
    fd = os.open(Path(run_record_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump({"expectation_sha256": expectation_digest(expected)}, file)
    return expectation_digest(expected)


# -- writing ---------------------------------------------------------------------------------------


def write(path, header, inventory):
    """Create the seal exclusively, then make it read-only; returns its SHA-256."""
    missing = [k for k in HEADER if k not in header]
    if missing:
        raise SealError("the seal header is incomplete")
    data = canonical({"header": header, "inventory": sorted(
        inventory, key=lambda e: (e["namespace"], e["arm"], e["role"], e["outer"], e["inner"], e["seed"]))})
    fd = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.chmod(path, 0o444)
    return sha256(data)


# -- verification ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Verified:
    """A seal that matched its expectation, held in memory (tokens and low-rate metrics need it)."""

    seal_sha256: str
    expectation_sha256: str
    entries: int


KEY = ("namespace", "arm", "role", "outer", "inner", "seed")


def _key(entry):
    return tuple(entry.get(k) for k in KEY)


def verify(seal_path, root, expected, run_record_path, declaration, amendment5):
    """Verify the seal against the expectation already recorded in the run record; ``root`` holds
    the model and prediction files. Returns ``Verified`` or raises ``SealError``."""
    record = json.loads(Path(run_record_path).read_text(encoding="utf-8"))
    if record.get("expectation_sha256") != expectation_digest(expected):
        raise SealError("the run record does not hold this expectation's digest")
    data = Path(seal_path).read_bytes()
    try:
        seal = json.loads(data)
        header, inventory = seal["header"], list(seal["inventory"])
    except (ValueError, KeyError, TypeError):
        raise SealError("the seal is malformed") from None
    for k, v in expected["header"].items():
        if header.get(k) != v:
            raise SealError(f"the seal's {k} differs from the declaration")
    seeds = POLICIES[declaration.policy]
    allowed = {"main", *declaration.reruns}
    sets = set(amendment5.get("sensitivity_sets", ()))
    expected_keys = {_key(e): e for e in expected["inventory"]}
    units = {(e["namespace"], e["outer"], e["inner"]) for e in expected["inventory"]}
    seen = set()
    for entry in inventory:
        k = _key(entry)
        if k in seen:
            raise SealError("the seal lists an entry twice")
        seen.add(k)
        ns, arm, role, seed = entry.get("namespace"), entry.get("arm"), entry.get("role"), entry.get("seed")
        if ns not in allowed:
            if ns in sets:
                raise SealError("the seal holds an undeclared rerun")
            raise SealError("the seal holds a rerun outside Amendment 5's sets")
        if arm not in ARMS or role not in ("inner", "outer", "final"):
            raise SealError("the seal holds a model identity outside the list")
        if seed not in seeds["inner" if role == "inner" else "outer"]:
            raise SealError("the seal holds a seed outside the declared policy's set")
        if (ns, entry.get("outer"), entry.get("inner")) not in units:
            raise SealError("the seal holds an evaluation unit that is not in the fold index")
        if k not in expected_keys:
            raise SealError("the seal holds a model identity outside the list")
        wanted = expected_keys[k]
        for field in ("identity", "model_file", "predictions_file"):
            if entry.get(field) != wanted[field]:
                raise SealError("the seal holds a relabelled entry")
    if seen != set(expected_keys):
        raise SealError("the seal is missing entries")
    on_disk = {str(p.relative_to(root)) for p in Path(root).rglob("*") if p.is_file()}
    listed = {e["model_file"] for e in inventory} | {e["predictions_file"] for e in inventory if e["predictions_file"]}
    if on_disk != listed:
        raise SealError("the files on disk differ from the seal's inventory")
    for entry in inventory:
        for field, digest in (("model_file", "model_sha256"), ("predictions_file", "predictions_sha256")):
            if entry[field] is None:
                continue
            if sha256((Path(root) / entry[field]).read_bytes()) != entry.get(digest):
                raise SealError("a model or prediction file does not match its hash in the seal")
    return Verified(sha256(data), expectation_digest(expected), len(inventory))
