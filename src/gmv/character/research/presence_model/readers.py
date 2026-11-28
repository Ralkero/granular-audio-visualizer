"""Step 2a: the strict readers of step 2's inputs (protocol Amendment 3, A3.4 and A3.6).

Each reader opens exactly one complete file, computes its SHA-256 from the same bytes it
parses, keeps only the fields Amendment 3 lets it pass on, and stops with a fixed message that
carries counts only (never a name, title, key, path or value) on any fault. JSON objects with a
repeated key are refused, so a later duplicate can never silently win.

**File level and field level.** Access recording (strace) shows which complete files a process
opens; it cannot show which fields were read. What each reader opens, keeps and must never
print:

=================  =====================================  ======================================
Reader             Keeps                                  Never printed, logged or reported
=================  =====================================  ======================================
``read_corpus``    MUSDB records: track id, corpus        track ids; every other record's
                   split, checksum; other records: their  fields
                   ``dataset`` only (as a count)
``read_registry``  development and validation entries:    group keys, names, recording
                   partition and nodes; every other       checksums of any entry
                   entry: partition only
``read_aliases``   the alias map and the links            every spelling and reason
                   (normalised)
``read_titles``    titles of the assessed groups'         every title; every other title
                   recordings only
``read_readings``  one reading per ambiguous credit       the credits and reasons
``read_decisions`` the rejected (track id, group) pairs   the pairs
=================  =====================================  ======================================

Built and tested on generated files only; step 2 itself needs its own authorisation.
"""

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from gmv.corpus.schema import CorpusManifest

from ..presence import identity, project
from ..presence.split import PARTITIONS

DATASET = "musdb18hq"
SHA256 = re.compile(r"^[0-9a-f]{64}$")
ASSESSED = ("development", "validation")
REGISTRY_KEYS = {"format", "version", "seed", "entries", "final_test_frozen_utc", "final_test_cohort_sha256"}
ENTRY_KEYS = {"partition", "nodes", "assigned_utc", "reason", "frozen_utc"}
NODE_PREFIXES = ("rec:", "name:", "mbid:", "leak:", "audio:", "version:", "a3:")
READINGS_FORMAT = "a5.2-musdb-credit-readings"
DECISIONS_FORMAT = "a5.2-musdb-review-decisions"
READINGS = ("whole-and-pieces", "single-name")
REJECTION_REASONS = ("not-the-same-artist",)


class ReaderError(ValueError):
    """An input is missing, malformed or ambiguous; step 2 stops. Messages carry counts only."""


@dataclass(frozen=True)
class Loaded:
    """One input: what the reader keeps, and the SHA-256 of the exact bytes it parsed."""

    value: object
    sha256: str

    def __repr__(self):
        return f"<{type(self.value).__name__} input, sha256 {self.sha256[:12]}…>"


def _unique_pairs(pairs):
    out = {}
    for key, value in pairs:
        if key in out:
            raise ReaderError("a JSON object repeats a key")
        out[key] = value
    return out


def _load(path, what):
    path = Path(path)
    if not path.is_file():
        raise ReaderError(f"the {what} file is missing")
    data = path.read_bytes()
    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=_unique_pairs)
    except ReaderError:
        raise ReaderError(f"the {what} file repeats a key in a JSON object") from None
    except (UnicodeDecodeError, ValueError):
        raise ReaderError(f"the {what} file is not valid UTF-8 JSON") from None
    return value, hashlib.sha256(data).hexdigest()


def _versioned(value, fmt, what, body):
    if not isinstance(value, dict) or set(value) != {"format", "version", body}:
        raise ReaderError(f"the {what} file must hold exactly format, version and {body}")
    if value["format"] != fmt or value["version"] != 1:
        raise ReaderError(f"the {what} file has an unknown format or version")
    return value[body]


# -- the corpus manifest -----------------------------------------------------------------------------------


def read_corpus(path):
    """The corpus manifest, validated with GMV's strict corpus schema (unexpected fields are
    refused). Keeps ``{"musdb": [(track id, split, sha256), …], "other": count}``. The split is
    passed on as written, ``unassigned`` included: the matcher refuses it (L4)."""
    value, digest = _load(path, "corpus manifest")
    try:
        manifest = CorpusManifest.model_validate(value)
    except ValidationError as error:
        raise ReaderError(f"the corpus manifest does not match the corpus schema ({error.error_count()} problem(s))") from None
    musdb, other = [], 0
    for r in manifest.recordings:
        if r.dataset == DATASET:
            if not (isinstance(r.dataset_track_id, str) and r.dataset_track_id.strip()):
                raise ReaderError("a MUSDB record has no usable track identifier")
            musdb.append((r.dataset_track_id, r.split, r.sha256))
        else:
            other += 1
    if not musdb:
        raise ReaderError("the corpus manifest holds no MUSDB18-HQ record")
    return Loaded({"musdb": sorted(musdb), "other": other}, digest)


# -- the split registry ------------------------------------------------------------------------------------


def read_registry(root):
    """The split registry through A5.1's fingerprint-checked reader. Keeps the development and
    validation entries' partition and nodes; every other entry only its partition."""
    path = Path(root) / "manifests" / project.REGISTRY
    if not path.is_file():
        raise ReaderError("the split registry file is missing")
    try:
        registry = project.read_registry(root)  # refuses bytes that differ from the recorded fingerprint
    except project.IntegrityError:
        raise ReaderError("the split registry does not match its recorded fingerprint") from None
    except ValueError:
        raise ReaderError("the split registry is not valid JSON") from None
    digest = project._fingerprints(root).get(project.REGISTRY)  # the verified digest of the bytes just parsed
    if not (isinstance(digest, str) and SHA256.match(digest)):
        raise ReaderError("the split registry's fingerprint is missing")
    if not isinstance(registry, dict) or set(registry) - REGISTRY_KEYS or not isinstance(registry.get("entries"), dict):
        raise ReaderError("the split registry has unexpected or missing top-level fields")
    kept, problems = {}, {}
    for key, entry in registry["entries"].items():
        why = None
        if not isinstance(entry, dict) or set(entry) - ENTRY_KEYS or "partition" not in entry or "nodes" not in entry:
            why = "unexpected or missing fields"
        elif entry["partition"] not in PARTITIONS:
            why = "an unknown partition"
        elif entry["partition"] in ASSESSED:
            nodes = entry["nodes"]
            if not isinstance(nodes, list) or not all(isinstance(n, str) and n.startswith(NODE_PREFIXES) for n in nodes):
                why = "an unknown node type"
            elif not all(SHA256.match(n[4:]) for n in nodes if n.startswith("rec:")):
                why = "a malformed recording checksum"
            else:
                kept[key] = {"partition": entry["partition"], "nodes": list(nodes)}
        else:
            kept[key] = {"partition": entry["partition"]}
        if why:
            problems[why] = problems.get(why, 0) + 1
    if problems:
        raise ReaderError("registry entries refused: " + ", ".join(f"{n} with {why}" for why, n in sorted(problems.items())))
    return Loaded({"entries": kept}, digest)


# -- the alias file ----------------------------------------------------------------------------------------


def read_aliases(path):
    """``identity/aliases.json``, **mandatory**: a missing file stops step 2; a file holding an
    empty alias map is accepted. Keeps ``(alias -> canonical, [(a, b), …])``, normalised. Stops
    on: unexpected fields; an entry or link end that normalises to an empty or generic name; two
    spellings of one key with different targets; a chained alias (a target that is itself an
    alias spelling); a malformed link; a link end that is an alias spelling (L5)."""
    value, digest = _load(path, "alias")
    if not isinstance(value, dict) or set(value) != {"aliases", "links"}:
        raise ReaderError("the alias file must hold exactly aliases and links")
    raw, links_raw = value["aliases"], value["links"]
    if not isinstance(raw, dict) or not isinstance(links_raw, list):
        raise ReaderError("the alias file's aliases are not a map or its links are not a list")
    alias_map, empty, conflicts = {}, 0, 0
    for spelling, canonical in raw.items():
        if not isinstance(canonical, str):
            raise ReaderError("an alias target is not text")
        a, b = identity.normalise(spelling), identity.normalise(canonical)
        if not a or not b:
            empty += 1
        elif alias_map.setdefault(a, b) != b:
            conflicts += 1
    if empty:
        raise ReaderError(f"{empty} alias entr(ies) normalise to an empty or generic name")
    if conflicts:
        raise ReaderError(f"{conflicts} alias spelling(s) normalise to one key with different targets")
    chained = sum(1 for a, b in alias_map.items() if b in alias_map and alias_map[b] != b)
    if chained:
        raise ReaderError(f"{chained} alias target(s) are themselves alias spellings")
    links, malformed, alias_ends = [], 0, 0
    for link in links_raw:
        if not (isinstance(link, list) and len(link) == 3 and all(isinstance(x, str) for x in link) and link[2].strip()):
            malformed += 1
            continue
        a, b = identity.normalise(link[0]), identity.normalise(link[1])
        if not a or not b or a == b:
            malformed += 1
        elif (a in alias_map and alias_map[a] != a) or (b in alias_map and alias_map[b] != b):
            alias_ends += 1
        else:
            links.append((a, b))
    if malformed:
        raise ReaderError(f"{malformed} link(s) are malformed or join a name to itself")
    if alias_ends:
        raise ReaderError(f"{alias_ends} link end(s) are alias spellings; correct the alias file")
    return Loaded((alias_map, sorted(set(links))), digest)


# -- titles ------------------------------------------------------------------------------------------------


def read_titles(path, keep):
    """``identity/titles.json``: recording sha256 -> title. Every key must be 64 lower-case hex
    digits and every value text; only the titles of the recordings in ``keep`` (the assessed
    groups' recordings) are kept. The whole file is opened."""
    value, digest = _load(path, "titles")
    if not isinstance(value, dict):
        raise ReaderError("the titles file is not a map")
    bad = sum(1 for k, v in value.items() if not (SHA256.match(k) and isinstance(v, str)))
    if bad:
        raise ReaderError(f"{bad} title entr(ies) have a malformed key or a value that is not text")
    keep = set(keep)
    return Loaded({sha: title for sha, title in value.items() if sha in keep}, digest)


# -- credit readings and review decisions ------------------------------------------------------------------


def read_readings(path):
    """The MUSDB credit-readings file (A3.6), always present (an empty map is valid): one
    reading per ambiguous MUSDB artist credit, ``whole-and-pieces`` (the default: the whole name
    and each piece are all credited) or ``single-name`` with a stated public reason. Which
    credits must have a reading is checked against the listing phase by the matcher."""
    value, digest = _load(path, "credit-readings")
    readings = _versioned(value, READINGS_FORMAT, "credit-readings", "readings")
    if not isinstance(readings, dict):
        raise ReaderError("the credit readings are not a map")
    kept, bad = {}, 0
    for credit, reading in readings.items():
        if not (isinstance(credit, str) and credit.strip() and isinstance(reading, dict)):
            bad += 1
        elif reading == {"reading": "whole-and-pieces"}:
            kept[credit] = "whole-and-pieces"
        elif (set(reading) == {"reading", "reason"} and reading["reading"] == "single-name"
              and isinstance(reading["reason"], str) and reading["reason"].strip()):
            kept[credit] = "single-name"
        else:
            bad += 1
    if bad:
        raise ReaderError(f"{bad} credit reading(s) are malformed")
    return Loaded(kept, digest)


def read_decisions(path):
    """The owner's review-decisions file (A3.6), always present (an empty list is valid): the
    name-based pairs the owner marked "not the same artist". Keeps a frozenset of (MUSDB track
    id, group key). Whether each pair is a name-based match is checked by the matcher."""
    value, digest = _load(path, "review-decisions")
    rejections = _versioned(value, DECISIONS_FORMAT, "review-decisions", "rejections")
    if not isinstance(rejections, list):
        raise ReaderError("the review rejections are not a list")
    pairs, bad = [], 0
    for item in rejections:
        if (isinstance(item, dict) and set(item) == {"track_id", "group", "reason"}
                and all(isinstance(item[k], str) and item[k] for k in item) and item["reason"] in REJECTION_REASONS):
            pairs.append((item["track_id"], item["group"]))
        else:
            bad += 1
    if bad:
        raise ReaderError(f"{bad} review decision(s) are malformed")
    if len(set(pairs)) != len(pairs):
        raise ReaderError("a review decision is repeated")
    return Loaded(frozenset(pairs), digest)
