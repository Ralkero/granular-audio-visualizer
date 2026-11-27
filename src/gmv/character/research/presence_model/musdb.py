"""Step 1b: the MUSDB18-HQ overlap matcher (protocol §18, Stage 1: "an aggregate check for
overlap with MUSDB18-HQ training material"). The rules are readiness proposal revision 4, §7,
carried unchanged to revision 4.4, §7. Built and tested on generated inputs only: this module
reads no file. Loading the real inputs and running the check is step 2, not implemented here.

**Inputs** (in memory):

* ``records``: corpus manifest recordings. Only those with ``dataset == "musdb18hq"`` are
  assessed. Each needs ``dataset_track_id`` (non-empty text), ``split`` (the corpus split,
  ``train``, ``validation`` or ``test``) and ``sha256`` (64 lower-case hex digits).
* ``registry``: the split registry. Only ``development`` and ``validation`` entries are assessed
  (their ``partition`` and ``nodes``). For every other entry, reserved ones included, only the
  ``partition`` field is read, to skip it. A partition outside the registry's five
  (``split.PARTITIONS``) is refused, so a misspelt one cannot hide a group.
* ``aliases``: ``(alias -> canonical, links)``, as ``identity.read_aliases`` returns them:
  every name non-empty and already normalised, or the check stops.
* ``titles``: recording sha256 -> title (``identity/titles.json``). Only titles of recordings in
  assessed groups are looked up.
* ``categories``: development group key -> ``paired`` or ``low-rate``, for every development
  entry. **Open discrepancy, for the owner:** the registered dimension needs this, but the
  registry does not record which groups B0 can run on (that comes from the excerpts' sample
  rates in the development manifest), and the registered step-2 inputs do not include the
  development manifest. The matcher takes it as an argument and resolves nothing; where it
  comes from, and whether it is fingerprinted with the other inputs, is for step 2's
  authorisation.

**Definitions** (revision 4, §7), for each MUSDB record r and assessed entry e:

* A_r and T_r: ``dataset_track_id`` split at the first " - " (space, hyphen, space). Without
  one, both are the whole id, and r is counted as unparsed. Taken as written: an id such as
  " - Title" is parsed, with an empty artist and so no credited name (only a checksum can
  match it).
* E(r) = ``identity.credited({artist: A_r, title: T_r}, aliases)``, plus one step of link
  expansion: b is added if (a, b) is a link and a is in the credited set, and the reverse.
  Names reached only through an added name are not added.
* Q(r) = {normalise(A_r)} ∪ the normalised pieces of A_r split at " and ", " with " and
  parentheses ∪ compact (spaces removed) versions of E(r) and of those pieces. The split
  strings are matched as written (lower case).
* N(e): the names of e's ``name:`` nodes; R(e): its ``rec:`` sha256 values; Ncomp(e): N(e)
  compacted.
* **exact:** E(r) ∩ N(e) ≠ ∅. **checksum:** sha256(r) ∈ R(e).
* **title confirmation** (reported only): exact, and normalise(T_r without its feature
  brackets) equals the same key for the title of some recording in R(e). An empty key never
  confirms.
* **permissive-only** (reported only): neither exact nor checksum, but
  (Q(r) ∪ compact(E(r))) ∩ (N(e) ∪ Ncomp(e)) ≠ ∅.

**Exposure**, on paired groups only: G_T holds the groups with an exact or checksum match to
a MUSDB-train record; G_V the same for MUSDB-validation. Title confirmation and
permissive-only never place a group in either.

**Test integrity:** an exact or checksum match from a MUSDB-test record to any assessed group
(development, paired or low-rate, or validation) stops the check (``TestSplitOverlap``). Test
records are assessed first, and no result is produced.

**Counts:** for every split × partition × match type, the number of distinct groups and of
distinct MUSDB records with that kind of match. Every cell is present, zeros included. Match
types overlap: a pair that is exact and checksum counts in both, and title confirmation is a
subset of exact.

**Fail closed:** malformed, missing, duplicated or conflicting inputs stop the check
(``MusdbInputError``), with fixed messages carrying counts only. Nothing is guessed: a split
outside train, validation and test (the corpus also knows ``unassigned``) is refused rather
than given an exposure class.

**Privacy:** ``public_report`` holds counts, the commit, a digest of the input hashes and the
limitations. Beyond the registered count table it adds two counts: unparsed records per split,
and the sizes of G_T, G_V and their overlap. The match list and G_T and G_V membership exist
only in ``Result.private``, which never prints its contents.
"""

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field

from ..presence import identity
from ..presence.split import PARTITIONS as REGISTRY_PARTITIONS

FORMAT = "a5.2-musdb-overlap-v1"
DATASET = "musdb18hq"
SPLITS = ("train", "validation", "test")
PARTITIONS = ("paired", "low-rate", "validation")
MATCH_TYPES = ("exact", "checksum", "title-confirmation", "permissive-only")
ASSESSED = ("development", "validation")
DEVELOPMENT_CATEGORIES = ("paired", "low-rate")
SEPARATOR = " - "
PIECES = re.compile(r" and | with |[()]")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
COMMIT = re.compile(r"^[0-9a-f]{7,40}$")
INPUTS = ("corpus_manifest", "registry", "aliases", "titles")
LIMITATIONS = (
    "Names transliterated or written in another script are not matched.",
    "Performers who are not credited are not matched.",
    "Covers and other recordings of the same song by other artists are not matched.",
    "Checksums only catch byte-identical files.",
    "Title confirmation and permissive-only matches are reported only; neither excludes a group.",
)


class MusdbInputError(ValueError):
    """The inputs are malformed, missing, duplicated or conflicting; the check stops."""


class TestSplitOverlap(RuntimeError):  # noqa: N818 - the name states the stop condition
    """A MUSDB-test record matches an assessed group by credit or checksum; everything stops
    for the owner's decision."""

    __test__ = False  # not a pytest test class


@dataclass(frozen=True)
class Group:
    category: str
    names: frozenset
    compact_names: frozenset
    recordings: frozenset
    title_keys: frozenset


class Private:
    """The private part of a result: G_T, G_V and the match list. It never prints its
    contents (repr, str, pytest's failure diffs) and is not JSON-serialisable; compare with
    ``==``."""

    __slots__ = ("g_t", "g_v", "matches")

    def __init__(self, g_t, g_v, matches):
        self.g_t, self.g_v, self.matches = frozenset(g_t), frozenset(g_v), tuple(matches)

    def __eq__(self, other):
        return isinstance(other, Private) and (self.g_t, self.g_v, self.matches) == (other.g_t, other.g_v, other.matches)

    def __hash__(self):
        return hash((self.g_t, self.g_v, self.matches))

    def __repr__(self):
        return "<private: withheld>"


@dataclass(frozen=True)
class Result:
    """Public aggregates, plus a ``Private`` part (G_T, G_V, the match list)."""

    counts: dict
    musdb_records: dict
    unparsed: dict
    exposed: dict
    private: Private = field(repr=False)

    @property
    def g_t(self):
        return self.private.g_t

    @property
    def g_v(self):
        return self.private.g_v

    @property
    def matches(self):
        """(track id, split, group key, match types), sorted."""
        return self.private.matches


# -- keys ------------------------------------------------------------------------------------------


def compact(key):
    return key.replace(" ", "")


def parse_track_id(track_id):
    """(artist, title, parsed): split at the first " - "; without one, both are the whole id."""
    if not isinstance(track_id, str) or not track_id.strip():
        raise MusdbInputError("a MUSDB record has no usable track identifier")
    artist, separator, title = track_id.partition(SEPARATOR)
    if not separator:
        return track_id, track_id, False
    return artist, title, True


def exact_keys(artist, title, aliases, links):
    """E(r): credited names (aliases applied) plus exactly one step of link expansion."""
    credited = identity.credited({"artist": artist, "title": title}, aliases)
    keys = set(credited)
    for a, b, *_reason in links:
        if a in credited:
            keys.add(b)
        if b in credited:
            keys.add(a)
    return frozenset(keys)


def permissive_keys(artist, exact):
    """Q(r) ∪ compact(E(r))."""
    pieces = [identity.normalise(p) for p in PIECES.split(artist)]
    keys = {identity.normalise(artist), *pieces}
    keys |= {compact(k) for k in exact} | {compact(p) for p in pieces}
    return frozenset(k for k in keys if k)


def title_key(title):
    """normalise(title without its feature brackets)."""
    return identity.normalise(identity.TITLE_FEATURE.sub(" ", title))


# -- inputs ----------------------------------------------------------------------------------------


def _musdb_records(records):
    if not isinstance(records, (list, tuple)):
        raise MusdbInputError("the corpus records are not a list")
    found, problems = [], Counter()
    for r in records:
        if not isinstance(r, dict):
            problems["not a record"] += 1
            continue
        if r.get("dataset") != DATASET:
            continue
        track_id, split, sha = r.get("dataset_track_id"), r.get("split"), r.get("sha256")
        if not (isinstance(track_id, str) and track_id.strip()):
            problems["no usable track identifier"] += 1
        elif split not in SPLITS:
            problems["a split outside train, validation and test"] += 1
        elif not (isinstance(sha, str) and SHA256.match(sha)):
            problems["a malformed checksum"] += 1
        else:
            found.append((track_id, split, sha))
    if problems:
        raise MusdbInputError("corpus records refused: " + ", ".join(f"{n} with {why}" for why, n in sorted(problems.items())))
    if not found:
        raise MusdbInputError("the corpus holds no MUSDB18-HQ record")
    ids, shas = Counter(t for t, _s, _h in found), Counter(h for _t, _s, h in found)
    if any(n > 1 for n in ids.values()) or any(n > 1 for n in shas.values()):
        raise MusdbInputError(f"{sum(n > 1 for n in ids.values())} track identifier(s) and "
                              f"{sum(n > 1 for n in shas.values())} checksum(s) occur in more than one MUSDB record")
    return sorted(found)


def _aliases(aliases):
    try:
        alias_map, links = aliases
    except (TypeError, ValueError):
        raise MusdbInputError("the aliases are not (alias map, links)") from None
    def name(x):  # a non-empty, already normalised name, as identity.read_aliases returns them
        return isinstance(x, str) and bool(x) and identity.normalise(x) == x

    if not isinstance(alias_map, dict) or not all(name(k) and name(v) for k, v in alias_map.items()):
        raise MusdbInputError("the alias map holds an empty or unnormalised name")
    if not isinstance(links, (list, tuple)) or not all(
            isinstance(link, (list, tuple)) and len(link) >= 2 and name(link[0]) and name(link[1]) for link in links):
        raise MusdbInputError("the links hold an empty or unnormalised name")
    return alias_map, [tuple(link[:2]) for link in links]


def _groups(registry, titles, categories):
    entries = registry.get("entries") if isinstance(registry, dict) else None
    if not isinstance(entries, dict):
        raise MusdbInputError("the registry has no entries")
    if not isinstance(titles, dict):
        raise MusdbInputError("the titles are not a mapping")
    if not isinstance(categories, dict) or not all(v in DEVELOPMENT_CATEGORIES for v in categories.values()):
        raise MusdbInputError("the development categories are malformed")
    groups, problems = {}, Counter()
    for key, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("partition"), str):
            problems["no partition"] += 1
            continue
        partition = entry["partition"]
        if partition not in REGISTRY_PARTITIONS:
            problems["an unknown partition"] += 1  # a misspelt partition must not hide a group
            continue
        if partition not in ASSESSED:
            continue  # reserved, final-test and excluded entries: never assessed, nodes never read
        nodes = entry.get("nodes")
        if not isinstance(nodes, list) or not all(isinstance(n, str) for n in nodes):
            problems["malformed nodes"] += 1
            continue
        recordings = {n[4:] for n in nodes if n.startswith("rec:")}
        if any(not SHA256.match(s) for s in recordings):
            problems["a malformed recording checksum"] += 1
            continue
        names = frozenset(n[5:] for n in nodes if n.startswith("name:") and n[5:])
        keys = set()
        for sha in recordings:
            title = titles.get(sha)
            if title is not None and not isinstance(title, str):
                problems["a malformed title"] += 1
            elif title:
                keys.add(title_key(title))
        if partition == "development":
            if key not in categories:
                problems["no paired or low-rate category"] += 1
                continue
            category = categories[key]
        else:
            category = "validation"
        groups[key] = Group(category, names, frozenset(compact(n) for n in names), frozenset(recordings),
                            frozenset(k for k in keys if k))
    development = {k for k, g in groups.items() if g.category in DEVELOPMENT_CATEGORIES}
    if set(categories) - development:
        problems["a category for a key that is not a development group"] += len(set(categories) - development)
    shared = Counter(n for g in groups.values() for n in {f"name:{x}" for x in g.names} | {f"rec:{s}" for s in g.recordings})
    if any(n > 1 for n in shared.values()):
        problems["an identity node shared by two assessed groups"] += sum(n > 1 for n in shared.values())
    if problems:
        raise MusdbInputError("registry entries refused: " + ", ".join(f"{n} with {why}" for why, n in sorted(problems.items())))
    if not groups:
        raise MusdbInputError("the registry has no development or validation group")
    return groups


# -- matching --------------------------------------------------------------------------------------


def _match_types(record_keys, sha, group):
    exact_set, permissive, tkey = record_keys
    exact, checksum = bool(exact_set & group.names), sha in group.recordings
    types = []
    if exact:
        types.append("exact")
    if checksum:
        types.append("checksum")
    if exact and tkey and tkey in group.title_keys:
        types.append("title-confirmation")
    if not exact and not checksum and permissive & (group.names | group.compact_names):
        types.append("permissive-only")
    return tuple(types)


def match(records, registry, aliases, titles, categories):
    """Run the overlap check on in-memory inputs; returns a ``Result`` or raises
    ``MusdbInputError`` or ``TestSplitOverlap``."""
    musdb = _musdb_records(records)
    alias_map, links = _aliases(aliases)
    groups = _groups(registry, titles, categories)
    unparsed = Counter()
    keyed = []
    for track_id, split, sha in musdb:
        artist, title, parsed = parse_track_id(track_id)
        if not parsed:
            unparsed[split] += 1
        exact = exact_keys(artist, title, alias_map, links)
        keyed.append((track_id, split, sha, (exact, permissive_keys(artist, exact), title_key(title))))
    pairs = []
    # test records first: any credit or checksum match stops everything before anything else
    for phase in (("test",), ("train", "validation")):
        for track_id, split, sha, record_keys in keyed:
            if split not in phase:
                continue
            for key in sorted(groups):
                types = _match_types(record_keys, sha, groups[key])
                if types:
                    pairs.append((track_id, split, key, types))
        if phase == ("test",):
            overlap = [(t, k) for t, s, k, types in pairs if s == "test" and {"exact", "checksum"} & set(types)]
            if overlap:
                raise TestSplitOverlap(f"MUSDB-test records match assessed groups by credit or checksum: "
                                       f"{len({t for t, _k in overlap})} record(s), {len({k for _t, k in overlap})} group(s)")
    counts = {s: {p: {m: {"groups": 0, "records": 0} for m in MATCH_TYPES} for p in PARTITIONS} for s in SPLITS}
    cells = {}
    for track_id, split, key, types in pairs:
        for m in types:
            cell = cells.setdefault((split, groups[key].category, m), (set(), set()))
            cell[0].add(key)
            cell[1].add(track_id)
    for (split, partition, m), (keys, tracks) in cells.items():
        counts[split][partition][m] = {"groups": len(keys), "records": len(tracks)}

    def exposed(split):
        return frozenset(k for _t, s, k, types in pairs
                         if s == split and groups[k].category == "paired" and {"exact", "checksum"} & set(types))

    g_t, g_v = exposed("train"), exposed("validation")
    return Result(
        counts=counts,
        musdb_records={s: sum(1 for _t, sp, _h in musdb if sp == s) for s in SPLITS},
        unparsed={s: unparsed.get(s, 0) for s in SPLITS},
        exposed={"training": len(g_t), "validation": len(g_v), "both": len(g_t & g_v)},
        private=Private(g_t, g_v, sorted(pairs)),
    )


def public_report(result, input_sha256, commit):
    """The publishable aggregate: counts, the commit, a digest of the four input files'
    SHA-256 values, and the limitations. No name, title, key, identifier or private hash."""
    if not isinstance(input_sha256, dict) or set(input_sha256) != set(INPUTS) or not all(
            isinstance(v, str) and SHA256.match(v) for v in input_sha256.values()):
        raise MusdbInputError("the input fingerprints must be the SHA-256 of exactly the four inputs")
    if not (isinstance(commit, str) and COMMIT.match(commit)):
        raise MusdbInputError("the commit is malformed")
    digest = hashlib.sha256(json.dumps(sorted(input_sha256.items()), separators=(",", ":")).encode()).hexdigest()
    return {
        "format": FORMAT,
        "commit": commit,
        "input_digest_sha256": digest,
        "musdb_records": dict(result.musdb_records),
        "unparsed_records": dict(result.unparsed),
        "counts": json.loads(json.dumps(result.counts)),
        "exposed_paired_groups": dict(result.exposed),
        "limitations": list(LIMITATIONS),
    }
