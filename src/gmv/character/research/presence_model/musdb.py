"""The MUSDB18-HQ overlap matcher (protocol Amendment 3, A3.6; built in Step 1b from readiness
proposal revision 4, §7, and brought to Amendment 3 in Step 2a). Built and tested on generated
inputs only: this module reads no file. The strict readers are ``readers``; running the check
on real inputs is step 2, which needs its own authorisation.

**Inputs** (in memory):

* ``records``: corpus manifest recordings. Only those with ``dataset == "musdb18hq"`` are
  assessed. Each needs ``dataset_track_id`` (non-empty text), ``split`` (the corpus split,
  ``train``, ``validation`` or ``test``) and ``sha256`` (64 lower-case hex digits).
* ``registry``: the split registry. Only ``development`` and ``validation`` entries are assessed
  (their ``partition`` and ``nodes``). For every other entry, reserved ones included, only the
  ``partition`` field is read, to skip it. A partition outside the registry's five
  (``split.PARTITIONS``) is refused, so a misspelt one cannot hide a group.
* ``aliases``: ``(alias -> canonical, links)``, as ``readers.read_aliases`` returns them:
  every name non-empty and already normalised, or the check stops.
* ``titles``: recording sha256 -> title (``identity/titles.json``). Only titles of recordings in
  assessed groups are looked up.
* ``categories``: development group key -> ``paired`` or ``low-rate``, or **None at step 2**
  (option C, L1): development groups are then filed under one private ``development``
  partition, and ``exposure`` classes them at step 3 from the development manifest, with
  exactly the same matches.
* ``readings``: one reading per ambiguous credit (L3; ``list_ambiguous`` is the listing phase).
* ``test_rejections``: the owner's "not the same artist" decisions for MUSDB-test pairs (C4),
  applied before the test-split check; train and validation decisions are applied at step 3.

**Definitions** (revision 4, §7), for each MUSDB record r and assessed entry e:

* A_r and T_r: ``dataset_track_id`` split at the first " - " (space, hyphen, space).
* **Unparsed (a stop, MUSDB records only):** no " - "; an artist with no credited name (empty or
  generic); or an ambiguous credit (``is_ambiguous``: "and" or "with" as a word in any
  capitalisation, a bracket, or letters outside the Latin alphabet) without a reading.
* E(r) = the names credited by the artist field and by title brackets (``identity.credited``,
  aliases applied), plus, under the ``whole-and-pieces`` reading, the credits of each piece of
  an ambiguous artist field, plus one step of link expansion: b is added if (a, b) is a link
  and a is in the credited set, and the reverse. Names reached only through an added name are
  not added. Each name-based pair records its sources (artist, bracket, piece, link) for the
  owner's review.
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
(development, paired or low-rate, or validation), after the owner's test-split rejections,
stops the check (``TestSplitOverlap``, which carries the test pairs privately for the review
file). Test records are assessed first, and no result is produced. A file (checksum) match can
never be rejected.

**Counts:** for every split × partition × match type, the number of distinct groups and of
distinct MUSDB records with that kind of match. Every cell is present, zeros included. Match
types overlap: a pair that is exact and checksum counts in both, and title confirmation is a
subset of exact.

**Fail closed:** malformed, missing, duplicated or conflicting inputs stop the check
(``MusdbInputError``), with fixed messages carrying counts only. Nothing is guessed: a split
outside train, validation and test (the corpus also knows ``unassigned``) is refused rather
than given an exposure class.

**Privacy:** the match list, the credit sources and G_T and G_V membership exist only in
``Result.private``, which never prints its contents. ``public_report`` is Step 1b's format; the
public report of Amendment 3 (L6) is built by the step 3 report with the disclosure check.
"""

import hashlib
import json
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field

from ..presence import identity
from ..presence.split import PARTITIONS as REGISTRY_PARTITIONS

FORMAT = "a5.2-musdb-overlap-v1"
DATASET = "musdb18hq"
SPLITS = ("train", "validation", "test")
PARTITIONS = ("paired", "low-rate", "validation")
DEFERRED_PARTITIONS = ("development", "validation")  # option C: the split is decided at step 3
MATCH_TYPES = ("exact", "checksum", "title-confirmation", "permissive-only")
ASSESSED = ("development", "validation")
DEVELOPMENT_CATEGORIES = ("paired", "low-rate")
SEPARATOR = " - "
PIECES = re.compile(r" and | with |[()]")  # the report-only permissive tier, as registered (lower case)
AMBIGUOUS_WORDS = re.compile(r"\b(?:and|with)\b")  # on case-folded text, so every capitalisation
BRACKETS = re.compile(r"[()\[\]{}]")
AMBIGUOUS_SPLIT = re.compile(r"\b(?:and|with)\b|[()\[\]{}]")
READINGS = ("whole-and-pieces", "single-name")
CREDIT_SOURCES = ("artist", "bracket", "piece", "link")
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


class UnparsedRecords(MusdbInputError):
    """MUSDB records the matcher cannot read with confidence (Amendment 3, A3.6): no " - "
    separator, an empty or generic artist, or an ambiguous credit without a registered reading.
    Step 2 stops before G_T and G_V are fixed. Other datasets never trigger it."""


class ReviewError(MusdbInputError):
    """A review decision does not fit the matches: it names a pair that is not a name-based
    match, or it rejects a file (checksum) match, which cannot be rejected."""


class TestSplitOverlap(RuntimeError):  # noqa: N818 - the name states the stop condition
    """A MUSDB-test record matches an assessed group by credit or checksum, after the owner's
    test-split rejections; everything stops for the owner's review. ``private`` holds the test
    pairs for the private review file and never prints."""

    __test__ = False  # not a pytest test class

    def __init__(self, message, private=None):
        super().__init__(message)
        self.private = private


@dataclass(frozen=True)
class Group:
    category: str
    names: frozenset
    compact_names: frozenset
    recordings: frozenset
    title_keys: frozenset


class Private:
    """The private part of a result: G_T, G_V, the match list and where each name-based credit
    came from. It never prints its contents (repr, str, pytest's failure diffs) and is not
    JSON-serialisable; compare with ``==``."""

    __slots__ = ("g_t", "g_v", "matches", "sources")

    def __init__(self, g_t, g_v, matches, sources=None):
        self.g_t = None if g_t is None else frozenset(g_t)
        self.g_v = None if g_v is None else frozenset(g_v)
        self.matches = tuple(matches)
        self.sources = dict(sources or {})

    def __eq__(self, other):
        return isinstance(other, Private) and (self.g_t, self.g_v, self.matches, self.sources) == (
            other.g_t, other.g_v, other.matches, other.sources)

    def __hash__(self):
        return hash((self.g_t, self.g_v, self.matches, tuple(sorted(self.sources.items()))))

    def __repr__(self):
        return "<private: withheld>"

    __str__ = __repr__


@dataclass(frozen=True)
class Result:
    """Public aggregates, plus a ``Private`` part (G_T, G_V, the match list, credit sources).
    In the deferred mode (option C, no categories) development groups are filed under one
    private ``development`` partition and G_T, G_V and ``exposed`` are None: they are derived at
    step 3 (``exposure``)."""

    counts: dict
    musdb_records: dict
    unparsed: dict
    exposed: dict | None
    private: Private = field(repr=False)
    readings_used: dict = field(default_factory=dict)

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

    @property
    def sources(self):
        """(track id, group key) -> where the name-based credit came from: ``artist`` (the
        artist field), ``bracket`` (a title bracket), ``piece`` (a piece of an ambiguous
        credit) or ``link`` (one link step)."""
        return self.private.sources


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


def _folded(text):
    """Text as the ambiguity check sees it: NFKC (full-width letters become plain ones),
    case-folded, every run of white space (non-breaking spaces included) one space."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text)).casefold()).strip()


def _non_latin(text):
    return any(c.isalpha() and "LATIN" not in unicodedata.name(c, "") for c in text)


def is_ambiguous(artist):
    """An artist credit the exact tier cannot read with confidence (L3): "and" or "with" as a
    separate word in any capitalisation, a bracket, or letters outside the Latin alphabet. The
    exact tier never splits an artist field at "and" or "with", so "X And Y" could otherwise
    never match a group "Y"."""
    folded = _folded(artist)
    return bool(AMBIGUOUS_WORDS.search(folded) or BRACKETS.search(folded) or _non_latin(folded))


def pieces(artist):
    """The pieces of an ambiguous credit: the folded text split at "and", "with" and brackets."""
    return [p.strip() for p in AMBIGUOUS_SPLIT.split(_folded(artist)) if p.strip()]


def credit_sources(artist, title, aliases, links, reading=None):
    """Every credited key of a record (aliases applied) with the sources it came from:
    ``artist``, ``bracket``, ``piece`` (only under the ``whole-and-pieces`` reading) and
    ``link`` (exactly one step of expansion from the others)."""
    out = {}

    def add(keys, source):
        for k in keys:
            out.setdefault(k, set()).add(source)

    add(identity.credited({"artist": artist}, aliases), "artist")
    add(identity.credited({"title": title}, aliases), "bracket")
    if reading == "whole-and-pieces":
        for piece in pieces(artist):
            add(identity.credited({"artist": piece}, aliases), "piece")
    base = set(out)
    for a, b, *_reason in links:
        if a in base:
            add({b}, "link")
        if b in base:
            add({a}, "link")
    return {k: frozenset(v) for k, v in out.items()}


def exact_keys(artist, title, aliases, links, reading=None):
    """E(r): credited names (aliases applied) plus exactly one step of link expansion."""
    return frozenset(credit_sources(artist, title, aliases, links, reading))


def permissive_keys(artist, exact):
    """Q(r) ∪ compact(E(r))."""
    parts = [identity.normalise(p) for p in PIECES.split(artist)]
    keys = {identity.normalise(artist), *parts}
    keys |= {compact(k) for k in exact} | {compact(p) for p in parts}
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
    deferred = categories is None  # option C: every development group is filed under "development"
    if not deferred and (not isinstance(categories, dict) or not all(v in DEVELOPMENT_CATEGORIES for v in categories.values())):
        raise MusdbInputError("the development categories are malformed")
    categories = categories or {}
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
        if partition == "development" and deferred:
            category = "development"
        elif partition == "development":
            if key not in categories:
                problems["no paired or low-rate category"] += 1
                continue
            category = categories[key]
        else:
            category = "validation"
        groups[key] = Group(category, names, frozenset(compact(n) for n in names), frozenset(recordings),
                            frozenset(k for k in keys if k))
    development = {k for k, g in groups.items() if g.category in (*DEVELOPMENT_CATEGORIES, "development")}
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


def list_ambiguous(records):
    """The listing phase (A3.6): the ambiguous artist credits among the parsed MUSDB records,
    sorted. It reads only the MUSDB records: these are public MUSDB names, nothing from the
    owner's library. Every credit listed here needs a reading before matching."""
    found = set()
    for track_id, _split, _sha in _musdb_records(records):
        artist, _title, parsed = parse_track_id(track_id)
        if parsed and is_ambiguous(artist):
            found.add(artist)
    return sorted(found)


def _readings(readings, listed):
    if readings is None:
        readings = {}
    if not isinstance(readings, dict) or not all(isinstance(k, str) and v in READINGS for k, v in readings.items()):
        raise MusdbInputError("the credit readings are malformed")
    stale = set(readings) - set(listed)
    if stale:
        raise MusdbInputError(f"{len(stale)} credit reading(s) name a credit that is not on the listing")
    return readings


def _pair_sources(credit_map, group):
    return tuple(sorted(set().union(*(credit_map[k] for k in credit_map if k in group.names)) or ()))


def match(records, registry, aliases, titles, categories=None, readings=None, test_rejections=frozenset()):
    """Run the overlap check on in-memory inputs; returns a ``Result`` or raises
    ``MusdbInputError`` (including ``UnparsedRecords`` and ``ReviewError``) or
    ``TestSplitOverlap``.

    ``categories``: development group key -> ``paired`` or ``low-rate``; None for the deferred
    mode (option C). ``readings``: credit -> reading for every listed ambiguous credit (None is an
    empty readings file). ``test_rejections``: (track id, group key) pairs of MUSDB-test records
    the owner marked "not the same artist"; they are applied before the test-split check."""
    musdb = _musdb_records(records)
    alias_map, links = _aliases(aliases)
    groups = _groups(registry, titles, categories)
    listed = list_ambiguous(records)
    readings = _readings(readings, listed)
    unparsed, keyed, used = Counter(), [], Counter()
    for track_id, split, sha in musdb:
        artist, title, parsed = parse_track_id(track_id)
        reading = None
        if not parsed:
            unparsed["no separator"] += 1
            continue
        if not identity.credited({"artist": artist}, alias_map):
            unparsed["an empty or generic artist"] += 1
            continue
        if is_ambiguous(artist):
            if artist not in readings:
                unparsed["an ambiguous credit without a reading"] += 1
                continue
            reading = readings[artist]
            used[split] += 1
        credit_map = credit_sources(artist, title, alias_map, links, reading)
        exact = frozenset(credit_map)
        keyed.append((track_id, split, sha, (exact, permissive_keys(artist, exact), title_key(title)), credit_map))
    if unparsed:
        raise UnparsedRecords("MUSDB records refused before matching: "
                              + ", ".join(f"{n} with {why}" for why, n in sorted(unparsed.items())))
    pairs, sources = [], {}
    # test records first: any credit or checksum match stops everything before anything else
    for phase in (("test",), ("train", "validation")):
        for track_id, split, sha, record_keys, credit_map in keyed:
            if split not in phase:
                continue
            for key in sorted(groups):
                types = _match_types(record_keys, sha, groups[key])
                if types:
                    pairs.append((track_id, split, key, types))
                    if "exact" in types:
                        sources[(track_id, key)] = _pair_sources(credit_map, groups[key])
        if phase == ("test",):
            pairs = _reject(pairs, test_rejections, splits=("test",))
            overlap = [(t, k, types) for t, s, k, types in pairs if s == "test" and {"exact", "checksum"} & set(types)]
            if overlap:
                review = [(t, "test", k, types) for t, k, types in overlap]
                raise TestSplitOverlap(
                    f"MUSDB-test records match assessed groups by credit or checksum: "
                    f"{len({t for t, _k, _x in overlap})} record(s), {len({k for _t, k, _x in overlap})} group(s)",
                    private=Private(None, None, review, {(t, k): sources.get((t, k), ()) for t, k, _x in overlap}))
    sources = {pair: s for pair, s in sources.items() if any((t, k) == pair for t, _s, k, _x in pairs)}
    deferred = categories is None
    g_t, g_v = (None, None) if deferred else _exposure(pairs, groups)
    return Result(
        counts=_counts(pairs, groups, DEFERRED_PARTITIONS if deferred else PARTITIONS),
        musdb_records={s: sum(1 for _t, sp, _h in musdb if sp == s) for s in SPLITS},
        unparsed={s: 0 for s in SPLITS},
        exposed=None if deferred else {"training": len(g_t), "validation": len(g_v), "both": len(g_t & g_v)},
        private=Private(g_t, g_v, sorted(pairs), sources),
        readings_used={s: used.get(s, 0) for s in SPLITS},
    )


def _reject(pairs, rejections, splits):
    """Apply the owner's "not the same artist" decisions for records of ``splits``. Every
    decision must name a name-based (exact) pair of those splits; a file (checksum) match cannot
    be rejected. A rejected pair is dropped."""
    rejections = frozenset(rejections or ())
    if not all(isinstance(p, tuple) and len(p) == 2 for p in rejections):
        raise ReviewError("the review decisions are malformed")
    found = {(t, k): types for t, s, k, types in pairs if s in splits}
    files = sum(1 for p in rejections if p in found and "checksum" in found[p])
    if files:
        raise ReviewError(f"{files} review decision(s) reject a file match, which cannot be rejected")
    unknown = sum(1 for p in rejections if p not in found or "exact" not in found[p])
    if unknown:
        raise ReviewError(f"{unknown} review decision(s) name a pair that is not a name-based match")
    return [p for p in pairs if (p[0], p[2]) not in rejections]


def _counts(pairs, groups, partitions):
    counts = {s: {p: {m: {"groups": 0, "records": 0} for m in MATCH_TYPES} for p in partitions} for s in SPLITS}
    cells = {}
    for track_id, split, key, types in pairs:
        for m in types:
            cell = cells.setdefault((split, groups[key].category, m), (set(), set()))
            cell[0].add(key)
            cell[1].add(track_id)
    for (split, partition, m), (keys, tracks) in cells.items():
        counts[split][partition][m] = {"groups": len(keys), "records": len(tracks)}
    return counts


def _exposure(pairs, groups):
    def exposed(split):
        return frozenset(k for _t, s, k, types in pairs
                         if s == split and groups[k].category == "paired" and {"exact", "checksum"} & set(types))

    return exposed("train"), exposed("validation")


def review_pairs(result):
    """The name-based pairs the owner reviews (C4), with where each credit came from, plus the
    file matches for information: [(track id, split, group key, types, sources)]. Private."""
    return [(t, s, k, types, result.sources.get((t, k), ())) for t, s, k, types in result.matches
            if {"exact", "checksum"} & set(types)]


def exposure(result, registry, categories, rejections=frozenset()):
    """Step 3 (option C): classify the deferred result's development groups with
    ``categories`` and apply the owner's train and validation rejections, then derive G_T, G_V
    and the counts. Returns a ``Result`` equal to matching with the categories from the start
    (minus the rejected pairs)."""
    if result.g_t is not None:
        raise MusdbInputError("the result is not a deferred (step 2) result")
    entries = registry.get("entries", {}) if isinstance(registry, dict) else {}
    development = {k for k, e in entries.items() if isinstance(e, dict) and e.get("partition") == "development"}
    if not isinstance(categories, dict) or set(categories) != development or not all(
            v in DEVELOPMENT_CATEGORIES for v in categories.values()):
        raise MusdbInputError("the categories must class exactly the registry's development groups")
    category_of = {k: categories.get(k, "validation") for k in {k for _t, _s, k, _x in result.matches}}
    groups = {k: Group(c, frozenset(), frozenset(), frozenset(), frozenset()) for k, c in category_of.items()}
    pairs = _reject(list(result.matches), rejections, splits=("train", "validation"))
    g_t, g_v = _exposure(pairs, groups)
    kept = {(t, k) for t, _s, k, _x in pairs}
    return Result(
        counts=_counts(pairs, groups, PARTITIONS),
        musdb_records=dict(result.musdb_records),
        unparsed=dict(result.unparsed),
        exposed={"training": len(g_t), "validation": len(g_v), "both": len(g_t & g_v)},
        private=Private(g_t, g_v, sorted(pairs), {p: s for p, s in result.sources.items() if p in kept}),
        readings_used=dict(result.readings_used),
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
