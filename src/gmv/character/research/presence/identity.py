"""Conservative artist groups: recordings that might share a performer stay together.

Two recordings land in the same group when they share any of these (a union over all of
them, so links chain):

* a credited name: the inventory's artist, album artist and MusicBrainz artist credit,
  split at "feat.", "&", ",", "/" and similar, plus any "(feat. ...)" in the title. Names are
  normalised (Unicode NFKC, accents and punctuation removed, case folded, a leading "the"
  dropped), so a different spelling or capitalisation never makes a new group;
* a MusicBrainz artist identifier;
* the inventory's leakage group, same-audio group or version group (duplicates, remasters,
  alternate mixes, live versions already linked by the inventory);
* the Phase A3 artist key of the recording;
* the owner's private alias file: ``{"aliases": {"spelling": "canonical name"},
  "links": [["name", "name", "reason"], ...]}`` for aliases and known shared vocalists.

Generic credits ("Various Artists", "Unknown Artist", "Soundtrack", ...) never link
anything. Where identity is unclear, merging is the safe direction: a merged group can
only make a split stricter. Vocalist identities are mostly unknown, so performer
disjointness is not claimed beyond these links.
"""

import json
import re
import unicodedata
from pathlib import Path

SEPARATORS = re.compile(r"\s*(?:,|&|;|/|\+|\bfeat\.?(?=\s)|\bft\.?(?=\s)|\bfeaturing\b|\bvs\.?(?=\s)|\bx\b(?=\s))\s*", re.I)
TITLE_FEATURE = re.compile(r"[(\[]\s*(?:feat\.?|ft\.?|featuring|with)\s+([^)\]]+)[)\]]", re.I)
GENERIC = {
    "", "various artists", "various", "va", "unknown artist", "unknown", "artist", "soundtrack",
    "original soundtrack", "ost", "original score", "various composers", "none",
}
NODE_FIELDS = ("leakage_group", "same_audio_group", "inventory_version_group")


def normalise(name):
    """A spelling-independent key for a name, or "" for a generic or empty credit."""
    text = unicodedata.normalize("NFKD", unicodedata.normalize("NFKC", str(name)))
    text = "".join(c for c in text if not unicodedata.combining(c)).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^the ", "", text)
    return "" if text in GENERIC else text


def credited(row, aliases=None):
    """Normalised names credited on an inventory row (aliases applied)."""
    aliases = aliases or {}
    names = []
    for field in ("artist", "album_artist", "mb_artist"):
        names += SEPARATORS.split(row.get(field) or "")
    for found in TITLE_FEATURE.findall(row.get("title") or ""):
        names += SEPARATORS.split(found)
    out = set()
    for name in names:
        key = normalise(name)
        if key:
            out.add(aliases.get(key, key))
    return out


def read_aliases(path):
    """(alias -> canonical, links) from the owner's private alias file; empty if absent."""
    if not path or not Path(path).is_file():
        return {}, []
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    aliases = {normalise(k): normalise(v) for k, v in (data.get("aliases") or {}).items() if normalise(k) and normalise(v)}
    links = [(normalise(a), normalise(b), str(reason)) for a, b, reason in (data.get("links") or [])]
    return aliases, [(a, b, r) for a, b, r in links if a and b]


class Groups:
    """Union-find over identity nodes: ``rec:<sha256>``, ``name:<key>``, ``mbid:<id>``,
    ``leak:``/``audio:``/``version:<group>`` and ``a3:<artist key>``."""

    def __init__(self):
        self.parent = {}

    def add(self, node):
        self.parent.setdefault(node, node)
        return node

    def find(self, node):
        self.add(node)
        while self.parent[node] != node:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node

    def join(self, a, b):
        ra, rb = sorted((self.find(a), self.find(b)))
        if ra != rb:
            self.parent[rb] = ra

    def members(self):
        out = {}
        for node in self.parent:
            out.setdefault(self.find(node), set()).add(node)
        return out


def build(inventory_rows, a3_recordings=None, aliases=None, links=()):
    """Group every inventory recording (and every Phase A3 recording) conservatively.

    ``inventory_rows``: dicts with at least ``sha256``; ``a3_recordings``: sha256 -> A3 artist
    key. Returns ``{"groups": {key: {"nodes": [...], "recordings": [sha256, ...]}},
    "recording_group": {sha256: key}}``. A group's key is its smallest A3 artist key if it has
    one, else its smallest inventory leakage group, else its smallest name: stable as long as
    those stay in the group."""
    groups = Groups()
    a3_recordings = a3_recordings or {}
    for row in inventory_rows:
        sha = row.get("sha256")
        if not sha:
            continue
        rec = groups.add(f"rec:{sha}")
        for name in credited(row, aliases):
            groups.join(rec, f"name:{name}")
        for mbid in re.split(r"[;,| ]+", row.get("mb_artist_ids") or ""):
            if mbid.strip():
                groups.join(rec, f"mbid:{mbid.strip()}")
        for field, prefix in zip(NODE_FIELDS, ("leak", "audio", "version"), strict=True):
            if (row.get(field) or "").strip():
                groups.join(rec, f"{prefix}:{row[field].strip()}")
    for sha, key in a3_recordings.items():
        groups.join(f"rec:{sha}", f"a3:{key}")
    for a, b, _reason in links:
        groups.join(f"name:{a}", f"name:{b}")
    out, recording_group = {}, {}
    for nodes in groups.members().values():
        recordings = sorted(n[4:] for n in nodes if n.startswith("rec:"))
        if not recordings:
            continue
        key = (min((n[3:] for n in nodes if n.startswith("a3:")), default=None)
               or min((n[5:] for n in nodes if n.startswith("leak:")), default=None)
               or "name-" + min((n[5:] for n in nodes if n.startswith("name:")), default=recordings[0][:12]))
        out[key] = {"nodes": sorted(nodes), "recordings": recordings}
        for sha in recordings:
            recording_group[sha] = key
    return {"groups": out, "recording_group": recording_group}
