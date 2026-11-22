"""Which recordings must stay on the same side of a train / evaluation split.

The split unit of this project is the artist folder. That is not enough on its own: the
same audio can sit in two artist folders (a compilation or soundtrack copy), and alternate
versions of one recording can be filed apart. Groups are therefore the connected components
of these links:

* same artist folder;
* same audio by fingerprint (``same_audio_groups.jsonl``);
* the inventory's exact-duplicate, potential-duplicate and alternate-version groups;
* the same MusicBrainz recording id, where the identity was matched reliably.

Two folders joined by any link become one group, with every recording in both.
"""

from collections import defaultdict

from . import read_jsonl, recording_id


def groups(rows, same_audio=(), identities=()):
    """{recording id: (group name, [reasons it was merged beyond its own folder])}."""
    from ..review.batch import group_of

    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def join(a, b):
        parent[find(a)] = find(b)

    by_id = {recording_id(row["sha256"]): row for row in rows}
    folder_of = {rid: group_of(row) for rid, row in by_id.items()}
    links = defaultdict(list)  # folder group -> reasons
    for rid in by_id:
        find(folder_of[rid])

    def link(members, reason):
        folders = sorted({folder_of[m] for m in members if m in folder_of})
        for other in folders[1:]:
            join(folders[0], other)
        if len(folders) > 1:
            for folder in folders:
                links[folder].append(reason)

    for group in same_audio:
        link(group["recordings"], f"same audio ({group['group']})")
    for key in ("exact_duplicate_group", "potential_duplicate_group", "alternate_version_group"):
        members = defaultdict(list)
        for rid, row in by_id.items():
            if row.get(key):
                members[row[key]].append(rid)
        for name, found in members.items():
            link(found, f"{key.replace('_', ' ')} {name}")
    by_mbid = defaultdict(list)
    for identity in identities:
        if identity.get("recording_mbid") and identity.get("confidence") in ("high", "medium"):
            by_mbid[identity["recording_mbid"]].append(identity["recording_id"])
    for mbid, found in by_mbid.items():
        link(found, f"same MusicBrainz recording {mbid[:8]}")
    components = defaultdict(set)
    for folder in set(folder_of.values()):
        components[find(folder)].add(folder)
    name_of = {}
    for members in components.values():
        ordered = sorted(members)
        name = ordered[0] if len(ordered) == 1 else "merged-" + "+".join(m.removeprefix("artist-") for m in ordered)[:90]
        for folder in ordered:
            name_of[folder] = name
    out = {}
    for rid, folder in folder_of.items():
        out[rid] = (name_of[folder], sorted(set(links.get(folder, []))))
    return out


def table(rows, workdir):
    same_audio = read_jsonl(workdir / "same_audio_groups.jsonl")
    identities = read_jsonl(workdir / "musicbrainz" / "identity.jsonl")
    found = groups(rows, same_audio, identities)
    by_id = {recording_id(row["sha256"]): row for row in rows}
    in_same_audio = {m: g["group"] for g in same_audio for m in g["recordings"]}
    return [
        {
            "recording_id": rid,
            "sha256": by_id[rid]["sha256"],
            "leakage_group": name,
            "same_audio_group": in_same_audio.get(rid, ""),
            "merged_because": "; ".join(reasons),
        }
        for rid, (name, reasons) in sorted(found.items())
    ]
