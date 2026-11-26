"""Building (or extending) the presence project from the Phase A3 project and the inventory.

Read only: the A3 folder (``dataset.json``, ``titles.json``; answers through the gated
development-only reader, so no held-out answer is ever decoded), the music inventory, the
owner's alias file and, optionally, the corpus manifest (for MUSDB18-HQ test artists). The
A3 files are fingerprinted before and after; any change stops with an error.

Written, inside the presence folder only: the identity groups, the partition registry, the
development and validation manifests (existing excerpts are always kept; new ones are
added), private titles, the selection report and candidate lists. The final-test manifest
is written only when the owner freezes the cohort, and only if enough never-used artist
groups exist.
"""

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import soundfile as sf

from ..delivery import data as delivery_data
from ..labeling import dataset as labelling
from ..labeling import titles as labelling_titles
from . import identity, project, selection, split

A3_FILES = ("dataset.json", "labels.jsonl", "titles.json", "session.json", "reference_vocalists.txt")
MINIMUM_FINAL_GROUPS = 20  # a cohort of fewer never-used artist groups is not frozen


class PrepareError(ValueError):
    pass


def fingerprint_files(folder, names=A3_FILES):
    folder = Path(folder)
    return {n: hashlib.sha256((folder / n).read_bytes()).hexdigest() for n in names if (folder / n).is_file()}


def read_inventory(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def musdb_test_artists(corpus_manifest):
    """Artist names of MUSDB18-HQ test tracks ("Artist - Title" track ids), if the corpus
    manifest is available."""
    if not corpus_manifest or not Path(corpus_manifest).is_file():
        return []
    manifest = json.loads(Path(corpus_manifest).read_text(encoding="utf-8"))
    return sorted({r["dataset_track_id"].split(" - ")[0] for r in manifest.get("recordings", [])
                   if r.get("dataset") == "musdb18hq" and r.get("split") == "test" and " - " in r.get("dataset_track_id", "")})


def local_path(path, drive_map):
    for source, target in drive_map:
        if path.startswith(source):
            path = target + path[len(source):]
    return path.replace("\\", "/")


def describe(path, sha_expected=None, verify=False):
    """(size, duration, rate, channels) of an audio file; with ``verify``, also checks its
    SHA-256 against the expected one."""
    path = Path(path)
    info = sf.info(str(path))
    if verify and sha_expected:
        digest = hashlib.sha256()
        with path.open("rb") as file:
            for chunk in iter(lambda: file.read(1 << 20), b""):
                digest.update(chunk)
        if digest.hexdigest() != sha_expected:
            raise PrepareError("a recording's contents differ from its recorded checksum")
    return path.stat().st_size, info.frames / info.samplerate, info.samplerate, info.channels


def prepare(root, a3_folder, inventory_path, aliases_path=None, corpus_manifest=None,
            drive_map=(), freeze=False,
            minimum_final=MINIMUM_FINAL_GROUPS, per_category=20, target=selection.TARGET, say=print, verify=True):
    root = project.layout(root)
    a3_folder = Path(a3_folder)
    before = fingerprint_files(a3_folder)
    dataset = labelling.load(a3_folder)
    found = delivery_data.load(a3_folder)  # development answers only; held-out never decoded
    if found["audit"]["held_out_values_decoded"]:
        raise PrepareError("the gated reader reported decoding a held-out answer")
    a3_titles = labelling_titles.load(a3_folder)
    inventory = read_inventory(inventory_path)
    aliases, links = identity.read_aliases(aliases_path or root / "identity" / "aliases.json")
    consumed_names = musdb_test_artists(corpus_manifest)

    a3_recordings = {r["sha256"]: r["artist"] for r in dataset["recordings"].values()}
    groups = identity.build(inventory, a3_recordings, aliases, links)
    evaluation_keys = {a for a, e in dataset["artists"].items() if e["split"] == "evaluation"}
    development_keys = {a for a, e in dataset["artists"].items() if e["split"] == "development"}
    consumed = split.consumed_nodes(evaluation_keys, consumed_names)
    registry = split.assign(groups["groups"], project.read_registry(root), consumed, development_keys)
    entry_of = {}
    for key, entry in registry["entries"].items():
        for node in entry["nodes"]:
            if node.startswith("rec:"):
                entry_of[node[4:]] = key

    # -- A: existing development excerpts ---------------------------------------------------
    category = {e["excerpt"]: e["label"] for e in found["examples"]}
    for row in found["private"]["excluded"]:
        if row["reason"] in ("gutturals", "sing_screams", "no_vocals"):
            category[row["excerpt"]] = row["reason"]
    side = delivery_data.sides(dataset)
    pool, dropped = [], Counter()
    for excerpt in dataset["excerpts"]:
        if side[excerpt["id"]] != "development" or excerpt["id"] not in category:
            continue
        sha = dataset["recordings"][excerpt["recording"]]["sha256"]
        key = entry_of.get(sha)
        partition = registry["entries"][key]["partition"] if key else None
        if partition not in ("development", "validation"):
            dropped[partition or "ungrouped"] += 1
            continue
        pool.append({"id": excerpt["id"], "recording": sha, "group": key, "partition": partition,
                     "category": category[excerpt["id"]], "excerpt": excerpt})
    batch = selection.development_batch(pool, target=target)

    # -- B: negative candidates in development-side recordings ------------------------------
    taken = defaultdict(list)
    for excerpt in dataset["excerpts"]:
        taken[dataset["recordings"][excerpt["recording"]]["sha256"]].append((excerpt["start_s"], excerpt["end_s"]))
    # Windows avoid the A3 excerpts only, so preparing again proposes the same windows (the
    # same identifiers) and existing manifest entries are simply kept.
    existing = {p: project.read_manifest(root, p) for p in ("development", "validation")}
    eligible = []
    for row in inventory:
        key = entry_of.get(row["sha256"])
        if not key or registry["entries"][key]["partition"] not in ("development", "validation"):
            continue
        try:
            duration = float(str(row.get("duration_s", "0")).replace(",", "."))
        except ValueError:
            continue
        eligible.append({"sha256": row["sha256"], "group": key, "duration_s": duration, "row": row,
                         "taken": taken[row["sha256"]], "path": local_path(row["file_path"], drive_map)})
    negatives = selection.negative_candidates(eligible, per_category=per_category)

    # -- manifests -------------------------------------------------------------------------
    by_sha = {r["sha256"]: r for r in eligible}
    a3_by_sha = {r["sha256"]: (k, r) for k, r in dataset["recordings"].items()}
    described, problems = {}, Counter()

    def entry(sha, start, length, source, group, reason):
        if sha in a3_by_sha:
            recording = a3_by_sha[sha][1]
            path, size, duration, rate, channels = (recording["path"], recording["size"], recording["duration_s"],
                                                    recording["sample_rate"], recording["channels"])
        else:
            path = by_sha[sha]["path"]
            if sha not in described:
                described[sha] = describe(path, sha, verify)
            size, duration, rate, channels = described[sha]
        return {"id": project.excerpt_id(sha, start), "source": source, "recording": sha, "path": str(path),
                "size": size, "duration_s": round(duration, 6), "sample_rate": rate, "channels": channels,
                "start_s": round(start, 3), "end_s": round(start + length, 3), "group": group, "selection": reason}

    new = {"development": [], "validation": []}
    for item in batch:
        e = item["excerpt"]
        new[item["partition"]].append(entry(item["recording"], e["start_s"], e["end_s"] - e["start_s"],
                                            f"a3:{item['id']}", item["group"], item["selection"]))
    say(f"checking {len({c['recording'] for v in negatives.values() for c in v} - set(a3_by_sha))} new candidate recordings")
    for items in negatives.values():
        for item in items:
            if not Path(by_sha[item["recording"]]["path"]).is_file():
                problems["candidate audio not found"] += 1
                continue
            partition = registry["entries"][item["group"]]["partition"]
            new[partition].append(entry(item["recording"], item["start_s"], selection.EXCERPT_S, "new", item["group"], item["selection"]))
    written = {}
    for partition in ("development", "validation"):
        keep = {e["id"]: e for e in (existing[partition] or {}).get("excerpts", [])}
        for e in new[partition]:
            keep.setdefault(e["id"], e)
        project.write_manifest(root, partition, list(keep.values()))
        written[partition] = list(keep.values())
    project.write_registry(root, registry)
    (root / "identity" / "groups.json").write_text(json.dumps(groups, indent=1, ensure_ascii=False), encoding="utf-8")
    titles = {}
    for key, recording in dataset["recordings"].items():
        title = (a3_titles.get(key) or {}).get("title")
        if title:
            titles[recording["sha256"]] = title
    for row in inventory:
        if row.get("title") and row["sha256"] not in titles:
            titles[row["sha256"]] = row["title"]
    (root / "identity" / "titles.json").write_text(json.dumps(titles, indent=1, ensure_ascii=False), encoding="utf-8")

    # -- final test ---------------------------------------------------------------------------
    partitions = Counter(e["partition"] for e in registry["entries"].values())
    reserved = [k for k, e in registry["entries"].items() if e["partition"] == "reserved"]
    reserved_recordings = sum(sum(1 for n in registry["entries"][k]["nodes"] if n.startswith("rec:")) for k in reserved)
    final = {"frozen": False, "reserved_groups": len(reserved), "reserved_recordings": reserved_recordings,
             "minimum_groups": minimum_final}
    if freeze:
        try:
            split.freeze_final_test(registry, set(reserved), minimum_final)
        except split.FreezeRefused as exc:
            final["refused"] = str(exc)
    after = fingerprint_files(a3_folder)
    if after != before:
        raise PrepareError("the Phase A3 files changed during preparation")

    summary = {
        "groups": len(groups["groups"]),
        "registry": dict(partitions),
        "development_pool": selection.coverage(pool),
        "pool_dropped_by_partition": dict(dropped),
        "batch": Counter(i["category"] for i in batch),
        "batch_groups": len({i["group"] for i in batch}),
        "batch_recordings": len({i["recording"] for i in batch}),
        "negatives": {k: len(v) for k, v in negatives.items()},
        "negative_groups": {k: len({c["group"] for c in v}) for k, v in negatives.items()},
        "manifests": {p: len(v) for p, v in written.items()},
        "problems": dict(problems),
        "final_test": final,
        "consumed_names": len(consumed_names),
        "a3_unchanged": True,
    }
    (root / "selection" / "summary.json").write_text(json.dumps(summary, indent=1, default=list), encoding="utf-8")
    with (root / "selection" / "candidates.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["partition", "id", "source", "group", "start_s", "end_s", "title", "selection"])
        for partition, excerpts in written.items():
            for e in sorted(excerpts, key=lambda x: (x["selection"], x["id"])):
                writer.writerow([partition, e["id"], e["source"], e["group"], e["start_s"], e["end_s"],
                                 titles.get(e["recording"], ""), e["selection"]])
    return summary
