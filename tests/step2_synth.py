"""A synthetic world for A5.2 Step 2a tests, laid out like the real one (Amendment 3, A3.4 and
A3.6), entirely under a temporary folder. Nothing here reads project data.

Layout under ``base``:

* ``corpus-data/corpus/manifest.json``: the corpus manifest (MUSDB18-HQ records plus decoy
  records of another dataset), valid under GMV's strict corpus schema;
* ``corpus/manifest.json``: the tracked, documented-empty manifest (it must fail closed);
* ``corpus-data/corpus-local/paths.json``, ``corpus-data/owned/manifest.json`` and a MUSDB audio
  file: decoys step 2 must never open;
* ``project/sonic-character/vocal-presence/``: the presence project (``presence_synth``),
  with registry name nodes, ``identity/aliases.json``, ``identity/titles.json`` and the Tier 3
  decoys ``identity/groups.json``, ``session.json`` and ``selection/``;
* ``inputs/credit-readings.json`` and ``inputs/review-decisions.json``: the step's private inputs.
"""

import hashlib
import json
from pathlib import Path

import presence_synth

from gmv.character.research.presence import identity, project
from gmv.character.research.presence_model import readers


def sha_of(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def corpus_record(track_id, split, sha=None, dataset="musdb18hq", group=None):
    sha = sha or sha_of(f"{dataset}:{track_id}")
    record = {"recording_id": "r-" + sha[:12], "sha256": sha, "duration_s": 180.0, "sample_rate": 44100,
              "channels": 2, "source_kind": "open_dataset", "license": "research licence (synthetic)",
              "permission_basis": "synthetic test fixture", "group_id": group or f"g-{sha[:10]}", "split": split}
    if dataset:
        record.update(dataset=dataset, dataset_track_id=track_id)
    return record


def write_corpus(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"corpus_schema_version": "1.0", "corpus_version": "1.0.0", "recordings": records}))
    return path


def write_readings(path, readings):
    """``readings``: credit -> "whole-and-pieces" or ("single-name", reason)."""
    body = {}
    for credit, reading in readings.items():
        body[credit] = ({"reading": "single-name", "reason": reading[1]} if isinstance(reading, tuple)
                        else {"reading": reading})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"format": readers.READINGS_FORMAT, "version": 1, "readings": body}))
    return Path(path)


def write_decisions(path, pairs):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps({"format": readers.DECISIONS_FORMAT, "version": 1, "rejections": [
        {"track_id": t, "group": g, "reason": "not-the-same-artist"} for t, g in pairs]}))
    return Path(path)


def build(base, names=None, records=(), aliases=None, links=(), titles=None, readings=None, decisions=(),
          spec=None, reserved=None):
    """Write the world. ``names``: group key -> list of artist names (stored normalised, as
    ``identity.build`` does); ``records``: corpus records (``corpus_record``); ``titles``: group
    key -> title for its first recording; ``reserved``: extra reserved group key -> names.
    Returns a dict of paths and facts."""
    base = Path(base)
    root = base / "project" / "sonic-character" / "vocal-presence"
    built = presence_synth.build(root, spec if spec is not None else presence_synth.default_spec(groups=4))
    registry = project.read_registry(root)
    recordings = {}
    for key, entry in registry["entries"].items():
        recordings[key] = sorted(n[4:] for n in entry["nodes"] if n.startswith("rec:"))
        for name in (names or {}).get(key, [f"artist {key}"]):
            entry["nodes"].append(f"name:{identity.normalise(name)}")
        entry["nodes"] = sorted(set(entry["nodes"]))
    for key, artist_names in (reserved or {}).items():
        registry["entries"][key] = {"partition": "reserved",
                                    "nodes": sorted(f"name:{identity.normalise(n)}" for n in artist_names)}
    project.write_registry(root, registry)
    (root / "identity").mkdir(parents=True, exist_ok=True)
    (root / "identity" / "aliases.json").write_text(json.dumps({"aliases": aliases or {}, "links": [list(x) for x in links]}))
    title_map = {recordings[k][0]: t for k, t in (titles or {}).items()}
    (root / "identity" / "titles.json").write_text(json.dumps(title_map))
    (root / "identity" / "groups.json").write_text(json.dumps({"note": "tier 3 decoy"}))
    (root / "session.json").write_text(json.dumps({"note": "decoy"}))
    (root / "selection").mkdir(exist_ok=True)
    (root / "selection" / "selection.json").write_text(json.dumps({"note": "decoy"}))
    corpus = write_corpus(base / "corpus-data" / "corpus" / "manifest.json", list(records) + [
        corpus_record("Other Dataset Track", "train", dataset="other-dataset")])
    tracked_empty = write_corpus(base / "corpus" / "manifest.json", [])
    (base / "corpus-data" / "corpus-local").mkdir(parents=True, exist_ok=True)
    (base / "corpus-data" / "corpus-local" / "paths.json").write_text(json.dumps({"note": "decoy"}))
    write_corpus(base / "corpus-data" / "owned" / "manifest.json", [])
    (base / "corpus-data" / "musdb").mkdir(parents=True, exist_ok=True)
    (base / "corpus-data" / "musdb" / "mixture.wav").write_bytes(b"RIFF decoy")
    readings_path = write_readings(base / "inputs" / "credit-readings.json", readings or {})
    decisions_path = write_decisions(base / "inputs" / "review-decisions.json", decisions)
    return {"base": base, "root": root, "corpus": corpus, "tracked_empty": tracked_empty,
            "aliases": root / "identity" / "aliases.json", "titles": root / "identity" / "titles.json",
            "readings": readings_path, "decisions": decisions_path, "recordings": recordings,
            "registry": root / "manifests" / project.REGISTRY, "expected": built["expected"]}
