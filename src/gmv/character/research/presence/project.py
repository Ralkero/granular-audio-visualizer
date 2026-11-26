"""The presence project on disk: folders, manifests and fingerprints.

Layout (all inside the ignored ``local-evidence/sonic-character/vocal-presence/``)::

    manifests/      development.json, validation.json, final-test.json (once frozen),
                    split-registry.json
    fingerprints/   manifests.json: SHA-256 of every manifest and of the registry
    identity/       groups.json (private artist groups); aliases.json (the owner's, optional)
    development/    annotations.jsonl, session.json
    validation/     annotations.jsonl, session.json
    final-test/     annotations.jsonl, session.json: read only by the annotation window
    backups/        zip copies
    selection/      selection report and candidate lists

A manifest lists a partition's excerpts: each with a stable identifier, its recording
(checksum, size, length, rate, channels, path), its times, its artist group and why it was
selected. A manifest is written atomically together with its fingerprint; reading checks the
fingerprint, so a hand-edited or half-written manifest is refused.

**A frozen final test** is two records that must agree: the registry (its ``final_test``
groups, the ``final_test_frozen_utc`` stamp and ``final_test_cohort_sha256``, a digest of the
cohort's excerpts) and ``final-test.json`` (the same stamp, the cohort's ``groups`` and the
excerpts themselves). ``write_frozen_cohort`` writes the manifest first and
the frozen registry last, so the registry is the commit point; on any failure it restores
the earlier registry and fingerprints and removes the new manifest. ``check_final_test``
refuses every other combination, so an interrupted freeze is reported, never used.
"""

import hashlib
import json
import os
from pathlib import Path

FOLDER = Path("local-evidence") / "sonic-character" / "vocal-presence"
FORMAT = "gmv-vocal-presence-manifest"
VERSION = 1
PARTITION_FOLDERS = {"development": "development", "validation": "validation", "final_test": "final-test"}
MANIFESTS = {"development": "development.json", "validation": "validation.json", "final_test": "final-test.json"}
REGISTRY = "split-registry.json"
SUBFOLDERS = ("manifests", "fingerprints", "identity", "development", "validation", "final-test", "backups", "selection")


class IntegrityError(ValueError):
    """A manifest, registry or history failed its integrity checks."""


def default_folder():
    from .... import host, paths

    if paths.checkout():
        return paths.REPOSITORY / FOLDER
    return host.data_folder() / "vocal-presence"


def layout(root):
    root = Path(root)
    for name in SUBFOLDERS:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def partition_folder(root, partition):
    return Path(root) / PARTITION_FOLDERS[partition]


def excerpt_id(sha256, start_s):
    """A stable excerpt identifier from the recording and the start time."""
    return "VP-" + hashlib.sha256(f"{sha256}:{float(start_s):.3f}".encode()).hexdigest()[:10]


def _fingerprints(root):
    path = Path(root) / "fingerprints" / "manifests.json"
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _write_atomic(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    with temporary.open("rb") as file:
        os.fsync(file.fileno())
    temporary.replace(path)


def _write_tracked(root, name, value):
    text = json.dumps(value, indent=1, ensure_ascii=False)
    _write_atomic(Path(root) / "manifests" / name, text)
    prints = _fingerprints(root)
    prints[name] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    _write_atomic(Path(root) / "fingerprints" / "manifests.json", json.dumps(prints, indent=1, sort_keys=True))


def _read_tracked(root, name):
    path = Path(root) / "manifests" / name
    if not path.is_file():
        return None
    data = path.read_bytes()
    expected = _fingerprints(root).get(name)
    if expected is None or hashlib.sha256(data).hexdigest() != expected:
        raise IntegrityError(f"{name} does not match its recorded fingerprint")
    return json.loads(data.decode("utf-8"))


def check_excerpts(partition, excerpts):
    """Structural checks of one manifest's excerpts; returns problems (empty when sound)."""
    problems, seen = [], set()
    for e in excerpts:
        for field in ("id", "recording", "path", "size", "duration_s", "start_s", "end_s", "group", "selection"):
            if field not in e:
                problems.append(f"an excerpt lacks {field}")
                break
        else:
            if e["id"] in seen:
                problems.append("two excerpts share an identifier")
            seen.add(e["id"])
            if e["id"] != excerpt_id(e["recording"], e["start_s"]):
                problems.append("an excerpt identifier does not match its recording and start")
            if not (0 <= e["start_s"] < e["end_s"] <= e["duration_s"] + 1e-6):
                problems.append("an excerpt lies outside its recording")
    return problems


def write_manifest(root, partition, excerpts, extra=None):
    excerpts = sorted(excerpts, key=lambda e: e["id"])
    problems = check_excerpts(partition, excerpts)
    if problems:
        raise IntegrityError("; ".join(sorted(set(problems))[:5]))
    _write_tracked(root, MANIFESTS[partition], {"format": FORMAT, "version": VERSION, "partition": partition,
                                                 **(extra or {}), "excerpts": excerpts})


def read_manifest(root, partition):
    """The partition's manifest, after its fingerprint and structure are checked; None if the
    partition has no manifest yet. Manifests hold no annotations."""
    manifest = _read_tracked(root, MANIFESTS[partition])
    if manifest is None:
        return None
    if manifest.get("format") != FORMAT or manifest.get("partition") != partition:
        raise IntegrityError(f"{MANIFESTS[partition]} is not a {partition} presence manifest")
    problems = check_excerpts(partition, manifest["excerpts"])
    if problems:
        raise IntegrityError("; ".join(sorted(set(problems))[:5]))
    return manifest


def write_registry(root, registry):
    _write_tracked(root, REGISTRY, registry)


def read_registry(root):
    return _read_tracked(root, REGISTRY) or {"entries": {}}


def cohort_digest(excerpts):
    """SHA-256 of a cohort's excerpts, independent of their order and of file formatting."""
    canonical = json.dumps(sorted(excerpts, key=lambda e: e["id"]), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def seal(registry, excerpts):
    """The frozen registry, recording the digest of the cohort's excerpts."""
    return {**registry, "final_test_cohort_sha256": cohort_digest(excerpts)}


def check_final_test(registry, manifest):
    """Problems (empty when sound) between the registry's frozen cohort and the final-test
    manifest: both or neither must record a freeze, with the same stamp, the same groups and
    the same excerpts (by digest), and every frozen group must have at least one excerpt."""
    entries = registry.get("entries", {})
    members = {k for k, e in entries.items() if e["partition"] == "final_test"}
    stamp = registry.get("final_test_frozen_utc")
    if manifest is None:
        if members or stamp or registry.get("final_test_cohort_sha256"):
            return ["the registry records a frozen final-test cohort but its manifest is missing (an interrupted freeze?)"]
        return []
    if not members or not stamp:
        return ["a final-test manifest exists but the registry records no frozen cohort (an interrupted freeze?)"]
    problems = []
    if manifest.get("frozen_utc") != stamp or any(entries[k].get("frozen_utc") != stamp for k in members):
        problems.append("the final-test manifest and the registry record different freezes")
    if set(manifest.get("groups") or ()) != members:
        problems.append("the final-test manifest's cohort differs from the registry's")
    if {e["group"] for e in manifest.get("excerpts", [])} != members:
        problems.append("a frozen final-test group has no excerpt, or an excerpt lies outside the cohort")
    if registry.get("final_test_cohort_sha256") != cohort_digest(manifest.get("excerpts", [])):
        problems.append("the final-test manifest's excerpts differ from the frozen cohort")
    return problems


def cohort_problems(excerpts, registry, extra):
    """Problems (empty when sound) of a planned final-test cohort, before anything is written.
    ``registry``: the frozen registry, sealed or not."""
    planned = {"format": FORMAT, "version": VERSION, "partition": "final_test", **extra, "excerpts": excerpts}
    return sorted(set(check_excerpts("final_test", excerpts) + check_final_test(seal(registry, excerpts), planned)))[:5]


def write_frozen_cohort(root, excerpts, registry, extra):
    """Persist a frozen final-test cohort: ``final-test.json`` (``excerpts`` and ``extra``,
    which must hold ``frozen_utc`` and ``groups``) and the frozen ``registry``, sealed with
    the digest of the excerpts. Returns (manifest, sealed registry) as read back.

    The cohort is checked before anything is written. The manifest is written first, then the
    registry, which is the commit point; both are read back and checked. On any failure or
    interruption inside this call, the new manifest is removed and the earlier registry and
    fingerprints are restored byte for byte, then the error is raised. An existing final-test
    manifest is never replaced."""
    root = Path(root)
    manifest_path = root / "manifests" / MANIFESTS["final_test"]
    if manifest_path.exists():
        raise IntegrityError("a final-test manifest already exists; a frozen cohort is never replaced")
    problems = cohort_problems(excerpts, registry, extra)
    if problems:
        raise IntegrityError("; ".join(problems))
    registry = seal(registry, excerpts)
    earlier = {path: (path.read_bytes() if path.is_file() else None)
               for path in (root / "manifests" / REGISTRY, root / "fingerprints" / "manifests.json")}
    try:
        write_manifest(root, "final_test", excerpts, extra)
        write_registry(root, registry)  # the commit point
        written = read_manifest(root, "final_test")
        if read_registry(root) != registry or check_final_test(registry, written):
            raise IntegrityError("the frozen cohort did not read back as written")
    except BaseException:
        for leftover in (manifest_path, manifest_path.with_suffix(manifest_path.suffix + ".tmp")):
            leftover.unlink(missing_ok=True)
        for path, data in earlier.items():
            if data is None:
                path.unlink(missing_ok=True)
            else:
                _write_atomic(path, data.decode("utf-8"))
        raise
    return written, registry


def durations(manifest):
    return {e["id"]: e["end_s"] - e["start_s"] for e in (manifest or {}).get("excerpts", [])}
