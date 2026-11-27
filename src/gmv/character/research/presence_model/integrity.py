"""Integrity records, recomputation checks and the computational/provenance split (readiness
proposal, revision 4.4, section 5). NumPy and the standard library only.

* **R-A** per excerpt: the audio file, the segment and grid, the SHA-256 of every
  frame array (labels, masks, quiet) and the frame counts. No input-manifest fingerprints.
* **R-B** the annotation history and the folded state used; **R-C** the development manifest
  (public) and, privately, the registry, fingerprints and other manifests; **R-D** code and
  environment; **R-E** the configuration; **R-F** caches and normalisation statistics.

``compare`` reports mismatches as kinds and counts only (never excerpt identifiers or
values). Artifact trees keep computational outputs (no provenance, no timestamps) apart from
provenance records, descriptive breakdown tables and run logs; ``compare_trees`` checks that
computational files are byte-identical and that provenance differs only in allowed fields.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

KINDS = ("computational", "provenance", "breakdown", "runlog")
EXCERPT_FIELDS = (
    "audio_sha256", "size", "sample_rate", "channels", "frames", "first", "end", "grid_frames",
    "f0", "f1", "labels_sha256", "evaluation_sha256", "training_sha256", "quiet_sha256",
    "n_voiced_training", "n_novoice_training", "n_voiced_evaluation", "n_novoice_evaluation",
)


class IntegrityMismatch(RuntimeError):
    """A recomputed record differs from the stored one; the run stops before training."""


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def array_sha256(array):
    array = np.ascontiguousarray(array)
    return sha256_bytes(f"{array.dtype.str}{tuple(array.shape)}".encode() + b"\0" + array.tobytes())


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def excerpt_record(audio_sha256, size, sample_rate, channels, frames, segment, arrays, quiet):
    labels, training, evaluation = arrays["labels"], arrays["training"], arrays["evaluation"]
    voice = labels == 1
    return {
        "audio_sha256": audio_sha256, "size": int(size), "sample_rate": int(sample_rate),
        "channels": int(channels), "frames": int(frames), "first": int(segment.first),
        "end": int(segment.end), "grid_frames": int(segment.frames), "f0": int(segment.f0),
        "f1": int(segment.f1), "labels_sha256": array_sha256(labels),
        "evaluation_sha256": array_sha256(evaluation), "training_sha256": array_sha256(training),
        "quiet_sha256": array_sha256(np.asarray(quiet, dtype=bool)),
        "n_voiced_training": int(np.count_nonzero(training & voice)),
        "n_novoice_training": int(np.count_nonzero(training & ~voice)),
        "n_voiced_evaluation": int(np.count_nonzero(evaluation & voice)),
        "n_novoice_evaluation": int(np.count_nonzero(evaluation & ~voice)),
    }


def record(excerpts, history_sha256, folded_state_sha256, development_manifest_sha256,
           private_inputs, environment, configuration, caches=None):
    return {
        "R-A": dict(excerpts),
        "R-B": {"history_sha256": history_sha256, "folded_state_sha256": folded_state_sha256},
        "R-C": {"development_manifest_sha256": development_manifest_sha256, "private": dict(private_inputs)},
        "R-D": dict(environment),
        "R-E": {"configuration": configuration, "configuration_sha256": sha256_bytes(canonical_json(configuration))},
        "R-F": dict(caches or {}),
    }


def public_digest(rec):
    """Development-scope values only: no identifiers, no private fingerprints."""
    excerpts = sorted(canonical_json(r).decode() for r in rec["R-A"].values())
    return {
        "development_manifest_sha256": rec["R-C"]["development_manifest_sha256"],
        "history_sha256": rec["R-B"]["history_sha256"],
        "folded_state_sha256": rec["R-B"]["folded_state_sha256"],
        "excerpts_sha256": sha256_bytes(canonical_json(excerpts)),
        "configuration_sha256": rec["R-E"]["configuration_sha256"],
        "environment": {k: v for k, v in rec["R-D"].items() if k != "private"},
    }


def flatten(value, prefix=""):
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            out.update(flatten(item, f"{prefix}{key}."))
        return out
    return {prefix[:-1]: value}


def compare(expected, actual):
    """{kind: count} of differences; empty when the records agree."""
    kinds = {}

    def bump(kind):
        kinds[kind] = kinds.get(kind, 0) + 1

    ea, aa = expected.get("R-A", {}), actual.get("R-A", {})
    for _key in ea.keys() - aa.keys():
        bump("R-A.missing_excerpt")
    for _key in aa.keys() - ea.keys():
        bump("R-A.extra_excerpt")
    for key in ea.keys() & aa.keys():
        for field in EXCERPT_FIELDS:
            if ea[key].get(field) != aa[key].get(field):
                bump(f"R-A.{field}")
    rest_e = flatten({k: v for k, v in expected.items() if k != "R-A"})
    rest_a = flatten({k: v for k, v in actual.items() if k != "R-A"})
    for path in rest_e.keys() | rest_a.keys():
        if rest_e.get(path) != rest_a.get(path):
            bump(path if not path.startswith("R-C.private") else "R-C.private")
    return dict(sorted(kinds.items()))


def require_equal(expected, actual):
    differences = compare(expected, actual)
    if differences:
        raise IntegrityMismatch("integrity mismatch: " + ", ".join(f"{k} x{n}" for k, n in differences.items()))


def environment(repository=None, lock=None):
    """R-D: versions, CPU, thread settings, lock-file hash and, given a repository, its commit
    and whether the worktree is clean. No timestamps."""
    import importlib
    import os
    import platform
    import subprocess

    out = {"python": platform.python_version(), "machine": platform.machine(), "numpy": np.__version__}
    for name in ("scipy", "soundfile", "torch"):
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        out[name] = module.__version__
        if name == "soundfile":
            out["libsndfile"] = module.__libsndfile_version__
    try:
        cpu = next(line for line in Path("/proc/cpuinfo").read_text().splitlines() if line.startswith("model name"))
        out["cpu"] = cpu.split(":", 1)[1].strip()
    except (OSError, StopIteration):
        out["cpu"] = platform.processor()
    out["threads"] = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    if lock is not None:
        out["lock_sha256"] = file_sha256(lock)
    if repository is not None:
        def git(*args):
            return subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True, check=True).stdout
        out["commit"] = git("rev-parse", "HEAD").strip()
        out["clean"] = git("status", "--porcelain") == ""
    return out


# -- artifact trees ------------------------------------------------------------------------------


def write_artifact(root, kind, name, data):
    if kind not in KINDS:
        raise ValueError("unknown artifact kind")
    path = Path(root) / kind / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data if isinstance(data, bytes) else canonical_json(data))
    return path


def compare_trees(a, b, allowed_provenance=(), breakdown_may_differ=False):
    """Differences between two artifact trees, as kinds and counts only. Computational files
    must be byte-identical; provenance JSON may differ only at ``allowed_provenance`` paths
    (exact dotted paths; ``*`` matches one path segment); breakdown files must be identical
    unless ``breakdown_may_differ``; run logs are not compared."""
    a, b = Path(a), Path(b)
    files_a = {p.relative_to(a) for p in a.rglob("*") if p.is_file()}
    files_b = {p.relative_to(b) for p in b.rglob("*") if p.is_file()}
    out = {"missing_or_extra_files": len(files_a ^ files_b), "computational": 0, "provenance_unexpected": 0,
           "provenance_allowed": 0, "breakdown": 0}
    for rel in sorted(files_a & files_b):
        kind = rel.parts[0]
        if kind == "runlog":
            continue
        da, db = (a / rel).read_bytes(), (b / rel).read_bytes()
        if da == db:
            continue
        if kind == "computational":
            out["computational"] += 1
        elif kind == "breakdown":
            out["breakdown"] += 0 if breakdown_may_differ else 1
        elif kind == "provenance":
            fa, fb = flatten(json.loads(da)), flatten(json.loads(db))
            for path in fa.keys() | fb.keys():
                if fa.get(path) != fb.get(path):
                    allowed = any(_matches(pattern, path) for pattern in allowed_provenance)
                    out["provenance_allowed" if allowed else "provenance_unexpected"] += 1
        else:
            out["missing_or_extra_files"] += 1
    return out


def _matches(pattern, path):
    p, q = pattern.split("."), path.split(".")
    return len(p) == len(q) and all(x == "*" or x == y for x, y in zip(p, q, strict=True))
