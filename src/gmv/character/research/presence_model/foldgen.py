"""Step 3's fold generation, built and tested on synthetic projects in Step 1 (readiness
proposal, revision 4.4, section 3). It reads only gateway records (Tier 2 fields), the
paired excerpts' audio and the development history and manifest bytes for provenance.

1. Check the aggregates against the expected values (the registered ones by default) and
   stop on any mismatch: statuses, excerpt and group counts, the low-rate group (excerpts
   whose rate B0 cannot even construct, all in one group), voiced and no-voice totals.
2. Stop unless every paired rate is supported by B0.
3. Per paired excerpt, in id order: the file's SHA-256 and size, its rate, channels and
   length against the manifest, at most 2 channels, the canonical segment (shared decoder),
   the engine's quiet mask, labels and masks on the grid, integer counts.
4. Folds: groups typed from integer counts, ordered by SHA-256 of ``{FOLD_SEED}:outer:{key}``;
   ``folds.for_outer`` per outer fold; an independent integer verifier.
5. Artifacts: computational (outer order, fold files, group counts, public summary) and
   provenance kept apart, plus a run log. Provenance: the integrity record (its R-E carries
   the seed format and enumeration fingerprint), the public digest (development scope only)
   and the fold index, whose ``public`` and ``chain`` fields hold the public development
   manifest hash and the hashes of the integrity record and of the public digest.
"""

import hashlib
import math
from pathlib import Path

import numpy as np

from . import folds, grid, integrity, seedplan
from ..presence.schema import union
from .seeds import FOLD_SEED

REGISTERED = {"excerpts": 171, "groups": 20, "low_rate_excerpts": 8, "paired_excerpts": 163,
              "paired_groups": 19, "voiced_groups": 15, "novoice_groups": 19, "voiced_s": 694, "novoice_s": 936}
MIN_FRAMES = 100
SEEK_VERIFIED = {"WAV", "FLAC"}  # containers where seek-and-read equals decode-and-slice (tested)
GATEWAY_FIELDS = {"id", "recording", "path", "size", "sample_rate", "channels", "duration_s", "start_s",
                  "end_s", "group", "status", "confirmed", "uncertain"}


class FoldGenerationError(RuntimeError):
    """Fold generation stopped; nothing was written (messages carry counts only)."""


def read_segment(path, first, end):
    """The shared decoder: samples [first, end) as float32, shaped (samples, channels)."""
    import soundfile as sf

    with sf.SoundFile(str(path)) as file:
        if file.format in SEEK_VERIFIED:
            file.seek(first)
            return file.read(end - first, dtype="float32", always_2d=True)
        audio = file.read(dtype="float32", always_2d=True)
    return audio[first:end]


def _voiced_seconds(record):
    duration = record["end_s"] - record["start_s"]
    return sum(min(e, duration) - max(s, 0.0) for s, e in union(record["confirmed"]))


def check_aggregates(records, rate_class, expected):
    if any(r["status"] not in ("positive", "negative") for r in records):
        raise FoldGenerationError("an excerpt is not marked positive or negative")
    low = [r for r in records if rate_class(r["sample_rate"]) == "not_constructible"]
    paired = [r for r in records if rate_class(r["sample_rate"]) != "not_constructible"]
    voiced = sum(_voiced_seconds(r) for r in paired)
    novoice = sum(r["end_s"] - r["start_s"] for r in paired) - voiced
    found = {
        "excerpts": len(records), "groups": len({r["group"] for r in records}),
        "low_rate_excerpts": len(low), "paired_excerpts": len(paired), "paired_groups": len({r["group"] for r in paired}),
        "voiced_groups": len({r["group"] for r in paired if _voiced_seconds(r) > 0}),
        "novoice_groups": len({r["group"] for r in paired if r["end_s"] - r["start_s"] - _voiced_seconds(r) > 0}),
        "voiced_s": round(voiced), "novoice_s": round(novoice),
    }
    if len({r["group"] for r in low}) > 1:
        raise FoldGenerationError("excerpts B0 cannot construct come from more than one group")
    mismatched = sorted(k for k in expected if found[k] != expected[k])
    if mismatched:
        raise FoldGenerationError(f"{len(mismatched)} registered aggregates differ: {', '.join(mismatched)}")
    if {r["group"] for r in low} & {r["group"] for r in paired}:
        raise FoldGenerationError("the low-rate group also has paired excerpts")
    return paired


def group_type(voiced, novoice):
    if voiced >= MIN_FRAMES and novoice >= MIN_FRAMES:
        return "both"
    if voiced >= MIN_FRAMES:
        return "voice_only"
    if novoice >= MIN_FRAMES:
        return "novoice_only"
    raise FoldGenerationError("a group has fewer than 100 eligible frames of either class")


def outer_order(keys):
    return sorted(keys, key=lambda k: hashlib.sha256(f"{FOLD_SEED}:outer:{k}".encode()).hexdigest())


def verify(counts, outer, assignment):
    """The registered fold checks again, in integers, without ``folds.check``."""
    keys = set(counts) - {outer}
    members = assignment.folds
    if set(members) != keys:
        raise FoldGenerationError("a fold assignment does not hold exactly the other groups")
    total_v = sum(counts[k][0] for k in keys)
    total_n = sum(counts[k][1] for k in keys)
    kinds = {k: group_type(*counts[k]) for k in keys}
    for fold in range(folds.FOLDS):
        held = [k for k in keys if members[k] == fold]
        if len(held) > folds.CAPACITY:
            raise FoldGenerationError("an inner fold holds more than 4 groups")
        kept_v = total_v - sum(counts[k][0] for k in held)
        kept_n = total_n - sum(counts[k][1] for k in held)
        if 100 * kept_v < 65 * total_v or 100 * kept_n < 65 * total_n:
            raise FoldGenerationError("an inner training partition keeps less than 65% of a class")
        training = keys - set(held)
        for c in (0, 1):
            if not any(counts[k][c] >= MIN_FRAMES for k in training):
                raise FoldGenerationError("an inner training set has no member group in a class")
    for kind in ("voice_only", "novoice_only"):
        n_type = sum(1 for k in keys if kinds[k] == kind)
        limit = math.ceil(n_type / folds.FOLDS) if n_type else 0
        for fold in range(folds.FOLDS):
            if sum(1 for k in keys if members[k] == fold and kinds[k] == kind) > limit:
                raise FoldGenerationError("an inner fold holds too many single-class groups of one type")


def generate(records, root, out, rate_class, quiet_of, expected=REGISTERED, environment=None,
             configuration=None, read=read_segment):
    """Run fold generation and write its artifacts under ``out``; returns the integrity
    record. ``records``: gateway records; ``rate_class``: rate -> B0 class; ``quiet_of``:
    (audio, rate) -> quiet mask."""
    import soundfile as sf

    if any(set(r) - GATEWAY_FIELDS for r in records):
        raise FoldGenerationError("records carry fields the gateway does not pass")
    paired = check_aggregates(records, rate_class, expected)
    if any(rate_class(r["sample_rate"]) != "supported" for r in paired):
        raise FoldGenerationError("a paired excerpt's rate is not supported by B0")
    excerpts, counts, folded = {}, {}, []
    for r in sorted(paired, key=lambda x: x["id"]):
        path = Path(r["path"])
        if path.stat().st_size != r["size"] or integrity.file_sha256(path) != r["recording"]:
            raise FoldGenerationError("an audio file differs from its manifest entry")
        info = sf.info(str(path))
        if info.samplerate != r["sample_rate"] or info.channels != r["channels"]:
            raise FoldGenerationError("an audio file's rate or channels differ from the manifest")
        if abs(info.frames - r["duration_s"] * info.samplerate) > 1 or info.channels > 2:
            raise FoldGenerationError("an audio file's length or channel count is not usable")
        seg = grid.canonical(info.frames, info.samplerate, r["start_s"], r["end_s"])
        audio = read(path, seg.first, seg.end)
        quiet = np.asarray(quiet_of(audio, info.samplerate), dtype=bool)
        arrays = grid.frame_arrays(seg, r["confirmed"], r["uncertain"], quiet)
        rec = integrity.excerpt_record(r["recording"], r["size"], info.samplerate, info.channels,
                                       info.frames, seg, arrays, quiet)
        excerpts[r["id"]] = rec
        v, n = counts.get(r["group"], (0, 0))
        counts[r["group"]] = (v + rec["n_voiced_training"], n + rec["n_novoice_training"])
        folded.append({"id": r["id"], "status": r["status"], "confirmed": r["confirmed"], "uncertain": r["uncertain"]})
    for key in counts:
        group_type(*counts[key])
    seconds = {k: (v / 50, n / 50) for k, (v, n) in counts.items()}
    order = outer_order(counts)
    assignments = []
    for key in order:
        try:
            assignment = folds.for_outer(seconds, key)
        except folds.FoldAssignmentError as exc:
            raise FoldGenerationError("an outer fold has no assignment that passes the registered checks") from exc
        verify(counts, key, assignment)
        assignments.append(assignment)
    root = Path(root)
    configuration = dict(configuration or {})
    for key, value in seedplan.configuration().items():
        if configuration.setdefault(key, value) != value:
            raise FoldGenerationError("the configuration's seed fields differ from the checked seed plan")
    record = integrity.record(
        excerpts,
        history_sha256=integrity.file_sha256(root / "development" / "annotations.jsonl"),
        folded_state_sha256=integrity.sha256_bytes(integrity.canonical_json(folded)),
        development_manifest_sha256=integrity.file_sha256(root / "manifests" / "development.json"),
        private_inputs={name: integrity.file_sha256(root / rel) for name, rel in (
            ("registry_sha256", "manifests/split-registry.json"),
            ("fingerprints_file_sha256", "fingerprints/manifests.json"),
            ("validation_manifest_sha256", "manifests/validation.json"),
            ("final_test_manifest_sha256", "manifests/final-test.json")) if (root / rel).is_file()},
        environment=environment or {}, configuration=configuration,
    )
    integrity.write_artifact(out, "computational", "outer_order.json", order)
    fold_hashes = []
    for o, assignment in enumerate(assignments):
        data = folds.serialise(assignment)
        integrity.write_artifact(out, "computational", f"folds/{o:02d}.json", data)
        fold_hashes.append(integrity.sha256_bytes(data))
    integrity.write_artifact(out, "computational", "group_counts.json", {k: list(v) for k, v in counts.items()})
    integrity.write_artifact(out, "computational", "summary.json",
                             [a.summary | {"outer": o} for o, a in enumerate(assignments)])
    public = integrity.public_digest(record)
    integrity.write_artifact(out, "provenance", "integrity.json", record)
    integrity.write_artifact(out, "provenance", "public_digest.json", public)
    integrity.write_artifact(out, "provenance", "fold_index.json", {
        "inputs": {"development_manifest_sha256": record["R-C"]["development_manifest_sha256"],
                   "history_sha256": record["R-B"]["history_sha256"]},
        "public": {"development_manifest_sha256": public["development_manifest_sha256"]},
        "folds": {"sha256": fold_hashes, "outer_order_sha256": integrity.sha256_bytes(integrity.canonical_json(order))},
        "chain": {"integrity_record_sha256": integrity.sha256_bytes(integrity.canonical_json(record)),
                  "public_digest_sha256": integrity.sha256_bytes(integrity.canonical_json(public))},
    })
    integrity.write_artifact(out, "runlog", "log.txt", f"folds written: {len(assignments)}\n".encode())
    return record
