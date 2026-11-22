"""Chromaprint fingerprints, computed locally with the official ``fpcalc`` utility.

Nothing here talks to a network. The fingerprints serve two purposes:

* finding recordings in the collection that are the same audio under different names
  (duplicates and reissues must not sit on both sides of a train / test split);
* being ready for an AcoustID lookup, which sends a fingerprint and a duration to
  acoustid.org and therefore runs only after the owner has approved it (``lookup`` refuses
  without ``approved=True``; it never submits anything).
"""

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from . import read_inventory, read_jsonl, recording_id, write_jsonl


def fpcalc(executable, path, length_s=120):
    """{"duration": s, "fingerprint": [uint32, ...]} of the first ``length_s`` seconds."""
    done = subprocess.run(
        [str(executable), "-raw", "-json", "-length", str(length_s), str(path)],
        capture_output=True,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip()[:200])
    return json.loads(done.stdout.decode("utf-8"))


def compute(inventory, workdir, executable, threads=6, limit=None, log=print):
    """Fingerprint every recording not fingerprinted yet. Returns counts."""
    target = Path(workdir) / "fingerprints.jsonl"
    known = {row["recording_id"]: row for row in read_jsonl(target)}
    rows = read_inventory(inventory)[: limit or None]
    todo = [r for r in rows if not known.get(recording_id(r["sha256"]), {}).get("fingerprint")]
    executable = Path(executable).resolve()

    def one(row):
        rid = recording_id(row["sha256"])
        try:
            found = fpcalc(executable, row["file_path"])
            return {
                "recording_id": rid,
                "duration_s": found["duration"],
                "fingerprint": found["fingerprint"],
                "error": "",
            }
        except Exception as error:
            return {"recording_id": rid, "duration_s": None, "fingerprint": [], "error": str(error)}

    with ThreadPoolExecutor(threads) as pool:
        for count, result in enumerate(pool.map(one, todo), 1):
            known[result["recording_id"]] = result
            if count % 200 == 0:
                log(f"{count}/{len(todo)} fingerprints", flush=True)
    write_jsonl(target, [known[k] for k in sorted(known)])
    return {
        "recordings": len(rows),
        "new": len(todo),
        "failed": sum(1 for r in known.values() if r["error"]),
    }


def similarity(a, b, max_shift=40):
    """Share of matching bits between two raw fingerprints at their best alignment (0 to 1).
    Unrelated music gives about 0.5; the same audio gives above 0.9."""
    a, b = np.asarray(a, dtype=np.uint32), np.asarray(b, dtype=np.uint32)
    best = 0.0
    for shift in range(-max_shift, max_shift + 1):
        x, y = (a[shift:], b) if shift >= 0 else (a, b[-shift:])
        size = min(len(x), len(y))
        if size < 100:
            continue
        differing = np.bitwise_xor(x[:size], y[:size])
        bits = np.unpackbits(differing.view(np.uint8)).sum()
        best = max(best, 1 - bits / (32 * size))
    return best


def same_audio_groups(workdir, threshold=0.85, duration_tolerance_s=8.0):
    """Groups of recordings whose fingerprints match: [[recording ids], ...]."""
    rows = [r for r in read_jsonl(Path(workdir) / "fingerprints.jsonl") if r["fingerprint"]]
    rows.sort(key=lambda r: r["duration_s"])
    parent = {r["recording_id"]: r["recording_id"] for r in rows}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    pairs = []
    for i, first in enumerate(rows):
        for second in rows[i + 1 :]:
            if second["duration_s"] - first["duration_s"] > duration_tolerance_s:
                break
            score = similarity(first["fingerprint"], second["fingerprint"])
            if score >= threshold:
                pairs.append((first["recording_id"], second["recording_id"], round(score, 3)))
                parent[find(first["recording_id"])] = find(second["recording_id"])
    groups = {}
    for name in parent:
        groups.setdefault(find(name), []).append(name)
    return [sorted(g) for g in groups.values() if len(g) > 1], pairs


def lookup(*_args, approved=False, **_kwargs):
    """AcoustID lookup: deliberately not implemented until the owner approves sending
    fingerprints. See docs/evidence/EVIDENCE_ENRICHMENT.md."""
    if not approved:
        raise PermissionError(
            "AcoustID lookups send an audio-derived fingerprint and the duration to "
            "acoustid.org; the owner has not approved that"
        )
    raise NotImplementedError("AcoustID lookup is not implemented in this version")
