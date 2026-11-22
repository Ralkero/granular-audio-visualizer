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


ACOUSTID_URL = "https://api.acoustid.org/v2/lookup"
ELIGIBLE = ("unmatched", "ambiguous")


def _post(url, fields):
    """POST form fields with the system curl; returns the decoded JSON answer."""
    command = ["curl", "-sS", "--max-time", "30", "-X", "POST", url]
    for key, value in fields.items():
        command += ["--data-urlencode", f"{key}={value}"]
    done = subprocess.run(
        command, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
    )
    if done.returncode:
        raise RuntimeError(done.stderr.decode("utf-8", "replace").strip()[:200])
    return json.loads(done.stdout.decode("utf-8"))


def read_lookup(answer, duration_s):
    """Reduce an AcoustID answer to (status, candidates). ``matched`` needs one result
    clearly ahead that names exactly one recording of fitting length; anything else that
    has candidates is ``ambiguous`` and keeps them all."""
    candidates = []
    for result in answer.get("results", []):
        for recording in result.get("recordings", []) or []:
            candidates.append(
                {
                    "acoustid": result.get("id", ""),
                    "score": round(float(result.get("score", 0)), 3),
                    "recording_mbid": recording.get("id", ""),
                    "title": recording.get("title", ""),
                    "artists": ", ".join(a.get("name", "") for a in recording.get("artists", [])),
                    "length_s": recording.get("duration"),
                }
            )
    candidates.sort(key=lambda c: c["score"], reverse=True)
    if not candidates:
        return "unmatched", []
    strong = [c for c in candidates if c["score"] >= 0.9]
    fitting = [
        c for c in strong if c["length_s"] is None or abs(c["length_s"] - duration_s) <= 5
    ]
    if len(strong) == len(fitting) and len({c["recording_mbid"] for c in fitting}) == 1:
        return "matched", candidates
    return "ambiguous", candidates


def lookup(
    inventory, workdir, executable, key, approved=False, post=_post, interval=0.4, log=print
):
    """Look up recordings that MusicBrainz could not identify.

    Sends, per recording, its Chromaprint fingerprint, its duration and the application key:
    nothing else. Never submits. Only recordings whose identity is ``unmatched`` or
    ``ambiguous`` are sent, each once (answers are cached). Results go to
    ``acoustid.jsonl``; nothing is assigned here.
    """
    import time

    if not approved:
        raise PermissionError(
            "AcoustID lookups send an audio-derived fingerprint and the duration to "
            "acoustid.org; the owner has not approved that"
        )
    if not key:
        raise ValueError(
            "An AcoustID application key is needed (https://acoustid.org/new-application)"
        )
    workdir = Path(workdir)
    wanted = {
        e["recording_id"]
        for e in read_jsonl(workdir / "musicbrainz" / "identity.jsonl")
        if e["status"] in ELIGIBLE
    }
    rows = [r for r in read_inventory(inventory) if recording_id(r["sha256"]) in wanted]
    cache = workdir / "cache" / "acoustid"
    cache.mkdir(parents=True, exist_ok=True)
    executable = Path(executable).resolve()
    out, sent = [], 0
    for row in rows:
        rid = recording_id(row["sha256"])
        file = cache / f"{rid}.json"
        if file.is_file():
            record = json.loads(file.read_text(encoding="utf-8"))
        else:
            done = subprocess.run(
                [str(executable), "-json", str(row["file_path"])],
                capture_output=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            if done.returncode:
                continue
            found = json.loads(done.stdout.decode("utf-8"))
            time.sleep(interval)  # the service allows three requests a second
            answer = post(
                ACOUSTID_URL,
                {
                    "client": key,
                    "duration": str(round(found["duration"])),
                    "fingerprint": found["fingerprint"],
                    "meta": "recordings",
                },
            )
            sent += 1
            if answer.get("status") != "ok":
                raise RuntimeError(f"AcoustID refused the lookup: {answer.get('error')}")
            record = {"duration_s": found["duration"], "answer": answer}
            file.write_text(json.dumps(record), encoding="utf-8")
            if sent % 50 == 0:
                log(f"{sent} lookups sent", flush=True)
        status, candidates = read_lookup(record["answer"], record["duration_s"])
        out.append({"recording_id": rid, "status": status, "candidates": candidates[:8]})
    write_jsonl(workdir / "acoustid.jsonl", out)
    counts = {}
    for item in out:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    return {"eligible": len(rows), "sent_this_run": sent, "results": counts}
