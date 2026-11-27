"""The A5.2 spy-test harness: run a target under recording, classify every access to the
synthetic world, and produce a sanitised report (category counts and evidence hashes only).

Raw evidence (access logs, the target's output, its exception) stays in a private folder
inside pytest's temporary directory (mode 0700) and is never printed. Assertions use the
classified counts and boolean canary scans, with fixed messages.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

BOOTSTRAP = Path(__file__).with_name("privacy_bootstrap.py")
ALLOWED_STEP3 = {"development-manifest", "development-history", "tier1:validation-manifest",
                 "tier1:registry", "tier1:final-test-manifest", "tier1:fingerprints", "directory",
                 "paired-audio"}
QUOTED_PATH = re.compile(r'"(/[^"]*)"')


def strace_available():
    return shutil.which("strace") is not None


def categorise(path, world, paired_audio):
    p = os.path.realpath(path)
    world = os.path.realpath(world)
    if p != world and not p.startswith(world + os.sep):
        return None
    rel = Path(p).relative_to(world).as_posix()
    files = {
        "presence/manifests/development.json": "development-manifest",
        "presence/manifests/validation.json": "tier1:validation-manifest",
        "presence/manifests/split-registry.json": "tier1:registry",
        "presence/manifests/final-test.json": "tier1:final-test-manifest",
        "presence/fingerprints/manifests.json": "tier1:fingerprints",
        "presence/development/annotations.jsonl": "development-history",
    }
    if rel in files:
        return files[rel]
    if rel in (".", "presence", "presence/manifests", "presence/fingerprints", "presence/development", "audio"):
        return "directory"
    for prefix, category in (("presence/validation", "tier3:validation"), ("presence/final-test", "tier3:final-test"),
                             ("presence/identity", "tier3:identity"), ("presence/backups", "tier3:backups"),
                             ("presence/selection", "tier3:selection")):
        if rel == prefix or rel.startswith(prefix + "/"):
            return category
    if rel.endswith("session.json"):
        return "tier3:session"
    if rel.startswith("audio/"):
        return "paired-audio" if p in paired_audio else "tier3:audio"
    return "unclassified"


def paths_in(evidence):
    out = []
    python_log = evidence / "python_access.log"
    if python_log.is_file():
        for line in python_log.read_text(encoding="utf-8", errors="replace").splitlines():
            _event, _tab, args = line.partition("\t")
            try:
                out += [a for a in json.loads(args) if a.startswith("/")]
            except ValueError:
                continue
    strace_log = evidence / "strace.log"
    if strace_log.is_file():
        out += QUOTED_PATH.findall(strace_log.read_text(encoding="utf-8", errors="replace"))
    return out


def run_target(world, evidence, target, arguments, paired_audio, timeout=600):
    """Run ``privacy_targets.<target>`` under recording; returns the sanitised report."""
    evidence.mkdir(parents=True, mode=0o700)
    command = [sys.executable, str(BOOTSTRAP), str(evidence), f"privacy_targets:{target}", json.dumps(arguments)]
    layers = ["audit-hook", "stat-probes"]
    if strace_available():
        command = ["strace", "-f", "-qq", "-y", "-e", "trace=%file,%process,getdents64",
                   "-o", str(evidence / "strace.log"), *command]
        layers.insert(0, "strace")
    done = subprocess.run(command, cwd=evidence, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                          timeout=timeout, check=False)
    counts = {}
    for path in paths_in(evidence):
        category = categorise(path, world, paired_audio)
        if category is not None:
            counts[category] = counts.get(category, 0) + 1
    exception = evidence / "exception.txt"
    report = {
        "layers": layers,
        "exit": done.returncode,
        "exception_type": exception.read_text(encoding="utf-8").splitlines()[0] if exception.is_file() else None,
        "counts": dict(sorted(counts.items())),
        "evidence_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in sorted(evidence.iterdir()) if p.is_file()},
    }
    (evidence.parent / f"{evidence.name}.report.json").write_text(json.dumps(report, indent=1))
    return report


def violations(report, allowed=ALLOWED_STEP3):
    return {c: n for c, n in report["counts"].items() if c not in allowed}


def canary_hits(canaries, folder_or_files):
    """How many of ``canaries`` occur in the files (read as bytes); never returns which."""
    files = []
    for item in folder_or_files:
        item = Path(item)
        files += [p for p in item.rglob("*") if p.is_file()] if item.is_dir() else [item]
    found = set()
    for path in files:
        data = path.read_bytes()
        found.update(c for c in canaries if c.encode() in data)
    return len(found)
