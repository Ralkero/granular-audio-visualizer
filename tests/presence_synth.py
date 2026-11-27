"""A synthetic presence project for A5.2 Step 1 tests: generated audio, manifests, registry,
fingerprints and annotation histories written with the real A5.1 project APIs, entirely
under a temporary folder. Nothing here reads project data.

Each spec item is (group, partition, rate, confirmed intervals); excerpts are 10 s windows
starting at 7 s in a 20 s generated stereo recording with a 0.4 s digital-silence gap.
"""

import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from gmv.character.research.presence import project
from gmv.character.research.presence.store import Store

EXCERPT_START, EXCERPT_S, RECORDING_S = 7.0, 10.0, 20.0


def write_audio(path, rate, seed):
    rng = np.random.default_rng(seed)
    n = round(RECORDING_S * rate)
    left = rng.normal(0, 0.05, n)
    audio = np.stack((left, 0.7 * left + rng.normal(0, 0.01, n)), axis=1)
    gap = round((EXCERPT_START + 5.0) * rate)
    audio[gap : gap + round(0.4 * rate)] = 0
    sf.write(path, audio, rate, subtype="PCM_16")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def default_spec(groups=8, rate=44100, low_rate_group=True):
    spec = []
    for g in range(groups):
        spec.append((f"group-{g:02d}", "development", rate, [(1.0 + 0.2 * g, 5.0 + 0.1 * g)]))
        spec.append((f"group-{g:02d}", "development", rate, []))
    if low_rate_group:
        spec.append(("group-low", "development", 22050, [(2.0, 6.0)]))
        spec.append(("group-low", "development", 22050, []))
    spec.append(("group-val", "validation", rate, [(3.0, 4.0)]))
    return spec


def build(root, spec=None, selection=lambda i: f"negative candidate, drums_only: synthetic {i}",
          source=lambda i: f"new-{i}"):
    """Write the project; returns {"root", "excerpts", "expected"} (expected aggregates)."""
    root = project.layout(Path(root))
    spec = default_spec() if spec is None else spec
    audio_dir = root.parent / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    by_partition, entries, annotations = {"development": [], "validation": []}, {}, {}
    for i, (group, partition, rate, confirmed) in enumerate(spec):
        path = audio_dir / f"rec-{i:03d}.wav"
        sha = write_audio(path, rate, seed=i)
        excerpt = {"id": project.excerpt_id(sha, EXCERPT_START), "source": source(i), "recording": sha,
                   "path": str(path), "size": path.stat().st_size, "duration_s": RECORDING_S,
                   "sample_rate": rate, "channels": 2, "start_s": EXCERPT_START,
                   "end_s": EXCERPT_START + EXCERPT_S, "group": group, "selection": selection(i)}
        by_partition[partition].append(excerpt)
        entries.setdefault(group, {"partition": partition, "nodes": []})["nodes"].append(f"rec:{sha}")
        annotations[excerpt["id"]] = (partition, confirmed)
    for partition, excerpts in by_partition.items():
        project.write_manifest(root, partition, excerpts)
    project.write_registry(root, {"format": "gmv-vocal-presence-registry", "version": 1, "entries": entries})
    for partition, excerpts in by_partition.items():
        store = Store(project.partition_folder(root, partition), project.durations({"excerpts": excerpts}))
        for e in excerpts:
            for start, end in annotations[e["id"]][1]:
                store.add_interval(e["id"], "confirmed", start, end)
            store.set_review(e["id"], "complete")
    development = [s for s in spec if s[1] == "development"]
    low = [s for s in development if s[2] < 23700]
    paired = [s for s in development if s[2] >= 23700]
    voiced = sum(sum(min(e, EXCERPT_S) - s for s, e in c) for *_x, c in paired)
    expected = {
        "excerpts": len(development), "groups": len({s[0] for s in development}),
        "low_rate_excerpts": len(low), "paired_excerpts": len(paired), "paired_groups": len({s[0] for s in paired}),
        "voiced_groups": len({s[0] for s in paired if s[3]}), "novoice_groups": len({s[0] for s in paired}),
        "voiced_s": round(voiced), "novoice_s": round(len(paired) * EXCERPT_S - voiced),
    }
    return {"root": root, "expected": expected}
