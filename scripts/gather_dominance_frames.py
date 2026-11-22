"""Gather what the dominance coefficients are fitted on.

    python scripts/gather_dominance_frames.py MOISESDB_FOLDER 60 frames.npz [progress.log]

For the first N song folders (sorted): the full mix of a 45 s excerpt (35 s to 80 s) goes
through the analysis pass with all four analyzers and the activity model, and through the
guitar recognition model. Saved per analysis frame, from 10 s into the excerpt: each
instrument's level, attack, energy, pulse and presence (the activity model's 0 or 1; for
guitar the recognition state as 1, 0.65 or 0), the mix's seven band energies (default
edges), and how loud each instrument's stem really is over 0.4 s. A song is left out when
fewer than three of its stems cover the excerpt. Run from the repository root; it reads
the models from .model-cache. Audio is only read.

This is the data ``fit_dominance.py`` fits ``gmv.visualizer.dominance.MODELS`` on. It
builds its inputs directly from the analyzer rows, not through the visual state; the
evaluation (``evaluate_dominance.py``) does go through the visual state.
"""

import sys
from pathlib import Path

import numpy as np

from diagnose_responsiveness_stems import read_stem

from gmv.evaluation import responsiveness as r
from gmv.evaluation.detect import run_recognition
from gmv.models import AnalysisConfig
from gmv.recognition import load_backend

KEYS = ("vocals", "guitar", "bass", "drums")
STATE = {"detected": 1.0, "candidate": 0.65, "absent": 0.0}


def stem_db(stem, rate, times, window=0.4):
    mono = stem.mean(axis=1)
    power = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    half = round(window * rate / 2)
    centre = np.clip((times * rate).astype(int), 0, len(mono) - 1)
    a, b = np.clip(centre - half, 0, len(mono)), np.clip(centre + half, 0, len(mono))
    return 10 * np.log10((power[b] - power[a]) / np.maximum(1, b - a) + 1e-12)


def main():
    root, songs, out = Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3])
    log = open(sys.argv[4], "w", encoding="utf-8", buffering=1) if len(sys.argv) > 4 else None
    backend = load_backend(".model-cache/guitar_mn10_v1.json")
    config = AnalysisConfig(
        instrument_analyzers=KEYS, spectrum_bands=0, activity_model_path=".model-cache/activity_v2.npz"
    )
    folders = [f for f in sorted(root.iterdir()) if f.is_dir()][:songs]
    if log:
        log.write(f"task Prominence data: {len(folders)} songs with stems\n")
    kept = []
    for number, folder in enumerate(folders, 1):
        if log:
            log.write(f"step {number}/{len(folders)} song {number}\n")
        parts = {}
        for stem in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            audio, rate = read_stem(folder / stem, 35.0, 45.0)
            if audio is not None and len(audio) >= rate * 40:
                parts[stem] = audio
        if len(parts) < 3:
            continue
        size = min(len(a) for a in parts.values())
        parts = {k: v[:size] for k, v in parts.items()}
        mix = sum(parts.values())
        frames = []
        rows, _ = r.trace(mix, rate, config, frames=frames)
        estimates, _ = run_recognition(mix, rate, backend, config)
        ends = np.array([e.end for e in estimates])
        states = np.array([STATE[dict(e.states).get("guitar", "absent")] for e in estimates])
        times = np.arange(10.0, size / rate - 1.0, 0.02)
        song = {"times": times, "song": number}
        for key in KEYS:
            table = rows[key]
            at = np.clip(np.searchsorted(table[:, 0], times), 0, len(table) - 1)
            for name in ("level", "attack", "energy", "pulse", "present", "flux"):
                song[f"{key}_{name}"] = table[at, r.F[name]]
            song[f"{key}_true_db"] = (
                stem_db(parts[key], rate, times) if key in parts else np.full(len(times), -120.0)
            )
        which = np.clip(np.searchsorted(ends, times), 0, len(ends) - 1)
        song["guitar_present"] = states[which]
        song["mix_db"] = stem_db(mix, rate, times)
        stamps = np.array([f.timestamp for f in frames])
        bands = np.array([f.band_energy for f in frames])
        near = np.clip(np.searchsorted(stamps, times), 0, len(stamps) - 1)
        for index in range(7):
            song[f"band_{index}"] = bands[near, index]
        song["density"] = np.array([f.rhythmic_density for f in frames])[near]
        kept.append(song)
    merged = {k: np.concatenate([np.full(len(s["times"]), s[k]) if k == "song" else s[k] for s in kept]) for k in kept[0]}
    np.savez_compressed(out, **merged)
    if log:
        log.write(f"line saved {len(merged['times'])} frames of {len(kept)} songs\ndone\n")
    print("saved", len(merged["times"]), "frames,", len(kept), "songs")


if __name__ == "__main__":
    main()
