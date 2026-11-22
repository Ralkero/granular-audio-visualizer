"""Does the dominant-instrument choice name the instrument that really is the loudest?

    python scripts/evaluate_dominance.py --moisesdb C:/Datasets/moisesdb_v0.1 --first 61 \
        --songs 40 --activity-model .model-cache/activity_v2.npz \
        --guitar-model .model-cache/guitar_mn10_v1.json [--json out.json] [--log progress.log]

Full mixes of multitrack songs go through the analysis pass and the guitar recognition
model as in the application; their output is fed to ``gmv.visualizer.dominance.Dominance``
exactly as the visual state feeds it. The reference is each song's own stems: the loudest
of vocals, guitar, bass and drums over 0.4 s, counted only where it leads the next by 3 dB
or more ("clear"). Songs from --first on were not used to fit the coefficients (the first
60 were). Reported: how often the leader is that stem, per instrument, how often nobody
leads, and how often the lead changes. Audio is only read.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from diagnose_responsiveness_stems import read_stem
from gmv.evaluation.detect import run_layers, run_recognition
from gmv.models import AnalysisConfig
from gmv.recognition import load_backend
from gmv.visualizer.dominance import KEYS, Dominance

STATE = {"detected": 1.0, "candidate": 0.65, "absent": 0.0}


def stem_db(stem, rate, times, window=0.4):
    mono = stem.mean(axis=1)
    power = np.concatenate(([0.0], np.cumsum(mono.astype(np.float64) ** 2)))
    half = round(window * rate / 2)
    centre = np.clip((times * rate).astype(int), 0, len(mono) - 1)
    a, b = np.clip(centre - half, 0, len(mono)), np.clip(centre + half, 0, len(mono))
    return 10 * np.log10((power[b] - power[a]) / np.maximum(1, b - a) + 1e-12)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moisesdb", required=True)
    parser.add_argument("--first", type=int, default=61)
    parser.add_argument("--songs", type=int, default=40)
    parser.add_argument("--activity-model", required=True)
    parser.add_argument("--guitar-model", required=True)
    parser.add_argument("--json")
    parser.add_argument("--log")
    args = parser.parse_args()
    log = open(args.log, "w", encoding="utf-8", buffering=1) if args.log else None
    backend = load_backend(args.guitar_model)
    config = AnalysisConfig(
        instrument_analyzers=KEYS, spectrum_bands=0, activity_model_path=args.activity_model
    )
    folders = [f for f in sorted(Path(args.moisesdb).iterdir()) if f.is_dir()]
    folders = folders[args.first - 1 : args.first - 1 + args.songs]
    if log:
        log.write(f"task Dominant instrument against stems: {len(folders)} songs not used for the fit\n")
    right = dict.fromkeys(KEYS, 0)
    cases = dict.fromkeys(KEYS, 0)
    led = dict.fromkeys((*KEYS, "nobody"), 0)
    total = clear_frames = changes = seconds = songs = 0
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
        songs += 1
        size = min(len(a) for a in parts.values())
        mix = sum(v[:size] for v in parts.values())
        frames, messages, _ = run_layers(mix, rate, config)
        estimates, _ = run_recognition(mix, rate, backend, config)
        ends = np.array([e.end for e in estimates])
        guitar = np.array([STATE[dict(e.states).get("guitar", "absent")] for e in estimates])
        messages.sort(key=lambda m: m.timestamp)
        dominance, fed, leaders, times = Dominance(), 0, [], []
        for frame in frames:
            while fed < len(messages) and messages[fed].timestamp <= frame.timestamp:
                m = messages[fed]
                fast = m.energy is not None
                dominance.row(m.instrument, m.timestamp, m.energy if fast else m.level, m.pulse if fast else m.attack)
                fed += 1
            weights = {key: float(key in frame.instruments_active) for key in ("vocals", "bass", "drums")}
            at = min(len(ends) - 1, int(np.searchsorted(ends, frame.timestamp)))
            weights["guitar"] = float(guitar[at]) if len(ends) else 0.0
            leaders.append(dominance.step(frame.timestamp, frame.band_energy, frame.silence, weights))
            times.append(frame.timestamp)
        times = np.asarray(times)
        keep = (times >= 10.0) & (times <= size / rate - 1.0)  # after the recognition warm-up
        true = np.stack(
            [stem_db(parts[k][:size], rate, times) if k in parts else np.full(len(times), -120.0) for k in KEYS],
            axis=1,
        )
        order = np.sort(true, axis=1)
        clear = keep & (order[:, -1] - order[:, -2] >= 3.0) & (order[:, -1] > -45.0)
        truth = true.argmax(axis=1)
        previous = None
        for index in np.flatnonzero(keep):
            leader = leaders[index]
            total += 1
            led[leader or "nobody"] += 1
            changes += previous is not None and leader != previous and leader is not None
            previous = leader if leader is not None else previous
            if clear[index] and leader is not None:
                clear_frames += 1
                cases[KEYS[truth[index]]] += 1
                right[KEYS[truth[index]]] += leader == KEYS[truth[index]]
        seconds += keep.sum() * 0.02
    result = {
        "songs": songs,
        "minutes": round(seconds / 60, 1),
        "clear_cases_with_a_leader": clear_frames,
        "leader_is_the_loudest_stem": round(sum(right.values()) / max(1, clear_frames), 3),
        "per_instrument": {k: round(right[k] / cases[k], 3) if cases[k] else None for k in KEYS},
        "share_of_clear_cases": {k: round(cases[k] / max(1, clear_frames), 3) for k in KEYS},
        "share_of_time_leading": {k: round(v / max(1, total), 3) for k, v in led.items()},
        "changes_of_leader_per_minute": round(changes / max(1e-9, seconds / 60), 1),
    }
    print(json.dumps(result, indent=1))
    if args.json:
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
    if log:
        log.write("line " + json.dumps(result)[:280] + "\ndone\n")


if __name__ == "__main__":
    main()
