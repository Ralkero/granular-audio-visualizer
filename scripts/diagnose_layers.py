"""Stem diagnostic for the drum, bass and vocal analyzers.

    python scripts/diagnose_layers.py --moisesdb C:/Datasets/moisesdb_v0.1 --songs 30 \
        --activity-model .model-cache/activity_v2.npz --log layers.log

For every song and instrument the analyzers run, exactly as in the engine, on

    mix      the full mix
    without  the same mix with the instrument's stem taken out
    alone    the instrument's stem by itself
    with     the stem together with one competing stem (guitar; drums for bass)

and each run is reduced to what its layer would draw:

    drums    hit strength per minute (sum of attack strengths)
    bass     mean weight (level), and the share of frames with a note
    vocals   mean level, and the share of frames with a pitch

"raw" is the analyzer alone. "shown" multiplies each frame by the activity model's
judgement for that instrument (1 or 0), which is what the application displays. The
question asked of every analyzer: does its output fall when its instrument is removed?
The reference for "follows the instrument" is the analyzer's own output on the clean stem,
not human annotation. Nothing is written except the optional progress log; audio is only
read.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from gmv.evaluation.detect import run_layers
from gmv.models import AnalysisConfig

TARGETS = ("drums", "bass", "vocals")
COMPETITOR = {"drums": "guitar", "bass": "drums", "vocals": "guitar"}


def read_stem(folder, start, seconds):
    total, rate = None, None
    for path in sorted(Path(folder).glob("*.wav")):
        with sf.SoundFile(path) as file:
            rate = file.samplerate
            file.seek(min(file.frames, round(start * rate)))
            audio = file.read(round(seconds * rate), dtype="float32", always_2d=True)
        audio = np.repeat(audio, 2, axis=1) if audio.shape[1] == 1 else audio[:, :2]
        if total is None:
            total = audio
        else:
            size = min(len(total), len(audio))
            total = total[:size] + audio[:size]
    return total, rate


def analyse(audio, rate, config):
    """instrument -> array of rows (time, level, flux, attack, pitch, shown weight)."""
    frames, messages, times = run_layers(audio, rate, config)
    active = {round(f.timestamp, 3): set(f.instruments_active) for f in frames}
    stamps = np.array(sorted(active))
    out = {}
    for name in TARGETS:
        rows = []
        for m in (m for m in messages if m.instrument == name):
            nearest = stamps[min(len(stamps) - 1, np.searchsorted(stamps, m.timestamp))]
            shown = float(name in active[float(nearest)]) if config.activity_model_path else 1.0
            rows.append(
                (m.timestamp, m.level, m.flux, m.attack, np.nan if m.pitch_midi is None else m.pitch_midi, shown)
            )
        out[name] = np.asarray(rows).reshape(-1, 6)
    return out, times


def amount(name, rows, shown):
    """What the layer draws, as one number per run."""
    weight = rows[:, 5] if shown else 1.0
    if name == "drums":
        return float((rows[:, 3] * weight).sum() / (len(rows) * 0.02 / 60))
    return float((rows[:, 1] * weight).mean())


def match(found, reference, tolerance=0.05):
    hits, used = 0, 0
    for time in found:
        while used < len(reference) and reference[used] < time - tolerance:
            used += 1
        if used < len(reference) and abs(reference[used] - time) <= tolerance:
            hits += 1
            used += 1
    return hits


def similarity(name, mix, alone):
    """How closely the mix's reading follows the clean stem's."""
    size = min(len(mix), len(alone))
    mix, alone = mix[:size], alone[:size]
    if name == "drums":
        reference, found = alone[alone[:, 3] > 0][:, 0], mix[mix[:, 3] > 0][:, 0]
        hits = match(found, reference)
        return {"hits_found": hits / max(1, len(reference)), "hits_right": hits / max(1, len(found))}
    pitched = np.isfinite(alone[:, 4])
    both = pitched & np.isfinite(mix[:, 4])
    agree = np.abs(mix[both, 4] - alone[both, 4]) < 1.0
    level = np.corrcoef(mix[:, 1], alone[:, 1])[0, 1] if alone[:, 1].std() > 0 and mix[:, 1].std() > 0 else 0.0
    return {
        "stem_pitched": float(pitched.mean()),
        "pitched_in_mix": float(both.sum() / max(1, pitched.sum())),
        "same_note": float(agree.mean()) if both.any() else 0.0,
        "level_correlation": float(level),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moisesdb", required=True)
    parser.add_argument("--songs", type=int, default=30)
    parser.add_argument("--start", type=float, default=45.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--activity-model")
    parser.add_argument("--log")
    parser.add_argument("--json")
    args = parser.parse_args()
    config = AnalysisConfig(
        instrument_analyzers=TARGETS, spectrum_bands=0, activity_model_path=args.activity_model
    )
    log = open(args.log, "w", encoding="utf-8", buffering=1) if args.log else None

    def say(text):
        print(text, flush=True)
        if log:
            log.write(text + "\n")

    results = {name: [] for name in TARGETS}
    costs = []
    folders = [f for f in sorted(Path(args.moisesdb).iterdir()) if all((f / t).is_dir() for t in TARGETS)]
    folders = folders[: args.songs]
    say(f"songs {len(folders)}")
    for number, folder in enumerate(folders, 1):
        parts = {}
        for stem in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            audio, rate = read_stem(folder / stem, args.start, args.seconds)
            if audio is not None and len(audio) >= rate * args.seconds * 0.9:
                parts[stem] = audio
        if not parts or not all(name in parts for name in TARGETS):
            say(f"song {number}/{len(folders)} {folder.name[:8]} skipped: a stem is shorter than the excerpt")
            continue
        size = min(len(a) for a in parts.values())
        parts = {k: v[:size] for k, v in parts.items()}
        mix = sum(parts.values())
        full, times = analyse(mix, rate, config)
        costs.extend(times)
        for name in TARGETS:
            target = parts.get(name)
            if target is None or float(np.sqrt(np.mean(target**2))) < 10 ** (-40 / 20):
                continue  # silent in this excerpt
            runs = {"mix": full, "without": analyse(mix - target, rate, config)[0]}
            runs["alone"] = analyse(target, rate, config)[0]
            other = parts.get(COMPETITOR[name])
            if other is not None:
                runs["with"] = analyse(target + other, rate, config)[0]
            row = {"song": folder.name[:8]}
            for label, run in runs.items():
                row[f"raw_{label}"] = amount(name, run[name], False)
                row[f"shown_{label}"] = amount(name, run[name], True)
            row.update(similarity(name, full[name], runs["alone"][name]))
            results[name].append(row)
        say(f"song {number}/{len(folders)} {folder.name[:8]}")
    summary = {"analysis_ms_per_frame_all_three": {
        "p50": float(np.percentile(costs, 50)), "p95": float(np.percentile(costs, 95))}}
    for name, rows in results.items():
        keys = [k for k in rows[0] if k != "song"] if rows else []
        median = {k: round(float(np.median([r[k] for r in rows if k in r])), 3) for k in keys}
        for kind in ("raw", "shown"):
            if median.get(f"{kind}_mix"):
                median[f"{kind}_without_over_mix"] = round(median[f"{kind}_without"] / median[f"{kind}_mix"], 2)
        summary[name] = {"songs": len(rows), **median}
        say(f"RESULT {name} {json.dumps(summary[name])}")
    say(f"RESULT cost {json.dumps(summary['analysis_ms_per_frame_all_three'])}")
    if args.json:
        Path(args.json).write_text(json.dumps({"summary": summary, "songs": results}, indent=1), encoding="utf-8")
    say("done")


if __name__ == "__main__":
    main()
