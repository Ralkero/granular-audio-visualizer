"""How soon, and whether, the instrument analyzers answer real playing: multitrack stems.

    python scripts/diagnose_responsiveness_stems.py --moisesdb C:/Datasets/moisesdb_v0.1 \
        --songs 20 --activity-model .model-cache/activity_v2.npz --json stems.json

For every song an excerpt of the full mix goes through the analysis pass with all four
analyzers and the frame-wise activity model, exactly as in the engine. The reference is
taken from each instrument's own stem:

    onsets    where the isolated stem's level rises by 6 dB or more within 30 ms
              (measured every 5 ms; an energy detector on a clean stem, not human labels)
    sounding  where the stem's level is within 40 dB of its peak in the excerpt

and the mix's analysis is reduced, per instrument, to

    attack found      share of stem onsets with a reported attack within 70 ms (mix, and
                      the stem analysed alone, to tell "cannot detect" from "lost in a mix")
    known             how much audio after the onset existed when it was reported
    level moved       share of stem onsets at which the continuous level rises by 0.1
    dip               how far the level falls back between consecutive onsets
    shown             share of stem onsets at which the activity model reports the
                      instrument (drums, bass, vocals; guitar's presence comes from the
                      recognition model, which is not run here)
    suppressed        share of *reported* attacks at stem onsets that the activity model
                      hides
    entrance          after the stem has been silent 1.5 s: seconds until the activity
                      model reports the instrument

Audio is only read. Nothing is written except the optional log and JSON.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d

from gmv.evaluation import responsiveness as r
from gmv.models import AnalysisConfig

TARGETS = ("bass", "drums", "vocals", "guitar")
BAND = {"bass": (35, 300), "drums": (40, 14000), "vocals": (150, 4000), "guitar": (200, 5000)}


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


def envelope(stem, rate, hop_s=0.005, window_s=0.01):
    """Level of the stem in dB every ``hop_s`` seconds, and those times."""
    mono = stem.mean(axis=1)
    hop, window = round(hop_s * rate), round(window_s * rate)
    count = (len(mono) - window) // hop
    index = np.arange(count)[:, None] * hop + np.arange(window)[None, :]
    level = 10 * np.log10(np.mean(mono[index] ** 2, axis=1) + 1e-12)
    return level, (np.arange(count) * hop + window / 2) / rate


def reference(stem, rate):
    """(onset times, times, whether the stem is sounding at each time)."""
    level, times = envelope(stem, rate)
    peak = float(level.max())
    sounding = maximum_filter1d(level, 21) > peak - 40
    before = np.concatenate((np.full(6, level[0]), level[:-6]))  # 30 ms earlier
    rise = level - np.maximum(before, peak - 45)
    onsets, last = [], -1.0
    for index in np.flatnonzero((rise >= 6) & (level > peak - 30)):
        if times[index] - last >= 0.08:
            onsets.append(float(times[index]) - 0.005)
            last = times[index]
    return onsets, times, sounding


def entrances(times, sounding, quiet_s=1.5):
    found, silent_since = [], times[0]
    for index in range(1, len(times)):
        if sounding[index] and not sounding[index - 1] and times[index] - silent_since >= quiet_s:
            found.append(float(times[index]))
        if sounding[index]:
            silent_since = times[index]
        elif sounding[index - 1]:
            silent_since = times[index]
    return found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--moisesdb", required=True)
    parser.add_argument("--songs", type=int, default=20)
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
    folders = [f for f in sorted(Path(args.moisesdb).iterdir()) if f.is_dir()][: args.songs]
    if log:
        log.write(f"task Responsiveness on stems: {len(folders)} songs, {args.seconds:.0f} s each\n")
    kept = {name: {"mix": [], "alone": [], "level": [], "dips": [], "known": [], "placed": [],
                   "shown": [], "suppressed": [], "entrance": [], "onsets": 0, "songs": 0,
                   "flux": [], "per_second": []} for name in TARGETS}
    for number, folder in enumerate(folders, 1):
        if log:
            log.write(f"step {number}/{len(folders)} song {number}\n")
        parts = {}
        for stem in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            audio, rate = read_stem(folder / stem, args.start, args.seconds)
            if audio is not None and len(audio) >= rate * args.seconds * 0.9:
                parts[stem] = audio
        if len(parts) < 2:
            continue
        size = min(len(a) for a in parts.values())
        parts = {k: v[:size] for k, v in parts.items()}
        mix = sum(parts.values())
        rows, _ = r.trace(mix, rate, config)
        for name in TARGETS:
            stem = parts.get(name)
            if stem is None or float(np.sqrt(np.mean(stem**2))) < 10 ** (-45 / 20):
                continue
            onsets, times, sounding = reference(stem, rate)
            onsets = [t for t in onsets if 1.0 < t < args.seconds - 1.0]
            if len(onsets) < 4:
                continue
            keep = kept[name]
            keep["songs"] += 1
            keep["onsets"] += len(onsets)
            keep["per_second"].append(len(onsets) / (args.seconds - 2))
            found = r.match(rows[name], onsets)
            keep["mix"].append(found["found"] / len(onsets))
            quick = r.fast(rows[name])
            if len(quick):
                pulses = r.match(quick, onsets)
                keep.setdefault("pulse", []).append(pulses["found"] / len(onsets))
                keep.setdefault("pulse_right", []).append(pulses["found"] / max(1, pulses["found"] + pulses["extra"]))
                keep.setdefault("attack_right", []).append(found["found"] / max(1, found["found"] + found["extra"]))
                if pulses["known_s"]["p50"] is not None:
                    keep.setdefault("pulse_known", []).append(pulses["known_s"]["p50"])
                    keep.setdefault("pulse_placed", []).append(pulses["placed_s"]["p50"])
                keep.setdefault("energy_dip", []).append(r.dips(quick, onsets)["p50"] or 0.0)
            alone, _ = r.trace(stem, rate, AnalysisConfig(instrument_analyzers=(name,), spectrum_bands=0))
            keep["alone"].append(r.match(alone[name], onsets)["found"] / len(onsets))
            keep["level"].append(r.movement(rows[name], onsets)["moved"] / len(onsets))
            keep["flux"].append(r.movement(rows[name], onsets, "flux", rise=0.15)["moved"] / len(onsets))
            dip = r.dips(rows[name], onsets)
            if dip["p50"] is not None:
                keep["dips"].append(dip["p50"])
            if found["known_s"]["p50"] is not None:
                keep["known"].append(found["known_s"]["p50"])
                keep["placed"].append(found["placed_s"]["p50"])
            table = rows[name]
            present = table[:, r.F["present"]]
            if np.isfinite(present).any():
                at = np.searchsorted(table[:, 0], onsets).clip(0, len(table) - 1)
                keep["shown"].append(float(np.mean(present[at] == 1.0)))
                attacks = table[table[:, r.F["attack"]] > 0]
                near = [a for a in attacks if np.min(np.abs(np.asarray(onsets) - a[0])) <= 0.07]
                if near:
                    keep["suppressed"].append(float(np.mean([a[r.F["present"]] == 0.0 for a in near])))
                for enter in entrances(times, sounding):
                    later = table[(table[:, 0] >= enter) & (present == 1.0)]
                    if len(later) and later[0, 0] - enter < 5.0:
                        keep["entrance"].append(float(later[0, r.F["known"]] - enter))
    summary = {}
    for name, keep in kept.items():
        def median(key, keep=keep):
            return round(float(np.median(keep[key])), 3) if keep.get(key) else None
        summary[name] = {
            "songs": keep["songs"],
            "stem_onsets": keep["onsets"],
            "stem_onsets_per_second": median("per_second"),
            "attack_found_in_mix": median("mix"),
            "attack_found_on_stem_alone": median("alone"),
            "attack_placed_s": median("placed"),
            "attack_known_s": median("known"),
            "level_moved": median("level"),
            "flux_moved": median("flux"),
            "level_dip": median("dips"),
            "shown_by_activity_model": median("shown"),
            "reported_attacks_hidden_by_model": median("suppressed"),
            "entrance_s": r.spread(keep["entrance"]),
        }
        if keep.get("pulse"):
            summary[name]["fast_path"] = {
                "pulse_found_in_mix": median("pulse"),
                "pulses_that_are_stem_onsets": median("pulse_right"),
                "attacks_that_are_stem_onsets": median("attack_right"),
                "pulse_placed_s": median("pulse_placed"),
                "pulse_known_s": median("pulse_known"),
                "energy_dip": median("energy_dip"),
            }
        print(f"RESULT {name} {json.dumps(summary[name])}", flush=True)
        if log:
            log.write(f"line {name}: {json.dumps(summary[name])[:260]}\n")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    if log:
        log.write("done\n")


if __name__ == "__main__":
    main()
