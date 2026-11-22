"""Diagnostic for a fast instrument analyzer, on stems or on any audio file.

    python scripts/diagnose_instrument_activity.py --moisesdb C:/Datasets/moisesdb_v0.1 --songs 12
    python scripts/diagnose_instrument_activity.py --file song.flac --start 0 --seconds 30

With stems it answers the question that matters for a fast analyzer working on a mix: does
its output follow the instrument, or everything that has attacks? For each song it runs the
analyzer on (a) the full mix, (b) the mix without the instrument, (c) the instrument alone
and (d) the drums alone, and compares attacks found in the mix with attacks of the stem.
Stem-derived attacks are a reference made by the same analyzer on clean material, not
human annotation. Nothing is written; audio is only read.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from gmv.evaluation.detect import run_instruments
from gmv.models import AnalysisConfig


def frames(audio, rate, name="guitar", sensitivity=0.5):
    """Run one analyzer over ``audio`` (frames, channels) exactly as the engine does.
    Returns times and an array of (level, flux, attack, rate or nan, regularity)."""
    config = AnalysisConfig(instrument_analyzers=(name,), spectrum_bands=0)
    found, _ = run_instruments(audio, rate, config, {name: sensitivity})
    rows = [
        (m.level, m.flux, m.attack, np.nan if m.rate_hz is None else m.rate_hz, m.regularity)
        for m in found
    ]
    return np.asarray([m.timestamp for m in found]), np.asarray(rows).reshape(-1, 5)


def match(found, reference, tolerance=0.05):
    """Greedy one-to-one matching of attack times; returns hits."""
    hits, used = 0, 0
    for time in found:
        while used < len(reference) and reference[used] < time - tolerance:
            used += 1
        if used < len(reference) and abs(reference[used] - time) <= tolerance:
            hits += 1
            used += 1
    return hits


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


def stems_report(root, songs, instrument, start, seconds, sensitivity, mono=False):
    results = []
    for folder in sorted(Path(root).iterdir()):
        if len(results) >= songs:
            break
        if not (folder / instrument).is_dir() or not (folder / "drums").is_dir():
            continue
        parts = {}
        for stem in sorted(p.name for p in folder.iterdir() if p.is_dir()):
            audio, rate = read_stem(folder / stem, start, seconds)
            if audio is not None and len(audio) >= rate * seconds * 0.9:
                parts[stem] = audio
        if instrument not in parts or "drums" not in parts:
            continue
        size = min(len(a) for a in parts.values())
        parts = {k: v[:size] for k, v in parts.items()}
        target = parts[instrument]
        if float(np.sqrt(np.mean(target**2))) < 10 ** (-40 / 20):
            continue  # the instrument is silent in this excerpt
        mix = sum(parts.values())
        without = mix - target

        def run(audio, rate=rate):
            if mono:  # what the analyzer makes of the same music without a stereo image
                audio = np.repeat(audio.mean(axis=1, keepdims=True), 2, axis=1)
            times, rows = frames(audio, rate, instrument, sensitivity)
            return times, rows

        t, full = run(mix)
        _, twin = run(without)
        _, alone = run(target)
        _, drums = run(parts["drums"])

        def at(rows, t=t):
            return t[: len(rows)][rows[:, 2] > 0]

        reference, found, leaked = at(alone), at(full), at(twin)
        hits = match(found, reference)
        minutes = seconds / 60
        results.append(
            {
                "song": folder.name[:8],
                "stem_attacks_per_min": round(len(reference) / minutes),
                "mix_attacks_per_min": round(len(found) / minutes),
                "recall": round(hits / max(1, len(reference)), 2),
                "precision": round(hits / max(1, len(found)), 2),
                "without_instrument_attacks_per_min": round(len(leaked) / minutes),
                "drums_alone_attacks_per_min": round(len(at(drums)) / minutes),
                "flux_mix": round(float(full[:, 1].mean()), 3),
                "flux_without": round(float(twin[:, 1].mean()), 3),
                "flux_correlation_with_stem": round(
                    float(np.corrcoef(full[:, 1], alone[: len(full), 1])[0, 1]), 2
                ),
                "flux_correlation_without": round(
                    float(np.corrcoef(twin[:, 1], alone[: len(twin), 1])[0, 1]), 2
                ),
            }
        )
        print(json.dumps(results[-1]), flush=True)
    keys = [k for k in results[0] if k != "song"] if results else []
    print("MEDIAN", json.dumps({k: float(np.median([r[k] for r in results])) for k in keys}))
    return results


def file_report(path, start, seconds, instrument, sensitivity):
    with sf.SoundFile(path) as file:
        rate = file.samplerate
        file.seek(round(start * rate))
        audio = file.read(round(seconds * rate), dtype="float32", always_2d=True)
    times, rows = frames(audio, rate, instrument, sensitivity)
    attacks = times[rows[:, 2] > 0]
    for second in range(int(seconds)):
        inside = (times >= second) & (times < second + 1)
        marks = "".join("|" if a > 0 else ("-" if f > 0.1 else ".") for f, a in rows[inside][:, 1:3])
        rate_hz = rows[inside][:, 3]
        shown = np.nanmedian(rate_hz) if np.isfinite(rate_hz).any() else float("nan")
        print(
            f"{start + second:6.0f}s {marks}  attacks {int((rows[inside][:, 2] > 0).sum()):2d}"
            f"  rate {shown:5.1f}/s  regularity {rows[inside][:, 4].mean():.2f}"
        )
    print(f"{len(attacks)} attacks in {seconds:.0f} s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--moisesdb")
    parser.add_argument("--file")
    parser.add_argument("--instrument", default="guitar")
    parser.add_argument("--songs", type=int, default=12)
    parser.add_argument("--start", type=float, default=45.0)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--sensitivity", type=float, default=0.5)
    parser.add_argument("--mono", action="store_true", help="sum to mono first")
    args = parser.parse_args()
    if args.file:
        file_report(args.file, args.start, args.seconds, args.instrument, args.sensitivity)
    else:
        stems_report(
            args.moisesdb,
            args.songs,
            args.instrument,
            args.start,
            args.seconds,
            args.sensitivity,
            args.mono,
        )
