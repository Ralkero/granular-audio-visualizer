"""Score the pitch tracker on GuitarSet (real solo guitar with note annotations).

    python scripts/evaluate_pitch.py E:/datasets/GuitarSet --split heldout --output result.json

GuitarSet has six players. Players 00 to 03 are the development set, used when choosing the
tracker's constants; players 04 and 05 are held out. Each excerpt exists as a solo (mostly
single notes) and as comping (chords), which are reported separately because they are very
different problems. The reference notes are the dataset's own annotations (from a hexaphonic
pickup), not anything produced by this project.

This measures pitch tracking on guitar alone. It says nothing about a full mix.
"""

import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from gmv.pitch import METHOD, PitchTracker

HOP_S = 0.02
DEVELOPMENT, HELD_OUT = ("00", "01", "02", "03"), ("04", "05")


def reference_notes(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    notes = [
        (item["time"], item["time"] + item["duration"], item["value"])
        for annotation in data["annotations"]
        if annotation["namespace"] == "note_midi"
        for item in annotation["data"]
    ]
    return sorted(notes)


def track(audio, rate, **options):
    """Run the tracker causally; returns (frame times, note sets, scores, attacks)."""
    hop = round(rate * HOP_S)
    tracker = PitchTracker(rate, hop / rate, **options)
    times, sets, attacks = [], [], []
    for start in range(0, len(audio) - hop + 1, hop):
        notes = tracker.step(audio[start : start + hop])
        described = (start + hop) / rate - tracker.lag_s  # the moment the window is about
        times.append(described)
        sets.append({note.midi: note.score for note in notes})
        for note in notes:
            if note.attack:
                attacks.append((described - note.attack_back * hop / rate, note.midi, note.score))
    return np.array(times), sets, attacks


def match(reference, estimated, tolerance):
    """One-to-one matching of (time, midi) pairs with the same pitch within ``tolerance``."""
    used, hits = set(), 0
    for time, midi in reference:
        best, distance = None, tolerance
        for index, (other, pitch) in enumerate(estimated):
            if index not in used and pitch == midi and abs(other - time) <= distance:
                best, distance = index, abs(other - time)
        if best is not None:
            used.add(best)
            hits += 1
    return hits


def score_file(audio, rate, notes, **options):
    times, sets, attacks = track(audio, rate, **options)
    counts = dict.fromkeys(
        ("tp", "fp", "fn", "chroma_tp", "chroma_fp", "chroma_fn", "frames", "silent_fp_frames"), 0
    )
    starts = np.array([n[0] for n in notes])
    ends = np.array([n[1] for n in notes])
    pitches = np.array([round(n[2]) for n in notes], dtype=int)
    for time, found in zip(times, sets, strict=True):
        if time < 0:
            continue
        active = set(pitches[(starts <= time) & (ends > time)].tolist())
        found = set(found)
        counts["frames"] += 1
        counts["tp"] += len(active & found)
        counts["fp"] += len(found - active)
        counts["fn"] += len(active - found)
        chroma_ref, chroma_found = {p % 12 for p in active}, {p % 12 for p in found}
        counts["chroma_tp"] += len(chroma_ref & chroma_found)
        counts["chroma_fp"] += len(chroma_found - chroma_ref)
        counts["chroma_fn"] += len(chroma_ref - chroma_found)
        counts["silent_fp_frames"] += bool(found and not active)
        counts["polyphony"] = counts.get("polyphony", 0) + len(active)
    onsets = [(n[0], round(n[2])) for n in notes]
    found = [(time, midi) for time, midi, _ in attacks]
    for name, tolerance in (("onset_50ms", 0.05), ("onset_100ms", 0.1)):
        counts[name + "_tp"] = match(onsets, found, tolerance)
    counts["onset_reference"], counts["onset_estimated"] = len(onsets), len(found)
    return counts


def prf(tp, fp, fn):
    return {
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
    }


def summarise(rows):
    total = {key: sum(row[key] for row in rows) for key in rows[0]}
    result = {
        "files": len(rows),
        "minutes": total["frames"] * HOP_S / 60,
        "mean_reference_polyphony": total["polyphony"] / total["frames"],
        "frame_notes": prf(total["tp"], total["fp"], total["fn"]),
        "frame_pitch_class": prf(total["chroma_tp"], total["chroma_fp"], total["chroma_fn"]),
        "frames_with_notes_reported_during_silence": total["silent_fp_frames"] / total["frames"],
    }
    for name in ("onset_50ms", "onset_100ms"):
        tp = total[name + "_tp"]
        result["note_" + name] = prf(
            tp, total["onset_estimated"] - tp, total["onset_reference"] - tp
        )
    return result


def evaluate(root, players, limit=None, **options):
    root = Path(root)
    groups = {"solo": [], "comp": []}
    files = sorted((root / "annotation").glob("*.jams"))
    files = [f for f in files if f.name[:2] in players][:limit]
    for jams in files:
        wav = root / "audio_mono-mic" / (jams.stem + "_mic.wav")
        audio, rate = sf.read(io.BytesIO(wav.read_bytes()), dtype="float64")
        row = score_file(audio, rate, reference_notes(jams), **options)
        groups["solo" if jams.stem.endswith("solo") else "comp"].append(row)
    return {name: summarise(rows) for name, rows in groups.items() if rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("--split", choices=("development", "heldout"), default="development")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    players = DEVELOPMENT if args.split == "development" else HELD_OUT
    result = {
        "dataset": "GuitarSet (mono microphone recordings)",
        "method": METHOD,
        "split": args.split,
        "players": players,
        "hop_seconds": HOP_S,
        "matching": "notes rounded to the nearest semitone; onsets matched one to one",
        "results": evaluate(args.root, players, args.limit),
    }
    text = json.dumps(result, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
