"""Score instrument activity read off HTDemucs' estimated stems against the stem-derived
reference, and set it beside the lightweight estimator on the same tracks.

Input is the level curves written by ``activity_worker.py`` (no separated audio). A stem is
called active where its estimated level is above ``max(floor, anchor - margin)``:

``own_peak``      anchor = the loudest level of that estimated stem in the whole track. This
                  is the rule the reference itself uses; it needs the whole track.
``running_peak``  anchor = the loudest level of that estimated stem so far. Causal.
``mix``           anchor = the level of the full mix at the same moment. Causal.

The margin is chosen per instrument and rule on the validation split (balanced accuracy,
whole-track separation), then applied unchanged to both passes of the test split. HTDemucs was trained on the validation tracks, so the
choice is made on audio the separator already knew; the test tracks from the dataset's own
test folder are the only ones it had not seen.
"""

import argparse
import json
from pathlib import Path

import numpy as np

from gmv.corpus import registry
from gmv.corpus.stems import FLOOR_DBFS, HOP_S, levels, mask_intervals
from gmv.evaluation.metrics import interval_score

LABELS = ("drums", "bass", "vocals", "other")
MARGINS = tuple(range(5, 65, 5))
RULES = {"own_peak": ("full",), "running_peak": ("full", "causal"), "mix": ("full", "causal")}


def anchor(rule, level, mix):
    if rule == "own_peak":
        return np.full_like(level, level.max())
    if rule == "running_peak":
        return np.maximum.accumulate(level)
    return mix


def load(manifest_path, levels_dir, split):
    manifest = registry.load_manifest(manifest_path)
    paths = registry.load_paths(manifest_path)
    tracks = []
    for recording in manifest.recordings:
        if recording.split != split or recording.dataset != "musdb18hq":
            continue
        cached = Path(levels_dir) / f"{recording.recording_id}.npz"
        if not cached.exists():
            continue
        path, problem = registry.resolve(recording, paths, verify_hash=False)
        if problem:
            raise ValueError(f"{recording.recording_id}: {problem}")
        audio, rate = registry.read_audio(path)
        annotation = registry.annotations_for(manifest_path, recording.recording_id)[0]
        if annotation.reference_kind != "stem_derived":
            raise ValueError("Expected the stem-derived reference")
        with np.load(cached, allow_pickle=False) as data:
            passes = {
                name: data[name].astype(np.float64) for name in ("full", "causal") if name in data
            }
            sources = data["sources"].tolist()
        tracks.append(
            {
                "id": recording.recording_id,
                "seen_in_separator_training": path.parts[-3] == "train",
                "duration": recording.duration_s,
                "mix": levels(audio, rate)[0],
                "passes": {
                    name: dict(zip(sources, value, strict=True)) for name, value in passes.items()
                },
                "reference": {
                    label: [(a.start, a.end) for a in annotation.activity if a.label == label]
                    for label in LABELS
                },
            }
        )
    return tracks


def score(track, name, rule, label, margin):
    level = track["passes"][name][label]
    mask = level > np.maximum(FLOOR_DBFS, anchor(rule, level, track["mix"]) - margin)
    predicted = mask_intervals(mask, HOP_S, track["duration"])
    return interval_score(track["reference"][label], predicted, [(0, track["duration"])])


def pooled(rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp_seconds", "fp_seconds", "fn_seconds"))
    inactive = sum(r["scored_seconds"] - r["reference_active_seconds"] for r in rows)
    f1s = [r["f1"] for r in rows if r["reference_active_seconds"] > 0]
    entrances = [
        r["entrances"]["median_absolute_error_ms"]
        for r in rows
        if r["entrances"].get("median_absolute_error_ms") is not None
    ]
    recall = tp / (tp + fn) if tp + fn else 0
    false_positive_rate = fp / inactive if inactive else None
    return {
        "recordings": len(rows),
        "precision": tp / (tp + fp) if tp + fp else 0,
        "recall": recall,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0,
        "false_positive_rate": false_positive_rate,
        "balanced_accuracy": (recall + 1 - (false_positive_rate or 0)) / 2,
        "always_active_f1": 2 * (tp + fn) / (2 * (tp + fn) + inactive) if tp + fn else 0,
        "median_track_f1": float(np.median(f1s)) if f1s else None,
        "worst_track_f1": min(f1s) if f1s else None,
        "tracks_below_f1_0.5": sum(f < 0.5 for f in f1s),
        "median_entrance_error_ms": float(np.median(entrances)) if entrances else None,
    }


def lightweight(report_path, ids):
    report = json.loads(Path(report_path).read_text(encoding="utf-8"))
    rows = {}
    for result in report["recordings"]:
        recording = result["recording"]
        identifier = recording["recording_id"] if isinstance(recording, dict) else recording
        if identifier in ids:
            for label, entry in result["instrument_activity"].items():
                if entry["status"] == "scored":
                    rows.setdefault(label, []).append(entry)
    return {
        "code_commit": report["code_commit"],
        "model": report["configuration"]["activity_model_path"],
        "summary": {label: pooled(entries) for label, entries in rows.items()},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--lightweight-report", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validation = load(args.corpus, args.levels, "validation")
    test = load(args.corpus, args.levels, "test")
    unseen = [t for t in test if not t["seen_in_separator_training"]]
    seen = [t for t in test if t["seen_in_separator_training"]]
    result = {
        "separator": "htdemucs/955717e8, shifts=0",
        "reference": "stem_derived (stem_energy_v1)",
        "margin_selection": "per instrument, balanced accuracy on the validation split",
        "validation_recordings": len(validation),
        "test_recordings_unseen_by_separator": len(unseen),
        "test_recordings_seen_by_separator": len(seen),
        "test_seconds_unseen": sum(t["duration"] for t in unseen),
        "setups": {},
    }
    for rule, passes in RULES.items():
        for name in passes:
            setup = {}
            for label in LABELS:
                sweep = {
                    margin: pooled([score(t, "full", rule, label, margin) for t in validation])
                    for margin in MARGINS
                }
                margin = max(sweep, key=lambda m: sweep[m]["balanced_accuracy"])
                setup[label] = {
                    "margin_db": margin,
                    "validation": sweep[margin],
                    "test_unseen": pooled([score(t, name, rule, label, margin) for t in unseen]),
                    "test_seen": pooled([score(t, name, rule, label, margin) for t in seen]),
                    # After the fact, for showing how much the margin matters. Not a result.
                    "test_unseen_by_margin": {
                        m: {
                            k: v
                            for k, v in pooled(
                                [score(t, name, rule, label, m) for t in unseen]
                            ).items()
                            if k in ("f1", "recall", "false_positive_rate")
                        }
                        for m in MARGINS
                    },
                }
            result["setups"][f"{name}/{rule}"] = setup
    result["lightweight_on_same_unseen_tracks"] = lightweight(
        args.lightweight_report, {t["id"] for t in unseen}
    )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    for key, setup in result["setups"].items():
        for label, entry in setup.items():
            row = entry["test_unseen"]
            print(
                f"{key:22s} {label:7s} margin {entry['margin_db']:2d} dB | "
                f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f} "
                f"FPR {row['false_positive_rate']:.3f} | worst track {row['worst_track_f1']:.2f}"
            )
    for label, row in result["lightweight_on_same_unseen_tracks"]["summary"].items():
        print(
            f"{'lightweight':22s} {label:7s}              | "
            f"P {row['precision']:.3f} R {row['recall']:.3f} F1 {row['f1']:.3f} "
            f"FPR {row['false_positive_rate']:.3f} | worst track {row['worst_track_f1']:.2f}"
        )


if __name__ == "__main__":
    main()
