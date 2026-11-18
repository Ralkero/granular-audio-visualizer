"""Score guitar activity read off the guitar stem of six-source HTDemucs on MoisesDB.

Input: level curves from ``activity_worker.py --six-source`` (no separated audio). The guitar
stem is called active where its level is above ``max(floor, anchor - margin)``; the anchors
are those of ``score_activity.py``. The margin is chosen on the validation songs (balanced
accuracy, same pass and rule) and applied unchanged to the held-out songs. Time where the
dataset's guitar files carry bleed is left out, as everywhere else.

    python experiments/separation/score_guitar.py --corpus corpus-data/corpus/manifest.json \
        --levels E:/datasets/gmv-cache/separation_6s --limit-seconds 150 --output OUT.json
"""

import argparse
import json
from pathlib import Path

import numpy as np
from score_activity import MARGINS, anchor

from gmv.corpus import registry
from gmv.corpus.stems import FLOOR_DBFS, HOP_S, levels, mask_intervals
from gmv.evaluation.metrics import interval_score

SETUPS = (("full", "own_peak"), ("full", "mix"), ("causal", "running_peak"), ("causal", "mix"))


def load(manifest_path, levels_dir, split, limit):
    manifest = registry.load_manifest(manifest_path)
    paths = registry.load_paths(manifest_path)
    tracks = []
    for recording in manifest.recordings:
        cached = Path(levels_dir) / f"{recording.recording_id}.npz"
        if recording.dataset != "moisesdb" or recording.split != split or not cached.exists():
            continue
        path, problem = registry.resolve(recording, paths, verify_hash=False)
        if problem:
            raise ValueError(f"{recording.recording_id}: {problem}")
        audio, rate = registry.read_audio(path)
        duration = min(recording.duration_s, limit or recording.duration_s)
        annotation = registry.annotations_for(manifest_path, recording.recording_id)[0]
        spans = [a for a in annotation.activity if a.label == "guitar" and a.start < duration]
        with np.load(cached, allow_pickle=False) as data:
            guitar = data["sources"].tolist().index("guitar")
            passes = {n: data[n][guitar].astype(np.float64) for n in ("full", "causal")}
        count = len(passes["full"])
        tracks.append(
            {
                "id": recording.recording_id,
                "duration": duration,
                "has_guitar": annotation.covers("activity.guitar"),
                "mix": levels(audio[: round(duration * rate)], rate)[0][:count],
                "passes": passes,
                "certain": [
                    (a.start, min(a.end, duration)) for a in spans if a.certainty == "certain"
                ],
                "uncertain": [
                    (a.start, min(a.end, duration)) for a in spans if a.certainty == "uncertain"
                ],
            }
        )
    return tracks


def score(track, name, rule, margin):
    level = track["passes"][name]
    mix = track["mix"][: len(level)]
    mask = level > np.maximum(FLOOR_DBFS, anchor(rule, level, mix) - margin)
    predicted = mask_intervals(mask, HOP_S, track["duration"])
    return interval_score(track["certain"], predicted, [(0, track["duration"])], track["uncertain"])


def pooled(tracks, rows):
    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp_seconds", "fp_seconds", "fn_seconds"))
    scored = sum(r["scored_seconds"] for r in rows)
    active = sum(r["reference_active_seconds"] for r in rows)
    recall = tp / (tp + fn) if tp + fn else 0.0
    rate = fp / (scored - active) if scored > active else None
    without = [
        (r["tp_seconds"] + r["fp_seconds"]) / r["scored_seconds"]
        for t, r in zip(tracks, rows, strict=True)
        if not t["has_guitar"] and r["scored_seconds"]
    ]
    f1s = [r["f1"] for r in rows if r["reference_active_seconds"] > 0]
    return {
        "recordings": len(rows),
        "scored_minutes": round(scored / 60, 1),
        "left_out_uncertain_minutes": round(sum(r["abstained_seconds"] for r in rows) / 60, 1),
        "active_share": round(active / scored, 3),
        "precision": round(tp / (tp + fp), 3) if tp + fp else 0.0,
        "recall": round(recall, 3),
        "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0,
        "always_active_f1": round(2 * active / (active + scored), 3),
        "false_positive_rate": None if rate is None else round(rate, 3),
        "balanced_accuracy": round((recall + 1 - (rate or 0)) / 2, 3),
        "median_song_f1": round(float(np.median(f1s)), 3) if f1s else None,
        "songs_below_f1_0.5": sum(f < 0.5 for f in f1s),
        "recordings_without_guitar": len(without),
        "lit_in_recordings_without_guitar": round(float(np.mean(without)), 3) if without else None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--levels", required=True)
    parser.add_argument("--limit-seconds", type=float)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    validation = load(args.corpus, args.levels, "validation", args.limit_seconds)
    test = load(args.corpus, args.levels, "test", args.limit_seconds)
    result = {
        "separator": "htdemucs_6s/5c90dfd2, shifts=0",
        "reference": "stem_derived (stem_energy_v1), guitar",
        "seconds_per_recording": args.limit_seconds,
        "margin_selection": "balanced accuracy on the validation songs, same pass and rule",
        "setups": {},
    }
    for name, rule in SETUPS:
        sweep = {
            m: pooled(validation, [score(t, name, rule, m) for t in validation]) for m in MARGINS
        }
        margin = max(sweep, key=lambda m: sweep[m]["balanced_accuracy"])
        held = pooled(test, [score(t, name, rule, margin) for t in test])
        result["setups"][f"{name}/{rule}"] = {
            "margin_db": margin,
            "validation": sweep[margin],
            "test": held,
            # After the fact, to show how much the margin matters. Not a result.
            "test_by_margin": {
                m: {
                    k: v
                    for k, v in pooled(test, [score(t, name, rule, m) for t in test]).items()
                    if k in ("recall", "false_positive_rate", "f1")
                }
                for m in MARGINS
            },
        }
        print(
            f"{name}/{rule:13s} margin {margin:2d} dB | validation BA "
            f"{sweep[margin]['balanced_accuracy']:.3f} | test P {held['precision']:.3f} "
            f"R {held['recall']:.3f} F1 {held['f1']:.3f} FPR {held['false_positive_rate']} "
            f"BA {held['balanced_accuracy']:.3f} | without guitar "
            f"{held['lit_in_recordings_without_guitar']}"
        )
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
