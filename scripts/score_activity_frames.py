"""Frame-level scores of activity weights over every recording of a split.

The corpus evaluator scores a label only on recordings that have a stem for it, so a
sub-label such as distorted guitar is never tested on songs without one. This script scores
every label on every recording: where a recording has no stem for a label the label counts
as absent (true for MoisesDB, where every file is typed). Frames whose reference is
uncertain (stems with bleed) are left out. Runs the real analyzer, hysteresis included.

    python scripts/score_activity_frames.py --corpus corpus-data/corpus/manifest.json \
        --dataset moisesdb --split test --weights .model-cache/guitar_v1_all.npz --output OUT.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

from gmv.corpus import registry
from gmv.evaluation.detect import run_detector
from gmv.models import AnalysisConfig


def trainer_module():
    spec = importlib.util.spec_from_file_location(
        "train_activity", Path(__file__).with_name("train_activity.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True, choices=("train", "validation", "test"))
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    trainer = trainer_module()
    labels = tuple(str(label) for label in np.load(args.weights)["labels"])
    trainer.configure(labels)
    manifest = registry.load_manifest(args.corpus)
    paths = registry.load_paths(args.corpus)
    config = AnalysisConfig(guitar_model_path=args.weights)
    scores, active, truth, known, songs = [], [], [], [], []
    for recording in manifest.recordings:
        if recording.dataset != args.dataset or recording.split != args.split:
            continue
        path, reason = registry.resolve(recording, paths, verify_hash=False)
        if not path:
            print(f"skipped {recording.recording_id}: {reason}")
            continue
        audio, rate = registry.read_audio(path)
        frames = [f for f in run_detector(audio, rate, "candidate", config)[0] if f.activity_method]
        frames = [f for f in frames if f.loudness_dbfs >= config.silence_dbfs]
        times = np.array([f.timestamp for f in frames]) - config.window_ms / 2000
        y, k = trainer.frame_labels(args.corpus, recording, times)
        scores.append(
            np.array([[dict(f.instrument_scores)[name] for name in labels] for f in frames])
        )
        active.append(np.array([[name in f.instruments_active for name in labels] for f in frames]))
        truth.append(y)
        known.append(k)
        songs.append((recording.recording_id, len(frames)))
        print(f"scored {recording.recording_id}: {len(frames)} frames", flush=True)
    scores, active = np.concatenate(scores), np.concatenate(active)
    truth, known = np.concatenate(truth), np.concatenate(known)
    result = {"weights": Path(args.weights).name, "split": args.split, "recordings": len(songs)}
    offsets = np.cumsum([0] + [count for _, count in songs])
    for index, name in enumerate(labels):
        use = known[:, index]
        actual, predicted = truth[use, index], active[use, index]
        tp, fp = int((predicted & actual).sum()), int((predicted & ~actual).sum())
        fn, tn = int((~predicted & actual).sum()), int((~predicted & ~actual).sum())
        without = [
            (active[a:b, index] & known[a:b, index]).sum() / max(1, known[a:b, index].sum())
            for a, b in zip(offsets[:-1], offsets[1:], strict=True)
            if not truth[a:b, index].any() and known[a:b, index].all()
        ]
        share = actual.mean()
        result[name] = {
            "scored_minutes": round(float(use.sum()) * 0.02 / 60, 1),
            "left_out_uncertain_minutes": round(float((~use).sum()) * 0.02 / 60, 1),
            "active_share": round(float(share), 3),
            "precision": round(tp / max(1, tp + fp), 3),
            "recall": round(tp / max(1, tp + fn), 3),
            "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3),
            "always_active_f1": round(float(2 * share / (1 + share)), 3),
            "false_positive_rate": round(fp / max(1, fp + tn), 3),
            "auc": round(trainer.auc(scores[use, index], actual), 3),
            "recordings_without_it": len(without),
            "reported_share_in_recordings_without_it": round(float(np.mean(without)), 3)
            if without
            else None,
        }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
