"""Score a context recognition model over corpus recordings, through the application's path.

Each recording is fed to ``gmv.recognition.Recognizer`` block by block exactly as the live
worker does (``run_recognition``), with the model description file the application loads.
Two steps:

``run``     writes one file of estimates per recording (scores, states, reference)
``report``  reads such folders and reports the pools of ``docs/evaluation/GUITAR_PROTOCOL.md``

A decision at the end of a hop is compared with the reference in the half second before it:
present when the label is active in at least half of that span. Spans touching uncertain
reference time (bleed) are left out. Guitar-free recordings (twins, classical sets) count
entirely as absent.

    python scripts/score_recognition.py run --corpus corpus-data/corpus/manifest.json \
        --model .model-cache/guitar_mn10_v1.json --datasets moisesdb,moisesdb_noguitar \
        --split validation --output OUT_DIR [--shard 0/8]
    python scripts/score_recognition.py report OUT_DIR [OUT_DIR ...] --output REPORT.json
"""

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np

SPAN = 0.5
LEVELS = {"absent": 0, "candidate": 1, "detected": 2}
GUITAR_FREE = ("moisesdb_noguitar", "musicnet", "spheres")


def trainer_module(labels):
    spec = importlib.util.spec_from_file_location(
        "train_activity", Path(__file__).with_name("train_activity.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.configure(labels)
    return module


def window_truth(y, known, hop, ends):
    span = max(1, round(SPAN / hop))
    last = np.clip(np.round(ends / hop).astype(int), span, len(y))
    active = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, 0)])
    sure = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(known, 0)])
    share = (active[last] - active[last - span]) / span
    return share >= 0.5, (sure[last] - sure[last - span]) == span


def run(args):
    from gmv.corpus import registry
    from gmv.evaluation.detect import run_recognition
    from gmv.recognition import load_backend

    backend = load_backend(args.model)
    trainer = trainer_module(backend.labels)
    manifest = registry.load_manifest(args.corpus)
    paths = registry.load_paths(args.corpus)
    index, count = (int(v) for v in args.shard.split("/"))
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    chosen = [
        r
        for r in manifest.recordings
        if r.dataset in args.datasets.split(",") and r.split == args.split
    ]
    for recording in chosen[index::count]:
        target = out / f"{recording.recording_id}.npz"
        if target.exists():
            continue
        path, problem = registry.resolve(recording, paths, verify_hash=False)
        if not path:
            print(f"skipped {recording.recording_id}: {problem}")
            continue
        audio, rate = registry.read_audio(path)
        estimates, times = run_recognition(audio, rate, backend)
        ends = np.array([e.end for e in estimates])
        grid = np.arange(0, recording.duration_s, 0.02) + 0.01
        y, known = trainer.frame_labels(args.corpus, recording, grid)
        truth, certain = window_truth(y, known, 0.02, ends)
        np.savez(
            target,
            t=ends,
            score=np.array([[s for _, s in e.scores] for e in estimates], np.float32),
            state=np.array([[LEVELS[s] for _, s in e.states] for e in estimates], np.int8),
            y=truth,
            certain=certain,
            frames=y,  # the reference every 20 ms, for entrance delays
            labels=np.array(backend.labels),
            dataset=recording.dataset,
            tags=np.array(recording.tags),
            inference_ms=np.array(times, np.float32),
            model=f"{backend.name} {backend.version}",
        )
        print(f"{recording.dataset} {recording.recording_id} {len(ends)} estimates", flush=True)


def auc(score, truth):
    if truth.all() or not truth.any():
        return None
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    ranks = (np.bincount(inverse, ranks) / counts)[inverse]
    positives = truth.sum()
    area = (ranks[truth].sum() - positives * (positives + 1) / 2) / (
        positives * (len(truth) - positives)
    )
    return round(float(area), 3)


def average_precision(score, truth):
    if not truth.any():
        return None
    hits = truth[np.argsort(-score, kind="stable")]
    return round(float((np.cumsum(hits) / np.arange(1, len(hits) + 1))[hits].mean()), 3)


def counts(hit, truth):
    tp, fp = int((hit & truth).sum()), int((hit & ~truth).sum())
    fn, tn = int((~hit & truth).sum()), int((~hit & ~truth).sum())
    return {
        "precision": round(tp / (tp + fp), 3) if tp + fp else None,
        "recall": round(tp / (tp + fn), 3) if tp + fn else None,
        "f1": round(2 * tp / (2 * tp + fp + fn), 3) if tp else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else None,
        "false_negative_rate": round(fn / (tp + fn), 3) if tp + fn else None,
    }


def longest_run(hit, hop):
    best = run_length = 0
    for value in hit:
        run_length = run_length + 1 if value else 0
        best = max(best, run_length)
    return best * hop


def entrance_delays(item, column):
    """Seconds from each guitar entrance (after at least 1 s without) to "detected"."""
    frames = item["frames"][:, column]
    starts = np.flatnonzero(frames[1:] & ~frames[:-1]) + 1
    delays, missed = [], 0
    for start in starts:
        if start < 50 or frames[start - 50 : start].any() or not frames[start : start + 100].all():
            continue  # needs a second of absence before and two seconds of guitar after
        onset = start * 0.02
        after = (item["t"] >= onset) & (item["t"] <= onset + 5) & (item["state"][:, column] == 2)
        if after.any():
            delays.append(float(item["t"][after][0] - onset))
        else:
            missed += 1
    return delays, missed


def pool(items, column, minimum):
    """One row of the report: decisions at or above ``minimum`` (2 detected, 1 candidate)."""
    if not items:
        return None
    hop = float(np.median(np.diff(items[0]["t"]))) if len(items[0]["t"]) > 1 else 0.5
    hit = np.concatenate([i["state"][i["certain"][:, column], column] >= minimum for i in items])
    truth = np.concatenate([i["y"][i["certain"][:, column], column] for i in items])
    score = np.concatenate([i["score"][i["certain"][:, column], column] for i in items])
    changes = sum(
        int(np.abs(np.diff((i["state"][:, column] >= minimum).astype(int))).sum()) for i in items
    )
    minutes = sum(len(i["t"]) for i in items) * hop / 60
    per_recording = [
        counts(
            i["state"][i["certain"][:, column], column] >= minimum,
            i["y"][i["certain"][:, column], column],
        )
        for i in items
    ]
    recalls = [c["recall"] for c in per_recording if c["recall"] is not None]
    alarms = [
        c["false_positive_rate"] for c in per_recording if c["false_positive_rate"] is not None
    ]
    row = {
        "recordings": len(items),
        "scored_minutes": round(len(truth) * hop / 60, 1),
        "active_share": round(float(truth.mean()), 3),
        **counts(hit, truth),
        "always_on_f1": round(float(2 * truth.mean() / (1 + truth.mean())), 3),
        "roc_auc": auc(score, truth),
        "pr_auc": average_precision(score, truth),
        "mean_recording_recall": round(float(np.mean(recalls)), 3) if recalls else None,
        "mean_recording_false_positive_rate": round(float(np.mean(alarms)), 3) if alarms else None,
        "state_changes_per_minute": round(changes / max(minutes, 1e-9), 2),
    }
    # Recordings known to be without it: no active reference and nothing uncertain.
    silent = [i for i in items if not i["y"][:, column].any() and i["certain"][:, column].all()]
    if silent:
        lit = [float((i["state"][:, column] >= minimum).mean()) for i in silent]
        runs = [longest_run(i["state"][:, column] >= minimum, hop) for i in silent]
        row["recordings_without_it"] = len(silent)
        row["lit_share_in_those_mean"] = round(float(np.mean(lit)), 3)
        row["those_lit_over_20pct"] = int(sum(v > 0.2 for v in lit))
        row["longest_false_activation_s_median"] = round(float(np.median(runs)), 1)
        row["longest_false_activation_s_max"] = round(float(np.max(runs)), 1)
    return row


def report(args):
    items = []
    for folder in args.folders:
        for path in sorted(Path(folder).glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                item = {key: data[key] for key in data.files}
            item["dataset"], item["tags"] = str(item["dataset"]), [str(t) for t in item["tags"]]
            items.append(item)
    labels = [str(label) for label in items[0]["labels"]]
    result = {"model": str(items[0]["model"]), "recordings": len(items), "labels": {}}
    times = np.concatenate([i["inference_ms"] for i in items])
    result["inference_ms"] = {
        "p50": round(float(np.percentile(times, 50)), 1),
        "p95": round(float(np.percentile(times, 95)), 1),
    }
    real = [i for i in items if i["dataset"] not in GUITAR_FREE]
    groups = {"real mixes": real, "all recordings": items}
    for dataset in sorted({i["dataset"] for i in items if i["dataset"] in GUITAR_FREE}):
        groups[f"guitar-free: {dataset}"] = [i for i in items if i["dataset"] == dataset]
    for genre in sorted({t for i in real for t in i["tags"] if t.startswith("genre_")}):
        chosen = [i for i in real if genre in i["tags"]]
        if len(chosen) >= 3:
            groups[f"real mixes, {genre[6:]}"] = chosen
    for column, label in enumerate(labels):
        section = {}
        for minimum, name in ((2, "detected"), (1, "detected_or_candidate")):
            section[name] = {
                group: pool(chosen, column, minimum) for group, chosen in groups.items() if chosen
            }
        delays, missed = [], 0
        for item in real:
            found, lost = entrance_delays(item, column)
            delays += found
            missed += lost
        section["entrances"] = {
            "count": len(delays) + missed,
            "not_detected_within_5s": missed,
            "delay_s_median": round(float(np.median(delays)), 2) if delays else None,
            "delay_s_p90": round(float(np.percentile(delays, 90)), 2) if delays else None,
        }
        result["labels"][label] = section
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    one = sub.add_parser("run")
    one.add_argument("--corpus", required=True)
    one.add_argument("--model", required=True)
    one.add_argument("--datasets", required=True)
    one.add_argument("--split", required=True, choices=("train", "validation", "test"))
    one.add_argument("--output", required=True)
    one.add_argument("--shard", default="0/1")
    two = sub.add_parser("report")
    two.add_argument("folders", nargs="+")
    two.add_argument("--output")
    args = parser.parse_args()
    (run if args.command == "run" else report)(args)


if __name__ == "__main__":
    main()
