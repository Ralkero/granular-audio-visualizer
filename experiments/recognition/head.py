"""Train and score a small head on cached frozen-model outputs, by artist folds.

A window ending at time t is labelled from the reference in its last ``SPAN`` seconds:
present when the label is active in at least half of that span. Windows whose span touches
uncertain reference time (bleed) are left out. For training only, windows with a mixed span
(active 20% to 80%) are left out as well.

Two pools are always reported apart:

``real``   the dataset's real mixes (guitar plays most of the time)
``twin``   the same songs summed without their guitar files, plus nothing else: every
           window is guitar-free, so anything reported there is a false alarm

    python experiments/recognition/head.py --jobs JOBS.json --labels LABELS_DIR \
        --embeddings E:/datasets/gmv-cache/recognition/mn10_as --output OUT.json
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

LABELS = ("guitar", "guitar_distorted", "guitar_clean", "guitar_acoustic")
SPAN = 0.5
# AudioSet outputs used as they come, without any training.
AUDIOSET = {"Guitar": 140, "Electric guitar": 141, "Acoustic guitar": 143}


def fold_of(group, folds):
    return int(hashlib.sha256(f"gmv-fold-1:{group}".encode()).hexdigest(), 16) % folds


def window_labels(labels_file, ends):
    """(target, certain, mixed) per window and label."""
    with np.load(labels_file, allow_pickle=False) as data:
        y, k, hop = data["y"], data["k"], float(data["hop"])
    span = max(1, round(SPAN / hop))
    last = np.clip(np.round(ends / hop).astype(int), span, len(y))
    total = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(y, 0)])
    known = np.concatenate([np.zeros((1, y.shape[1])), np.cumsum(k, 0)])
    share = (total[last] - total[last - span]) / span
    certain = (known[last] - known[last - span]) == span
    return share >= 0.5, certain, (share > 0.2) & (share < 0.8)


def load(jobs, labels_dir, embeddings, parts, raw):
    """One entry per job and version: features, targets, masks."""
    items = []
    for job in jobs:
        for twin in (False, True):
            name = f"{job['id']}.noguitar.npz" if twin else f"{job['id']}.npz"
            path = Path(embeddings) / name
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as data:
                ends = data["t"]
                times = ends
                x = np.concatenate([data[p] for p in parts], 1).astype(np.float32)
                logits = data[raw].astype(np.float32)
            y, certain, mixed = window_labels(Path(labels_dir) / f"{job['id']}.npz", ends)
            n = min(len(x), len(y))
            if twin:  # the guitar files were left out: absent, whatever the stems contained
                y, certain, mixed = np.zeros_like(y), np.ones_like(certain), np.zeros_like(mixed)
            items.append(
                {
                    "id": job["id"],
                    "group": job["group"],
                    "split": job["split"],
                    "twin": twin,
                    "t": times[:n],
                    "x": x[:n],
                    "logits": logits[:n],
                    "y": y[:n],
                    "certain": certain[:n],
                    "mixed": mixed[:n],
                }
            )
    return items


def auc(score, truth):
    truth = np.asarray(truth, bool)
    if truth.all() or not truth.any():
        return None
    order = np.argsort(score, kind="stable")
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks of ties
    values, inverse, counts = np.unique(score, return_inverse=True, return_counts=True)
    sums = np.bincount(inverse, ranks)
    ranks = (sums / counts)[inverse]
    positives = truth.sum()
    return float(
        (ranks[truth].sum() - positives * (positives + 1) / 2)
        / (positives * (len(truth) - positives))
    )


def average_precision(score, truth):
    truth = np.asarray(truth, bool)
    if not truth.any():
        return None
    order = np.argsort(-score, kind="stable")
    hits = truth[order]
    precision = np.cumsum(hits) / np.arange(1, len(hits) + 1)
    return float(precision[hits].mean())


def fit(x, y, weight, hidden, epochs, decay, seed):
    torch.manual_seed(seed)
    mean, scale = x.mean(0), x.std(0) + 1e-6
    data = torch.from_numpy((x - mean) / scale)
    target, weight = torch.from_numpy(y.astype(np.float32)), torch.from_numpy(weight)
    layers = [torch.nn.Linear(x.shape[1], y.shape[1])]
    if hidden:
        layers = [
            torch.nn.Linear(x.shape[1], hidden),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden, y.shape[1]),
        ]
    net = torch.nn.Sequential(*layers)
    optimiser = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=decay)
    # Each label: positives and negatives weigh the same in total.
    positive = (target * weight).sum(0) / weight.sum(0).clamp(min=1)
    balance = torch.where(target > 0, 0.5 / positive.clamp(min=1e-3), 0.5 / (1 - positive))
    steps = epochs * (len(data) // 512)
    schedule = torch.optim.lr_scheduler.OneCycleLR(optimiser, 3e-3, total_steps=steps)
    net.train()
    for _ in range(steps):
        index = torch.randint(len(data), (512,))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(data[index]), target[index], weight[index] * balance[index]
        )
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
        schedule.step()
    net.eval()

    def predict(rows):
        with torch.inference_mode():
            return torch.sigmoid(net(torch.from_numpy((rows - mean) / scale))).numpy()

    return predict, net, mean, scale


def at_false_alarm_rate(score, truth, rate):
    """Threshold with at most ``rate`` false alarms among the negatives, and its recall."""
    negatives = np.sort(score[~truth])
    if not len(negatives):
        return None, None
    threshold = float(negatives[min(len(negatives) - 1, int(np.ceil(len(negatives) * (1 - rate))))])
    return threshold, float((score[truth] > threshold).mean()) if truth.any() else None


def summary(items, scores, index):
    """Scores of one label over a set of items. ``scores[i]`` belongs to ``items[i]``."""
    real = [
        (s[:, index], i["y"][:, index], i["certain"][:, index])
        for s, i in zip(scores, items, strict=True)
        if not i["twin"]
    ]
    twins = [s[:, index] for s, i in zip(scores, items, strict=True) if i["twin"]]
    score = np.concatenate([s[c] for s, _, c in real])
    truth = np.concatenate([y[c] for _, y, c in real])
    twin = np.concatenate(twins) if twins else np.zeros(0, np.float32)
    everything = np.concatenate([score, twin])
    labels = np.concatenate([truth, np.zeros(len(twin), bool)])
    threshold, recall = at_false_alarm_rate(everything, labels, 0.05)
    out = {
        "real_active_share": round(float(truth.mean()), 3),
        "real_auc": auc(score, truth),
        "real_recall_at_0.5": float((score[truth] >= 0.5).mean()) if truth.any() else None,
        "real_fpr_at_0.5": float((score[~truth] >= 0.5).mean()) if (~truth).any() else None,
        "twin_fpr_at_0.5": float((twin >= 0.5).mean()) if len(twin) else None,
        "all_auc": auc(everything, labels),
        "all_pr_auc": average_precision(everything, labels),
        "threshold_for_5pct_false_alarms": threshold,
        "recall_at_5pct_false_alarms": recall,
        "twin_songs_lit_over_20pct": int(sum((t >= 0.5).mean() > 0.2 for t in twins)),
        "twin_songs": len(twins),
    }
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--features", default="e3,eall,lall", help="cached arrays to use")
    parser.add_argument("--raw", default="lall", help="cached AudioSet outputs (untrained row)")
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--decay", type=float, default=0.05)
    parser.add_argument("--folds", type=int, default=4)
    parser.add_argument("--no-twins-in-training", action="store_true")
    parser.add_argument("--output", required=True)
    parser.add_argument("--scores", help="folder for the held-out scores of every recording")
    args = parser.parse_args()
    torch.set_num_threads(8)
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8"))
    items = load(jobs, args.labels, args.embeddings, args.features.split(","), args.raw)
    result = {
        "embeddings": Path(args.embeddings).name,
        "features": args.features,
        "hidden": args.hidden,
        "decay": args.decay,
        "epochs": args.epochs,
        "span_s": SPAN,
        "twins_in_training": not args.no_twins_in_training,
        "folds": [],
    }
    held_scores, held_items, zero = [], [], []
    for fold in range(args.folds):
        train = [i for i in items if fold_of(i["group"], args.folds) != fold]
        if args.no_twins_in_training:
            train = [i for i in train if not i["twin"]]
        test = [i for i in items if fold_of(i["group"], args.folds) == fold]
        x = np.concatenate([i["x"] for i in train])
        y = np.concatenate([i["y"] for i in train])
        weight = np.concatenate([i["certain"] & ~i["mixed"] for i in train]).astype(np.float32)
        predict, *_ = fit(x, y, weight, args.hidden, args.epochs, args.decay, fold)
        scores = [predict(i["x"]) for i in test]
        row = {
            "fold": fold,
            "songs": sum(not i["twin"] for i in test),
            "artists": len({i["group"] for i in test}),
        }
        for index, label in enumerate(LABELS):
            row[label] = summary(test, scores, index)
        result["folds"].append(row)
        held_scores += scores
        held_items += test
        g = row["guitar"]
        print(
            f"fold {fold} ({row['songs']} songs): guitar real AUC {g['real_auc']} | all AUC "
            f"{g['all_auc']} | twin FPR@0.5 {g['twin_fpr_at_0.5']} | recall at 5% false alarms "
            f"{g['recall_at_5pct_false_alarms']}",
            flush=True,
        )
    if args.scores:
        Path(args.scores).mkdir(parents=True, exist_ok=True)
        for item, scores in zip(held_items, held_scores, strict=True):
            name = item["id"] + (".noguitar" if item["twin"] else "")
            np.savez(
                Path(args.scores) / f"{name}.npz",
                t=item["t"],
                score=scores.astype(np.float32),
                y=item["y"],
                certain=item["certain"],
            )
    result["pooled"] = {
        label: summary(held_items, held_scores, n) for n, label in enumerate(LABELS)
    }
    # The pretrained AudioSet outputs as they are (no training): same pools, guitar only.
    for name, column in AUDIOSET.items():
        raw = [1 / (1 + np.exp(-i["logits"][:, [column]])) for i in held_items]
        zero.append((name, summary(held_items, raw, 0) | {"scored_against": "guitar"}))
    result["audioset_outputs_untrained"] = dict(zero)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("pooled:", json.dumps(result["pooled"]["guitar"]))
    print(
        "AudioSet 'Guitar' untrained:", json.dumps(result["audioset_outputs_untrained"]["Guitar"])
    )


if __name__ == "__main__":
    main()
