"""Summarise held-out scores written by ``head.py`` or ``finetune.py``.

Reads every ``<id>.npz`` / ``<id>.noguitar.npz`` in the given folders (several folds can be
given together) and reports, per label, the pools of the protocol: real mixes and
guitar-free twins. With ``--labels`` the real mixes are also scored against the "audible"
reference (guitar within 15 dB of the mix is present, more than 30 dB below is absent,
in between is not scored) and recall is broken down by how loud the guitar is in the mix.

    python experiments/recognition/summarise.py SCORES_DIR [SCORES_DIR ...] \
        --labels LABELS_DIR --output OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from head import LABELS, at_false_alarm_rate, auc, average_precision  # noqa: E402

LEVELS = ((-200, -30), (-30, -20), (-20, -15), (-15, -10), (-10, -5), (-5, 50))


def relative_level(labels_file, times, span=0.5):
    """Guitar level minus mix level (dB), averaged over the span before each time."""
    with np.load(labels_file, allow_pickle=False) as data:
        if "guitar_db" not in data.files:
            return None
        relative = data["guitar_db"] - data["mix_db"][: len(data["guitar_db"])]
    width = round(span / 0.01)
    last = np.clip(np.round(times / 0.01).astype(int), width, len(relative))
    total = np.concatenate([[0.0], np.cumsum(relative)])
    return (total[last] - total[last - width]) / width


def load(folders, labels_dir):
    items = []
    for folder in folders:
        for path in sorted(Path(folder).glob("*.npz")):
            with np.load(path, allow_pickle=False) as data:
                item = {k: data[k] for k in ("t", "score", "y", "certain")}
            item["id"] = path.name.split(".")[0]
            item["twin"] = ".noguitar" in path.name
            if labels_dir and not item["twin"]:
                item["relative"] = relative_level(Path(labels_dir) / f"{item['id']}.npz", item["t"])
            items.append(item)
    return items


def rates(score, truth, threshold):
    hit = score > threshold
    tp, fp = int((hit & truth).sum()), int((hit & ~truth).sum())
    fn, tn = int((~hit & truth).sum()), int((~hit & ~truth).sum())
    return {
        "precision": round(tp / max(1, tp + fp), 3),
        "recall": round(tp / max(1, tp + fn), 3),
        "f1": round(2 * tp / max(1, 2 * tp + fp + fn), 3),
        "false_positive_rate": round(fp / max(1, fp + tn), 3),
    }


def summarise(items, index):
    real = [i for i in items if not i["twin"]]
    twins = [i["score"][:, index] for i in items if i["twin"]]
    score = np.concatenate([i["score"][i["certain"][:, index], index] for i in real])
    truth = np.concatenate([i["y"][i["certain"][:, index], index] for i in real])
    twin = np.concatenate(twins) if twins else np.zeros(0, np.float32)
    everything = np.concatenate([score, twin])
    labels = np.concatenate([truth, np.zeros(len(twin), bool)])
    threshold, _ = at_false_alarm_rate(everything, labels, 0.05)
    out = {
        "real_mixes": len(real),
        "twins": len(twins),
        "real_active_share": round(float(truth.mean()), 3),
        "always_on_f1_real": round(float(2 * truth.mean() / (1 + truth.mean())), 3),
        "roc_auc_real": auc(score, truth),
        "roc_auc_all": auc(everything, labels),
        "pr_auc_all": average_precision(everything, labels),
        "threshold_for_5pct_false_alarms": threshold,
        "at_that_threshold": {
            "real": rates(score, truth, threshold),
            "all": rates(everything, labels, threshold),
            "twin_false_positive_rate": round(float((twin > threshold).mean()), 3),
            "twins_lit_over_20pct": int(sum((t > threshold).mean() > 0.2 for t in twins)),
        },
        "at_0.5": {
            "real": rates(score, truth, 0.5),
            "all": rates(everything, labels, 0.5),
            "twin_false_positive_rate": round(float((twin > 0.5).mean()), 3),
            "twins_lit_over_20pct": int(sum((t > 0.5).mean() > 0.2 for t in twins)),
        },
    }
    if index == 0 and any(i.get("relative") is not None for i in real):
        relative = np.concatenate(
            [
                i["relative"][: len(i["t"])][i["certain"][:, 0]]
                if i.get("relative") is not None
                else np.full(int(i["certain"][:, 0].sum()), -200.0)
                for i in real
            ]
        )
        out["recall_by_guitar_level_db"] = {
            f"{low}..{high}": {
                "share_of_guitar_time": round(
                    float(
                        (truth & (relative >= low) & (relative < high)).sum() / max(1, truth.sum())
                    ),
                    3,
                ),
                "recall_at_5pct": round(
                    float(
                        (score[truth & (relative >= low) & (relative < high)] > threshold).mean()
                    ),
                    3,
                )
                if (truth & (relative >= low) & (relative < high)).any()
                else None,
            }
            for low, high in LEVELS
        }
        audible = truth & (relative > -15)
        scored = audible | ~truth | (relative < -30)
        mix = np.concatenate([score[scored], twin])
        mixed_truth = np.concatenate([audible[scored], np.zeros(len(twin), bool)])
        audible_threshold, _ = at_false_alarm_rate(mix, mixed_truth, 0.05)
        out["audible_reference"] = {
            "roc_auc_all": auc(mix, mixed_truth),
            "threshold_for_5pct_false_alarms": audible_threshold,
            "all": rates(mix, mixed_truth, audible_threshold),
        }
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("scores", nargs="+")
    parser.add_argument("--labels")
    parser.add_argument("--output")
    args = parser.parse_args()
    items = load(args.scores, args.labels)
    result = {
        label: summarise(items, n)
        for n, label in enumerate(LABELS)
        if n < items[0]["score"].shape[1]
    }
    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
