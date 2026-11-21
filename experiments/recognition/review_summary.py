"""Compare the model before and after ``finetune_review.py`` on material it was not shown,
and choose decision thresholds for the model trained on everything.

Two held-out pools, never merged in the tables:

* reviewed excerpts of the artists each ``--hold`` run kept out (core excerpts answered
  "present throughout" or "absent"), scored before and after that run's training;
* the MoisesDB artist fold the starting weights never saw, as songs and as their
  guitar-free twins; "before" is the starting model's own scores of that fold.

Scores are averaged over two hops, as the application does. Because a threshold moves
recall and false alarms together, the comparison is made at equal false-alarm rates.

Thresholds: "detected" is the score exceeded in 5% of guitar-free held-out time and
"candidate" in 20%, with the two pools given equal weight.

    python experiments/recognition/review_summary.py --review REVIEW_CACHE \
        --runs OUT_hold0 OUT_hold1 OUT_hold2 --before SCORES_FT_V1 \
        --old-thresholds .model-cache/guitar_mn10_v1.thresholds.json \
        --thresholds NEW.json --report summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from finetune_review import TYPES, load_review  # noqa: E402
from head import LABELS, auc  # noqa: E402


def smooth(scores):
    """Average of each hop and the one before it (the first hop stands alone)."""
    return (scores + np.concatenate([scores[:1], scores[:-1]])) / 2


def review_pool(items, run, which, label):
    """(positive scores, negative scores, per-excerpt rows) of one run's kept-out excerpts."""
    index = LABELS.index(label)
    pos, neg, rows = [], [], []
    with np.load(Path(run) / f"review_{which}.npz") as data:
        for item in items:
            if item["id"] not in data.files or item["category"] != "core":
                continue
            if item["presence"] == "absent":
                truth = False
            elif item["presence"] == "present_throughout" and (
                index == 0 or TYPES.get(item["type"]) == index
            ):
                truth = True
            else:
                continue
            values = smooth(data[item["id"]])[:, index]
            (pos if truth else neg).append(values)
            rows.append((item, truth, values))
    return pos, neg, rows


def moises_pool(folder, names, label):
    index = LABELS.index(label)
    pos, neg = [], []
    for name in names:
        with np.load(Path(folder) / name) as data:
            values = smooth(data["score"])[:, index]
            y, certain = data["y"][:, index], data["certain"][:, index]
        pos.append(values[certain & y])
        neg.append(values[certain & ~y])
    return pos, neg


def join(parts):
    return np.concatenate(parts) if parts else np.zeros(0)


def quantile_weighted(pools, rate):
    """Score exceeded in ``rate`` of the time, each pool counting the same."""
    values = np.concatenate(pools)
    weights = np.concatenate([np.full(len(p), 1 / (len(pools) * len(p))) for p in pools])
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, 1 - rate)])


def at(pos, neg, threshold):
    return {
        "recall": round(float((pos >= threshold).mean()), 3) if len(pos) else None,
        "false_alarms": round(float((neg >= threshold).mean()), 3) if len(neg) else None,
    }


def describe(pos, neg):
    """Threshold-free figures: area under the curve and recall at fixed false-alarm rates."""
    out = {"guitar_hops": len(pos), "guitar_free_hops": len(neg)}
    if not len(pos) or not len(neg):
        return out
    out["auc"] = round(float(auc(np.concatenate([pos, neg]), np.arange(len(pos) + len(neg)) < len(pos))), 4)
    ordered = np.sort(neg)
    for rate in (0.05, 0.10):
        threshold = ordered[min(len(neg) - 1, int(np.ceil(len(neg) * (1 - rate))))]
        out[f"recall_at_{int(rate * 100)}pct_false_alarms"] = round(
            float((pos > threshold).mean()), 3
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--review", required=True)
    parser.add_argument("--runs", nargs="+", required=True)
    parser.add_argument("--before", required=True, help="starting model's scores of its fold")
    parser.add_argument("--old-thresholds", required=True)
    parser.add_argument("--thresholds", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    items = load_review(args.review)
    for item in items:
        del item["audio"]
    old = json.loads(Path(args.old_thresholds).read_text(encoding="utf-8"))
    names = sorted(p.name for p in Path(args.runs[0]).glob("r-*.npz"))
    report, thresholds = {}, {}
    for label in LABELS:
        pools = {}
        for which in ("before", "after"):
            pos, neg, rows = [], [], []
            for run in args.runs:
                p, n, r = review_pool(items, run, which, label)
                pos, neg, rows = pos + p, neg + n, rows + r
            pools[f"reviewed {which}"] = (join(pos), join(neg), rows)
        pools["moisesdb before"] = (*map(join, moises_pool(args.before, names, label)), None)
        pos, neg = [], []
        for run in args.runs:  # three models, each trained without a third of the reviewed artists
            p, n = moises_pool(run, names, label)
            pos, neg = pos + p, neg + n
        pools["moisesdb after"] = (join(pos), join(neg), None)

        negatives = [pools["reviewed after"][1], pools["moisesdb after"][1]]
        negatives = [n for n in negatives if len(n)]
        detected = quantile_weighted(negatives, 0.05)
        candidate = min(detected, quantile_weighted(negatives, 0.20))
        thresholds[label] = {"detected": round(detected, 3), "candidate": round(candidate, 3)}
        entry = {}
        for name, (pos, neg, rows) in pools.items():
            chosen = old[label] if name.endswith("before") else thresholds[label]
            entry[name] = {
                **describe(pos, neg),
                "thresholds": chosen,
                "detected": at(pos, neg, chosen["detected"]),
                "detected_or_candidate": at(pos, neg, chosen["candidate"]),
            }
            if rows and label == "guitar":
                for key, test in (
                    ("starts_10s_or_later", lambda i: i["start"] >= 10),
                    ("starts_before_10s", lambda i: i["start"] < 10),
                ):
                    p = join([v for i, t, v in rows if t and test(i)])
                    n = join([v for i, t, v in rows if not t and test(i)])
                    entry[name][key] = {
                        "guitar_hops": len(p),
                        "guitar_free_hops": len(n),
                        **at(p, n, chosen["detected"]),
                    }
        report[label] = entry
    Path(args.thresholds).write_text(json.dumps(thresholds, indent=1), encoding="utf-8")
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report["guitar"], indent=1))
    print(json.dumps(thresholds))


if __name__ == "__main__":
    main()
