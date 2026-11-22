"""How well each audio model, alone, separates the human-labelled excerpts.

Reference: core excerpts of the review batch answered "present throughout" or "absent"
(chosen without looking at any model). Compared, each on its own: the project's recogniser
(``guitar_mn10_v1``, through the application's own ``Recognizer`` path with its level
step) and PANNs CNN14 frame-wise classes. Nothing here is fitted to the labels except the
operating points that are reported as such.

Runs in the application environment (ONNX Runtime).

    python experiments/evidence/compare_models.py --cache REVIEW_CACHE \
        --panns research-local/evidence-v1/panns/review_pilot.npz --report OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
from gmv.recognition import levelled, load_backend  # noqa: E402

TYPES = ("distorted_electric", "clean_electric", "acoustic")


def auc(pos, neg):
    """Probability that a guitar excerpt scores above a guitar-free one (ties count half)."""
    if not len(pos) or not len(neg):
        return None
    values = np.concatenate([pos, neg])
    order = values.argsort(kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    for value in np.unique(values):  # average ranks of ties
        same = values == value
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at(pos, neg, rate):
    threshold = np.sort(neg)[min(len(neg) - 1, int(np.ceil(len(neg) * (1 - rate))))]
    return float((pos > threshold).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--panns", required=True)
    parser.add_argument("--model", default=".model-cache/guitar_mn10_v1.json")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    backend = load_backend(args.model)
    size = round(backend.context_s * backend.sample_rate)
    step = round(backend.hop_s * backend.sample_rate)
    with np.load(args.panns, allow_pickle=False) as data:
        classes = [str(c) for c in data["classes"]]
        panns = {k: data[k] for k in data.files if k != "classes"}
    rows = []
    for path in sorted(Path(args.cache).glob("*.npz")):
        with np.load(path, allow_pickle=False) as data:
            audio, meta, group = data["audio"], data["meta"], str(data["group"])
        for clip, (eid, category, presence, kind, start, _length) in zip(audio, meta, strict=True):
            if category != "core" or presence not in ("present_throughout", "absent"):
                continue
            clip = clip.astype(np.float32)
            ends = range(size + step, len(clip) + 1, step)
            own = np.array([backend.infer(levelled(clip[e - size : e])) for e in ends])
            mean = panns[str(eid)][0]
            scores = {f"gmv_{label}": float(own[:, i].mean()) for i, label in enumerate(backend.labels)}
            scores.update({f"panns_{c}": float(mean[:, i].mean()) for i, c in enumerate(classes)})
            scores["panns_any_guitar"] = float(
                mean[:, [classes.index(c) for c in ("guitar", "electric_guitar", "acoustic_guitar")]]
                .max(axis=1)
                .mean()
            )
            rows.append(
                {
                    "id": str(eid),
                    "group": group,
                    "guitar": presence != "absent",
                    "type": str(kind),
                    "opening": float(start) < 10,
                    **scores,
                }
            )
    names = [k for k in rows[0] if k.startswith(("gmv_", "panns_"))]
    pos = [r for r in rows if r["guitar"]]
    neg = [r for r in rows if not r["guitar"]]
    report = {
        "excerpts": {"guitar": len(pos), "guitar_free": len(neg)},
        "artists": len({r["group"] for r in rows}),
        "separation": {},
        "by_type": {},
        "guitar_free_means": {},
    }

    def column(items, name):
        return np.array([r[name] for r in items])

    for name in names:
        a = auc(column(pos, name), column(neg, name))
        report["separation"][name] = {
            "auc": round(a, 3),
            "recall_at_5pct_false_alarms": round(recall_at(column(pos, name), column(neg, name), 0.05), 3),
            "recall_at_10pct_false_alarms": round(recall_at(column(pos, name), column(neg, name), 0.10), 3),
        }
    for kind in TYPES:
        some = [r for r in pos if r["type"] == kind]
        report["by_type"][kind] = {
            "excerpts": len(some),
            **{
                name: round(auc(column(some, name), column(neg, name)), 3)
                for name in ("gmv_guitar", "panns_guitar", "panns_any_guitar", "panns_electric_guitar", "panns_acoustic_guitar")
            },
        }
    opening = [r for r in pos if r["opening"]]
    report["openings"] = {
        "excerpts": len(opening),
        **{
            name: round(auc(column(opening, name), column(neg, name)), 3)
            for name in ("gmv_guitar", "panns_guitar", "panns_any_guitar")
        },
    }
    # Do the two models make the same mistakes? Rank correlation within each class.
    def spearman(items, a, b):
        x, y = column(items, a).argsort().argsort(), column(items, b).argsort().argsort()
        return round(float(np.corrcoef(x, y)[0, 1]), 3)

    report["rank_correlation_gmv_vs_panns"] = {
        "guitar_excerpts": spearman(pos, "gmv_guitar", "panns_guitar"),
        "guitar_free_excerpts": spearman(neg, "gmv_guitar", "panns_guitar"),
    }
    # A fixed, unfitted combination: mean of the two models' ranks.
    both = rows
    rank = lambda name: column(both, name).argsort().argsort() / (len(both) - 1)  # noqa: E731
    combined = (rank("gmv_guitar") + rank("panns_guitar")) / 2
    truth = np.array([r["guitar"] for r in both])
    report["separation"]["rank_mean_gmv_panns"] = {
        "auc": round(auc(combined[truth], combined[~truth]), 3),
        "recall_at_5pct_false_alarms": round(recall_at(combined[truth], combined[~truth], 0.05), 3),
        "recall_at_10pct_false_alarms": round(recall_at(combined[truth], combined[~truth], 0.10), 3),
    }
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    np.savez_compressed(
        Path(args.report).with_suffix(".scores.npz"),
        ids=np.array([r["id"] for r in rows]),
        groups=np.array([r["group"] for r in rows]),
        guitar=truth,
        types=np.array([r["type"] for r in rows]),
        names=np.array(names),
        scores=np.array([[r[n] for n in names] for r in rows], np.float32),
    )
    keep = ("gmv_guitar", "panns_guitar", "panns_any_guitar", "panns_electric_guitar", "panns_acoustic_guitar",
            "panns_plucked_string", "panns_piano", "panns_synthesizer", "rank_mean_gmv_panns")
    print(json.dumps({k: report["separation"][k] for k in keep}, indent=1))
    print(json.dumps({k: report[k] for k in ("excerpts", "artists", "by_type", "openings", "rank_correlation_gmv_vs_panns")}, indent=1))


if __name__ == "__main__":
    main()
