"""What the separated guitar stem's level is worth, measured on human-labelled excerpts of
the pilot recordings, next to the two audio models on the same excerpts.

Per excerpt: the median over its hops of the guitar stem's level relative to the mix (dB),
the recogniser's mean guitar score and PANNs' mean guitar score. Reported: how well each
separates "present throughout" from "absent", and whether the stem level still separates
them among the excerpts the two models leave undecided, which is the only place a third
source can add anything.

    python experiments/evidence/separation_eval.py --workdir research-local/evidence-v1 \
        --review review-local/pilot-v1 --report OUT.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).parent))
from compare_models import auc  # noqa: E402

from gmv.evidence import fusion  # noqa: E402

HOP = fusion.HOP_S


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--review", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    work = Path(args.workdir)
    human = fusion.human_intervals(args.review)
    gmv_dir = sorted(p for p in (work / "timelines").glob("*") if p.is_dir())[0]
    rows = []
    for path in sorted((work / "separation").glob("*.npz")):
        rid = path.stem
        stem = fusion.load_npz(path)
        gmv = fusion.load_npz(gmv_dir / f"{rid}.npz")
        panns = fusion.load_npz(work / "panns" / "timelines" / f"{rid}.npz")
        if gmv is None or panns is None:
            continue
        names = [str(c) for c in panns["classes"]]
        classes = fusion.hop_classes(gmv["guitar"], panns["mean"][:, names.index("guitar")], gmv["level_dbfs"])
        for start, end, presence, kind in human.get(rid, []):
            if presence not in ("present_throughout", "absent"):
                continue
            a, b = round(start / HOP), round(end / HOP)
            piece = stem["guitar_db"][a:b]
            if not np.isfinite(piece).any():
                continue
            decided = classes[a:b]
            rows.append(
                {
                    "recording": rid,
                    "guitar": presence == "present_throughout",
                    "type": kind,
                    "stem_db": float(np.nanmedian(piece)),
                    "gmv": float(gmv["guitar"][a:b].mean()),
                    "panns": float(panns["mean"][a:b, names.index("guitar")].mean()),
                    "models_decided": bool((np.abs(decided) == 2).mean() >= 0.5),
                }
            )

    def column(items, key):
        return np.array([r[key] for r in items], float)

    pos = [r for r in rows if r["guitar"]]
    neg = [r for r in rows if not r["guitar"]]
    report = {
        "recordings": len({r["recording"] for r in rows}),
        "excerpts": {"guitar": len(pos), "guitar_free": len(neg)},
        "auc": {k: round(auc(column(pos, k), column(neg, k)), 3) for k in ("stem_db", "gmv", "panns")},
        "stem_db_quartiles": {
            "guitar": [round(float(v), 1) for v in np.quantile(column(pos, "stem_db"), [0.1, 0.25, 0.5, 0.75, 0.9])],
            "guitar_free": [round(float(v), 1) for v in np.quantile(column(neg, "stem_db"), [0.1, 0.25, 0.5, 0.75, 0.9])],
        },
        "by_type_auc_stem_db": {},
        "cut_offs": {},
    }
    for kind in ("distorted_electric", "clean_electric", "acoustic"):
        some = [r for r in pos if r["type"] == kind]
        if len(some) >= 5:
            report["by_type_auc_stem_db"][kind] = {
                "excerpts": len(some),
                "auc": round(auc(column(some, "stem_db"), column(neg, "stem_db")), 3),
            }
    stem_pos, stem_neg = column(pos, "stem_db"), column(neg, "stem_db")
    for cut in (-10, -15, -20):
        said = int((stem_pos >= cut).sum() + (stem_neg >= cut).sum())
        report["cut_offs"][f"present_if_at_least_{cut}"] = {
            "excerpts_selected": said,
            "precision": round(float((stem_pos >= cut).sum() / said), 3) if said else None,
            "coverage_of_guitar": round(float((stem_pos >= cut).mean()), 3),
        }
    for cut in (-30, -40, -50):
        said = int((stem_pos <= cut).sum() + (stem_neg <= cut).sum())
        report["cut_offs"][f"absent_if_at_most_{cut}"] = {
            "excerpts_selected": said,
            "precision": round(float((stem_neg <= cut).sum() / said), 3) if said else None,
            "coverage_of_guitar_free": round(float((stem_neg <= cut).mean()), 3),
        }
    open_pos = [r for r in pos if not r["models_decided"]]
    open_neg = [r for r in neg if not r["models_decided"]]
    report["where_the_two_models_are_undecided"] = {
        "excerpts": {"guitar": len(open_pos), "guitar_free": len(open_neg)},
        "auc_stem_db": round(auc(column(open_pos, "stem_db"), column(open_neg, "stem_db")), 3)
        if open_pos and open_neg
        else None,
        "auc_gmv": round(auc(column(open_pos, "gmv"), column(open_neg, "gmv")), 3)
        if open_pos and open_neg
        else None,
    }
    # Does the stem level add to the recogniser? Rank-average of the two, unfitted.
    everything = pos + neg
    truth = np.array([r["guitar"] for r in everything])

    def rank(key):
        return column(everything, key).argsort().argsort() / (len(everything) - 1)

    for name, keys in (("gmv+stem", ("gmv", "stem_db")), ("gmv+panns", ("gmv", "panns")), ("all three", ("gmv", "panns", "stem_db"))):
        combined = sum(rank(k) for k in keys) / len(keys)
        report["auc"][f"rank mean {name}"] = round(auc(combined[truth], combined[~truth]), 3)
    Path(args.report).write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
