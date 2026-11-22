"""Compare the starting model with the trained arms on excerpts none of them trained on.

Reads the runs written by ``finetune_review.py``. Every excerpt is scored by the baseline
run and, for each arm, by the one run that kept the excerpt's fold out; the report says
which run that was. Before any figure is computed the runs are checked against each other:
same starting weights, same settings, same seeds per fold, the intended training data per
arm, and nothing shared between a run's training data and what it scored. If a check fails,
the report lists the problems and gives no verdict.

Reported apart, never as one figure: the primary pool (excerpts chosen without model
output), the secondary pool (chosen because of model output; it can show a regression, not
a gain), the primary pool by guitar type and by start in the first ten seconds, and the
MoisesDB control fold. Thresholds are the shipped model's own, read from its description
and applied unchanged to every arm. No threshold is chosen here.

Runs in either environment (no torch).

    python experiments/recognition/review_summary.py --review CACHE --runs RUNS \
        --model-config .model-cache/guitar_mn10_v1.json --report RUNS/summary.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import review_data as rd  # noqa: E402
import review_stats as rs  # noqa: E402

FOLDS = 5
SAME_CONFIG = (
    "batch", "learning_rate", "head_learning_rate", "weight_decay", "schedule", "review_share",
    "silenced_lead_in_share", "sampling", "loss", "hop_s", "scoring", "moisesdb_control_fold",
)  # fmt: skip


def load_run(folder):
    folder = Path(folder)
    if not (folder / "run.json").is_file():
        return None
    run = json.loads((folder / "run.json").read_text(encoding="utf-8"))
    run["folder"] = folder
    return run


def check_runs(baseline, runs, items, cache):
    """Everything that must hold before the arms may be compared. ``runs``: {(arm, fold)}."""
    problems = []
    every = [baseline, *runs.values()]
    for run in every:
        if run["status"] != "complete":
            problems.append(f"{run['folder']}: status is {run['status']}")
        if run["initial_weights"]["sha256"] != baseline["initial_weights"]["sha256"]:
            problems.append(f"{run['folder']}: different starting weights")
        if run["dataset"]["cache_manifest"]["segments_table_sha256"] != cache["segments_table_sha256"]:
            problems.append(f"{run['folder']}: made from a different segment table")
        for key in SAME_CONFIG:
            if run["config"][key] != baseline["config"][key]:
                problems.append(f"{run['folder']}: setting {key} differs")
    human = {i["id"] for i in items if i["source"] == "human"}
    automatic = {i["id"] for i in items if i["source"] in rd.AUTOMATIC and i["label"] == "present"}
    group_of = {i["id"]: i["group"] for i in items}
    steps = {run["config"]["steps"] for run in runs.values()}
    if len(steps) > 1:
        problems.append(f"training steps differ between runs: {sorted(steps)}")
    for (arm, fold), run in sorted(runs.items()):
        where = run["folder"]
        data = run["dataset"]
        if run["arm"] != arm or run["held_out_fold"] != fold:
            problems.append(f"{where}: is not arm {arm}, fold {fold}")
        if run["seeds"] != rd.seeds(fold):
            problems.append(f"{where}: seeds are not the fold's seeds")
        expected = sorted(i["id"] for i in rd.evaluation_items(items, fold))
        if data["scored_ids"] != expected:
            problems.append(f"{where}: scored excerpts are not the fold's human excerpts")
        trained = set(data["training_ids"])
        if trained & set(data["scored_ids"]):
            problems.append(f"{where}: scored an excerpt it trained on")
        if {group_of[i] for i in trained} & {group_of[i] for i in data["scored_ids"]}:
            problems.append(f"{where}: a leakage group is on both sides")
        if trained != {i["id"] for i in rd.training_items(items, arm, fold)}:
            problems.append(f"{where}: training data is not what arm {arm} should have")
        if arm == "H" and not trained <= human:
            problems.append(f"{where}: arm H trained on something a person did not confirm")
        if arm == "H+A" and not (trained - human) <= automatic:
            problems.append(f"{where}: arm H+A holds a segment that is neither human nor an automatic positive")
    for fold in range(FOLDS):
        pair = [runs.get((arm, fold)) for arm in ("H", "H+A")]
        if all(pair):
            if pair[0]["seeds"] != pair[1]["seeds"]:
                problems.append(f"fold {fold}: the arms used different seeds")
            first, second = (set(r["dataset"]["training_ids"]) for r in pair)
            if first != second & human:
                problems.append(f"fold {fold}: the arms do not share the same human segments")
    return problems


def moisesdb(folder):
    """(guitar hops, guitar-free hops, songs) of the control fold as one run scored it."""
    pos, neg, names = [], [], []
    for path in sorted((Path(folder) / "moisesdb").glob("*.npz")):
        with np.load(path) as data:
            values = rs.smooth(data["score"])[:, 0]
            y, certain = data["y"][:, 0].astype(bool), data["certain"][:, 0].astype(bool)
        pos.append(values[certain & y])
        neg.append(values[certain & ~y])
        names.append(path.name)
    return np.concatenate(pos), np.concatenate(neg), names


def rounded(value):
    if isinstance(value, float):
        return round(value, 4)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--review", required=True, help="segment cache written by cache_review.py")
    parser.add_argument("--runs", required=True)
    parser.add_argument("--model-config", required=True, help="the shipped model's description (thresholds)")
    parser.add_argument("--report", required=True)
    parser.add_argument("--replicates", type=int, default=rs.REPLICATES)
    args = parser.parse_args()
    root = Path(args.runs)
    items = rd.load_segments(args.review, audio=False)
    cache = json.loads((Path(args.review) / "manifest.json").read_text(encoding="utf-8"))
    shipped = json.loads(Path(args.model_config).read_text(encoding="utf-8"))
    thresholds = {k: float(v) for k, v in shipped["thresholds"]["guitar"].items()}
    baseline = load_run(root / "baseline")
    if baseline is None:
        raise SystemExit("No baseline run: score the starting model first (--arm baseline).")
    runs = {}
    for arm in rd.ARMS:
        for fold in range(FOLDS):
            run = load_run(root / arm / f"hold{fold}")
            if run:
                runs[(arm, fold)] = run
    arms = ["baseline"] + [arm for arm in rd.ARMS if all((arm, f) in runs for f in range(FOLDS))]
    partial = sorted({arm for arm, _ in runs} - set(arms))
    report = {
        "shipped_model": {"name": shipped["name"], "version": shipped["version"], "thresholds_guitar": thresholds},
        "arms_compared": arms,
        "arms_with_missing_runs": partial,
        "problems": check_runs(baseline, {k: v for k, v in runs.items() if k[0] in arms}, items, cache),
        "scored_by": {arm: {f"fold {f}": str(runs[(arm, f)]["folder"]) for f in range(FOLDS)} for arm in arms[1:]},
        "starting_weights": baseline["initial_weights"],
        "bootstrap": {"unit": "leakage group", "replicates": args.replicates, "seed": rs.BOOTSTRAP_SEED, "paired": True},
    }
    if report["problems"]:
        report["verdict"] = "none: the runs failed the checks above"
        rd.write_json(args.report, report)
        raise SystemExit("The runs cannot be compared:\n  " + "\n  ".join(report["problems"]))

    before = dict(np.load(baseline["folder"] / "review.npz"))
    after = {key: dict(np.load(run["folder"] / "review.npz")) for key, run in runs.items() if key[0] in arms}

    def scores_of(item, arm):
        """An excerpt's scores from the one run of ``arm`` that kept its fold out."""
        found = before if arm == "baseline" else after[(arm, item["fold"])]
        return rs.smooth(found[item["id"]])[:, 0]

    excerpts = []
    for item in rd.evaluation_items(items):
        excerpts.append(
            {
                "group": item["group"],
                "guitar": item["label"] == "present",
                "pool": item["pool"],
                "selection": item["selection"],
                "type": item["type"],
                "early": item["start"] < 10.0,
                "scores": {arm: scores_of(item, arm) for arm in arms},
            }
        )

    def pool(test):
        chosen = [e for e in excerpts if test(e)]
        return rs.compare(chosen, arms, thresholds, args.replicates)

    primary = pool(lambda e: e["pool"] == "primary")
    secondary = pool(lambda e: e["pool"] == "secondary")
    report["primary_pool"] = primary
    report["secondary_pool"] = secondary
    report["secondary_pool_by_selection"] = {
        reason: pool(lambda e, reason=reason: e["selection"] == reason)
        for reason in sorted({e["selection"] for e in excerpts if e["pool"] == "secondary"})
    }
    report["primary_pool_by_guitar_type"] = {
        kind: pool(lambda e, kind=kind: e["pool"] == "primary" and (not e["guitar"] or e["type"] == kind))
        for kind in rd.TYPES
    }
    report["primary_pool_by_start"] = {
        "first_10_s": pool(lambda e: e["pool"] == "primary" and e["early"]),
        "later": pool(lambda e: e["pool"] == "primary" and not e["early"]),
    }
    control = {}
    pos, neg, names = moisesdb(baseline["folder"])
    control["baseline"] = {**rs.figures(pos, neg, thresholds), "songs_and_twins": len(names)}
    for arm in arms[1:]:
        each = []
        for fold in range(FOLDS):
            p, n, found = moisesdb(runs[(arm, fold)]["folder"])
            if found != names:
                raise SystemExit(f"{arm} fold {fold}: the control fold was scored on different songs")
            each.append(rs.figures(p, n, thresholds))
        control[arm] = {
            "auc_mean": float(np.mean([e["auc"] for e in each])),
            "auc_min": float(min(e["auc"] for e in each)),
            "per_run": each,
        }
    report["moisesdb_control"] = control
    report["verdicts"] = {
        arm: rs.verdict(primary, secondary, control, arm) for arm in arms[1:]
    }
    report["note"] = (
        "The pre-registered decision is the H+A verdict. H is reported to show what the "
        "human segments alone do. The secondary pool was selected by the baseline model's "
        "own output and cannot show a gain."
    )
    rd.write_json(args.report, rounded(report))
    brief = {
        "arms": arms,
        "primary": rounded(primary["arms"]),
        "primary_counts": primary["counts"],
        "secondary": rounded(secondary["arms"]),
        "moisesdb_control": rounded({k: {a: b for a, b in v.items() if a != "per_run"} for k, v in control.items()}),
        "verdicts": report["verdicts"],
    }
    print(json.dumps(brief, indent=1))


if __name__ == "__main__":
    main()
