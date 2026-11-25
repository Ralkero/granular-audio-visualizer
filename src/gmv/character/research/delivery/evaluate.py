"""Leave-one-artist-group-out validation, metrics and abstention.

Every fold holds out all the excerpts of one artist group; the model is fitted on the
others, with preprocessing and class weights learned from those others only, and predicts
the held-out group. Each excerpt is predicted once, by a model that never saw its artist
group (nor, therefore, its recording). These are the development side's out-of-fold
predictions. Comparing models on them is **exploratory model selection**: an untouched
final test needs the held-out evaluation artists, which this experiment does not use.
"""

import numpy as np

from . import models

BOOTSTRAP = 2000
COVERAGES = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5)
TARGET_COVERAGE = 0.8  # fixed in advance for the abstention rule


class LeakageError(AssertionError):
    """A fold broke group separation."""


def folds(groups):
    """(held-out group, training rows, test rows) for each group, in sorted order."""
    groups = np.asarray(groups)
    return [(g, np.flatnonzero(groups != g), np.flatnonzero(groups == g)) for g in sorted(set(groups.tolist()))]


def check_fold(examples, train, test):
    """No artist group, artist or recording may be on both sides of a fold."""
    for key in ("group", "artist", "recording"):
        inside = {examples[i][key] for i in train}
        outside = {examples[i][key] for i in test}
        if inside & outside:
            raise LeakageError(f"a fold has the same {key} in training and test data")


def cross_validate(X, y, examples, name, seed, classes=3, fitted=None):
    """Out-of-fold probabilities (rows, classes) of one model kind, and per-fold facts.
    ``fitted`` (optional list) receives (group, preprocess, model, weights) per fold."""
    proba = np.full((len(y), classes), np.nan)
    facts = []
    for number, (group, train, test) in enumerate(folds([e["group"] for e in examples])):
        check_fold(examples, train, test)
        prepare = models.Preprocess().fit(X[train])
        weights = models.class_weights(y[train], classes)
        model = models.make(name, [seed, number]).fit(prepare.transform(X[train]), y[train], weights, classes)
        proba[test] = model.predict_proba(prepare.transform(X[test]))
        facts.append({
            "group": group, "train": len(train), "test": len(test),
            "train_counts": np.bincount(y[train], minlength=classes).tolist(),
            "converged": getattr(model, "converged", True),
        })
        if fitted is not None:
            fitted.append((group, prepare, model, weights))
    return proba, facts


def predict(proba):
    """The predicted answer: highest probability, ties to the lower index."""
    return np.argmax(proba, axis=1)


def metrics(y, prediction, classes=3):
    """Confusion matrix (rows: true, columns: predicted), per-class precision, recall and
    F1, balanced accuracy (mean recall), macro F1 and plain accuracy."""
    y, prediction = np.asarray(y), np.asarray(prediction)
    confusion = np.zeros((classes, classes), dtype=int)
    np.add.at(confusion, (y, prediction), 1)
    true = confusion.sum(axis=1)
    predicted = confusion.sum(axis=0)
    hits = np.diag(confusion)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(true > 0, hits / true, np.nan)
        precision = np.where(predicted > 0, hits / predicted, 0.0)
        f1 = np.where(precision + np.nan_to_num(recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    present = true > 0
    return {
        "confusion": confusion.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "balanced_accuracy": float(np.nanmean(recall[present])) if present.any() else float("nan"),
        "macro_f1": float(np.mean(f1[present])) if present.any() else float("nan"),
        "accuracy": float(hits.sum() / max(1, len(y))),
        "examples": int(len(y)),
    }


def per_group(examples, y, prediction, classes=3):
    """Per held-out artist group: examples, answers present, accuracy, and recall of each
    answer it has. Private: groups are named."""
    out = {}
    groups = np.array([e["group"] for e in examples])
    for group in sorted(set(groups.tolist())):
        rows = groups == group
        recall = {}
        for c in range(classes):
            mine = rows & (y == c)
            if mine.any():
                recall[c] = float((prediction[mine] == c).mean())
        out[group] = {
            "examples": int(rows.sum()),
            "counts": np.bincount(y[rows], minlength=classes).tolist(),
            "accuracy": float((prediction[rows] == y[rows]).mean()),
            "recall": recall,
        }
    return out


def consistency(per, classes=3):
    """How evenly the model does across unseen artist groups. Counts and spreads only."""
    accuracy = np.array([p["accuracy"] for p in per.values()])
    out = {
        "groups": len(per),
        "accuracy_median": float(np.median(accuracy)),
        "accuracy_quartiles": [float(np.percentile(accuracy, 25)), float(np.percentile(accuracy, 75))],
        "accuracy_min": float(accuracy.min()),
        "accuracy_max": float(accuracy.max()),
        "groups_at_least_half_right": int((accuracy >= 0.5).sum()),
        "by_answer": {},
    }
    for c in range(classes):
        recalls = np.array([p["recall"][c] for p in per.values() if c in p["recall"]])
        if len(recalls):
            out["by_answer"][c] = {
                "groups": int(len(recalls)),
                "recall_median": float(np.median(recalls)),
                "recall_min": float(recalls.min()),
                "recall_max": float(recalls.max()),
                "groups_with_recall_at_least_half": int((recalls >= 0.5).sum()),
                "groups_with_no_example_right": int((recalls == 0).sum()),
            }
    return out


def bootstrap(examples, y, prediction, seed, rounds=BOOTSTRAP, classes=3):
    """95% intervals for balanced accuracy and macro F1, resampling whole artist groups
    with replacement (the unit that varies between unseen artists)."""
    groups = np.array([e["group"] for e in examples])
    names = sorted(set(groups.tolist()))
    rows = {g: np.flatnonzero(groups == g) for g in names}
    rng = np.random.default_rng(seed)
    balanced, macro = [], []
    for _round in range(rounds):
        picked = np.concatenate([rows[names[i]] for i in rng.integers(0, len(names), len(names))])
        found = metrics(y[picked], prediction[picked], classes)
        balanced.append(found["balanced_accuracy"])
        macro.append(found["macro_f1"])
    return {
        "balanced_accuracy": [float(np.nanpercentile(balanced, 2.5)), float(np.nanpercentile(balanced, 97.5))],
        "macro_f1": [float(np.nanpercentile(macro, 2.5)), float(np.nanpercentile(macro, 97.5))],
        "rounds": rounds,
    }


def collapsed(y, prediction):
    """Two readings of the same three-way predictions: clean against harsh (High and Low
    together), and, among harsh excerpts predicted harsh, high against low."""
    harsh_true, harsh_said = y > 0, prediction > 0
    out = {"clean_vs_harsh": metrics(harsh_true.astype(int), harsh_said.astype(int), 2)}
    both = harsh_true & harsh_said
    out["high_vs_low_among_harsh"] = metrics((y[both] == 2).astype(int), (prediction[both] == 2).astype(int), 2)
    return out


def within(examples, y, prediction):
    """The same predictions where artist and production are held constant. Clean against
    harsh, counted only in artist groups (and recordings) that have both: if the model only
    told a metal mix from another kind of mix, it would fail here. High against low, among
    harsh excerpts predicted harsh, in artist groups that have both."""
    out = {}
    for key in ("group", "recording"):
        units = np.array([e[key] for e in examples])
        mixed = [u for u in sorted(set(units.tolist())) if (y[units == u] == 0).any() and (y[units == u] > 0).any()]
        chosen = np.isin(units, mixed)
        out[f"clean_vs_harsh_within_{key}s"] = {
            "units": len(mixed),
            **metrics((y[chosen] > 0).astype(int), (prediction[chosen] > 0).astype(int), 2),
        }
    groups = np.array([e["group"] for e in examples])
    names = sorted(set(groups.tolist()))
    mixed = [u for u in names if (y[groups == u] == 1).any() and (y[groups == u] == 2).any()]
    chosen = np.isin(groups, mixed) & (y > 0) & (prediction > 0)
    out["high_vs_low_within_groups"] = {
        "units": len(mixed),
        **metrics((y[chosen] == 2).astype(int), (prediction[chosen] == 2).astype(int), 2),
    }
    out["groups_by_content"] = {
        "clean_only": sum(1 for u in names if (y[groups == u] == 0).all()),
        "harsh_only": sum(1 for u in names if (y[groups == u] > 0).all()),
        "both": sum(1 for u in names if (y[groups == u] == 0).any() and (y[groups == u] > 0).any()),
    }
    return out


def abstention(y, proba, classes=3, coverages=COVERAGES, target=TARGET_COVERAGE):
    """What happens when the least confident predictions are withheld. Confidence is the
    highest class probability. The rule's threshold is the confidence that keeps the
    ``target`` share of the development out-of-fold predictions: chosen and measured on the
    same development predictions, so it is optimistic."""
    confidence = proba.max(axis=1)
    prediction = predict(proba)
    order = np.argsort(-confidence, kind="stable")
    curve = []
    for coverage in coverages:
        keep = np.zeros(len(y), dtype=bool)
        keep[order[: int(round(coverage * len(y)))]] = True
        found = metrics(y[keep], prediction[keep], classes)
        withheld = np.bincount(y[~keep], minlength=classes)
        totals = np.bincount(y, minlength=classes)
        curve.append({
            "coverage": coverage,
            "threshold": float(confidence[keep].min()) if keep.any() else None,
            "kept": int(keep.sum()),
            "accuracy": found["accuracy"],
            "balanced_accuracy": found["balanced_accuracy"],
            "macro_f1": found["macro_f1"],
            "recall": found["recall"],
            "withheld_share_by_answer": (withheld / np.maximum(totals, 1)).tolist(),
        })
    threshold = float(np.quantile(confidence, 1.0 - target))
    decided = confidence >= threshold
    rule = metrics(y[decided], prediction[decided], classes)
    return {
        "curve": curve,
        "rule": {
            "target_coverage": target,
            "threshold": threshold,
            "coverage": float(decided.mean()),
            "accuracy": rule["accuracy"],
            "balanced_accuracy": rule["balanced_accuracy"],
            "macro_f1": rule["macro_f1"],
            "recall": rule["recall"],
            "abstained_share_by_answer": [
                float((~decided & (y == c)).sum() / max(1, (y == c).sum())) for c in range(classes)
            ],
        },
    }


def outcome(proba, threshold):
    """Per excerpt: ("predicted", answer index) or ("abstained", None) under a threshold.
    An abstention means "not confident among the three supported answers", never "a
    different technique"."""
    confidence = proba.max(axis=1)
    return [("predicted", int(a)) if c >= threshold else ("abstained", None)
            for a, c in zip(predict(proba), confidence, strict=True)]
