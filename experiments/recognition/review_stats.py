"""Figures and decisions for the comparison of the starting model with the trained arms.

No torch. Everything works on hop scores of excerpts that the scoring model never trained
on, grouped by leakage group: groups are the unit of leakage protection and therefore the
unit that is resampled. A replicate draws groups once and uses the same draw for every arm,
so differences between arms are paired.

Thresholds are given from outside (the shipped model's own). Nothing here chooses one.
"""

import numpy as np

REPLICATES = 2000
BOOTSTRAP_SEED = 20261122
MIN_GROUPS = 5  # per class; with fewer, an interval over groups says nothing
# Fixed before any run. A rise in false positives at the shipped "detected" threshold on
# the primary pool counts as meaningful if its interval lies above zero, or if the rise is
# larger than this share of guitar-free time whatever the interval says.
FALSE_POSITIVE_RISE = 0.02
MOISESDB_MARGIN = 0.01


def smooth(scores):
    """Average of each hop and the one before it, as the application does."""
    scores = np.asarray(scores, np.float64)
    return (scores + np.concatenate([scores[:1], scores[:-1]])) / 2


def auc(pos, neg):
    """Probability that a guitar hop scores above a guitar-free hop (ties count half)."""
    if not len(pos) or not len(neg):
        return None
    values = np.concatenate([pos, neg])
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    ranks = (np.bincount(inverse, ranks) / counts)[inverse]
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def recall_at_false_alarms(pos, neg, rate):
    """Recall at the threshold that ``rate`` of guitar-free hops exceed. The threshold is a
    property of the same scores; this is a ranking figure, not an operating point."""
    if not len(pos) or not len(neg):
        return None
    ordered = np.sort(neg)
    threshold = ordered[min(len(neg) - 1, int(np.ceil(len(neg) * (1 - rate))))]
    return float((pos > threshold).mean())


def at_threshold(pos, neg, threshold):
    """Recall and false-positive rate at a threshold fixed elsewhere."""
    return (
        float((pos >= threshold).mean()) if len(pos) else None,
        float((neg >= threshold).mean()) if len(neg) else None,
    )


def figures(pos, neg, thresholds):
    out = {
        "auc": auc(pos, neg),
        "recall_at_5pct_false_alarms": recall_at_false_alarms(pos, neg, 0.05),
        "recall_at_10pct_false_alarms": recall_at_false_alarms(pos, neg, 0.10),
    }
    for name, value in thresholds.items():
        out[f"recall_at_{name}"], out[f"false_positive_rate_at_{name}"] = at_threshold(pos, neg, value)
    return out


def _join(parts):
    return np.concatenate(parts) if parts else np.zeros(0)


def by_group(excerpts, arm):
    """{group: (guitar hops, guitar-free hops)} of one arm's scores."""
    out = {}
    for excerpt in excerpts:
        entry = out.setdefault(excerpt["group"], ([], []))
        entry[0 if excerpt["guitar"] else 1].append(excerpt["scores"][arm])
    return {g: (_join(p), _join(n)) for g, (p, n) in out.items()}


def compare(excerpts, arms, thresholds, replicates=REPLICATES, seed=BOOTSTRAP_SEED):
    """Figures per arm, and paired changes between arms with 95% intervals over groups.

    ``excerpts``: [{group, guitar: bool, scores: {arm: hop scores}}]. ``arms``: names, the
    first being the reference every other arm is compared with.
    """
    grouped = {arm: by_group(excerpts, arm) for arm in arms}
    groups = sorted(grouped[arms[0]])
    counts = {
        "excerpts_guitar": sum(e["guitar"] for e in excerpts),
        "excerpts_guitar_free": sum(not e["guitar"] for e in excerpts),
        "groups": len(groups),
        "groups_with_guitar": sum(len(grouped[arms[0]][g][0]) > 0 for g in groups),
        "groups_with_guitar_free": sum(len(grouped[arms[0]][g][1]) > 0 for g in groups),
        "hops_guitar": int(sum(len(grouped[arms[0]][g][0]) for g in groups)),
        "hops_guitar_free": int(sum(len(grouped[arms[0]][g][1]) for g in groups)),
    }

    def measure(chosen):
        return {
            arm: figures(
                _join([grouped[arm][g][0] for g in chosen]),
                _join([grouped[arm][g][1] for g in chosen]),
                thresholds,
            )
            for arm in arms
        }

    out = {"counts": counts, "arms": measure(groups), "changes": {}}
    enough = min(counts["groups_with_guitar"], counts["groups_with_guitar_free"]) >= MIN_GROUPS
    out["interval_possible"] = enough
    pairs = [(arm, arms[0]) for arm in arms[1:]] + ([(arms[2], arms[1])] if len(arms) > 2 else [])
    keys = list(out["arms"][arms[0]])
    samples = {pair: {k: [] for k in keys} for pair in pairs}
    if enough:
        rng = np.random.default_rng(seed)
        for _ in range(replicates):
            drawn = measure([groups[i] for i in rng.integers(len(groups), size=len(groups))])
            for a, b in pairs:
                for k in keys:
                    if drawn[a][k] is not None and drawn[b][k] is not None:
                        samples[(a, b)][k].append(drawn[a][k] - drawn[b][k])
    for a, b in pairs:
        entry = {}
        for k in keys:
            first, second = out["arms"][a][k], out["arms"][b][k]
            change = None if first is None or second is None else first - second
            values = samples[(a, b)][k]
            usable = enough and len(values) >= 0.9 * replicates
            entry[k] = {
                "change": change,
                "interval_95": [float(v) for v in np.quantile(values, [0.025, 0.975])] if usable else None,
            }
        out["changes"][f"{a} minus {b}"] = entry
    return out


def verdict(primary, secondary, moisesdb, arm="H+A", reference="baseline"):
    """Each criterion as pass, fail or cannot be evaluated. Fixed before any run; a criterion
    that cannot be evaluated is never counted as passed."""
    key = f"{arm} minus {reference}"

    def change(pool, figure):
        return pool["changes"][key][figure]

    out = {}
    item = change(primary, "auc")
    if item["interval_95"] is None:
        out["auc_gain_on_primary_pool"] = ("cannot be evaluated", "too few groups for an interval")
    else:
        out["auc_gain_on_primary_pool"] = (
            "pass" if item["interval_95"][0] > 0 else "fail",
            f"change {item['change']:+.4f}, 95% interval {item['interval_95'][0]:+.4f} to {item['interval_95'][1]:+.4f}; must lie above zero",
        )
    item = change(primary, "recall_at_5pct_false_alarms")
    out["recall_at_5pct_false_alarms_on_primary_pool"] = (
        ("cannot be evaluated", "no figure")
        if item["change"] is None
        else ("pass" if item["change"] >= 0 else "fail", f"change {item['change']:+.4f}; must not decline")
    )
    if moisesdb is None or moisesdb.get(arm) is None:
        out["moisesdb_control"] = ("cannot be evaluated", "control scores missing")
    else:
        loss = moisesdb[reference]["auc"] - moisesdb[arm]["auc_mean"]
        out["moisesdb_control"] = (
            "pass" if loss <= MOISESDB_MARGIN else "fail",
            f"AUC {moisesdb[reference]['auc']:.4f} before, {moisesdb[arm]['auc_mean']:.4f} after (mean of the runs; lowest {moisesdb[arm]['auc_min']:.4f}); may lose at most {MOISESDB_MARGIN}",
        )
    item = change(secondary, "auc")
    if item["interval_95"] is None:
        out["no_regression_on_model_selected_pool"] = ("cannot be evaluated", "too few groups for an interval")
    else:
        out["no_regression_on_model_selected_pool"] = (
            "fail" if item["interval_95"][1] < 0 else "pass",
            f"change {item['change']:+.4f}, 95% interval {item['interval_95'][0]:+.4f} to {item['interval_95'][1]:+.4f}; fails if the whole interval is below zero. A pass here is not evidence of a gain",
        )
    item = change(primary, "false_positive_rate_at_detected")
    if item["change"] is None or item["interval_95"] is None:
        out["false_positives_at_shipped_detected_threshold"] = ("cannot be evaluated", "too few groups with guitar-free excerpts")
    else:
        risen = item["interval_95"][0] > 0 or item["change"] > FALSE_POSITIVE_RISE
        out["false_positives_at_shipped_detected_threshold"] = (
            "fail" if risen else "pass",
            f"change {item['change']:+.4f}, 95% interval {item['interval_95'][0]:+.4f} to {item['interval_95'][1]:+.4f}; fails if the interval lies above zero or the rise exceeds {FALSE_POSITIVE_RISE}",
        )
    states = [state for state, _ in out.values()]
    overall = "fail" if "fail" in states else "cannot be decided" if "cannot be evaluated" in states else "pass"
    flags = []
    for figure, worse in (("recall_at_detected", -1), ("recall_at_candidate", -1), ("false_positive_rate_at_candidate", 1)):
        item = change(primary, figure)
        if item["interval_95"] and (item["interval_95"][1] < 0 if worse < 0 else item["interval_95"][0] > 0):
            flags.append(f"{figure} on the primary pool moved the wrong way: {item['change']:+.4f}, interval {item['interval_95'][0]:+.4f} to {item['interval_95'][1]:+.4f}")
    return {
        "arm": arm,
        "criteria": {k: {"result": state, "detail": detail} for k, (state, detail) in out.items()},
        "overall": overall,
        "fixed_threshold_flags": flags,
    }
