"""Step 4, S4.5: ROC-AUC and PR-AUC (protocol §11; Amendment 3, A3.5 B20) *[tested before use]*.
NumPy and exact fractions. Built and tested on generated scores only.

* **Score:** each arm's per-frame voice probability after the silence gate (quiet frames scored
  0) and before hysteresis; C's is computed in float64 from its logit (``ctrain.probabilities``).
* **Frames:** the evaluation mask (inside the excerpt, outside the collars and uncertain time),
  quiet frames included.
* **ROC-AUC:** ranked by value, ties counting half: (Σ ranks of voiced frames − n₁(n₁+1)/2) /
  (n₁·n₀) with mid-ranks, in exact fractions.
* **PR-AUC (average precision):** AP = Σ (R_k − R_{k−1})·P_k over the distinct scores from the
  highest down, tied frames entering together (one step), no interpolation.
* **Eligibility:** a group with at least 100 evaluated frames (2 s) of each class. Its value comes
  from the one model that predicted it.
* **Macro:** over the eligible groups per seed index, then the mean over O. B0 and B0ₜ share one
  value (no threshold is involved). No pooled AUC.
* **Public:** the macro values, the eligible-group count and the mean voiced share over eligible
  groups. **Private:** every group's value and prevalence.
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import labels as labelling
from . import metrics


def frames(scored):
    """(scores, labels) on one excerpt's evaluated frames, quiet frames scored 0."""
    keep = labelling.evaluation_mask(scored.centres, scored.duration, scored.confirmed, scored.uncertain)
    voice = labelling.frame_labels(scored.centres, scored.confirmed, scored.duration) == 1
    scores = np.where(np.asarray(scored.silent, dtype=bool), 0.0, np.asarray(scored.scores, dtype=np.float64))
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    return scores[keep], voice[keep]


def roc_auc(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    n1, n0 = int(labels.sum()), int((~labels).sum())
    if not n1 or not n0:
        raise ValueError("ROC-AUC needs both classes")
    order = np.argsort(scores, kind="stable")
    sorted_scores = scores[order]
    twice_ranks = np.empty(len(scores), dtype=np.int64)  # 2 × mid-rank, an integer
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        twice_ranks[order[i : j + 1]] = (i + 1) + (j + 1)
        i = j + 1
    twice_sum = int(twice_ranks[labels].sum())
    return Fraction(twice_sum - n1 * (n1 + 1), 2 * n1 * n0)


def average_precision(scores, labels):
    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels, dtype=bool)
    n1 = int(labels.sum())
    if not n1:
        raise ValueError("average precision needs voiced frames")
    ap, tp, fp, previous_recall = Fraction(0), 0, 0, Fraction(0)
    for value in np.unique(scores)[::-1]:
        at = scores == value
        tp += int((at & labels).sum())
        fp += int((at & ~labels).sum())
        recall = Fraction(tp, n1)
        ap += (recall - previous_recall) * Fraction(tp, tp + fp)
        previous_recall = recall
    return ap


@dataclass(frozen=True)
class GroupAuc:
    roc: Fraction
    ap: Fraction
    prevalence: Fraction  # the voiced share of the group's evaluated frames (private)


def group_values(scored):
    """{group: GroupAuc} for the eligible groups of one model's (or one aggregate's) predictions."""
    by_group = {}
    for s in scored:
        sc, lab = frames(s)
        a, b = by_group.get(s.group, (np.zeros(0), np.zeros(0, dtype=bool)))
        by_group[s.group] = (np.concatenate([a, sc]), np.concatenate([b, lab]))
    out = {}
    for g, (sc, lab) in sorted(by_group.items()):
        n1, n0 = int(lab.sum()), int((~lab).sum())
        if n1 >= metrics.MIN_CLASS_FRAMES and n0 >= metrics.MIN_CLASS_FRAMES:
            out[g] = GroupAuc(roc_auc(sc, lab), average_precision(sc, lab), Fraction(n1, n1 + n0))
    return out


@dataclass(frozen=True)
class MacroAuc:
    roc: float
    ap: float
    eligible_groups: int
    mean_voiced_share: float
    per_seed: dict  # {seed index: (roc macro, ap macro)}, for the private record


def macro(per_seed):
    """``per_seed``: {seed index: {group: GroupAuc}} (for B0 and B0ₜ, a single entry)."""
    if not per_seed:
        raise ValueError("no predictions")
    eligible = {s: sorted(v) for s, v in per_seed.items()}
    first = next(iter(eligible.values()))
    if any(e != first for e in eligible.values()):
        raise ValueError("eligibility depends on labels only, so it must agree across seed indices")
    if not first:
        raise ValueError("no group is eligible for AUC")
    seeds = {s: (sum((v[g].roc for g in first), Fraction(0)) / len(first),
                 sum((v[g].ap for g in first), Fraction(0)) / len(first)) for s, v in sorted(per_seed.items())}
    share = sum((next(iter(per_seed.values()))[g].prevalence for g in first), Fraction(0)) / len(first)
    return MacroAuc(float(sum(r for r, _a in seeds.values()) / len(seeds)),
                    float(sum(a for _r, a in seeds.values()) / len(seeds)), len(first), float(share),
                    {s: (float(r), float(a)) for s, (r, a) in seeds.items()})


def public(result):
    """The figures A3.5 allows in public (each still through the disclosure check)."""
    return {"roc_auc_macro": result.roc, "pr_auc_macro": result.ap, "eligible_groups": result.eligible_groups,
            "mean_voiced_share": result.mean_voiced_share}
