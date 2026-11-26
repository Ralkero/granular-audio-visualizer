"""The low-denominator rule and the screening classes (protocol §12.1, §13, Amendment 1).

Inputs are point estimates for a candidate (C, or B1) and the tuned baseline B0ₜ:
group-averaged false vocal activity FPA (s/min) and group-averaged recall R. Then
D_FA = FPA(candidate) − FPA(B0ₜ) and Δ_R = R(candidate) − R(B0ₜ).

* **FPA(B0ₜ) ≥ 3 s/min ("relative" mode, §13).** The ratio FPA(candidate)/FPA(B0ₜ) is
  computed as it is, with no constant added. *Not promising*: D_FA ≥ 0 or Δ_R < −0.05.
  *Promising*: ratio ≤ 0.50, Δ_R ≥ −0.05, and robust. *Mixed*: otherwise (lower false
  activity without meeting "promising").
* **FPA(B0ₜ) < 3 s/min, including zero ("absolute" mode, Amendment 1, A1.1).** No ratio is
  computed and the 50% criterion is *not applicable*. Rows, first match wins:
  1. recall below margin (Δ_R < −0.05);
  2. worsened (D_FA > 0);
  3. improved (D_FA < 0, FPA(candidate) ≤ 3 s/min, Δ_R ≥ −0.05, robust);
  4. maintained (D_FA = 0): not an improvement, retain B0;
  5. mixed.

**Robust** means D_FA < 0 and Δ_R + 0.05 ≥ 0 in every supplied recomputation: the 5
single-seed models and every leave-one-group-out recomputation. With none supplied,
robustness cannot be shown, so it is not met.
"""

import math
from dataclasses import dataclass

READINESS_S_PER_MIN = 3.0
RATIO_TARGET = 0.50
RECALL_MARGIN = 0.05
EPSILON = 1e-12  # float tolerance for the registered boundaries (≤ 0.50, ≥ −0.05)


@dataclass(frozen=True)
class Screening:
    mode: str  # "relative" or "absolute"
    fpa_candidate: float
    fpa_baseline: float
    d_fa: float
    ratio: float | None
    fifty_percent: str  # "met", "not met" or "not applicable"
    delta_recall: float
    robust: bool
    result: str  # "promising", "mixed" or "not promising"
    reason: str


def _number(name, value, low=None, high=None):
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def robust(recomputations):
    rows = list(recomputations)
    return bool(rows) and all(d_fa < 0 and d_recall + RECALL_MARGIN >= -EPSILON for d_fa, d_recall in rows)


def screen(fpa_candidate, fpa_baseline, recall_candidate, recall_baseline, recomputations=()):
    """Classify a candidate against B0ₜ. ``recomputations``: (D_FA, Δ_R) pairs."""
    fpa_c = _number("FPA(candidate)", fpa_candidate, low=0.0)
    fpa_b = _number("FPA(baseline)", fpa_baseline, low=0.0)
    d_recall = _number("recall(candidate)", recall_candidate, 0.0, 1.0) - _number("recall(baseline)", recall_baseline, 0.0, 1.0)
    d_fa = fpa_c - fpa_b
    is_robust = robust(recomputations)
    recall_ok = d_recall + RECALL_MARGIN >= -EPSILON
    if fpa_b >= READINESS_S_PER_MIN:
        ratio = fpa_c / fpa_b
        fifty = "met" if ratio <= RATIO_TARGET + EPSILON else "not met"
        if d_fa >= 0:
            result, reason = "not promising", "false activity is not lower than the baseline's"
        elif not recall_ok:
            result, reason = "not promising", "recall is more than 5 points below the baseline's"
        elif fifty == "met" and is_robust:
            result, reason = "promising", "at least a 50% reduction, recall within the margin, robust"
        else:
            result, reason = "mixed", "lower false activity without meeting every promising condition"
        return Screening("relative", fpa_c, fpa_b, d_fa, ratio, fifty, d_recall, is_robust, result, reason)
    # Amendment 1, A1.1: the baseline already meets the readiness target.
    if not recall_ok:
        result, reason = "not promising", "recall below margin"
    elif d_fa > 0:
        result, reason = "not promising", "worsened: more false activity than the baseline"
    elif d_fa < 0 and fpa_c <= READINESS_S_PER_MIN and is_robust:
        result, reason = "promising", "improved: strictly lower false activity, readiness target, recall and robustness met"
    elif d_fa == 0:
        result, reason = "not promising", "maintained: equal false activity is not an improvement; retain B0"
    else:
        result, reason = "mixed", "lower false activity without robustness"
    return Screening("absolute", fpa_c, fpa_b, d_fa, None, "not applicable", d_recall, is_robust, result, reason)
