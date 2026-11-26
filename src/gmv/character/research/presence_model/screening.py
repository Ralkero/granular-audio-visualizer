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

**Robust** (both modes, §13 (c)) requires two complete sets of recomputations, each row a
(D_FA, Δ_R) pair:

* exactly ``SEED_RECOMPUTATIONS`` (5) single-seed recomputations;
* one leave-one-group-out recomputation for every expected group, and only for those:
  each row is (group, D_FA, Δ_R), and the caller supplies the expected groups. A missing,
  duplicated or unexpected group makes the set incomplete.

Every row in both sets must satisfy D_FA < 0 and Δ_R ≥ −0.05. An incomplete set is never
robust; there is no fallback that treats missing rows as passing. Malformed or non-finite
rows raise ``ValueError``. Results and messages report counts only, never group keys.
"""

import math
from collections import Counter
from dataclasses import dataclass

READINESS_S_PER_MIN = 3.0
RATIO_TARGET = 0.50
RECALL_MARGIN = 0.05
SEED_RECOMPUTATIONS = 5
EPSILON = 1e-12  # float tolerance for the registered boundaries (≤ 0.50, ≥ −0.05)


@dataclass(frozen=True)
class Robustness:
    """The robustness check, in counts only (no group keys)."""

    passed: bool
    reason: str
    seeds_supplied: int
    groups_expected: int
    groups_supplied: int


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
    robustness: str  # the robustness check's reason, counts only
    result: str  # "promising", "mixed" or "not promising"
    reason: str


def _number(name, value, low=None, high=None):
    value = float(value)
    if not math.isfinite(value) or (low is not None and value < low) or (high is not None and value > high):
        raise ValueError(f"{name} is out of range")
    return value


def _pair(row):
    try:
        d_fa, d_recall = (float(v) for v in row)
    except (TypeError, ValueError):
        raise ValueError("a recomputation must be a (D_FA, Δ_R) pair of numbers") from None
    if not (math.isfinite(d_fa) and math.isfinite(d_recall)):
        raise ValueError("recomputations must be finite numbers")
    return d_fa, d_recall


def _holds(d_fa, d_recall):
    return d_fa < 0 and d_recall + RECALL_MARGIN >= -EPSILON


def check_robustness(seed_recomputations, group_recomputations, expected_groups):
    """``seed_recomputations``: (D_FA, Δ_R) pairs, exactly 5. ``group_recomputations``:
    (group, D_FA, Δ_R) triples, one per expected group. ``expected_groups``: the groups the
    leave-one-group-out analysis must cover, given by the caller."""
    seeds = [_pair(row) for row in seed_recomputations]
    expected = list(expected_groups)
    if len(set(expected)) != len(expected):
        raise ValueError("the expected groups contain a duplicate")
    rows = []
    for row in group_recomputations:
        try:
            group, d_fa, d_recall = row
        except (TypeError, ValueError):
            raise ValueError("a leave-one-group-out recomputation must be (group, D_FA, Δ_R)") from None
        rows.append((group, *_pair((d_fa, d_recall))))

    def result(passed, reason):
        return Robustness(passed, reason, len(seeds), len(expected), len(rows))

    if len(seeds) != SEED_RECOMPUTATIONS:
        return result(False, f"{len(seeds)} seed recomputations supplied; exactly {SEED_RECOMPUTATIONS} are required")
    if not expected:
        return result(False, "no expected groups were given for the leave-one-group-out set")
    counts, wanted = Counter(group for group, _, _ in rows), set(expected)
    missing = sum(1 for group in wanted if group not in counts)
    duplicated = sum(1 for group, n in counts.items() if n > 1)
    unexpected = sum(1 for group in counts if group not in wanted)
    if missing or duplicated or unexpected:
        return result(False, f"the leave-one-group-out set is incomplete or invalid: {missing} missing, "
                             f"{duplicated} duplicated, {unexpected} unexpected")
    failed_seeds = sum(1 for d_fa, d_recall in seeds if not _holds(d_fa, d_recall))
    failed_groups = sum(1 for _, d_fa, d_recall in rows if not _holds(d_fa, d_recall))
    if failed_seeds or failed_groups:
        return result(False, f"the direction failed in {failed_seeds} of {len(seeds)} seed and {failed_groups} of "
                             f"{len(rows)} leave-one-group-out recomputations")
    return result(True, f"the direction held in all {len(seeds)} seed and all {len(rows)} leave-one-group-out recomputations")


def robust(seed_recomputations, group_recomputations, expected_groups):
    return check_robustness(seed_recomputations, group_recomputations, expected_groups).passed


def screen(fpa_candidate, fpa_baseline, recall_candidate, recall_baseline,
           seed_recomputations=(), group_recomputations=(), expected_groups=()):
    """Classify a candidate against B0ₜ; robustness needs the complete sets described above."""
    fpa_c = _number("FPA(candidate)", fpa_candidate, low=0.0)
    fpa_b = _number("FPA(baseline)", fpa_baseline, low=0.0)
    d_recall = _number("recall(candidate)", recall_candidate, 0.0, 1.0) - _number("recall(baseline)", recall_baseline, 0.0, 1.0)
    d_fa = fpa_c - fpa_b
    check = check_robustness(seed_recomputations, group_recomputations, expected_groups)
    recall_ok = d_recall + RECALL_MARGIN >= -EPSILON
    if fpa_b >= READINESS_S_PER_MIN:
        ratio = fpa_c / fpa_b
        fifty = "met" if ratio <= RATIO_TARGET + EPSILON else "not met"
        if d_fa >= 0:
            result, reason = "not promising", "false activity is not lower than the baseline's"
        elif not recall_ok:
            result, reason = "not promising", "recall is more than 5 points below the baseline's"
        elif fifty == "met" and check.passed:
            result, reason = "promising", "at least a 50% reduction, recall within the margin, robust"
        else:
            result, reason = "mixed", "lower false activity without meeting every promising condition"
        return Screening("relative", fpa_c, fpa_b, d_fa, ratio, fifty, d_recall, check.passed, check.reason, result, reason)
    # Amendment 1, A1.1: the baseline already meets the readiness target.
    if not recall_ok:
        result, reason = "not promising", "recall below margin"
    elif d_fa > 0:
        result, reason = "not promising", "worsened: more false activity than the baseline"
    elif d_fa < 0 and fpa_c <= READINESS_S_PER_MIN and check.passed:
        result, reason = "promising", "improved: strictly lower false activity, readiness target, recall and robustness met"
    elif d_fa == 0:
        result, reason = "not promising", "maintained: equal false activity is not an improvement; retain B0"
    else:
        result, reason = "mixed", "lower false activity without robustness"
    return Screening("absolute", fpa_c, fpa_b, d_fa, None, "not applicable", d_recall, check.passed, check.reason, result, reason)
