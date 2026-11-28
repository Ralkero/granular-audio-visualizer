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

**Robust** (both modes, §13 (c)) requires two complete sets of recomputations:

* the seed-index set: one (seed_index, D_FA, Δ_R) row for each seed-index aggregate of the
  policy in force (Amendment 3, A3.1): indices 0-4 under policy C, 0-2 under policy D, and only
  those. A missing, duplicated or unexpected index makes the set incomplete, so four rows, six
  rows, or one seed's result repeated never pass under policy C;
* the leave-one-group-out set: one (group, D_FA, Δ_R) row for every expected group, and
  only those; the caller supplies the expected groups. A missing, duplicated or unexpected
  group makes the set incomplete.

Every row in both sets must satisfy D_FA < 0 and Δ_R ≥ −0.05. An incomplete set is never
robust; there is no fallback that treats missing rows as passing. Malformed or non-finite
rows raise ``ValueError``. Results and messages report counts only, never seed indices or
group keys.
"""

import math
import numbers
from collections import Counter
from dataclasses import dataclass

READINESS_S_PER_MIN = 3.0
RATIO_TARGET = 0.50
RECALL_MARGIN = 0.05
POLICY_SEEDS = {"C": frozenset(range(5)), "D": frozenset(range(3))}  # A3.1: O under each policy
SEED_INDICES = POLICY_SEEDS["C"]  # §7: outer and final training seed indices 0-4 (policy C)
SEED_RECOMPUTATIONS = len(SEED_INDICES)
POLICY_D_NOTE = "Policy D was in force; its seed criterion (3 seed-index aggregates) is less demanding than policy C's (5)."


def seed_indices(policy):
    if policy not in POLICY_SEEDS:
        raise ValueError("the policy must be C or D")
    return POLICY_SEEDS[policy]
EPSILON = 1e-12  # float tolerance for the registered boundaries (≤ 0.50, ≥ −0.05)


@dataclass(frozen=True)
class Robustness:
    """The robustness check, in counts only (no group keys)."""

    passed: bool
    reason: str
    seeds_expected: int
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
    policy: str = "C"

    @property
    def policy_note(self):
        """§12: the report names the policy and says when the seed criterion is less demanding."""
        return POLICY_D_NOTE if self.policy == "D" else "Policy C was in force (5 seed-index aggregates)."


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


def _seed_row(row):
    try:
        index, d_fa, d_recall = row
    except (TypeError, ValueError):
        raise ValueError("a seed recomputation must be (seed_index, D_FA, Δ_R)") from None
    if isinstance(index, bool) or not isinstance(index, numbers.Integral):
        raise ValueError("a seed index must be an integer")
    return (int(index), *_pair((d_fa, d_recall)))


def _completeness(supplied, wanted):
    """(missing, duplicated, unexpected) counts of ``supplied`` identities against ``wanted``."""
    counts = Counter(supplied)
    return (sum(1 for key in wanted if key not in counts),
            sum(1 for _key, n in counts.items() if n > 1),
            sum(1 for key in counts if key not in wanted))


def check_robustness(seed_recomputations, group_recomputations, expected_groups, policy="C"):
    """``seed_recomputations``: (seed_index, D_FA, Δ_R) triples, one for each index of the
    policy (0-4 under C, 0-2 under D).
    ``group_recomputations``: (group, D_FA, Δ_R) triples, one per expected group.
    ``expected_groups``: the groups the leave-one-group-out analysis must cover, given by
    the caller."""
    seeds = [_seed_row(row) for row in seed_recomputations]
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

    wanted = seed_indices(policy)

    def result(passed, reason):
        return Robustness(passed, reason, len(wanted), len(seeds), len(expected), len(rows))

    missing, duplicated, unexpected = _completeness([index for index, _, _ in seeds], wanted)
    if missing or duplicated or unexpected or len(seeds) != len(wanted):
        return result(False, f"the seed set is incomplete or invalid: {len(seeds)} of {len(wanted)} rows, "
                             f"{missing} missing, {duplicated} duplicated, {unexpected} unexpected")
    if not expected:
        return result(False, "no expected groups were given for the leave-one-group-out set")
    missing, duplicated, unexpected = _completeness([group for group, _, _ in rows], set(expected))
    if missing or duplicated or unexpected:
        return result(False, f"the leave-one-group-out set is incomplete or invalid: {missing} missing, "
                             f"{duplicated} duplicated, {unexpected} unexpected")
    failed_seeds = sum(1 for _, d_fa, d_recall in seeds if not _holds(d_fa, d_recall))
    failed_groups = sum(1 for _, d_fa, d_recall in rows if not _holds(d_fa, d_recall))
    if failed_seeds or failed_groups:
        return result(False, f"the direction failed in {failed_seeds} of {len(seeds)} seed and {failed_groups} of "
                             f"{len(rows)} leave-one-group-out recomputations")
    return result(True, f"the direction held in all {len(seeds)} seed and all {len(rows)} leave-one-group-out recomputations")


def robust(seed_recomputations, group_recomputations, expected_groups, policy="C"):
    return check_robustness(seed_recomputations, group_recomputations, expected_groups, policy).passed


def screen(fpa_candidate, fpa_baseline, recall_candidate, recall_baseline,
           seed_recomputations=(), group_recomputations=(), expected_groups=(), policy="C"):
    """Classify a candidate against B0ₜ; robustness needs the complete sets described above."""
    fpa_c = _number("FPA(candidate)", fpa_candidate, low=0.0)
    fpa_b = _number("FPA(baseline)", fpa_baseline, low=0.0)
    d_recall = _number("recall(candidate)", recall_candidate, 0.0, 1.0) - _number("recall(baseline)", recall_baseline, 0.0, 1.0)
    d_fa = fpa_c - fpa_b
    check = check_robustness(seed_recomputations, group_recomputations, expected_groups, policy)
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
        return Screening("relative", fpa_c, fpa_b, d_fa, ratio, fifty, d_recall, check.passed, check.reason, result, reason,
                         policy)
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
    return Screening("absolute", fpa_c, fpa_b, d_fa, None, "not applicable", d_recall, check.passed, check.reason, result,
                     reason, policy)
