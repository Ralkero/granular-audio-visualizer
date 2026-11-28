"""Step 4, S4.4: single-model scoring, operating points, seed-index aggregates and effect sizes
(protocol §8, §11-§13; Amendment 3, A3.1). NumPy and exact fractions. Built and tested on
generated scores only.

* ``Scored``: one model's per-frame voice probability on one excerpt's whole canonical segment
  (pre-roll included), with the frame centres on the excerpt's timeline, the quiet flags and
  the excerpt's labels. Post-processing (§8) runs over the whole segment, with hysteresis
  inactive at the segment's start; only excerpt frames outside the collars are counted (§10,
  §11).
* ``counts_at``: per-group frame counts at one operating point (or one per group).
* ``j_exact``: J(t) = ½ (mean group recall + mean group specificity), each group counting once
  in its class, in exact fractions, so that ties are exact.
* ``choose``: t maximising J̄(t), the mean over the inner seed indices of J(t) on each seed's
  group-disjoint predictions (§8 step 4; under policy D one seed, J̄ = J); ties go to the t
  closest to 0.5, then the higher one.
* ``seed_aggregates``: aggregate s is the cross-validation result in which every outer fold
  uses its outer model with seed index s (§12): per-group counts, then group averages.
* ``effects``: the primary effect sizes are the means of the seed-index aggregates (the ratio is
  a ratio of these means; §12.1 decides whether it is used); the seed rows and the
  leave-one-group-out rows (from each group's mean over O, models and thresholds fixed) feed
  ``screening.screen``.
* ``t_final``: the final models' operating point, maximising the mean over s ∈ O of J on seed
  s's 19 outer predictions (same candidates and tie rule).

Values that the disclosure check later reads (per-group FPA and recall) are also available as
exact fractions (``group_values``).
"""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from . import metrics, postprocess, screening

FIXED_POINT = 0.6  # B0's shipped on 0.6 / off 0.4, reported for every arm (§11)
SHIFT = 0.05  # the effects at t ± 0.05 (§12)


@dataclass(frozen=True)
class Scored:
    group: str
    centres: np.ndarray  # the whole segment's frame centres, excerpt time
    duration: float
    confirmed: tuple
    uncertain: tuple
    silent: np.ndarray  # the engine's quiet flag per frame
    scores: np.ndarray  # voice probability per frame

    def __post_init__(self):
        n = len(self.centres)
        if np.asarray(self.scores).shape != (n,) or np.asarray(self.silent).shape != (n,):
            raise ValueError("scores and the quiet flag need one value per frame")


def _point(t):
    return round(float(t), 2)


def counts_at(scored, t):
    """{group: metrics.Counts}; ``t`` is one operating point or {group: t}."""
    rows = []
    for s in scored:
        point = _point(t[s.group] if isinstance(t, dict) else t)
        active = postprocess.decide(s.scores, point, s.silent)
        rows.append((s.group, metrics.excerpt_counts(s.centres, active, s.duration, s.confirmed, s.uncertain)))
    return metrics.group_counts(rows)


def counts_grid(scored, grid=postprocess.GRID):
    return {t: counts_at(scored, t) for t in grid}


def group_values(groups):
    """Exact per-group values: {group: {"fpa": Fraction | None, "recall": Fraction | None}}."""
    out = {}
    for g, c in groups.items():
        out[g] = {"fpa": Fraction(60 * c.false, c.novoice) if c.novoice >= metrics.MIN_CLASS_FRAMES else None,
                  "recall": Fraction(c.detected, c.voiced) if c.voiced >= metrics.MIN_CLASS_FRAMES else None}
    return out


def j_exact(groups):
    recall = [Fraction(c.detected, c.voiced) for c in groups.values() if c.voiced >= metrics.MIN_CLASS_FRAMES]
    spec = [1 - Fraction(c.false, c.novoice) for c in groups.values() if c.novoice >= metrics.MIN_CLASS_FRAMES]
    if not recall or not spec:
        raise ValueError("J needs at least one group of each class")
    return (sum(recall, Fraction(0)) / len(recall) + sum(spec, Fraction(0)) / len(spec)) / 2


def choose(j_by_t):
    """The t with the largest value; ties to the t closest to 0.5, then the higher one."""
    if not j_by_t:
        raise ValueError("no candidate operating point")
    best = max(j_by_t.values())
    tied = [t for t, v in j_by_t.items() if v == best]
    return max(tied, key=lambda t: (-abs(Fraction(str(t)) - Fraction(1, 2)), Fraction(str(t))))


def j_bar(grids_by_seed):
    """{t: J̄(t)}: the mean over the inner seeds of J(t), each from its own predictions.
    ``grids_by_seed``: {seed index: {t: {group: Counts}}}."""
    seeds = sorted(grids_by_seed)
    if not seeds:
        raise ValueError("no inner seed")
    grid = sorted(grids_by_seed[seeds[0]])
    if any(sorted(grids_by_seed[s]) != grid for s in seeds):
        raise ValueError("every inner seed must cover the same candidates")
    return {t: sum((j_exact(grids_by_seed[s][t]) for s in seeds), Fraction(0)) / len(seeds) for t in grid}


def operating_point(grids_by_seed, policy="C", inner_seeds=None):
    """§8 step 4 for one outer fold: t from J̄ over the policy's inner seed indices."""
    wanted = set(inner_seeds if inner_seeds is not None else {"C": (0, 1), "D": (0,)}[policy])
    if set(grids_by_seed) != wanted:
        raise ValueError("the inner predictions must come from exactly the policy's inner seed indices")
    return choose(j_bar(grids_by_seed))


def inner_agreement(grids_by_seed):
    """Operating-point stability (§12, policy C only): whether each inner seed alone would
    choose the same t."""
    alone = {s: choose({t: j_exact(g) for t, g in grid.items()}) for s, grid in grids_by_seed.items()}
    return len(set(alone.values())) == 1


def seed_aggregates(outer_counts):
    """``outer_counts``: {seed index: {group: Counts}} (each group from its own outer model at
    its fold's t). Returns {seed index: metrics.macro(...)}."""
    return {s: metrics.macro(groups) for s, groups in sorted(outer_counts.items())}


def _mean(values):
    values = [v for v in values if v is not None]
    return float(np.mean(values)) if values else None


def _means_over_seeds(outer_counts, key):
    """{group: mean over seed indices of the group's value} (None where undefined)."""
    seeds = sorted(outer_counts)
    groups = sorted(outer_counts[seeds[0]])
    out = {}
    for g in groups:
        values = [getattr(outer_counts[s][g], key) for s in seeds]
        out[g] = None if values[0] is None else float(np.mean(values))
    return out


@dataclass(frozen=True)
class Effects:
    fpa_candidate: float
    fpa_baseline: float
    recall_candidate: float
    recall_baseline: float
    d_fa: float
    delta_recall: float
    ratio: float | None  # only when FPA_macro(B0ₜ) ≥ 3 s/min (§12.1)
    seed_rows: tuple  # (seed index, D_FA, Δ_R)
    group_rows: tuple  # (group, D_FA, Δ_R) without that group

    def screen(self, policy="C"):
        return screening.screen(self.fpa_candidate, self.fpa_baseline, self.recall_candidate, self.recall_baseline,
                                self.seed_rows, self.group_rows, [g for g, _d, _r in self.group_rows], policy)


def effects(candidate, baseline, policy="C"):
    """Candidate (C or B1) against B0ₜ. ``candidate``: {seed index: {group: Counts}} over the
    policy's outer seed indices; ``baseline``: {group: Counts} (B0ₜ is fixed, one value)."""
    seeds = sorted(candidate)
    if set(seeds) != set(screening.seed_indices(policy)):
        raise ValueError("the candidate must cover exactly the policy's outer seed indices")
    groups = sorted(baseline)
    if any(sorted(candidate[s]) != groups for s in seeds):
        raise ValueError("every seed-index aggregate must cover the same groups as the baseline")
    base = metrics.macro(baseline)
    aggregates = seed_aggregates(candidate)
    fpa_c = float(np.mean([aggregates[s]["fpa"] for s in seeds]))
    recall_c = float(np.mean([aggregates[s]["recall"] for s in seeds]))
    seed_rows = tuple((s, aggregates[s]["fpa"] - base["fpa"], aggregates[s]["recall"] - base["recall"]) for s in seeds)
    fpa_g, recall_g = _means_over_seeds(candidate, "fpa"), _means_over_seeds(candidate, "recall")
    rows = []
    for g in groups:
        rest = [h for h in groups if h != g]
        parts = (_mean(fpa_g[h] for h in rest), _mean(baseline[h].fpa for h in rest),
                 _mean(recall_g[h] for h in rest), _mean(baseline[h].recall for h in rest))
        if any(v is None for v in parts):
            raise ValueError("a leave-one-group-out recomputation is undefined (F5)")
        rows.append((g, parts[0] - parts[1], parts[2] - parts[3]))
    ratio = fpa_c / base["fpa"] if base["fpa"] >= screening.READINESS_S_PER_MIN else None
    return Effects(fpa_c, base["fpa"], recall_c, base["recall"], fpa_c - base["fpa"], recall_c - base["recall"], ratio,
                   seed_rows, tuple(rows))


def shifted(points, shift):
    """Each fold's operating point moved by ``shift`` (the effects at t ± 0.05; off stays t − 0.2)."""
    return {g: _point(t + shift) for g, t in points.items()}


def t_final(outer_grids):
    """``outer_grids``: {seed index: {t: {group: Counts}}} on the outer predictions."""
    return choose(j_bar(outer_grids))
