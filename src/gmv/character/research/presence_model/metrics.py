"""Per-group metrics, group averages and events (protocol §10, §11).

For each group, over the frames ``labels.evaluation_mask`` keeps (inside the excerpt,
outside the ±0.3 s collars and uncertain-only time), at 20 ms per frame:

* voiced time V and no-voice time N, counted in frames (100 frames = 2 s);
* recall R = detected voiced frames ÷ voiced frames, defined when V ≥ 2 s;
* false vocal activity FPA = 60 × false vocal seconds ÷ no-voice seconds (seconds per minute
  of no-voice time), defined when N ≥ 2 s; specificity S = 1 − false frames ÷ no-voice frames.

A group with only one class counts only in that class's average; it is never dropped.
Group averages (``macro``) are the main view; duration-weighted figures (``pooled``) are
always reported beside them.

**Events.** References are the confirmed intervals merged across gaps shorter than 0.25 s.
Predictions are clipped to the excerpt. A reference and a prediction match when their
onsets differ by at most 0.5 s and their offsets by at most max(0.5 s, 20% of the reference
length); matching is one-to-one and maximises the number of matches, then minimises the
total onset difference. Timing errors are reported only for interior boundaries (strictly
inside the excerpt). Whether a boundary truncated by the excerpt edge should also be
exempt from the matching collars is not specified by the protocol; it is applied as written
(the Stage 1 report notes this).
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment

from . import labels as labelling
from ..presence.schema import union

FRAME_S = labelling.HOP_S
MIN_CLASS_S = 2.0
MIN_CLASS_FRAMES = 100  # 2 s at 20 ms per frame
MERGE_GAP_S = 0.25
ONSET_COLLAR_S = 0.5
OFFSET_COLLAR_S = 0.5
OFFSET_COLLAR_SHARE = 0.2
EPSILON_S = 1e-9


@dataclass(frozen=True)
class Counts:
    """Frame counts (20 ms each) of one excerpt or group; seconds and rates on demand."""

    voiced: int = 0
    novoice: int = 0
    detected: int = 0
    false: int = 0

    def __add__(self, other):
        return Counts(self.voiced + other.voiced, self.novoice + other.novoice,
                      self.detected + other.detected, self.false + other.false)

    @property
    def voiced_s(self):
        return self.voiced * FRAME_S

    @property
    def novoice_s(self):
        return self.novoice * FRAME_S

    @property
    def recall(self):
        return self.detected / self.voiced if self.voiced >= MIN_CLASS_FRAMES else None

    @property
    def fpa(self):
        return 60.0 * self.false / self.novoice if self.novoice >= MIN_CLASS_FRAMES else None

    @property
    def specificity(self):
        return 1.0 - self.false / self.novoice if self.novoice >= MIN_CLASS_FRAMES else None


def excerpt_counts(centres, active, duration, confirmed, uncertain=(), collar_s=labelling.COLLAR_S):
    """Counts for one excerpt from its frame centres (excerpt time) and active decisions."""
    centres = np.asarray(centres, dtype=float)
    active = np.asarray(active, dtype=bool)
    if active.shape != centres.shape:
        raise ValueError("one decision per frame is needed")
    keep = labelling.evaluation_mask(centres, duration, confirmed, uncertain, collar_s)
    voice = labelling.frame_labels(centres, confirmed, duration) == 1
    return Counts(
        voiced=int(np.count_nonzero(keep & voice)),
        novoice=int(np.count_nonzero(keep & ~voice)),
        detected=int(np.count_nonzero(keep & voice & active)),
        false=int(np.count_nonzero(keep & ~voice & active)),
    )


def group_counts(rows):
    """{group: Counts} from (group, Counts) rows."""
    out = {}
    for group, counts in rows:
        out[group] = out.get(group, Counts()) + counts
    return out


def macro(groups):
    """Group averages: FPA over groups with ≥ 2 s of no-voice time, recall over groups with
    ≥ 2 s of voice; None when no group qualifies."""
    fpa = [c.fpa for c in groups.values() if c.fpa is not None]
    recall = [c.recall for c in groups.values() if c.recall is not None]
    spec = [c.specificity for c in groups.values() if c.specificity is not None]
    return {
        "fpa": float(np.mean(fpa)) if fpa else None,
        "recall": float(np.mean(recall)) if recall else None,
        "specificity": float(np.mean(spec)) if spec else None,
        "groups_fpa": len(fpa),
        "groups_recall": len(recall),
    }


def pooled(groups):
    """Duration-weighted figures over every frame of every group."""
    total = sum(groups.values(), Counts())
    return {"fpa": total.fpa, "recall": total.recall, "specificity": total.specificity,
            "voiced_s": total.voiced_s, "novoice_s": total.novoice_s}


def objective(groups):
    """J = ½ · (mean group recall + mean group specificity), each group once per class."""
    summary = macro(groups)
    if summary["recall"] is None or summary["specificity"] is None:
        raise ValueError("J needs at least one group of each class")
    return 0.5 * (summary["recall"] + summary["specificity"])


def reference_events(confirmed, duration, gap_s=MERGE_GAP_S):
    """The confirmed union with gaps shorter than ``gap_s`` closed."""
    merged = []
    for start, end in union((float(s), float(e)) for s, e in confirmed):
        if not (0 <= start < end <= duration + EPSILON_S):
            raise ValueError("an interval must lie inside the excerpt and run forwards")
        if merged and start - merged[-1][1] < gap_s - EPSILON_S:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def match_events(reference, predicted, duration):
    """Event matching for one excerpt; returns counts and interior timing errors (s)."""
    reference = [(float(s), float(e)) for s, e in reference]
    clipped = []
    for start, end in predicted:
        start, end = max(0.0, float(start)), min(float(duration), float(end))
        if end > start:
            clipped.append((start, end))
    pairs = []
    if reference and clipped:
        big = 1e6
        cost = np.full((len(reference), len(clipped)), big)
        for i, (rs, re) in enumerate(reference):
            offset_collar = max(OFFSET_COLLAR_S, OFFSET_COLLAR_SHARE * (re - rs))
            for j, (ps, pe) in enumerate(clipped):
                if abs(ps - rs) <= ONSET_COLLAR_S + EPSILON_S and abs(pe - re) <= offset_collar + EPSILON_S:
                    cost[i, j] = -10.0 + abs(ps - rs)  # a match always beats no match
        rows, cols = linear_sum_assignment(cost)
        pairs = [(i, j) for i, j in zip(rows, cols, strict=True) if cost[i, j] < big]
    onset_errors = [clipped[j][0] - reference[i][0] for i, j in pairs if 0 < reference[i][0] < duration]
    offset_errors = [clipped[j][1] - reference[i][1] for i, j in pairs if 0 < reference[i][1] < duration]
    return {
        "references": len(reference),
        "predictions": len(clipped),
        "detected": len(pairs),
        "missed": len(reference) - len(pairs),
        "false": len(clipped) - len(pairs),
        "onset_errors_s": onset_errors,
        "offset_errors_s": offset_errors,
    }
