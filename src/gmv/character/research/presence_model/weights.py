"""The training objective shared by B1 and C (protocol §6), in NumPy.

For a model trained on groups 𝒯, F₍g,c₎ is group g's eligible frames of class c and n₍g,c₎
their number. A group is a member of class c (Aᶜ) when n₍g,c₎ ≥ 100 frames (2 s). With ℓ the
binary cross-entropy computed from logits:

    L = ½ · mean over g in A¹ of [mean over F₍g,1₎ of ℓ(z, 1)]
      + ½ · mean over g in A⁰ of [mean over F₍g,0₎ of ℓ(z, 0)]

Equivalently each eligible frame of a member group weighs w = 1 / (2·|Aᶜ|·n₍g,c₎), every
other frame 0, and L = Σ w·ℓ. Every member group's total weight in a class is exactly
1/(2·|Aᶜ|). Excluded frames weigh 0 and are not counted in n₍g,c₎.

**Batches.** Each epoch visits every unit (an excerpt for C, a frame for B1) exactly once.
With J batches per epoch, a batch's loss is J · Σ (w·ℓ) over its frames, *not* normalised
by the batch's own weight. The batch losses of an epoch then average to L exactly.
"""

from dataclasses import dataclass

import numpy as np

MIN_FRAMES = 100  # 2 s at 50 frames per second
CLASSES = (0, 1)


@dataclass(frozen=True)
class Unit:
    """One training unit: its group, per-frame labels (0/1) and per-frame eligibility."""

    group: str
    labels: np.ndarray
    eligible: np.ndarray

    def __post_init__(self):
        labels, eligible = np.asarray(self.labels), np.asarray(self.eligible)
        if labels.ndim != 1 or labels.shape != eligible.shape:
            raise ValueError("labels and eligibility need one value per frame")
        if not np.isin(labels, CLASSES).all():
            raise ValueError("labels must be 0 or 1")
        if eligible.dtype != bool:
            raise ValueError("eligibility must be boolean")


def class_counts(units):
    """{(group, class): eligible frame count}."""
    counts = {}
    for unit in units:
        for c in CLASSES:
            n = int(np.count_nonzero(unit.eligible & (np.asarray(unit.labels) == c)))
            counts[(unit.group, c)] = counts.get((unit.group, c), 0) + n
    return counts


def members(units, min_frames=MIN_FRAMES):
    """{class: sorted groups with at least ``min_frames`` eligible frames of that class}."""
    counts = class_counts(units)
    return {c: sorted(g for (g, cc), n in counts.items() if cc == c and n >= min_frames) for c in CLASSES}


def frame_weights(units, min_frames=MIN_FRAMES):
    """Per-unit arrays of frame weights w (see the module docstring)."""
    units = list(units)
    counts, member = class_counts(units), members(units, min_frames)
    for c in CLASSES:
        if not member[c]:
            raise ValueError(f"no group has {min_frames} eligible frames of class {c}; the objective is undefined")
    size = {c: len(member[c]) for c in CLASSES}
    weights = []
    for unit in units:
        w = np.zeros(len(unit.labels))
        for c in CLASSES:
            if unit.group in member[c]:
                frames = unit.eligible & (np.asarray(unit.labels) == c)
                w[frames] = 1.0 / (2 * size[c] * counts[(unit.group, c)])
        weights.append(w)
    return weights


def bce_with_logits(logits, labels):
    """Binary cross-entropy from logits, numerically stable."""
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(labels, dtype=np.float64)
    return np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))


def objective(units, logits, min_frames=MIN_FRAMES):
    """L = Σ w·ℓ over every unit's frames."""
    units = list(units)
    weights = frame_weights(units, min_frames)
    return float(sum(np.dot(w, bce_with_logits(z, u.labels)) for u, w, z in zip(units, weights, logits, strict=True)))


def registered_loss(units, logits, min_frames=MIN_FRAMES):
    """L computed directly from the protocol's formula (group means, then class means)."""
    units, logits = list(units), list(logits)
    member = members(units, min_frames)
    total = 0.0
    for c in CLASSES:
        group_means = []
        for group in member[c]:
            losses = [bce_with_logits(z, u.labels)[u.eligible & (np.asarray(u.labels) == c)]
                      for u, z in zip(units, logits, strict=True) if u.group == group]
            group_means.append(float(np.concatenate(losses).mean()))
        total += 0.5 * float(np.mean(group_means))
    return total


def batch_losses(units, logits, order, batch_size, min_frames=MIN_FRAMES):
    """The losses of one epoch's batches: units in ``order`` (each exactly once), batches of
    ``batch_size`` units, each batch's loss J · Σ (w·ℓ)."""
    units, logits, order = list(units), list(logits), list(order)
    if sorted(order) != list(range(len(units))):
        raise ValueError("an epoch must visit every unit exactly once")
    if batch_size < 1:
        raise ValueError("a batch holds at least one unit")
    weights = frame_weights(units, min_frames)
    batches = [order[i : i + batch_size] for i in range(0, len(order), batch_size)]
    count = len(batches)
    return [count * float(sum(np.dot(weights[i], bce_with_logits(logits[i], units[i].labels)) for i in batch))
            for batch in batches]
