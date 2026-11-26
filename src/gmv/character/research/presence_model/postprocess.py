"""The post-processing every arm shares (protocol §8): the silence gate, then hysteresis.

As in the engine (``gmv.activity.ActivityBank.step``): a silent frame's score is forced to 0
before the hysteresis update; a detector starts inactive; an inactive detector turns on at
a score ≥ on, and an active one stays on while the score ≥ off. For a chosen t, on = t and
off = t − 0.2. Phrases then come from the unchanged ``VocalContour`` (Stage 2).
"""

import numpy as np

GAP = 0.2
GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(11))  # 0.30, 0.35, ..., 0.80


def hysteresis(scores, on, off):
    if not off <= on:
        raise ValueError("the off threshold must not exceed the on threshold")
    scores = np.asarray(scores, dtype=float)
    active = np.zeros(len(scores), dtype=bool)
    state = False
    for i, score in enumerate(scores):
        state = score >= off if state else score >= on
        active[i] = state
    return active


def decide(scores, t, silent=None):
    """Active frames for operating point t: silent frames forced to 0, then hysteresis."""
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")
    if silent is not None:
        silent = np.asarray(silent, dtype=bool)
        if silent.shape != scores.shape:
            raise ValueError("the silence mask must have one value per frame")
        scores = np.where(silent, 0.0, scores)
    return hysteresis(scores, on=t, off=t - GAP)
