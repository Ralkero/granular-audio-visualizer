"""Frame labels and the frames each use leaves out (protocol §6, §10, §11).

Times are seconds on the excerpt's own timeline (0 is the excerpt start). A frame is
labelled by its centre: voice (1) when the centre lies inside the union of confirmed
intervals, half-open [start, end), as ``presence.schema.targets`` does; otherwise no voice
(0).

**Collars.** Boundaries are the starts and ends of the confirmed union lying strictly inside
the excerpt (0 < time < duration). Excerpt edges are not boundaries. A frame whose centre
is within 0.3 s of a boundary, inclusive, is left out. Uncertain time is left out as a
whole. Collars around the edges of uncertain intervals are not specified by the protocol
and are not added (the Stage 1 report notes this).

**Training** additionally leaves out frames below the silence level and pre-roll frames
(``training_mask``). **Evaluation** keeps silent frames, whose predictions the shared
post-processing forces to "no voice" (``evaluation_mask``).
"""

import numpy as np

from ..presence.schema import union

HOP_S = 0.02
COLLAR_S = 0.3
EPSILON_S = 1e-9  # float tolerance for "within 0.3 s, inclusive" and interval edges


def _spans(intervals, duration):
    spans = []
    for start, end in intervals:
        start, end = float(start), float(end)
        if not (0 <= start < end <= duration + EPSILON_S):
            raise ValueError("an interval must lie inside the excerpt and run forwards")
        spans.append((start, end))
    return union(spans)


def frame_labels(centres, confirmed, duration):
    """1 where the frame centre lies inside the confirmed union [start, end), else 0."""
    centres = np.asarray(centres, dtype=float)
    labels = np.zeros(len(centres), dtype=np.int8)
    for start, end in _spans(confirmed, duration):
        labels[(centres >= start - EPSILON_S) & (centres < end - EPSILON_S)] = 1
    return labels


def interior_boundaries(confirmed, duration):
    return [b for start, end in _spans(confirmed, duration) for b in (start, end) if 0 < b < duration]


def collar_mask(centres, confirmed, duration, collar_s=COLLAR_S):
    """True for frames within ``collar_s`` (inclusive) of an interior boundary."""
    centres = np.asarray(centres, dtype=float)
    mask = np.zeros(len(centres), dtype=bool)
    for boundary in interior_boundaries(confirmed, duration):
        mask |= np.abs(centres - boundary) <= collar_s + EPSILON_S
    return mask


def uncertain_mask(centres, uncertain, confirmed, duration):
    """True inside uncertain time that is not also confirmed (confirmed wins)."""
    centres = np.asarray(centres, dtype=float)
    inside = np.zeros(len(centres), dtype=bool)
    for start, end in _spans(uncertain, duration):
        inside |= (centres >= start - EPSILON_S) & (centres < end - EPSILON_S)
    return inside & (frame_labels(centres, confirmed, duration) == 0)


def in_excerpt(centres, duration):
    centres = np.asarray(centres, dtype=float)
    return (centres >= 0) & (centres < duration)


def evaluation_mask(centres, duration, confirmed, uncertain=(), collar_s=COLLAR_S):
    """Frames that count in evaluation: inside the excerpt, outside collars and uncertain-only time."""
    return (in_excerpt(centres, duration) & ~collar_mask(centres, confirmed, duration, collar_s)
            & ~uncertain_mask(centres, uncertain, confirmed, duration))


def training_mask(centres, duration, confirmed, uncertain=(), silent=None, collar_s=COLLAR_S):
    """Frames that count in training: as ``evaluation_mask``, and not silent."""
    mask = evaluation_mask(centres, duration, confirmed, uncertain, collar_s)
    if silent is not None:
        silent = np.asarray(silent, dtype=bool)
        if silent.shape != mask.shape:
            raise ValueError("the silence mask must have one value per frame")
        mask &= ~silent
    return mask
