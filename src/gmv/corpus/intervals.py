"""Small interval helpers shared by stem drafting, annotator comparison and metrics."""

import numpy as np


def merge(intervals, gap=0.0):
    """Sorted union of (start, end) pairs, joining pieces separated by at most ``gap``."""
    result = []
    for start, end in sorted((float(s), float(e)) for s, e in intervals if e > s):
        if result and start - result[-1][1] <= gap:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return [(s, e) for s, e in result]


def intersect(a, b):
    a, b, result, i, j = merge(a), merge(b), [], 0, 0
    while i < len(a) and j < len(b):
        start, end = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if end > start:
            result.append((start, end))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return result


def rasterize(intervals, duration, hop=0.01):
    """Boolean frame mask; a frame is on when its centre lies inside an interval."""
    centres = (np.arange(max(1, int(np.ceil(duration / hop)))) + 0.5) * hop
    mask = np.zeros(len(centres), dtype=bool)
    for start, end in intervals:
        mask |= (centres >= start) & (centres < end)
    return mask


def from_mask(mask, hop=0.01):
    edges = np.flatnonzero(np.diff(np.concatenate(([0], mask.astype(np.int8), [0]))))
    return [(float(a * hop), float(b * hop)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def total(intervals):
    return float(sum(end - start for start, end in merge(intervals)))
