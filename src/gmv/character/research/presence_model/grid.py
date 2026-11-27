"""The canonical segment and frame grid (readiness proposal, revision 4.4, section 2).

Each excerpt has exactly one segment: ``frontend.segment`` with up to 5 s of pre-roll. Its
engine frame grid carries every frame-dependent array (labels, masks, the quiet flag, fold
counts, B1 features and C's log-mel rows). Nothing is computed on any other segment.

Frame centres are on the excerpt's own timeline. Pre-roll frames have centres before 0;
excerpt frames have centres in [0, duration). ``f0`` is the first excerpt frame and ``f1``
one past the last. Model C trains on rows [s0, f1) with s0 = max(0, f0 - 75): 75 frames is
the registered "up to 1.5 s" of context, and C's output at frame t depends on frames
t-72 ... t only (its receptive field is 73 frames).
"""

from dataclasses import dataclass

import numpy as np

from . import frontend
from . import labels as labelling

CONTEXT_FRAMES = 75
RECEPTIVE_HISTORY = 72


class GridError(ValueError):
    """The excerpt cannot be placed on the registered 20 ms grid."""


@dataclass(frozen=True)
class Segment:
    rate: int
    total_samples: int
    start_s: float
    end_s: float
    first: int
    end: int
    frames: int
    centres: np.ndarray
    f0: int
    f1: int

    @property
    def duration(self):
        return self.end_s - self.start_s

    def context_start(self, context=CONTEXT_FRAMES):
        return max(0, self.f0 - context)

    def c_rows(self, context=CONTEXT_FRAMES):
        """Rows of the segment's log-mel that model C trains on."""
        return slice(self.context_start(context), self.f1)


def hop_is_20ms(rate):
    return frontend.engine_hop(rate) * 50 == rate


def canonical(total_samples, rate, start_s, end_s):
    if not hop_is_20ms(rate):
        raise GridError("the engine hop is not exactly 20 ms at this sample rate")
    first, end = frontend.segment(total_samples, rate, start_s, end_s)
    frames = frontend.frame_count(end - first, rate)
    centres = frontend.frame_centres(frames, rate, first / rate) - start_s
    f0 = int(np.count_nonzero(centres < 0))
    f1 = int(np.count_nonzero(centres < end_s - start_s))
    return Segment(rate, total_samples, start_s, end_s, first, end, frames, centres, f0, f1)


def frame_arrays(segment, confirmed, uncertain=(), quiet=None):
    """Labels and masks on the segment's grid (``labels`` module rules)."""
    c, d = segment.centres, segment.duration
    if quiet is not None and len(quiet) != segment.frames:
        raise GridError("the quiet mask does not match the segment's frames")
    return {
        "labels": labelling.frame_labels(c, confirmed, d),
        "evaluation": labelling.evaluation_mask(c, d, confirmed, uncertain),
        "training": labelling.training_mask(c, d, confirmed, uncertain, silent=quiet),
    }
