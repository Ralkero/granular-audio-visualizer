"""Generated training excerpts for the A5.2 Step 4 tests (model C's inputs). No project data."""

import numpy as np

from gmv.character.research.presence_model import cbatches


def excerpt(key, group, voiced_frames, rows=160, context=20, seed=0, voice_bands=(10, 20), quiet_rows=3):
    """``rows`` rows: ``context`` context rows, then the excerpt's frames; the first
    ``voiced_frames`` excerpt frames are voice, and voice adds power in ``voice_bands``."""
    rng = np.random.default_rng(seed)
    power = rng.gamma(2.0, 1e-3, (rows, 64))
    labels = np.zeros(rows, dtype=np.int8)
    labels[context : context + voiced_frames] = 1
    lo, hi = voice_bands
    power[labels == 1, lo:hi] *= 30
    eligible = np.zeros(rows, dtype=bool)
    eligible[context:] = True
    selected = eligible.copy()
    selected[context : context + quiet_rows] = False  # a few quiet frames: not used for statistics
    eligible[context : context + quiet_rows] = False
    return cbatches.Excerpt(key, group, bool(voiced_frames), power, labels, eligible, selected, context)


def training_set(seed=0, rows=160):
    """Three groups: a voiced group with a no-voice excerpt, a no-voice group with three
    excerpts (mixing partners), and a group with both classes in one excerpt."""
    return [
        excerpt("a-1", "group_a", 130, rows, seed=seed + 1),
        excerpt("a-2", "group_a", 0, rows, seed=seed + 2),
        excerpt("b-1", "group_b", 0, rows, seed=seed + 3),
        excerpt("b-2", "group_b", 0, rows - 20, context=10, seed=seed + 4),
        excerpt("b-3", "group_b", 0, rows, seed=seed + 5),
        excerpt("c-1", "group_c", 110, rows + 40, seed=seed + 6),
    ]
