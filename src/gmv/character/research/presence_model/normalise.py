"""Normalisation interfaces (readiness proposal, revision 4.4, section 8). Which procedure each
arm uses is still an owner decision; these implement the proposed ones over explicit arrays.

* ``pooled``: model C's proposed statistic, the pooled mean and population spread plus 1e-6
  per band, in float64, over the frames each training unit selects. ``statistics_mask``
  gives the proposed selection: frames centred inside the excerpt that are not quiet
  (collar and uncertain frames included, pre-roll and context excluded). Units are taken in
  key order, so the result does not depend on the order they are passed in.
* ``fixed``: B1's proposed use of B0's stored constants (548 means and spreads), validated.
* ``apply``: (x - mean) / scale, as float32.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from . import labels as labelling

SPREAD_EPSILON = 1e-6


@dataclass(frozen=True)
class Stats:
    mean: np.ndarray
    scale: np.ndarray
    frames: int

    @property
    def sha256(self):
        data = np.ascontiguousarray(self.mean, dtype="<f8").tobytes() + np.ascontiguousarray(self.scale, dtype="<f8").tobytes()
        return hashlib.sha256(data).hexdigest()


def statistics_mask(segment, quiet):
    return labelling.in_excerpt(segment.centres, segment.duration) & ~np.asarray(quiet, dtype=bool)


def pooled(units):
    """``units``: (key, features shaped (frames, bands), selection mask) for the model's
    training excerpts only."""
    selected = [np.asarray(f, dtype=np.float64)[np.asarray(m, dtype=bool)] for _k, f, m in sorted(units, key=lambda u: u[0])]
    x = np.concatenate(selected) if selected else np.zeros((0, 0))
    if len(x) == 0:
        raise ValueError("no frames to compute statistics from")
    return Stats(x.mean(axis=0), x.std(axis=0) + SPREAD_EPSILON, len(x))


def fixed(mean, scale, width=548):
    mean, scale = np.asarray(mean, dtype=np.float64), np.asarray(scale, dtype=np.float64)
    if mean.shape != (width,) or scale.shape != (width,):
        raise ValueError("normalisation constants have the wrong shape")
    if not (np.isfinite(mean).all() and np.isfinite(scale).all()) or (scale <= 0).any():
        raise ValueError("normalisation constants must be finite with positive spreads")
    return Stats(mean, scale, 0)


def apply(x, stats):
    return ((np.asarray(x, dtype=np.float64) - stats.mean) / stats.scale).astype(np.float32)
