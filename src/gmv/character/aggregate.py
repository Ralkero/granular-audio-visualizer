"""Running averages of the measurements over several lengths of time at once.

Each window forgets exponentially, by the audio time that has passed and not by the number
of updates, so frames that come 10 ms or 40 ms apart, evenly or not, build the same
averages. A window keeps, for every measurement, the weight it has gathered (seconds of
music), the weighted sum and the weighted sum of squares. The mean is the sum over the
weight: at the start of a recording it is simply the average of what has been heard so far,
with no pull towards zero and no warm-up bias, and the weight says how much that is.

A window of infinite length forgets nothing: it is the whole recording.
"""

import math

import numpy as np


class Windows:
    """``lengths`` are time constants in seconds (``math.inf`` for "everything")."""

    def __init__(self, count, lengths):
        self.lengths = tuple(float(length) for length in lengths)
        self._rate = np.array([[0.0 if math.isinf(t) else -1.0 / t] for t in self.lengths])
        self.count = count
        self.clear()

    def clear(self):
        shape = (len(self.lengths), self.count)
        self.weight, self.total, self.square = np.zeros(shape), np.zeros(shape), np.zeros(shape)

    def add(self, values, seconds):
        """Take ``values`` (NaN where there is none) as standing for ``seconds`` of music."""
        values = np.asarray(values, dtype=np.float64)
        known = np.isfinite(values)
        weight = known * seconds
        values = np.where(known, values, 0.0)
        weighted = values * weight
        decay = np.exp(self._rate * seconds)
        self.weight *= decay
        self.weight += weight
        self.total *= decay
        self.total += weighted
        self.square *= decay
        self.square += weighted * values

    def loosen(self, share):
        """Keep the averages but only ``share`` of their weight: what comes next counts for
        more, and less is claimed to be known."""
        self.weight *= share
        self.total *= share
        self.square *= share

    def mean(self, window):
        weight = self.weight[window]
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(weight > 0, self.total[window] / weight, np.nan)

    def spread(self, window):
        """Standard deviation of each measurement inside the window."""
        weight = self.weight[window]
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = self.total[window] / weight
            variance = self.square[window] / weight - mean * mean
        return np.where(weight > 0, np.sqrt(np.maximum(variance, 0.0)), np.nan)


class LevelDistribution:
    """How the level of the music is distributed inside each window, in steps of a quarter
    of a decibel, for the quantiles the dynamics are read from. Only differences between
    its quantiles are used, so the playback volume cancels out."""

    LOW, HIGH, STEP = -100.0, 12.0, 0.25  # dBFS covered; levels outside go to the ends
    DEPTH_DB = 30.0  # how far under the loud level music still counts (see ``quantiles``)

    def __init__(self, lengths):
        self.lengths = tuple(float(length) for length in lengths)
        self._rate = np.array([[0.0 if math.isinf(t) else -1.0 / t] for t in self.lengths])
        self._count = round((self.HIGH - self.LOW) / self.STEP)
        self._edges = self.LOW + self.STEP * np.arange(self._count + 1)
        self.clear()

    def clear(self):
        self.bins = np.zeros((len(self.lengths), self._count))

    def add(self, level, seconds):
        self.bins *= np.exp(self._rate * seconds)
        index = min(max(int((level - self.LOW) / self.STEP), 0), self._count - 1)
        self.bins[:, index] += seconds

    def loosen(self, share):
        self.bins *= share

    def quantiles(self, window, shares, depth=None):
        """Levels in dB under which the given shares of the window's music lie, or NaN.

        Music more than ``depth`` under the loud level of the window (its 95% level) is
        left out first. Such passages are the ones a lower playback volume would push
        under the silence threshold, where they are not heard at all; counting them only
        when the volume happens to be high would make the result depend on the volume.
        """
        bins = self.bins[window]
        total = float(bins.sum())
        if total <= 0:
            return [math.nan] * len(shares)
        top = self._levels(bins, total, (0.95,))[0]
        depth = self.DEPTH_DB if depth is None else depth
        bins = np.where(self._edges[1:] > top - depth, bins, 0.0)
        return self._levels(bins, float(bins.sum()), shares)

    def _levels(self, bins, total, shares):
        cumulative = np.concatenate(([0.0], np.cumsum(bins) / total))
        # Only the edges of steps that hold something: a share is never placed in the empty
        # stretch between a quiet passage and a loud one.
        filled = bins > 0
        edges = np.concatenate(([filled[0]], filled[1:] | filled[:-1], [filled[-1]]))
        return [float(v) for v in np.interp(shares, cumulative[edges], self._edges[edges])]
