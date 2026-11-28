"""Descriptive group-bootstrap intervals (protocol §12, §12.1).

Paired: each draw resamples the groups with replacement, and both arms' values of a drawn
group enter that draw. Each statistic averages the drawn groups that have it (single-class
groups count only where defined, with multiplicity). A draw with no such group leaves the
statistic undefined for that draw; the count is reported, never hidden. The interval is the
2.5th to 97.5th percentile of the defined draws.

Every interval carries the registered caveat. It is a field fixed at construction, cannot
be passed in or left out, and appears in ``to_dict`` and in ``str``.
"""

from dataclasses import asdict, dataclass, field

import numpy as np

from .screening import READINESS_S_PER_MIN
from .seeds import BOOTSTRAP_SEED, numpy_seed

DRAWS = 10_000
CAVEAT = ("Descriptive only. These treat dependent cross-validation results as independent, ignore "
          "training-set variation, and are probably too narrow. They are not evidence of superiority, "
          "non-inferiority or equivalence.")


@dataclass(frozen=True)
class DescriptiveInterval:
    statistic: str
    estimate: float | None
    low: float | None
    high: float | None
    draws: int
    undefined_draws: int
    note: str = ""
    caveat: str = field(default=CAVEAT, init=False)

    def to_dict(self):
        return asdict(self)

    def __str__(self):
        span = "not available" if self.low is None else f"{self.low:.4g} to {self.high:.4g}"
        text = f"{self.statistic}: {self.estimate if self.estimate is None else f'{self.estimate:.4g}'} ({span})"
        return f"{text}{' ' + self.note if self.note else ''} [{self.caveat}]"


def _means(values, drawn):
    """Mean of each arm over the drawn groups that have a value."""
    kept = [values[g] for g in drawn if g in values]
    if not kept:
        return None
    candidate, baseline = zip(*kept, strict=True)
    return float(np.mean(candidate)), float(np.mean(baseline))


def _draws(groups, draws, seed):
    rng = np.random.default_rng(numpy_seed(seed))
    return [list(rng.choice(groups, size=len(groups), replace=True)) for _ in range(draws)]


def difference(values, draws=DRAWS, seed=BOOTSTRAP_SEED, name="difference"):
    """Interval for mean(candidate) − mean(baseline). ``values``: {group: (candidate, baseline)}
    for the groups where the per-group statistic is defined."""
    groups = sorted(values)
    if not groups:
        raise ValueError("no group has this statistic")
    point = _means(values, groups)
    results, undefined = [], 0
    for drawn in _draws(groups, draws, seed):
        means = _means(values, drawn)
        if means is None:
            undefined += 1
        else:
            results.append(means[0] - means[1])
    low, high = (np.percentile(results, [2.5, 97.5]) if results else (None, None))
    return DescriptiveInterval(name, point[0] - point[1], None if low is None else float(low),
                               None if high is None else float(high), draws, undefined)


def ratio(values, draws=DRAWS, seed=BOOTSTRAP_SEED, name="false-activity ratio"):
    """Interval for mean(candidate) / mean(baseline) false activity. Shown only when the
    point-estimate baseline is at least 3 s/min; then the share of draws whose baseline mean
    fell below 3 s/min is reported in ``note``, and those draws stay in the interval (their
    ratios are finite unless the baseline mean is exactly 0, which leaves that draw undefined)."""
    groups = sorted(values)
    if not groups:
        raise ValueError("no group has this statistic")
    candidate, baseline = _means(values, groups)
    if baseline < READINESS_S_PER_MIN:
        return DescriptiveInterval(name, None, None, None, 0, 0,
                                   note="not shown: the baseline is below 3 s/min (absolute difference applies)")
    results, undefined, below = [], 0, 0
    for drawn in _draws(groups, draws, seed):
        means = _means(values, drawn)
        if means is None or means[1] == 0:
            undefined += 1
            continue
        below += means[1] < READINESS_S_PER_MIN
        results.append(means[0] / means[1])
    low, high = (np.percentile(results, [2.5, 97.5]) if results else (None, None))
    return DescriptiveInterval(name, candidate / baseline, None if low is None else float(low),
                               None if high is None else float(high), draws, undefined,
                               note=f"{below / draws:.1%} of draws had a baseline below 3 s/min")
