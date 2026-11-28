"""Step 4, S4.9: the disclosure check for continuous figures (protocol Amendment 3, A3.5 N1.1-N1.9)
*[tested before use]*. Exact (``exact``) and decided from public information only. Built and
tested on generated worlds only; the counts of step 2 and 3 are ``disclosure_counts``.

**Figures** are linear forms of protected per-group values (group averages, effect sizes,
duration totals), or ratios of two such forms. Each has its reporting unit, its natural limits,
an optional zone z (N1.6: "below z" within z of the lower limit, "above limit − z" within z of
the upper one, zone edges rounded up to the reporting unit) and its registered fixed unit.

**Protected quantities** (N1.3) are linear forms with a target: a width (N1.4) or, under Fix 1, the
prior range; or a sign quantity (a direction: both strict signs must remain possible).

**The decision for a figure** (N1.7), in the registered release order:

1. Its possible values are bracketed exactly over the public region (registered limits, the
   public record and every figure already released, each as the closed window of what it
   showed). If they cannot be bracketed, its fixed unit is withheld whole (item 7).
2. Every outcome it could show (each code, each rounded number) is checked: the public region
   with the figure inside that outcome's window must keep every protected quantity at its
   target, in every reader case (item 8). An outcome whose region is proven empty cannot occur
   and needs no check. A failed or uncertain verdict makes the figure unsafe: it is suppressed
   (item 6). Too many checks for the registered size limit withholds the figure's unit (items 7, 8).
3. Only then does the true value decide what a released figure shows (item 5); the shown
   outcome's window joins the public region.

The decision code never receives private values (``Checker.decide`` reads only the public
region); ``Checker.run`` passes the true values only to ``show``.

**The extreme-value shortcut** (item 5) is sound for linear figures and linear protected
quantities over a polyhedral public region: for windows [v − h, v + h] of fixed width sliding with
v, the largest value of a protected quantity q is a concave function of v (its hypograph is a
Minkowski sum of convex sets) and the smallest is convex, so the range is concave and smallest at
an end of the interval of possible v; a sign quantity's two conditions (max > 0, min < 0) hold on
intervals of v by the same argument. Only the windows at the two ends of the run of possible
numbers, the windows truncated by a zone edge (which are those end windows) and each code need
checking. It is never used for ratios (their windows rotate rather than slide) or whole-number
counts; those are checked exhaustively (``shortcut=False`` checks every outcome everywhere).

**Stops** (item 9), decided from public information: the already-public record failing on its own
(``check_record``), and a prior range narrower than its width outside Fix 1 (``Protected`` refuses
it). A self-check failure (the true values outside the public region) is a software fault
(``SelfCheckError``).
"""

from dataclasses import dataclass, field
from fractions import Fraction
from math import ceil, floor

from . import exact

WITHHELD = "withheld"  # every withheld or suppressed figure looks the same (N1.7 item 9)
CHECK_LIMIT = 200_000  # registered size limit: protected-quantity checks per figure


class StopForReview(RuntimeError):
    """N1.7 item 9: a stop decided from public information; the message names public facts only."""


class SelfCheckError(RuntimeError):
    """N1.8 item 7: the true values are outside the public region: a software fault."""


@dataclass(frozen=True)
class Figure:
    name: str
    form: dict
    unit: Fraction
    low: Fraction
    high: Fraction
    zone: Fraction | None = None
    fixed_unit: str = ""
    ratio: tuple | None = None  # (numerator form, denominator form); ``form`` unused
    cuts: tuple | None = None  # ((label, lo, hi), ...): a verdict shown as one of these intervals

    def zone_edges(self):
        if self.zone is None:
            return None, None
        z = ceil(self.zone / self.unit) * self.unit  # rounded up to the reporting unit
        return self.low + z, self.high - z


@dataclass(frozen=True)
class Protected:
    name: str
    form: dict
    target: Fraction = Fraction(0)
    kind: str = "width"  # "width" or "signs"
    width: Fraction | None = None  # the registered width when ``target`` is a Fix 1 prior range
    fix1: bool = False

    def __post_init__(self):
        if self.kind not in ("width", "signs"):
            raise ValueError("a protected quantity is a width or a sign quantity")
        if self.kind == "width" and self.width is not None and self.target < self.width and not self.fix1:
            raise StopForReview("a prior range is narrower than its width outside Fix 1")


@dataclass(frozen=True)
class Outcome:
    label: object  # a code string or a rounded number
    lo: Fraction
    hi: Fraction


def _round(value, unit):
    """Round half to even on the unit (Python's rule)."""
    return round(Fraction(value) / unit) * unit


def _numbers_span(figure):
    low_edge, high_edge = figure.zone_edges()
    return (figure.low if low_edge is None else low_edge), (figure.high if high_edge is None else high_edge)


def _number(figure, k):
    """The outcome of the rounded number k·unit, or None if its window is empty."""
    numbers_lo, numbers_hi = _numbers_span(figure)
    v = k * figure.unit
    w_lo, w_hi = max(v - figure.unit / 2, numbers_lo), min(v + figure.unit / 2, numbers_hi)
    return Outcome(v, w_lo, w_hi) if w_lo <= w_hi else None


def _codes(figure, lo, hi):
    low_edge, high_edge = figure.zone_edges()
    out = []
    if low_edge is not None and lo <= low_edge:
        out.append(Outcome(f"below {low_edge}", figure.low, low_edge))
    if high_edge is not None and hi >= high_edge:
        out.append(Outcome(f"above {high_edge}", high_edge, figure.high))
    return out


def _number_range(figure, lo, hi):
    """(first k, last k) of the rounded numbers whose windows meet [lo, hi], or None."""
    numbers_lo, numbers_hi = _numbers_span(figure)
    a, b = max(lo, numbers_lo), min(hi, numbers_hi)
    if a > b:
        return None
    first, last = floor(_round(a, figure.unit) / figure.unit), ceil(_round(b, figure.unit) / figure.unit)
    while first <= last and (_number(figure, first) is None or _number(figure, first).hi < lo):
        first += 1
    while last >= first and (_number(figure, last) is None or _number(figure, last).lo > hi):
        last -= 1
    return (first, last) if first <= last else None


def outcomes(figure, lo, hi):
    """Every outcome of ``figure`` whose window meets [lo, hi]: codes and rounded numbers (or its
    cuts, for a verdict)."""
    if figure.cuts is not None:
        return [Outcome(label, a, b) for label, a, b in figure.cuts if b >= lo and a <= hi]
    span = _number_range(figure, lo, hi)
    numbers = [] if span is None else [o for o in (_number(figure, k) for k in range(span[0], span[1] + 1)) if o]
    codes = _codes(figure, lo, hi)
    return [o for o in codes if o.label.startswith("below")] + numbers + [o for o in codes if o.label.startswith("above")]


def outcome_of(figure, label):
    """The outcome a shown label stands for."""
    if figure.cuts is not None:
        match = [Outcome(lab, a, b) for lab, a, b in figure.cuts if lab == label]
        if len(match) != 1:
            raise ValueError("the shown label is not one of the verdict's cuts")
        return match[0]
    if isinstance(label, str):
        match = [o for o in _codes(figure, figure.low, figure.high) if o.label == label]
        if len(match) != 1:
            raise ValueError("the shown label is not one of the figure's codes")
        return match[0]
    k = Fraction(label) / figure.unit
    o = None if k.denominator != 1 else _number(figure, int(k))
    if o is None:
        raise ValueError("the shown label is not one of the figure's numbers")
    return o


def show(figure, value):
    """What a released figure shows for its true value (codes at the zones, else the rounded number;
    for a verdict, the first cut whose upper end lies above the value, or the last)."""
    if figure.cuts is not None:
        for i, (label, _a, b) in enumerate(figure.cuts):
            if value < b or i == len(figure.cuts) - 1:
                return label
    low_edge, high_edge = figure.zone_edges()
    if low_edge is not None and value < low_edge:
        return f"below {low_edge}"
    if high_edge is not None and value > high_edge:
        return f"above {high_edge}"
    return _round(value, figure.unit)


@dataclass
class Decision:
    figure: str
    released: bool
    reason: str  # public terms only
    checks: int = 0
    outcomes_checked: int = 0


@dataclass
class Report:
    shown: dict = field(default_factory=dict)  # figure name: label or WITHHELD
    decisions: list = field(default_factory=list)
    withheld_units: set = field(default_factory=set)


class Checker:
    """``limits``: {variable: (lo, hi)}; ``record``: [(form, lo, hi)] already public; ``protected``:
    the protected quantities; ``cases``: a function region → (count, iterable of extra
    constraint lists) for reader knowledge that is not public (default: none)."""

    def __init__(self, limits, protected, record=(), shortcut=True, cases=None, check_limit=CHECK_LIMIT):
        self.region = exact.Region(limits)
        for form, lo, hi in record:
            self.region.add_window(form, lo, hi)
        self.protected = list(protected)
        self.shortcut = shortcut
        self.cases = cases
        self.check_limit = check_limit
        self.report = Report()

    # -- public decisions --------------------------------------------------------------------------

    def check_record(self):
        """N1.7 item 9: stop when the public record already fails on its own."""
        for p in self.protected:
            verdict = self._verdict(self.region, p, [])
            if verdict == exact.EMPTY:
                raise StopForReview("the public record is inconsistent with the registered structure")
            if verdict != exact.PASS:
                raise StopForReview("the already-public record fails the disclosure check on its own")

    def _verdict(self, region, p, pool):
        if p.kind == "signs":
            return exact.signs(region, p.form, pool)
        return exact.width(region, p.form, p.target, pool)

    def _windowed(self, figure, outcome):
        region = self.region.copy()
        if figure.ratio is None:
            region.add_window(figure.form, outcome.lo, outcome.hi)
        else:
            numerator, denominator = figure.ratio
            region.add_ge(_combine(numerator, denominator, -outcome.lo), Fraction(0))
            region.add_le(_combine(numerator, denominator, -outcome.hi), Fraction(0))
        return region

    def _range(self, figure):
        if figure.ratio is None:
            e = self.region.extent(figure.form)
            if e.empty or not e.certain:
                return None
            return e.min_lo, e.max_hi
        return self._ratio_range(figure)

    def _ratio_possible(self, figure, side, r):
        """Whether the ratio can be ≥ r (side +1) or ≤ r (side −1); None if uncertain."""
        numerator, denominator = figure.ratio
        region = self.region.copy()
        form = _combine(numerator, denominator, -r)
        if side > 0:
            region.add_ge(form, Fraction(0))
        else:
            region.add_le(form, Fraction(0))
        empty = region.empty()
        return None if empty is None else not empty

    def _ratio_range(self, figure):
        """The ratio's possible values, bracketed to one reporting unit by exact bisection."""
        lo, hi = figure.low, figure.high
        out = []
        for side in (-1, 1):
            a, b = lo, hi  # invariant: the bracket holds the extreme on this side
            while b - a > figure.unit:
                m = _round((a + b) / 2, figure.unit)
                if m in (a, b):
                    break
                possible = self._ratio_possible(figure, side, m)
                if possible is None:
                    return None
                if side > 0:
                    a, b = (m, b) if possible else (a, m)
                else:
                    a, b = (a, m) if possible else (m, b)
            out.append((a, b))
        (min_a, _min_b), (_max_a, max_b) = out
        return min_a, max_b

    def _withhold_unit(self, figure, reason, checks=0, checked=0):
        if not figure.fixed_unit:
            raise StopForReview("a figure with no registered fixed unit could not be decided from public information")
        self.report.withheld_units.add(figure.fixed_unit)
        return Decision(figure.name, False, reason, checks, checked)

    def decide(self, figure):
        """Release or not, from the public region only."""
        if figure.fixed_unit and figure.fixed_unit in self.report.withheld_units:
            return Decision(figure.name, False, "its fixed unit is withheld")
        bracket = self._range(figure)
        if bracket is None:
            return self._withhold_unit(figure, "its possible values could not be bracketed: its fixed unit is withheld")
        if self.shortcut and figure.ratio is None and figure.cuts is None:
            candidates = self._shortcut(figure, *bracket)
            if candidates is None:
                return self._withhold_unit(figure, "its possible outcomes could not be decided: its fixed unit is withheld")
        else:
            candidates = outcomes(figure, *bracket)
        checks = checked = 0
        for outcome in candidates:
            region = self._windowed(figure, outcome)
            if region.empty() is True:
                continue
            checked += 1
            count, case_constraints = (1, [[]]) if self.cases is None else self.cases(region)
            if checks + count * len(self.protected) > self.check_limit:
                return self._withhold_unit(figure, "more checks than the registered size limit: its fixed unit is withheld",
                                           checks, checked)
            for extra in case_constraints:
                case = region.copy()
                for form, lo, hi in extra:
                    case.add_window(form, lo, hi)
                if extra and case.empty() is True:
                    continue
                pool = [p for p in (case.interior(),) if p is not None]  # witnesses found in this case
                for p in self.protected:
                    checks += 1
                    verdict = self._verdict(case, p, pool)
                    if verdict == exact.EMPTY:
                        break
                    if verdict != exact.PASS:
                        return Decision(figure.name, False, "an outcome it could show would narrow a protected quantity",
                                        checks, checked)
        return Decision(figure.name, True, "every outcome it could show keeps every protected quantity", checks, checked)

    def _shortcut(self, figure, lo, hi):
        """Each code, the first and last possible numbers, and next to each of those the first full
        window when it was truncated (see the module docstring). None if emptiness is uncertain."""
        codes = _codes(figure, lo, hi)
        span = _number_range(figure, lo, hi)
        if span is None:
            return codes
        keep = []
        for ks in (range(span[0], span[1] + 1), range(span[1], span[0] - 1, -1)):
            for k in ks:
                o = _number(figure, k)
                empty = self._windowed(figure, o).empty()
                if empty is None:
                    return None
                if not empty:
                    keep.append(k)
                    if o.hi - o.lo < figure.unit:  # truncated by a zone edge or a limit
                        nxt = k + ks.step
                        if span[0] <= nxt <= span[1]:
                            keep.append(nxt)
                    break
        return codes + [_number(figure, k) for k in sorted(set(keep))]

    def release(self, figure, label):
        """Add a released figure's shown outcome to the public region."""
        self.region = self._windowed(figure, outcome_of(figure, label))

    # -- the driver ------------------------------------------------------------------------------

    def run(self, figures, truth):
        """Decide each figure in the registered order; ``truth`` ({variable: Fraction}) is used
        only by ``show`` and the self-check."""
        x = self.region.vector(truth)
        if not self.region.contains(x):
            raise SelfCheckError("the true values are outside the public record")
        self.check_record()
        for figure in figures:
            decision = self.decide(figure)
            self.report.decisions.append(decision)
            if decision.released:
                label = show(figure, _value(figure, truth))
                self.report.shown[figure.name] = label
                self.release(figure, label)
                if not self.region.contains(x):
                    raise SelfCheckError("the true values left the public region after a release")
            else:
                self.report.shown[figure.name] = WITHHELD
        return self.report


def _combine(numerator, denominator, factor):
    out = dict((k, Fraction(v)) for k, v in numerator.items())
    for k, v in denominator.items():
        out[k] = out.get(k, Fraction(0)) + factor * Fraction(v)
    return out


def _value(figure, truth):
    if figure.ratio is None:
        return sum((Fraction(c) * truth[k] for k, c in figure.form.items()), Fraction(0))
    numerator, denominator = figure.ratio
    num = sum((Fraction(c) * truth[k] for k, c in numerator.items()), Fraction(0))
    den = sum((Fraction(c) * truth[k] for k, c in denominator.items()), Fraction(0))
    return num / den
