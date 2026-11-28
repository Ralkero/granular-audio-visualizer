"""Step 4, S4.9 (first part): exact rational verification for the disclosure check (protocol
Amendment 3, A3.5 N1.8) *[tested before use]*. SciPy's HiGHS proposes; every verdict is proven in
exact rational arithmetic. Built and tested on generated systems only.

**A region** is {x : lo ≤ x ≤ hi, a·x ≤ b for each row, a·x = b for each equality}, with every
number an exact fraction. It holds only public information: registered limits, already
published figures (as the intervals their rounding allows) and the windows of a figure being
checked. True values are never an input.

**The extent of a linear form q** is bracketed exactly: [min_lo, min_hi] holds the minimum and
[max_lo, max_hi] the maximum.

* *Upper side of the maximum (and lower side of the minimum):* weak duality. For any multipliers
  y ≥ 0 on the rows and μ on the equalities, max q ≤ y·b + μ·e + Σⱼ max(rⱼ·loⱼ, rⱼ·hiⱼ) with
  r = q − Aᵀy − Eᵀμ. HiGHS's multipliers are turned into fractions and clipped to the right
  sign, so the bound is exact whatever their accuracy (N1.8 (3)).
* *Lower side of the maximum:* an exactly feasible point. HiGHS's dual simplex returns a vertex;
  the variables at a limit are fixed there and the binding rows are solved exactly for the rest
  (N1.8 (4)). If that point is not exactly feasible, the proposal is repaired toward an exact
  point in the relative interior of the region (N1.8 (5)): x = c + λ(p − c) with the largest λ ≤ 1
  that keeps every row, computed exactly.
* *The relative-interior point* is computed from the region alone (public information, N1.7 (3)):
  forced equalities are found and proven exactly (a row whose largest possible slack is proven to
  be 0), proposals are projected exactly onto them, and every other row must then hold with
  positive slack (N1.8 (5), (6)).
* *Emptiness* is proven by a positive exact lower bound on the smallest total violation.

**Verdicts:** ``width`` passes when two exactly feasible points differ by at least the width (a
range exactly equal to its width passes) and fails when the exact bounds prove the range
narrower; ``signs`` passes when exactly feasible points of both strict signs exist. Anything else
is ``UNCERTAIN``, which every caller treats as a failure (N1.8 (8)).
"""

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import csr_matrix

PASS, FAIL, UNCERTAIN = "pass", "fail", "uncertain"
TOLERANCE = 1e-9
DENOMINATOR = 10**12
INTERIOR_ROUNDS = 64


def frac(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, str)):
        return Fraction(value)
    raise TypeError("exact inputs must be fractions, integers or decimal strings")


def _rational(x):
    return Fraction(float(x)).limit_denominator(DENOMINATOR)


@dataclass
class Extent:
    empty: bool = False
    min_lo: Fraction | None = None
    min_hi: Fraction | None = None
    max_lo: Fraction | None = None
    max_hi: Fraction | None = None
    points: list = field(default_factory=list)  # exactly feasible points found on the way

    @property
    def certain(self):
        return self.empty or None not in (self.min_lo, self.min_hi, self.max_lo, self.max_hi)


class Region:
    """A polyhedron over named variables, in exact fractions (see the module docstring)."""

    def __init__(self, limits):
        """``limits``: {name: (lo, hi)} with lo ≤ hi."""
        self.names = sorted(limits)
        self.index = {n: i for i, n in enumerate(self.names)}
        self.lo = [frac(limits[n][0]) for n in self.names]
        self.hi = [frac(limits[n][1]) for n in self.names]
        if any(a > b for a, b in zip(self.lo, self.hi, strict=True)):
            raise ValueError("a variable's lower limit is above its upper limit")
        self.rows = []  # (terms {index: coefficient}, b): terms·x ≤ b
        self.equalities = []  # (terms, b): terms·x = b
        self._interior = None
        self._arrays = None

    def copy(self):
        other = Region.__new__(Region)
        other.names, other.index, other.lo, other.hi = self.names, self.index, list(self.lo), list(self.hi)
        other.rows, other.equalities = list(self.rows), list(self.equalities)
        other._interior, other._arrays = None, None
        return other

    def _terms(self, form):
        terms = {}
        for name, coefficient in form.items():
            c = frac(coefficient)
            if c:
                terms[self.index[name]] = terms.get(self.index[name], Fraction(0)) + c
        return {k: v for k, v in terms.items() if v}

    def _changed(self):
        self._interior, self._arrays = None, None

    def add_le(self, form, b):
        self.rows.append((self._terms(form), frac(b)))
        self._changed()

    def add_ge(self, form, b):
        self.add_le({k: -frac(v) for k, v in form.items()}, -frac(b))

    def add_eq(self, form, b):
        self.equalities.append((self._terms(form), frac(b)))
        self._changed()

    def add_window(self, form, lo, hi):
        """lo ≤ form ≤ hi (closed; rounding windows enter as their closures)."""
        if lo is not None:
            self.add_ge(form, lo)
        if hi is not None:
            self.add_le(form, hi)

    # -- exact evaluation -----------------------------------------------------------------------

    def vector(self, values):
        return [frac(values[n]) for n in self.names]

    def value(self, form, x):
        return sum((frac(c) * x[self.index[n]] for n, c in form.items()), Fraction(0))

    @staticmethod
    def _dot(terms, x):
        return sum((c * x[j] for j, c in terms.items()), Fraction(0))

    def contains(self, x):
        if any(not (lo <= v <= hi) for lo, v, hi in zip(self.lo, x, self.hi, strict=True)):
            return False
        return all(self._dot(t, x) <= b for t, b in self.rows) and all(self._dot(t, x) == b for t, b in self.equalities)

    # -- floating-point proposals ------------------------------------------------------------------

    def _matrices(self):
        if self._arrays is None:
            n = len(self.names)

            def sparse(rows):
                data, ri, ci = [], [], []
                for r, (t, _b) in enumerate(rows):
                    for j, c in t.items():
                        data.append(float(c))
                        ri.append(r)
                        ci.append(j)
                return csr_matrix((data, (ri, ci)), shape=(len(rows), n)) if rows else None

            self._arrays = (sparse(self.rows), np.array([float(b) for _t, b in self.rows]) if self.rows else None,
                            sparse(self.equalities),
                            np.array([float(b) for _t, b in self.equalities]) if self.equalities else None,
                            [(float(a), float(b)) for a, b in zip(self.lo, self.hi, strict=True)])
        return self._arrays

    def _propose(self, c, maximise):
        a_ub, b_ub, a_eq, b_eq, bounds = self._matrices()
        objective = -np.asarray(c, dtype=float) if maximise else np.asarray(c, dtype=float)
        return linprog(objective, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")

    # -- exact proofs ----------------------------------------------------------------------------

    def _dual_bound(self, terms, result, maximise):
        """An exact bound on max (upper) or min (lower) of ``terms``·x from HiGHS's multipliers."""
        sign = 1 if maximise else -1
        best = None
        rows_m = np.asarray(result.ineqlin.marginals) if self.rows else np.zeros(0)
        eq_m = np.asarray(result.eqlin.marginals) if self.equalities else np.zeros(0)
        for flip in (1, -1):
            y = [max(Fraction(0), _rational(flip * m)) for m in rows_m]
            mu = [_rational(flip * m) for m in eq_m]
            r = {j: sign * c for j, c in terms.items()}
            bound = Fraction(0)
            for yi, (t, b) in zip(y, self.rows, strict=True):
                if yi:
                    bound += yi * b
                    for j, c in t.items():
                        r[j] = r.get(j, Fraction(0)) - yi * c
            for mi, (t, b) in zip(mu, self.equalities, strict=True):
                if mi:
                    bound += mi * b
                    for j, c in t.items():
                        r[j] = r.get(j, Fraction(0)) - mi * c
            for j in range(len(self.names)):
                rj = r.get(j, Fraction(0))
                if rj:
                    bound += max(rj * self.lo[j], rj * self.hi[j])
            best = bound if best is None or bound < best else best
        return sign * best

    def _vertex(self, xf):
        """The vertex HiGHS proposed, solved exactly from its limits and binding rows."""
        n = len(self.names)
        x = [None] * n
        for j in range(n):
            scale = TOLERANCE * (1 + abs(float(self.hi[j])) + abs(float(self.lo[j])))
            if abs(xf[j] - float(self.lo[j])) <= scale:
                x[j] = self.lo[j]
            elif abs(xf[j] - float(self.hi[j])) <= scale:
                x[j] = self.hi[j]
        free = [j for j in range(n) if x[j] is None]
        if free:
            binding = list(self.equalities) + [(t, b) for t, b in self.rows
                                               if abs(sum(float(c) * xf[j] for j, c in t.items()) - float(b))
                                               <= TOLERANCE * (1 + abs(float(b)))]
            system = []
            for t, b in binding:
                rhs = b - sum((c * x[j] for j, c in t.items() if x[j] is not None), Fraction(0))
                row = {j: c for j, c in t.items() if x[j] is None}
                if row:
                    system.append((row, rhs))
            solved = _solve(system, free, {j: _rational(xf[j]) for j in free})
            if solved is None:
                return None
            for j in free:
                x[j] = solved[j]
        return x if self.contains(x) else None

    def interior(self):
        """An exact point of the relative interior, from the region alone; None if uncertain or
        empty."""
        if self._interior is None:
            self._interior = _interior(self) or False
        return self._interior[0] if self._interior else None

    def forced_equalities(self):
        """The region's equalities plus every row proven to be forced (after ``interior``)."""
        self.interior()
        return list(self._interior[1]) if self._interior else None

    def _repair(self, p):
        c = self.interior()
        if c is None:
            return None
        p = _project(self.forced_equalities(), p)
        if p is None:
            return None
        lam = Fraction(1)
        rows = list(self.rows) + [({j: Fraction(1)}, self.hi[j]) for j in range(len(self.names))] + \
            [({j: Fraction(-1)}, -self.lo[j]) for j in range(len(self.names))]
        for t, b in rows:
            step = self._dot(t, p) - self._dot(t, c)
            if step > 0:
                lam = min(lam, (b - self._dot(t, c)) / step)
        if lam < 0:
            return None
        x = [ci + lam * (pi - ci) for ci, pi in zip(c, p, strict=True)]
        return x if self.contains(x) else None

    def _point(self, result):
        x = self._vertex(result.x)
        if x is None:
            x = self._repair([_rational(v) for v in result.x])
        return x

    def extent(self, form):
        terms = self._terms(form)
        c = np.zeros(len(self.names))
        for j, v in terms.items():
            c[j] = float(v)
        out = Extent()
        for maximise in (True, False):
            result = self._propose(c, maximise)
            if result.status == 2:
                if self.empty() is True:
                    out.empty = True
                    return out
                return out  # uncertain
            if result.status != 0:
                return out
            point = self._point(result)
            bound = self._dual_bound(terms, result, maximise)
            value = None if point is None else self._dot(terms, point)
            if point is not None:
                out.points.append(point)
            if maximise:
                out.max_lo, out.max_hi = value, bound
            else:
                out.min_lo, out.min_hi = bound, value
        return out

    def empty(self):
        """True if proven empty, False if a point is known, None if uncertain."""
        if self.interior() is not None:
            return False
        violation = _violation_bound(self)
        if violation is not None and violation > 0:
            return True
        return None


# -- exact linear algebra ------------------------------------------------------------------------


def _solve(system, unknowns, fallback):
    """Solve ``system`` ([(terms, rhs)]) exactly for ``unknowns`` (Gauss-Jordan). Unknowns the
    system leaves free take their ``fallback`` value. None if the system is inconsistent."""
    rows = [(dict(t), r) for t, r in system]
    pivots = []  # (unknown, row index)
    used = set()
    for j in unknowns:
        k = next((i for i, (t, _r) in enumerate(rows) if i not in used and t.get(j)), None)
        if k is None:
            continue
        t, r = rows[k]
        p = t[j]
        t, r = {kk: v / p for kk, v in t.items()}, r / p
        rows[k] = (t, r)
        for i, (u, s) in enumerate(rows):
            f = u.get(j)
            if i != k and f:
                u = dict(u)
                for kk, v in t.items():
                    u[kk] = u.get(kk, Fraction(0)) - f * v
                rows[i] = ({kk: v for kk, v in u.items() if v}, s - f * r)
        pivots.append((j, k))
        used.add(k)
    if any(i not in used and not t and r != 0 for i, (t, r) in enumerate(rows)):
        return None
    pivot_of = {j for j, _k in pivots}
    values = {j: fallback[j] for j in unknowns if j not in pivot_of}
    for j, k in pivots:
        t, r = rows[k]
        values[j] = r - sum((v * values[kk] for kk, v in t.items() if kk != j), Fraction(0))
    if any(sum((c * values[kk] for kk, c in t.items()), Fraction(0)) != rhs for t, rhs in system):
        return None
    return values


def _project(equalities, p):
    """The exact orthogonal projection of ``p`` onto the equalities' affine set (None if empty)."""
    if not equalities:
        return list(p)
    rows = [(t, b - sum((c * p[j] for j, c in t.items()), Fraction(0))) for t, b in equalities]
    m = len(rows)
    system = []
    for i in range(m):
        gram = {}
        for k in range(m):
            v = sum((c * rows[k][0].get(j, Fraction(0)) for j, c in rows[i][0].items()), Fraction(0))
            if v:
                gram[k] = v
        system.append((gram, rows[i][1]))
    mu = _solve(system, list(range(m)), {k: Fraction(0) for k in range(m)})
    if mu is None:
        return None
    x = list(p)
    for i, (t, _r) in enumerate(rows):
        if mu[i]:
            for j, c in t.items():
                x[j] += mu[i] * c
    return x


def _bound_rows(region):
    rows = []
    for j in range(len(region.names)):
        if region.lo[j] != region.hi[j]:
            rows += [({j: Fraction(1)}, region.hi[j]), ({j: Fraction(-1)}, -region.lo[j])]
    return rows


def _interior(region):
    """(point, equalities) for ``Region.interior``: rows proven to be forced equalities join the
    region's own equalities; every other row and limit holds with positive slack at the point."""
    n = len(region.names)
    fixed = [({j: Fraction(1)}, region.lo[j]) for j in range(n) if region.lo[j] == region.hi[j]]
    equalities = list(region.equalities) + fixed
    inequalities = list(region.rows) + _bound_rows(region)
    for _round in range(INTERIOR_ROUNDS):
        candidate = _max_slack_point(region, inequalities, equalities)
        if candidate is None:
            return None
        point, slack = candidate
        if slack > 1e-7:
            x = _project(equalities, [_rational(v) for v in point])
            if x is not None and all(Region._dot(t, x) < b for t, b in inequalities) and \
                    all(Region._dot(t, x) == b for t, b in equalities):
                return x, equalities
        probe = region.copy()
        probe.equalities = list(equalities)
        forced = [row for row in inequalities if _proven_tight(probe, row)]
        if not forced:
            return None
        equalities += forced
        inequalities = [row for row in inequalities if not any(row is f for f in forced)]
    return None


def _max_slack_point(region, inequalities, equalities):
    """max s subject to a·x + s ≤ b for every inequality (limits included), the equalities, and
    0 ≤ s ≤ 1."""
    n = len(region.names)
    data, ri, ci = [], [], []
    for r, (t, _b) in enumerate(inequalities):
        for j, c in t.items():
            data.append(float(c))
            ri.append(r)
            ci.append(j)
        data.append(1.0)
        ri.append(r)
        ci.append(n)
    a_ub = csr_matrix((data, (ri, ci)), shape=(len(inequalities), n + 1)) if inequalities else None
    b_ub = np.array([float(b) for _t, b in inequalities]) if inequalities else None
    if equalities:
        data, ri, ci = [], [], []
        for r, (t, _b) in enumerate(equalities):
            for j, c in t.items():
                data.append(float(c))
                ri.append(r)
                ci.append(j)
        a_eq = csr_matrix((data, (ri, ci)), shape=(len(equalities), n + 1))
        b_eq = np.array([float(b) for _t, b in equalities])
    else:
        a_eq = b_eq = None
    c = np.zeros(n + 1)
    c[n] = -1.0
    bounds = [(float(a), float(b)) for a, b in zip(region.lo, region.hi, strict=True)] + [(0.0, 1.0)]
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, A_eq=a_eq, b_eq=b_eq, bounds=bounds, method="highs-ds")
    if result.status != 0:
        return None
    return result.x[:n], result.x[n]


def _proven_tight(region, row):
    """True if ``row`` (a·x ≤ b) is proven to hold with equality everywhere in ``region``: an
    exact lower bound on min a·x reaches b."""
    t, b = row
    c = np.zeros(len(region.names))
    for j, v in t.items():
        c[j] = float(v)
    result = region._propose(c, maximise=False)
    if result.status != 0:
        return False
    return region._dual_bound(t, result, maximise=False) >= b


def _violation_bound(region):
    """An exact lower bound on the smallest total violation of the rows and equalities over the
    limits (Farkas): −y·b + Σⱼ min(rⱼ·loⱼ, rⱼ·hiⱼ) with r = Aᵀy for any y ≥ 0. Above 0 proves the
    region empty."""
    n = len(region.names)
    rows = list(region.rows) + list(region.equalities) + [({j: -c for j, c in t.items()}, -b) for t, b in region.equalities]
    if not rows:
        return None
    data, ri, ci = [], [], []
    for r, (t, _b) in enumerate(rows):
        for j, c in t.items():
            data.append(float(c))
            ri.append(r)
            ci.append(j)
        data.append(-1.0)  # a·x − v ≤ b, v ≥ 0 the violation
        ri.append(r)
        ci.append(n)
    a_ub = csr_matrix((data, (ri, ci)), shape=(len(rows), n + 1))
    b_ub = np.array([float(b) for _t, b in rows])
    c = np.zeros(n + 1)
    c[n] = 1.0
    bounds = [(float(a), float(b)) for a, b in zip(region.lo, region.hi, strict=True)] + [(0.0, None)]
    result = linprog(c, A_ub=a_ub, b_ub=b_ub, bounds=bounds, method="highs-ds")
    if result.status != 0:
        return None
    y = [max(Fraction(0), _rational(-m)) for m in result.ineqlin.marginals]
    bound = Fraction(0)
    r = {}
    for yi, (t, b) in zip(y, rows, strict=True):
        if yi:
            bound -= yi * b
            for j, c in t.items():
                r[j] = r.get(j, Fraction(0)) + yi * c
    for j in range(n):
        rj = r.get(j, Fraction(0))
        if rj:
            bound += min(rj * region.lo[j], rj * region.hi[j])
    return bound


# -- verdicts ----------------------------------------------------------------------------------------


EMPTY = "empty"
POOL = 64  # witnesses kept per region


def _keep(pool, points):
    """Keep exactly feasible points as witnesses for later checks in the same region."""
    if isinstance(pool, list):
        pool.extend(points)
        del pool[:-POOL]


def width(region, form, target, pool=()):
    """PASS if the form's range in ``region`` is at least ``target``, FAIL if it is proven
    narrower, UNCERTAIN otherwise, EMPTY if the region is proven empty. ``pool``: points already
    known to be exactly inside ``region`` (witnesses; they decide only a pass)."""
    target = frac(target)
    values = [region.value(form, x) for x in pool]
    if values and max(values) - min(values) >= target:
        return PASS
    e = region.extent(form)
    _keep(pool, e.points)
    if e.empty:
        return EMPTY
    if e.max_lo is not None and e.min_hi is not None and e.max_lo - e.min_hi >= target:
        return PASS
    if e.max_hi is not None and e.min_lo is not None and e.max_hi - e.min_lo < target:
        return FAIL
    return UNCERTAIN


def signs(region, form, pool=()):
    """PASS if the form can be both above and below zero in ``region``; FAIL if one side is proven
    impossible; UNCERTAIN otherwise; EMPTY if the region is proven empty."""
    values = [region.value(form, x) for x in pool]
    if any(v > 0 for v in values) and any(v < 0 for v in values):
        return PASS
    e = region.extent(form)
    _keep(pool, e.points)
    if e.empty:
        return EMPTY
    if e.max_lo is not None and e.min_hi is not None and e.max_lo > 0 and e.min_hi < 0:
        return PASS
    if (e.max_hi is not None and e.max_hi <= 0) or (e.min_lo is not None and e.min_lo >= 0):
        return FAIL
    return UNCERTAIN
