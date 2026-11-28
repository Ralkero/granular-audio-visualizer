"""Step 4, S4.9: the main results tables as the disclosure check sees them (protocol §12, §13;
Amendment 3, A3.5 N1). Built and tested on generated worlds only.

**Variables** are each group's values per seed index: C's false activity c[g,s] (s/min, 0-60)
and recall r[g,s] (0-1), and B0ₜ's b[g] and q[g]. Groups fall into two public classes: those with
both classes eligible (``both``) and those with no-voice time only (``fpa_only``); recall exists
only for the first.

**Figures**, in the registered release order (N1.7 item 4), each with its unit (2 decimals for
s/min, 3 on 0-to-1 scales) and zones (N1.6: width ÷ the smallest count of its band):

* ``main``: FPA_macro and R_macro of C (means of the seed-index aggregates) and of B0ₜ, D_FA,
  Δ_R, and the ratio when B0ₜ's shown FPA_macro is at least 3 s/min (§12.1);
* ``seeds``: each seed-index aggregate's FPA_macro, D_FA, R_macro and Δ_R (§12);
* ``verdicts``: §13 as inequalities: D_FA < 0, Δ_R ≥ −0.05 and (when the ratio is used) ratio ≤ 0.5.

Direction counts and the leave-one-group-out count and verdict (with the screening class that
depends on it) are whole-number counts of per-group conditions; they form the registered fixed unit
``counts``, which this layout does not model, so it is withheld whole (N1.7 item 7).

**Protected quantities** (N1.3): each group's value behind each figure (c and r per seed, their
means over seeds, b, q, and the per-group differences), the sum and the difference of any two
groups' values, and each group's direction (both signs possible).

**Exchangeability.** Every figure is a symmetric function of the groups within each class and
every group has the same limits, so the public region is unchanged when groups of one class swap
places. One representative group per class (and one representative pair per pair of classes)
then stands for all: a range shown for the representative holds for every group of its class.
``check_symmetry`` verifies the condition on the built figures.

**Reader knowledge** (N1.1, N1.7 item 8): ``registered_cases`` counts the possibilities the
registered reader would need checked (every group's directions, leave-one-group-out outcomes and
eligibility, none taken from actual values), even after the exchangeability reduction.
``reader="public"`` is an *alternative for comparison, not the registered rule*: a reader who knows
the public record only.
"""

from dataclasses import dataclass
from fractions import Fraction as Q
from math import comb

from . import disclosure

RATE_UNIT, SHARE_UNIT, RATIO_UNIT = Q(1, 100), Q(1, 1000), Q(1, 1000)
RATE_WIDTH, SHARE_WIDTH = Q(1), Q(1, 10)
FPA_HI = Q(60)
READINESS = Q(3)


def band_minimum(n):
    """The smallest group count of the band a count of n falls in (3-4, 5-9, 10+)."""
    return 10 if n >= 10 else 5 if n >= 5 else 3


@dataclass(frozen=True)
class Structure:
    both: int  # groups with both classes eligible
    fpa_only: int  # groups with no-voice time only
    seeds: tuple  # the policy's outer seed indices

    @property
    def groups(self):
        return [f"g{i}" for i in range(self.both + self.fpa_only)]

    @property
    def voiced(self):
        return self.groups[: self.both]

    def classes(self):
        return {"both": self.voiced, "fpa_only": self.groups[self.both :]}


def limits(st):
    out = {}
    for g in st.groups:
        out[f"b[{g}]"] = (Q(0), FPA_HI)
        for s in st.seeds:
            out[f"c[{g},{s}]"] = (Q(0), FPA_HI)
    for g in st.voiced:
        out[f"q[{g}]"] = (Q(0), Q(1))
        for s in st.seeds:
            out[f"r[{g},{s}]"] = (Q(0), Q(1))
    return out


def _mean(names):
    return {n: Q(1, len(names)) for n in names}


def _minus(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, Q(0)) - v
    return out


def forms(st):
    """The linear forms behind the figures."""
    g, v, seeds = st.groups, st.voiced, st.seeds
    f = {"C FPA": _mean([f"c[{x},{s}]" for x in g for s in seeds]), "B0t FPA": _mean([f"b[{x}]" for x in g]),
         "C recall": _mean([f"r[{x},{s}]" for x in v for s in seeds]), "B0t recall": _mean([f"q[{x}]" for x in v])}
    f["D_FA"] = _minus(f["C FPA"], f["B0t FPA"])
    f["Delta_R"] = _minus(f["C recall"], f["B0t recall"])
    for s in seeds:
        f[f"C FPA seed {s}"] = _mean([f"c[{x},{s}]" for x in g])
        f[f"C recall seed {s}"] = _mean([f"r[{x},{s}]" for x in v])
        f[f"D_FA seed {s}"] = _minus(f[f"C FPA seed {s}"], f["B0t FPA"])
        f[f"Delta_R seed {s}"] = _minus(f[f"C recall seed {s}"], f["B0t recall"])
    return f


def _average(name, form, n, rate, cap=None):
    unit, width, hi = (RATE_UNIT, RATE_WIDTH, FPA_HI) if rate else (SHARE_UNIT, SHARE_WIDTH, Q(1))
    return disclosure.Figure(name, form, unit, Q(0), hi, width / band_minimum(n), "main" if "seed" not in name else "seeds",
                             cap=cap if rate else None)


def _difference(name, form, rate):
    unit, hi = (RATE_UNIT, FPA_HI) if rate else (SHARE_UNIT, Q(1))
    return disclosure.Figure(name, form, unit, -hi, hi, None, "main" if "seed" not in name else "seeds")


def figures(st, with_ratio, rule="registered", cap=None):
    """The release order (the ratio only when B0ₜ's shown FPA_macro is ≥ 3 s/min). ``rule="3B"``:
    the proposed Amendment 3B: false-activity averages capped at ``cap``, and the seed-index
    aggregates replaced by one verdict (``seed_verdict``), decided last."""
    f, n, nv = forms(st), len(st.groups), len(st.voiced)
    cap = cap if rule == "3B" else None
    out = [_average("C FPA", f["C FPA"], n, True, cap), _average("B0t FPA", f["B0t FPA"], n, True, cap),
           _difference("D_FA", f["D_FA"], True), _average("C recall", f["C recall"], nv, False),
           _average("B0t recall", f["B0t recall"], nv, False), _difference("Delta_R", f["Delta_R"], False)]
    if with_ratio:
        out.append(disclosure.Figure("ratio", {}, RATIO_UNIT, Q(0), Q(20), None, "main",
                                     ratio=(f["C FPA"], f["B0t FPA"])))
    for s in st.seeds if rule == "registered" else ():
        out += [_average(f"C FPA seed {s}", f[f"C FPA seed {s}"], n, True),
                _difference(f"D_FA seed {s}", f[f"D_FA seed {s}"], True),
                _average(f"C recall seed {s}", f[f"C recall seed {s}"], nv, False),
                _difference(f"Delta_R seed {s}", f[f"Delta_R seed {s}"], False)]
    out.append(disclosure.Figure("D_FA < 0", f["D_FA"], RATE_UNIT, -FPA_HI, FPA_HI, None, "verdicts",
                                 cuts=(("below 0", -FPA_HI, Q(0)), ("not below 0", Q(0), FPA_HI))))
    out.append(disclosure.Figure("Delta_R >= -0.05", f["Delta_R"], SHARE_UNIT, Q(-1), Q(1), None, "verdicts",
                                 cuts=(("below -0.05", Q(-1), Q(-1, 20)), ("at least -0.05", Q(-1, 20), Q(1)))))
    if with_ratio:
        out.append(disclosure.Figure("ratio <= 0.5", {}, RATIO_UNIT, Q(0), Q(20), None, "verdicts",
                                     ratio=(f["C FPA"], f["B0t FPA"]),
                                     cuts=(("at most 0.5", Q(0), Q(1, 2)), ("above 0.5", Q(1, 2), Q(20)))))
    return out


CAPS_3B = {3: Q(35), 5: Q(45), 10: Q(50)}  # proposed 3B, option B: by the smallest count of the band


def proposed_cap(n):
    """The proposed Amendment 3B cap for a false-activity average over n groups: below the value
    60·(m − 1)/m (m the smallest count of n's band) above which a code window could fix every
    group's direction, with a margin."""
    return CAPS_3B[band_minimum(n)]


SEED_VERDICT = "requirement held in every seed-index aggregate"
CODED = "coded with its sources"


def seed_verdict(st):
    """Proposed Amendment 3B, option C: the seed spread as one verdict. "yes": every seed-index
    aggregate has D_FA < 0 and Δ_R ≥ −0.05 (closures); "no": the union over seeds of D_FA ≥ 0 or
    Δ_R ≤ −0.05."""
    f = forms(st)
    holds = [w for s in st.seeds for w in ((f[f"D_FA seed {s}"], -FPA_HI, Q(0)),
                                           (f[f"Delta_R seed {s}"], Q(-1, 20), Q(1)))]
    fails = [[(f[f"D_FA seed {s}"], Q(0), FPA_HI)] for s in st.seeds] + \
        [[(f[f"Delta_R seed {s}"], Q(-1), Q(-1, 20))] for s in st.seeds]
    return disclosure.Verdict(SEED_VERDICT, (("yes", [holds]), ("no", fails)), "verdicts")


def seed_verdict_value(st, truth):
    f = forms(st)

    def value(form):
        return sum((c * truth[k] for k, c in form.items()), Q(0))

    ok = all(value(f[f"D_FA seed {s}"]) < 0 and value(f[f"Delta_R seed {s}"]) >= Q(-1, 20) for s in st.seeds)
    return "yes" if ok else "no"


COUNTS = ("C lower in groups (false activity)", "C higher in groups (false activity)",
          "C lower in groups (recall)", "C higher in groups (recall)",
          "leave-one-group-out recomputations holding", "leave-one-group-out verdict", "screening class")


def _families(st):
    """(family name, {group: form}, class names it covers, width)."""
    g, v, seeds = st.groups, st.voiced, st.seeds
    fam = []
    for s in seeds:
        fam.append((f"c seed {s}", {x: {f"c[{x},{s}]": Q(1)} for x in g}, ("both", "fpa_only"), RATE_WIDTH))
        fam.append((f"r seed {s}", {x: {f"r[{x},{s}]": Q(1)} for x in v}, ("both",), SHARE_WIDTH))
        fam.append((f"c-b seed {s}", {x: {f"c[{x},{s}]": Q(1), f"b[{x}]": Q(-1)} for x in g}, ("both", "fpa_only"),
                    RATE_WIDTH))
        fam.append((f"r-q seed {s}", {x: {f"r[{x},{s}]": Q(1), f"q[{x}]": Q(-1)} for x in v}, ("both",), SHARE_WIDTH))
    k = Q(1, len(seeds))
    fam.append(("c mean", {x: {f"c[{x},{s}]": k for s in seeds} for x in g}, ("both", "fpa_only"), RATE_WIDTH))
    fam.append(("r mean", {x: {f"r[{x},{s}]": k for s in seeds} for x in v}, ("both",), SHARE_WIDTH))
    fam.append(("b", {x: {f"b[{x}]": Q(1)} for x in g}, ("both", "fpa_only"), RATE_WIDTH))
    fam.append(("q", {x: {f"q[{x}]": Q(1)} for x in v}, ("both",), SHARE_WIDTH))
    d = {x: {**{f"c[{x},{s}]": k for s in seeds}, f"b[{x}]": Q(-1)} for x in g}
    e = {x: {**{f"r[{x},{s}]": k for s in seeds}, f"q[{x}]": Q(-1)} for x in v}
    fam.append(("direction FPA", d, ("both", "fpa_only"), None))
    fam.append(("direction recall", e, ("both",), None))
    fam.append(("c mean - b", d, ("both", "fpa_only"), RATE_WIDTH))
    fam.append(("r mean - q", e, ("both",), SHARE_WIDTH))
    return fam


def _add(a, b, sign=1):
    out = dict(a)
    for kk, vv in b.items():
        out[kk] = out.get(kk, Q(0)) + sign * vv
    return {kk: vv for kk, vv in out.items() if vv}


def protected(st):
    """Representative protected quantities (see Exchangeability)."""
    classes = st.classes()
    out = []
    for name, per_group, covered, width in _families(st):
        reps = {c: [x for x in classes[c] if x in per_group][:2] for c in covered if classes[c]}
        if width is None:
            out += [disclosure.Protected(f"{name} [{c}]", per_group[r[0]], kind="signs") for c, r in reps.items() if r]
            continue
        out += [disclosure.Protected(f"{name} [{c}]", per_group[r[0]], width) for c, r in reps.items() if r]
        pairs = [(r[0], r[1]) for r in reps.values() if len(r) > 1]
        cs = [c for c in reps if reps[c]]
        pairs += [(reps[a][0], reps[b][0]) for i, a in enumerate(cs) for b in cs[i + 1 :]]
        for x, y in pairs:
            out.append(disclosure.Protected(f"{name} [{x}+{y}]", _add(per_group[x], per_group[y]), width))
            out.append(disclosure.Protected(f"{name} [{x}-{y}]", _add(per_group[x], per_group[y], -1), width))
    return out


def check_symmetry(st, figure_list):
    """Every figure must weigh the groups of one class equally (the exchangeability condition)."""
    classes = st.classes()
    for fig in figure_list:
        parts = [fig.form] if fig.ratio is None else list(fig.ratio)
        for form in parts:
            for members in classes.values():
                for template in ("c[{},{}]", "r[{},{}]"):
                    for s in st.seeds:
                        values = {form.get(template.format(x, s), Q(0)) for x in members if f"{x}" in st.groups}
                        if len(values) > 1:
                            return False
                for template in ("b[{}]", "q[{}]"):
                    values = {form.get(template.format(x), Q(0)) for x in members}
                    if len(values) > 1:
                        return False
    return True


def registered_cases(st, comparisons=1):
    """The number of reader cases for one representative protected group under the registered
    reader (N1.1, N1.7 item 8), after the exchangeability reduction: each other group's state is
    its direction on false activity and on recall for each comparison (lower, tied, higher), its two
    leave-one-group-out conditions (hold or fail each) and its two eligibilities; the cases are the
    ways of distributing the other groups over those states, times the representative's own state."""
    states = (3 ** (2 * comparisons)) * 4 * 4
    n = len(st.groups)
    return comb(n - 1 + states - 1, states - 1) * states


def canonical(st, values):
    """{variable: Fraction} from per-group values: ``values`` maps the canonical group names (the
    voiced groups first) to {"c": {s: fpa}, "r": {s: recall}, "b": fpa, "q": recall}."""
    out = {}
    for g in st.groups:
        out[f"b[{g}]"] = Q(values[g]["b"])
        for s in st.seeds:
            out[f"c[{g},{s}]"] = Q(values[g]["c"][s])
    for g in st.voiced:
        out[f"q[{g}]"] = Q(values[g]["q"])
        for s in st.seeds:
            out[f"r[{g},{s}]"] = Q(values[g]["r"][s])
    return out


@dataclass
class Outcome:
    reader: str
    shown: dict
    withheld_units: set
    registered_cases: int | None = None
    seconds: float | None = None


def check(st, values, reader="registered", case_limit=disclosure.CHECK_LIMIT, rule="registered", cap=None):
    """Run the disclosure check on the main tables. ``reader="registered"`` applies N1.1 and N1.7
    item 8; ``reader="public"`` is the reader of the proposed Amendment 3B (option A).
    ``rule="3B"`` with ``cap`` applies the proposed options B and C (``figures``, ``seed_verdict``);
    a difference or the ratio whose sources are shown as codes is shown as ``CODED`` (N1.6:
    coded consistently with the figures they come from), which adds nothing to the public record."""
    import time

    start = time.monotonic()
    if rule == "3B" and (reader != "public" or cap is None):
        raise ValueError("the proposed 3B rule uses the public-record reader and a cap")
    truth = canonical(st, values)
    prot = protected(st)
    cases = None
    count = None
    if reader == "registered":
        count = registered_cases(st)
        cases = lambda region: (count, ())  # noqa: E731 - the count alone decides against the size limit
    elif reader != "public":
        raise ValueError("the reader is 'registered' or 'public'")
    checker = disclosure.Checker(limits(st), prot, cases=cases, check_limit=case_limit)
    if not check_symmetry(st, figures(st, with_ratio=True)):
        raise ValueError("the layout's figures are not symmetric within classes")
    first = figures(st, False, rule, cap)[:2]
    report = checker.run(first, truth)
    shown_c, shown_b = report.shown["C FPA"], report.shown["B0t FPA"]
    coded = any(isinstance(v, str) and v != disclosure.WITHHELD for v in (shown_c, shown_b))
    b_above = isinstance(shown_b, str) and shown_b.startswith("above")
    with_ratio = b_above or (not isinstance(shown_b, str) and shown_b >= READINESS)
    rest = [f for f in figures(st, with_ratio, rule, cap) if f.name not in report.shown]
    for fig in rest:
        if rule == "3B" and coded and fig.name in ("D_FA", "ratio"):
            report.shown[fig.name] = CODED
            continue
        decision = checker.decide(fig)
        report.decisions.append(decision)
        if decision.released:
            label = disclosure.show(fig, disclosure._value(fig, truth))
            report.shown[fig.name] = label
            checker.release(fig, label)
        else:
            report.shown[fig.name] = disclosure.WITHHELD
    if rule == "3B":
        verdict = seed_verdict(st)
        decision = checker.decide_verdict(verdict)
        report.decisions.append(decision)
        if decision.released:
            label = seed_verdict_value(st, truth)
            report.shown[verdict.name] = label
            checker.release_verdict(verdict, label)
        else:
            report.shown[verdict.name] = disclosure.WITHHELD
    for name in COUNTS:
        report.shown[name] = disclosure.WITHHELD
    report.withheld_units.add("counts")
    return Outcome(reader, report.shown, report.withheld_units, count, time.monotonic() - start)
