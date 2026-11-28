"""A5.2 Step 4, S4.9: the disclosure check for continuous figures (protocol Amendment 3, A3.5
N1.1-N1.9; A3.10 "Disclosure checker (N1)"), on generated worlds only.

The worlds are small on purpose (two to four groups, values in s/min on [0, 60]) so that every
case can be listed; the mechanisms are those of the regression cases from the synthetic width
tests, rebuilt here."""

import itertools
import os
import subprocess
import sys
from fractions import Fraction as Q

import pytest

from gmv.character.research.presence_model import disclosure, exact

U = Q(1, 100)  # s/min, 2 decimals
W = Q(1)  # the width for false-activity rates


def limits(n, hi=60):
    return {f"x{i}": (Q(0), Q(hi)) for i in range(n)}


def mean_of(groups):
    return {f"x{i}": Q(1, len(groups)) for i in groups}


def protected(n, width=W, pairs=True):
    out = [disclosure.Protected(f"x{i}", {f"x{i}": 1}, width) for i in range(n)]
    if pairs:
        for i, j in itertools.combinations(range(n), 2):
            out.append(disclosure.Protected(f"x{i}+x{j}", {f"x{i}": 1, f"x{j}": 1}, width))
            out.append(disclosure.Protected(f"x{i}-x{j}", {f"x{i}": 1, f"x{j}": -1}, width))
    return out


def figure(name, groups, zone="registered", unit="row", high=Q(60)):
    """A group average; by default with the registered zone at both limits (N1.6), width ÷ count."""
    groups = tuple(groups)
    zone = W / len(groups) if zone == "registered" else zone
    return disclosure.Figure(name, mean_of(groups), U, Q(0), high, zone, unit)


def truth(*values):
    return {f"x{i}": Q(v) for i, v in enumerate(values)}


def run(n, figures, values, **kwargs):
    return disclosure.Checker(limits(n), protected(n), **kwargs).run(figures, truth(*values))


# -- regression mechanisms ------------------------------------------------------------------------


def test_one_group_left_out_of_a_row_would_rebuild_it_and_is_suppressed():
    figures = [figure("all", range(3)), figure("without x2", (0, 1))]
    report = run(3, figures, ("2.5", "3.1", "0.4"))
    assert report.shown["all"] == Q(2) and report.shown["without x2"] == disclosure.WITHHELD


def test_without_a_zone_an_average_near_a_limit_is_suppressed():
    """N1.6's zones exist because the outcome "0.00" would pin every group near 0."""
    report = run(3, [figure("all", range(3), zone=None)], ("2.5", "3.1", "0.4"))
    assert report.shown["all"] == disclosure.WITHHELD


def test_two_groups_sum_rebuilt_from_two_rows_is_suppressed():
    figures = [figure("all", range(4)), figure("first two", (0, 1))]
    report = run(4, figures, ("2.5", "3.1", "0.4", "1.0"))
    assert report.shown["first two"] == disclosure.WITHHELD


def test_a_range_exactly_equal_to_the_width_passes():
    c = disclosure.Checker({"x0": (Q(0), Q(1))}, [disclosure.Protected("x0", {"x0": 1}, Q(1))])
    c.check_record()  # [0, 1] has width exactly 1: a tie passes
    tight = disclosure.Checker({"x0": (Q(0), Q(1) - Q(1, 10**6))}, [disclosure.Protected("x0", {"x0": 1}, Q(1))])
    with pytest.raises(disclosure.StopForReview):
        tight.check_record()


@pytest.mark.parametrize("n", (3, 4))
def test_the_registered_zone_keeps_every_group_at_its_width(n):
    """z = width ÷ the smallest count of the band (3 for 3-4 groups): "below 0.34"."""
    zone = W / 3
    report = run(n, [figure("all", range(n), zone)], ["0.1"] * n)
    assert report.shown["all"] == "below 17/50"  # 0.34 s/min, the edge rounded up to the unit


def test_an_earlier_smaller_zone_divisor_fails_and_the_figure_is_suppressed():
    """A zone of width ÷ (count + 1) lets a near-zero average pin each group below the width."""
    report = run(3, [figure("all", range(3), W / 4)], ["0.1"] * 3)
    assert report.shown["all"] == disclosure.WITHHELD


def test_a_prior_range_narrower_than_its_width_stops_outside_fix_1():
    """Missed voice protected in seconds: a group's possible range of 5.27 s is below 10 s."""
    with pytest.raises(disclosure.StopForReview):
        disclosure.Protected("missed seconds", {"m": 1}, Q("5.27"), width=Q(10))
    disclosure.Protected("one-excerpt group", {"m": 1}, Q("5.27"), width=Q(10), fix1=True)  # Fix 1 keeps it


def test_fix_1_a_prior_limited_group_must_keep_its_whole_prior_range():
    """A one-excerpt group's voiced time can be at most 6.4 s, below the 10 s width: under Fix 1 it
    must keep exactly [0, 6.4]; without Fix 1 the record already fails and the run stops."""
    lim = {"v0": (Q(0), Q("6.4")), "v1": (Q(0), Q(100)), "v2": (Q(0), Q(100))}
    prot = [disclosure.Protected("v0", {"v0": 1}, Q("6.4"), width=Q(10), fix1=True),
            disclosure.Protected("v1", {"v1": 1}, Q(10)), disclosure.Protected("v2", {"v2": 1}, Q(10))]
    total = disclosure.Figure("total", {"v0": 1, "v1": 1, "v2": 1}, Q(1, 10), Q(0), Q("206.4"), Q(10), "durations")
    report = disclosure.Checker(lim, prot).run([total], {"v0": Q(3), "v1": Q(50), "v2": Q(60)})
    assert report.shown["total"] == Q(113)
    low = disclosure.Checker(lim, prot).run([total], {"v0": Q(1), "v1": Q(1), "v2": Q(2)})
    assert low.shown["total"] == "below 10"  # N1.6: step 3 totals below 10 s
    strict = [disclosure.Protected("v0", {"v0": 1}, Q(10))] + prot[1:]
    with pytest.raises(disclosure.StopForReview):
        disclosure.Checker(lim, strict).run([total], {"v0": Q(3), "v1": Q(50), "v2": Q(60)})


def test_nested_rows_are_suppressed():
    """Rows without a set and without a subset of it (nested G_V inside G_T)."""
    figures = [figure("all", range(4)), figure("without 2, 3", (0, 1)), figure("without 3", (0, 1, 2))]
    report = run(4, figures, ("2.5", "3.1", "0.4", "1.0"))
    assert [report.shown[f.name] == disclosure.WITHHELD for f in figures] == [False, True, True]


# -- T1-T7, T9 -------------------------------------------------------------------------------------


def true_value_rule(n, figure_, values):
    """The superseded rule (positive control): release if the protected quantities are safe given
    the value the figure actually shows."""
    region = exact.Region(limits(n))
    label = disclosure.show(figure_, sum(Q(c) * Q(values[int(k[1:])]) for k, c in figure_.form.items()))
    window = disclosure.outcome_of(figure_, label)
    region.add_window(figure_.form, window.lo, window.hi)
    return all(exact.width(region, p.form, p.target) == exact.PASS for p in protected(n, pairs=False))


def test_t1_the_true_value_rule_leaks_through_its_withholding():
    f = figure("all", range(2), zone=None)  # no zone: a small average pins both groups
    grid = [Q(k, 4) for k in range(0, 13)]  # 0 … 3 s/min
    withheld = [(a, b) for a in grid for b in grid if not true_value_rule(2, f, (a, b))]
    released = [(a, b) for a in grid for b in grid if true_value_rule(2, f, (a, b))]
    assert withheld and released
    # a reader who sees "withheld" learns the average was small: each group is below the width
    assert max(max(a, b) for a, b in withheld) < W


@pytest.mark.parametrize("values", (("0.1", "0.2", "0.3"), ("5", "9", "40"), ("59", "60", "58"), ("0", "0", "0")))
def test_t2_decisions_are_identical_across_private_values_with_the_same_public_information(values):
    figures = [figure("all", range(3), W / 3), figure("first two", (0, 1))]
    first = disclosure.Checker(limits(3), protected(3)).decide(figures[0])
    assert first.released == disclosure.Checker(limits(3), protected(3)).decide(figures[0]).released
    report = run(3, figures, values)
    assert report.decisions[0].released == first.released


class SpyTruth(dict):
    deciding = False

    def __getitem__(self, key):
        if SpyTruth.deciding:
            raise AssertionError("the decision code read a private value")
        return super().__getitem__(key)


def test_t3_the_decision_code_receives_no_private_values(monkeypatch):
    real = disclosure.Checker.decide

    def spied(self, figure_):
        SpyTruth.deciding = True
        try:
            return real(self, figure_)
        finally:
            SpyTruth.deciding = False

    monkeypatch.setattr(disclosure.Checker, "decide", spied)
    spy = SpyTruth(truth("2.5", "3.1", "0.4"))
    report = disclosure.Checker(limits(3), protected(3)).run([figure("all", range(3)), figure("pair", (0, 1))], spy)
    assert report.decisions[0].released


def observable(report):
    return tuple(report.shown.items())


def test_t4_exhaustive_small_worlds_grouped_by_what_a_reader_sees():
    """Every world on a fine grid, grouped by the full observable outcome; within each group every
    protected quantity still spans its width (up to the grid spacing). Decisions depend only on
    what is public, so they are computed once per observable history (a decision tree)."""
    n, top = 4, Q(2)  # the world's whole range is 0 … 2 s/min, and the grid covers all of it
    figures = [figure("all", range(n), W / 3, high=top), figure("without x3", (0, 1, 2), W / 3, high=top)]
    spacing = Q(1, 4)
    grid = [k * spacing for k in range(0, 9)]
    tree = {}

    def observe(values):
        checker = disclosure.Checker(limits(n, hi=top), protected(n))
        history = ()
        for f in figures:
            if history + (f.name,) not in tree:
                tree[history + (f.name,)] = checker.decide(f).released
            label = disclosure.WITHHELD
            if tree[history + (f.name,)]:
                label = disclosure.show(f, sum(Q(c) * values[int(k[1:])] for k, c in f.form.items()))
                checker.release(f, label)
            history += ((f.name, label),)
        return history

    classes = {}
    for values in itertools.product(grid, repeat=n):
        classes.setdefault(observe(values), []).append(values)
    assert len(classes) > 1
    assert all(dict(h)["without x3"] == disclosure.WITHHELD for h in classes)  # it would pin x3
    for members in classes.values():
        for p in protected(n):
            values = [sum(Q(c) * v[int(k[1:])] for k, c in p.form.items()) for v in members]
            assert max(values) - min(values) >= W - 2 * spacing


def test_t5_stops_use_public_information_and_name_only_public_facts():
    record = [(mean_of((0, 1, 2)), Q(0), Q(0))]  # the public record already pins every group
    c = disclosure.Checker(limits(3), protected(3), record=record)
    with pytest.raises(disclosure.StopForReview) as caught:
        c.check_record()
    message = str(caught.value)
    assert "x0" not in message and "0." not in message
    later = disclosure.Checker(limits(3), protected(3, pairs=False))  # a later release, checked afresh
    later.check_record()


def test_t6_an_uncertifiable_figure_withholds_its_unit_identically_and_without_a_unit_the_run_stops(monkeypatch):
    figures = [figure("a", range(3), unit="table 1"), figure("b", range(3), unit="table 1"),
               figure("c", range(3), unit="table 2")]
    no_unit = figure("d", range(3), unit="")
    real = exact.Region.extent

    def broken(self, form):
        if form is figures[0].form or form is no_unit.form:
            return exact.Extent()  # uncertain: "a" and "d" cannot be bracketed
        return real(self, form)

    monkeypatch.setattr(exact.Region, "extent", broken)
    for values in (("2.5", "3.1", "0.4"), ("30", "1", "7")):
        report = run(3, figures, values)
        assert report.shown["a"] == report.shown["b"] == disclosure.WITHHELD and report.shown["c"] != disclosure.WITHHELD
        assert report.withheld_units == {"table 1"}
    with pytest.raises(disclosure.StopForReview):
        run(3, [no_unit], ("2.5", "3.1", "0.4"))


@pytest.mark.parametrize("seed", range(4))
def test_t7_the_shortcut_agrees_with_checking_every_outcome(seed):
    import random

    rnd = random.Random(seed)
    n, top = 3, Q(3)  # a small range keeps checking every outcome affordable (300 windows)
    figures = [figure("all", range(n), W / 3, high=top), figure(f"pair {seed}", (seed % n, (seed + 1) % n), W / 2, high=top)]
    values = [Q(rnd.randint(0, 300), 100) for _ in range(n)]
    fast = disclosure.Checker(limits(n, hi=top), protected(n), shortcut=True).run(figures, truth(*values))
    slow = disclosure.Checker(limits(n, hi=top), protected(n), shortcut=False).run(figures, truth(*values))
    assert [d.released for d in fast.decisions] == [d.released for d in slow.decisions]
    assert fast.shown == slow.shown


def test_t7_note_passing_small_worlds_does_not_prove_the_shortcut():
    """The shortcut's soundness rests on the concavity argument in the module docstring, which
    holds for linear figures and protected quantities; it is not used for ratios or counts."""
    assert "concave" in disclosure.__doc__ and "never used for ratios" in disclosure.__doc__


def test_t9_determinism_across_runs_and_thread_counts(tmp_path):
    script = ("import sys; sys.path.insert(0, 'tests'); import test_presence_model_disclosure as t; "
              "r = t.run(3, [t.figure('all', range(3), t.W / 3), t.figure('pair', (0, 1))], ('2.5', '3.1', '0.4')); "
              "print(sorted((k, str(v)) for k, v in r.shown.items()), [d.checks for d in r.decisions])")
    outputs = set()
    for threads in ("1", "4"):
        env = dict(os.environ, OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads)
        outputs.add(subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, env=env,
                                   check=True).stdout)
    assert len(outputs) == 1


def test_the_true_values_must_lie_in_the_public_record():
    record = [(mean_of((0, 1, 2)), Q(10), Q(20))]
    c = disclosure.Checker(limits(3), protected(3), record=record)
    with pytest.raises(disclosure.SelfCheckError):
        c.run([figure("all", range(3))], truth("1", "1", "1"))


def test_a_ratio_is_checked_on_every_window_it_could_show():
    """The ratio of two published group averages: by then both are public, so its windows add
    little; it is decided on every window it could show (no shortcut)."""
    lim = {**{f"x{i}": (Q(0), Q(60)) for i in range(3)}, **{f"y{i}": (Q(0), Q(60)) for i in range(3)}}
    prot = [disclosure.Protected(k, {k: 1}, W) for k in lim]
    mx, my = mean_of(range(3)), {f"y{i}": Q(1, 3) for i in range(3)}
    figures = [disclosure.Figure("C", mx, U, Q(0), Q(60), W / 3, "main"),
               disclosure.Figure("B", my, U, Q(0), Q(60), W / 3, "main"),
               disclosure.Figure("ratio", {}, Q(1, 1000), Q(0), Q(20), None, "ratio", ratio=(mx, my))]
    values = {"x0": Q(1), "x1": Q(2), "x2": Q(3), "y0": Q(4), "y1": Q(5), "y2": Q(6)}
    report = disclosure.Checker(lim, prot).run(figures, values)
    assert report.shown["C"] == 2 and report.shown["B"] == 5 and report.shown["ratio"] == Q(2, 5)
