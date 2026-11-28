"""A5.2 Step 4, S4.9: exact rational verification for the disclosure check (protocol Amendment 3,
A3.5 N1.8; A3.10's exact cases). Generated systems only."""

from fractions import Fraction as Q

import pytest

from gmv.character.research.presence_model import exact


def region(n=3, lo=0, hi=10):
    return exact.Region({f"x{i}": (Q(lo), Q(hi)) for i in range(n)})


def avg(n):
    return {f"x{i}": Q(1, n) for i in range(n)}


def test_a_box_alone_has_its_limits_exactly():
    r = region()
    e = r.extent({"x0": 1})
    assert (e.min_lo, e.min_hi, e.max_lo, e.max_hi) == (0, 0, 10, 10)
    assert all(r.contains(p) for p in e.points)


def test_a_published_average_bounds_each_group_exactly():
    r = region()
    r.add_window(avg(3), Q(1), Q(101, 100))  # the average shown as 1.0 at 0.01 precision, closed window
    e = r.extent({"x0": 1})
    assert e.max_lo == e.max_hi == Q(303, 100) and e.min_lo == e.min_hi == 0
    assert exact.width(r, {"x0": 1}, Q(303, 100)) == exact.PASS  # exactly equal to the width passes
    assert exact.width(r, {"x0": 1}, Q(303, 100) + Q(1, 10**9)) == exact.FAIL


def test_awkward_fractions_stay_exact():
    r = region(3, 0, 1)
    r.add_window({"x0": Q(1, 3), "x1": Q(1, 7), "x2": Q(1, 11)}, Q(1, 5), Q(1, 5))
    e = r.extent({"x0": 1, "x1": -1})
    assert e.max_lo == e.max_hi and e.min_lo == e.min_hi
    assert all(r.contains(p) for p in e.points)


def test_the_interior_point_is_strictly_inside_and_public():
    r = region()
    r.add_window(avg(3), Q(1), Q(2))
    c = r.interior()
    assert r.contains(c)
    assert all(Q(0) < v < Q(10) for v in c) and Q(1) < sum(c) / 3 < Q(2)


def test_true_values_on_a_constraint_do_not_enter():
    """The true point can sit on a limit (a group with zero false activity); the region and the
    verdict are the same whatever it is, because it is never an input."""
    r = region()
    r.add_window(avg(3), Q(0), Q(1, 100))
    assert exact.width(r, {"x0": 1}, Q(1, 10)) == exact.FAIL
    assert r.interior() is not None and all(v > 0 for v in r.interior())


def test_coinciding_limits_make_a_single_valued_quantity_that_fails():
    r = exact.Region({"x0": (Q(2), Q(2)), "x1": (Q(0), Q(5))})
    assert exact.width(r, {"x0": 1}, Q(1, 10)) == exact.FAIL
    assert exact.width(r, {"x0": 1}, Q(0)) == exact.PASS
    assert r.interior()[0] == 2


def test_a_figure_that_pins_its_groups_is_found_as_a_forced_equality():
    r = region(3, 0, 60)
    r.add_window(avg(3), Q(0), Q(0))  # an average of 0 s/min forces every group to 0
    assert exact.width(r, {"x1": 1}, Q(1, 10)) == exact.FAIL
    c = r.interior()
    assert c == [0, 0, 0]
    assert len(r.forced_equalities()) >= 3


def test_an_equality_forced_by_two_inequalities():
    r = region(2, 0, 1)
    r.add_le({"x0": 1, "x1": 1}, Q(1))
    r.add_ge({"x0": 1, "x1": 1}, Q(1))
    c = r.interior()
    assert c is not None and c[0] + c[1] == 1 and 0 < c[0] < 1
    assert exact.width(r, {"x0": 1}, Q(1)) == exact.PASS


def test_a_difference_pinned_to_zero_has_one_value():
    r = region(2, 0, 1)
    r.add_eq({"x0": 1, "x1": -1}, Q(0))
    assert exact.width(r, {"x0": 1, "x1": -1}, Q(1, 10)) == exact.FAIL
    assert exact.signs(r, {"x0": 1, "x1": -1}) == exact.FAIL


def test_an_empty_region_is_proven_empty():
    r = region(2, 0, 1)
    r.add_ge({"x0": 1, "x1": 1}, Q(3))
    assert r.empty() is True
    assert exact.width(r, {"x0": 1}, Q(1)) == exact.EMPTY


def test_signs():
    r = region(2, 0, 1)
    assert exact.signs(r, {"x0": 1, "x1": -1}) == exact.PASS
    r.add_ge({"x0": 1, "x1": -1}, Q(0))
    assert exact.signs(r, {"x0": 1, "x1": -1}) == exact.FAIL  # cannot be below zero


def test_a_deliberately_unprovable_case_is_uncertain_never_passed(monkeypatch):
    r = region()
    r.add_window(avg(3), Q(1), Q(2))

    class Broken:
        status, x = 4, None  # the solver reports a numerical failure

    monkeypatch.setattr(exact.Region, "_propose", lambda self, c, maximise: Broken())
    assert exact.width(r.copy(), {"x0": 1}, Q(1, 10)) == exact.UNCERTAIN


def test_witnesses_decide_only_a_pass():
    r = region()
    pool = [[Q(0), Q(0), Q(0)], [Q(5), Q(0), Q(0)]]
    assert exact.width(r, {"x0": 1}, Q(5), pool) == exact.PASS
    assert exact.width(r, {"x0": 1}, Q(10), pool) == exact.PASS  # the solver finds the rest
    assert exact.width(r, {"x0": 1}, Q(11), pool) == exact.FAIL


def test_inputs_must_be_exact():
    with pytest.raises(TypeError):
        exact.Region({"x": (0.1, 1)})
