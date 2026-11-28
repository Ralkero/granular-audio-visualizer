"""A5.2 Step 4, S4.9: the main results tables through the disclosure check (protocol §12, §13;
Amendment 3, A3.5 N1; A3.10 T8), on generated worlds only."""

import random
from fractions import Fraction as Q

import pytest

from gmv.character.research.presence_model import disclosure, exact, results


def small(both=4, fpa_only=1, seeds=(0, 1), seed=1):
    st = results.Structure(both, fpa_only, seeds)
    rnd = random.Random(seed)
    values = {}
    for g in st.groups:
        base = Q(rnd.randint(300, 800), 100)
        values[g] = {"b": base, "c": {s: base * Q(rnd.randint(30, 70), 100) for s in seeds},
                     "q": Q(rnd.randint(600, 900), 1000), "r": {s: Q(rnd.randint(600, 900), 1000) for s in seeds}}
    return st, values


def test_the_layout_is_symmetric_within_classes_and_an_asymmetric_one_is_caught():
    st, _ = small()
    assert results.check_symmetry(st, results.figures(st, with_ratio=True))
    lopsided = disclosure.Figure("x", {"c[g0,0]": Q(1, 2), "c[g1,0]": Q(1, 4)}, Q(1, 100), Q(0), Q(60))
    assert not results.check_symmetry(st, [lopsided])


def test_zones_follow_the_band_of_each_average():
    st, _ = small()
    figs = {f.name: f for f in results.figures(st, False)}
    assert figs["C FPA"].zone == Q(1, 5)  # 5 groups: band 5-9
    assert figs["C recall"].zone == Q(1, 10) / 3  # 4 groups: band 3-4
    assert results.band_minimum(19) == 10 and results.band_minimum(15) == 10 and results.band_minimum(4) == 3


def test_the_registered_reader_needs_astronomically_many_cases_so_every_unit_is_withheld():
    st, values = small()
    out = results.check(st, values, reader="registered")
    assert out.registered_cases > 10**9
    assert all(v == disclosure.WITHHELD for v in out.shown.values())
    assert results.registered_cases(results.Structure(15, 4, (0, 1, 2))) > 10**20


def test_with_the_public_record_reader_b0ts_average_is_still_suppressed_by_directions():
    """Once C's average is public, an extreme value B0ₜ's average could show (near 60 s/min) would
    reveal every group's direction, so it is suppressed whatever its true value (N1.7 item 5)."""
    st, values = small()
    out = results.check(st, values, reader="public")
    assert out.shown["C FPA"] != disclosure.WITHHELD
    assert out.shown["B0t FPA"] == disclosure.WITHHELD and out.shown["D_FA"] == disclosure.WITHHELD
    checker = disclosure.Checker(results.limits(st), results.protected(st))
    figs = results.figures(st, False)
    truth = results.canonical(st, values)
    checker.release(figs[0], disclosure.show(figs[0], disclosure._value(figs[0], truth)))
    top = disclosure.outcome_of(figs[1], f"above {figs[1].zone_edges()[1]}")
    region = checker._windowed(figs[1], top)
    failing = [p.name for p in results.protected(st) if checker._verdict(region, p, []) != exact.PASS]
    assert failing and all(name.startswith("direction FPA") for name in failing)


def test_seed_aggregates_are_suppressed_because_an_extreme_one_would_pin_the_others():
    st, values = small()
    out = results.check(st, values, reader="public")
    assert all(out.shown[f"C FPA seed {s}"] == disclosure.WITHHELD for s in st.seeds)


def test_verdicts_are_coarse_enough_to_be_released():
    st, values = small()
    out = results.check(st, values, reader="public")
    assert out.shown["D_FA < 0"] == "below 0" and out.shown["Delta_R >= -0.05"] in ("at least -0.05", "below -0.05")


def test_the_counts_unit_is_not_modelled_and_withheld_whole():
    st, values = small()
    out = results.check(st, values, reader="public")
    assert "counts" in out.withheld_units and all(out.shown[name] == disclosure.WITHHELD for name in results.COUNTS)


def test_cuts_show_the_interval_holding_the_value():
    f = disclosure.Figure("v", {"x": 1}, Q(1, 100), Q(-1), Q(1), None, "verdicts",
                          cuts=(("below 0", Q(-1), Q(0)), ("not below 0", Q(0), Q(1))))
    assert disclosure.show(f, Q(-1, 1000)) == "below 0" and disclosure.show(f, Q(0)) == "not below 0"
    assert disclosure.outcome_of(f, "below 0").hi == 0
    with pytest.raises(ValueError):
        disclosure.outcome_of(f, "maybe")


def test_an_unknown_reader_is_refused():
    st, values = small()
    with pytest.raises(ValueError):
        results.check(st, values, reader="nobody")
