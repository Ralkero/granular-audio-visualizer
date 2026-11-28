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
    out = results.check(st, values, reader="registered", rule="3")
    assert out.registered_cases > 10**9
    assert all(v == disclosure.WITHHELD for v in out.shown.values())
    assert results.registered_cases(results.Structure(15, 4, (0, 1, 2))) > 10**20


def test_with_the_public_record_reader_b0ts_average_is_still_suppressed_by_directions():
    """Once C's average is public, an extreme value B0ₜ's average could show (near 60 s/min) would
    reveal every group's direction, so it is suppressed whatever its true value (N1.7 item 5)."""
    st, values = small()
    out = results.check(st, values, reader="public", rule="3")
    assert out.shown["C FPA"] != disclosure.WITHHELD
    assert out.shown["B0t FPA"] == disclosure.WITHHELD and out.shown["D_FA"] == disclosure.WITHHELD
    checker = disclosure.Checker(results.limits(st), results.protected(st))
    figs = results.figures(st, False, rule="3")
    truth = results.canonical(st, values)
    checker.release(figs[0], disclosure.show(figs[0], disclosure._value(figs[0], truth)))
    top = disclosure.outcome_of(figs[1], f"above {figs[1].zone_edges()[1]}")
    region = checker._windowed(figs[1], top)
    failing = [p.name for p in results.protected(st) if checker._verdict(region, p, []) != exact.PASS]
    assert failing and all(name.startswith("direction FPA") for name in failing)


def test_seed_aggregates_are_suppressed_because_an_extreme_one_would_pin_the_others():
    st, values = small()
    out = results.check(st, values, reader="public", rule="3")
    assert all(out.shown[f"C FPA seed {s}"] == disclosure.WITHHELD for s in st.seeds)


def test_verdicts_are_coarse_enough_to_be_released():
    st, values = small()
    out = results.check(st, values, reader="public", rule="3")
    assert out.shown["D_FA < 0"] == "below 0" and out.shown["Delta_R >= -0.05"] in ("at least -0.05", "below -0.05")


def test_the_counts_unit_is_not_modelled_and_withheld_whole():
    st, values = small()
    out = results.check(st, values)
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


# -- the proposed Amendment 3B (options A, B and C), measured on generated worlds ----------------------


def test_3b_publishes_the_main_comparison_the_verdicts_and_the_seed_verdict():
    st, values = small()
    out = results.check(st, values, reader="public", rule="3B", cap=Q(30))
    for name in ("C FPA", "B0t FPA", "D_FA", "C recall", "B0t recall", "Delta_R", "D_FA < 0", "Delta_R >= -0.05",
                 results.SEED_VERDICT):
        assert out.shown[name] != disclosure.WITHHELD, name
    assert out.shown[results.SEED_VERDICT] == results.seed_verdict_value(st, results.canonical(st, values))
    assert not any(name.startswith("C FPA seed") for name in out.shown)  # seed aggregates stay private


def test_3b_a_baseline_above_the_cap_is_shown_as_a_code_and_its_difference_follows_it():
    st, values = small()
    for g in st.groups:
        values[g]["b"] = Q(45)
    out = results.check(st, values, reader="public", rule="3B", cap=Q(30))
    assert out.shown["B0t FPA"] == "above 30"
    assert out.shown["D_FA"] == results.CODED and out.shown["ratio"] == results.CODED


def test_3b_is_the_default_needs_the_public_record_reader_and_caps_by_table_size():
    st, values = small()
    with pytest.raises(ValueError):
        results.check(st, values, reader="registered", rule="3B")
    with pytest.raises(ValueError):
        results.figures(st, False, rule="registered")
    default = results.check(st, values)
    assert results.SEED_VERDICT in default.shown and not any("seed 0" in k for k in default.shown)
    assert {f.cap for f in results.figures(st, False) if f.name in ("C FPA", "B0t FPA")} == {results.proposed_cap(5)}


def test_a_cap_moves_the_upper_zone_edge():
    f = disclosure.Figure("x", {"a": 1}, Q(1, 100), Q(0), Q(60), Q(1, 10), cap=Q(30))
    assert f.zone_edges() == (Q(1, 10), Q(30)) and disclosure.show(f, Q(31)) == "above 30"
    g = disclosure.Figure("y", {"a": 1}, Q(1, 100), Q(0), Q(60), None, cap=Q(30))
    assert g.zone_edges() == (None, Q(30))


def test_a_union_verdict_uses_the_range_over_the_union():
    """Two alternatives that each pin x0 but together leave it its width."""
    lim = {"x0": (Q(0), Q(10)), "x1": (Q(0), Q(10))}
    prot = [disclosure.Protected("x0", {"x0": 1}, Q(1))]
    v = disclosure.Verdict("v", (("yes", [[({"x0": 1}, Q(3), Q(5))]]),
                                 ("no", [[({"x0": 1}, Q(0), Q("0.5"))], [({"x0": 1}, Q("9.5"), Q(10))]])))
    assert disclosure.Checker(lim, prot).decide_verdict(v).released
    narrow = disclosure.Verdict("w", (("yes", [[({"x0": 1}, Q(3), Q(5))]]), ("no", [[({"x0": 1}, Q(0), Q("0.5"))]])))
    assert not disclosure.Checker(lim, prot).decide_verdict(narrow).released


def test_no_figure_is_decided_after_a_union_verdict_is_released():
    lim = {"x0": (Q(0), Q(10))}
    prot = [disclosure.Protected("x0", {"x0": 1}, Q(1))]
    v = disclosure.Verdict("v", (("yes", [[({"x0": 1}, Q(0), Q(5))]]),
                                 ("no", [[({"x0": 1}, Q(5), Q(7))], [({"x0": 1}, Q(8), Q(10))]])))
    c = disclosure.Checker(lim, prot)
    c.release_verdict(v, "no")
    with pytest.raises(disclosure.StopForReview):
        c.decide(disclosure.Figure("x", {"x0": 1}, Q(1, 100), Q(0), Q(10)))


def test_the_proposed_caps_sit_below_the_direction_threshold_of_each_band():
    for n in (3, 4, 5, 9, 10, 19):
        m = results.band_minimum(n)
        assert results.proposed_cap(n) < Q(60) * (m - 1) / m
    assert [results.proposed_cap(n) for n in (3, 5, 19)] == [Q(35), Q(45), Q(50)]
