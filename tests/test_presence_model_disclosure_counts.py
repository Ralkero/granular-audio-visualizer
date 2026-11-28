"""A5.2 Step 2a, S2a.6 (first part): the exact disclosure check for step 2 and 3's MUSDB count
figures (protocol Amendment 3, A3.5 N1.7 and A3.6 L6), on generated worlds only.

These tests also record a consequence of the approved rules (T8, cost): the size of G_T (and
likewise G_V) can be released only when the item D figure is 0-2 or 24-26."""

import random

import numpy as np

from gmv.character.research.presence_model import disclosure_counts as dc


def world(pa=0, la=0, va=0, pb=0, lb=0, vb=0, pab=0):
    return {"p_a": pa, "l_a": la, "v_a": va, "p_b": pb, "l_b": lb, "v_b": vb, "p_ab": pab}


def random_world(rng):
    while True:
        n = [rng.randrange(20) for _ in range(4)]
        if sum(n[:3]) <= 19:
            n00, n10, n01 = n[:3]
            n11 = 19 - n00 - n10 - n01
            m = [rng.randrange(7) for _ in range(3)]
            if sum(m) <= 6:
                m11 = 6 - sum(m)
                la, lb = rng.randrange(2), rng.randrange(2)
                return world(n10 + n11, la, m[1] + m11, n01 + n11, lb, m[2] + m11, n11)


def released(decisions):
    return {d.figure: (d.released, d.shown) for d in decisions}


def test_the_model_enumerates_every_world():
    assert len(dc.worlds()["p_a"]) == 1540 * 84 * 4 and len(dc.Model(3, 1, 2).w["p_a"]) == 20 * 4 * 10


def test_item_d_figures_are_always_released_exactly():
    rng = random.Random(5)
    for _ in range(20):
        t = random_world(rng)
        r = released(dc.decide(t))
        assert r["D_train"] == (True, t["p_a"] + t["l_a"] + t["v_a"])
        assert r["D_validation"] == (True, t["p_b"] + t["l_b"] + t["v_b"])


def test_release_decisions_depend_only_on_public_figures():
    """Worlds with the same item D figures get the same release decisions (T2)."""
    rng = random.Random(9)
    by_public = {}
    for _ in range(150):
        t = random_world(rng)
        decisions = dc.decide(t)
        key = (decisions[0].shown, decisions[1].shown)
        pattern = tuple(d.released for d in decisions)
        assert by_public.setdefault(key, pattern) == pattern


def test_g_t_size_is_released_only_at_the_ends_of_the_training_figure():
    """The cost of the approved rules: for item D training figures 3-23, some value the G_T size
    could show would pin the low-rate and validation groups' training status, so it is
    withheld; the C5 trigger can then never fire on a published code of 3 or more except at
    the top end, where F1 fails from the size alone."""
    for d in range(27):
        pa = min(d, 19)
        va = min(d - pa, 6)
        t = world(pa=pa, va=va, la=d - pa - va)
        g_t = released(dc.decide(t))["G_T"]
        assert g_t[0] == (d <= 2 or d >= 24), d
        if d <= 2:
            assert g_t[1] == "0–2"
        if d >= 24:
            assert g_t[1] == "17–19"


def test_an_exact_g_t_equal_to_the_training_figure_would_pin_seven_groups():
    w = dc.worlds()
    mask = (w["D_train"] == 10) & (w["p_a"] == 10)
    failures = dc.unsafe(mask)
    assert "one low-rate group's a" in failures and "one validation group's a" in failures


def test_the_fix_1_extension_applies_at_0_and_26_only():
    assert not [f for f in dc.unsafe(dc.worlds()["D_train"] == 0, fixed=frozenset({"a"})) if f.endswith("'s a")]
    zero = released(dc.decide(world()))
    assert zero["G_T"] == (True, "0–2")
    one = dc.worlds()["D_train"] == 1
    assert not [f for f in dc.unsafe(one) if f.endswith("'s a")]  # D = 1 fixes no single group's bit
    assert [f for f in dc.unsafe(dc.worlds()["D_train"] == 0) if f.endswith("'s a")]  # D = 0 would, without the extension


def test_a_true_value_rule_leaks_through_its_withholding_positive_control():
    """T1: deciding from the true world (withhold only when the true value is unsafe) makes the
    withholding itself informative. With D_train = 5, the only unsafe G_T value is 5, so a
    withheld G_T tells the reader that no low-rate or validation group matched."""
    w = dc.worlds()
    public = w["D_train"] == 5
    codes = dc.shown("G_T", w)
    values = sorted(set(codes[public].tolist()), key=str)
    unsafe_values = [c for c in values if dc.unsafe(public & (codes == c))]
    assert unsafe_values == ["5"]  # "0–2" is safe; an exact 5 says every match is a paired group
    withheld_worlds = public & np.isin(codes, unsafe_values)
    assert "one low-rate group's a" in dc.unsafe(withheld_worlds)  # the leak the public-information rule prevents
