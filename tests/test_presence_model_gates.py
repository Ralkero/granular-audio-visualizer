"""A5.2 Step 4, S4.6: the training-length gate, the 16 h gate and the caps (protocol §15;
Amendment 3, A3.1, A3.2, A3.8). Generated values only: every branch of the rule."""

import pytest

from gmv.character.research.presence_model import gates, timing


def curve(final, epochs, rising=0.0):
    """Per-epoch losses ending at ``final``; ``rising`` > 0 makes the last window rise."""
    out = [final * (1 + 0.5 * (epochs - k) / epochs) for k in range(1, epochs + 1)]
    if rising:
        out[-5:] = [x * (1 + rising) for x in out[-5:]]
        out[-1] = final
    return tuple(out)


def runs(arm="C", losses=None, epoch_count=None, flags=None):
    """``losses``: {(size, seed, horizon): L}; default: E and E⁺ equal, both sizes and seeds."""
    e, e1, _e2 = gates.LENGTHS[arm]
    losses = losses or {(z, s, h): 0.30 + 0.001 * s for z in gates.SIZES for s in gates.SEEDS for h in (e, e1)}
    out, flags = [], flags or {}
    for (z, s, h), loss in losses.items():
        out.append(gates.Run(arm, z, s, h, loss, curve(loss, epoch_count or h), **flags.get((z, s, h), {})))
    return out


def with_horizons(arm, values):
    """{horizon: (L_outer_0, L_outer_1, L_inner_0, L_inner_1)}."""
    losses = {}
    for h, (o0, o1, i0, i1) in values.items():
        losses.update({("outer", 0, h): o0, ("outer", 1, h): o1, ("inner", 0, h): i0, ("inner", 1, h): i1})
    return runs(arm, losses)


def test_equal_losses_adopt_e():
    v = gates.decide("C", runs())
    assert (v.outcome, v.epochs, v.reasons) == ("adopt", 30, ())


def test_a_material_improvement_at_one_size_asks_for_stage_2():
    v = gates.decide("C", with_horizons("C", {30: (0.30, 0.302, 0.30, 0.302), 45: (0.28, 0.281, 0.30, 0.302)}))
    assert v.outcome == "stage 2" and v.epochs is None


def test_an_improvement_inside_the_seed_spread_does_not_extend():
    # ε = |0.30 − 0.33| / 0.315 ≈ 0.095; Δ ≈ 0.05 is inside it
    v = gates.decide("C", with_horizons("C", {30: (0.30, 0.33, 0.30, 0.33), 45: (0.285, 0.3135, 0.30, 0.33)}))
    assert v.outcome == "adopt" and v.epochs == 30


def test_both_seeds_worse_at_one_size_stop_s4():
    v = gates.decide("C", with_horizons("C", {30: (0.30, 0.301, 0.30, 0.301), 45: (0.33, 0.331, 0.30, 0.301)}))
    assert v.outcome == "stop" and any(r.startswith("S4") for r in v.reasons)


def test_stage_2_confirming_adopts_e_plus_and_still_improving_stops():
    base = {30: (0.30, 0.302, 0.30, 0.302), 45: (0.28, 0.281, 0.30, 0.302)}
    confirmed = gates.decide("C", with_horizons("C", {**base, 68: (0.28, 0.281, 0.30, 0.302)}))
    assert (confirmed.outcome, confirmed.epochs) == ("adopt", 45)
    still = gates.decide("C", with_horizons("C", {**base, 68: (0.25, 0.251, 0.30, 0.302)}))
    assert still.outcome == "stop" and any(r.startswith("EXT′") for r in still.reasons)
    worse = gates.decide("C", with_horizons("C", {**base, 68: (0.32, 0.321, 0.30, 0.302)}))
    assert worse.outcome == "stop" and any(r.startswith("S4") for r in worse.reasons)


def test_b1_uses_its_own_lengths():
    v = gates.decide("B1", runs("B1"))
    assert (v.outcome, v.epochs) == ("adopt", 60)
    assert gates.LENGTHS == {"C": (30, 45, 68), "B1": (60, 90, 135)}


def test_s1_s2_and_s5_stop():
    bad = runs(flags={("outer", 0, 30): {"finite": False}})
    assert any(r.startswith("S1") for r in gates.decide("C", bad).reasons)
    assert any(r.startswith("S5") for r in gates.decide("C", runs(flags={("inner", 1, 45): {"checks_passed": False}})).reasons)
    zero = runs(losses={(z, s, h): (0.0 if (z, s, h) == ("outer", 0, 30) else 0.3) for z in gates.SIZES
                        for s in gates.SEEDS for h in (30, 45)})
    assert any(r.startswith("S2") for r in gates.decide("C", zero).reasons)


def test_s3_stops_when_both_seeds_rise_and_warns_for_one():
    e = 30
    losses = {(z, s, h): 0.3 for z in gates.SIZES for s in gates.SEEDS for h in (e, 45)}
    one = runs(losses=losses)
    one = [gates.Run(r.arm, r.size, r.seed, r.horizon, r.loss, curve(r.loss, r.horizon, 0.2)
                     if (r.size, r.seed, r.horizon) == ("outer", 0, e) else r.epoch_losses) for r in one]
    v = gates.decide("C", one)
    assert v.outcome == "adopt" and any(w.startswith("S3") for w in v.warnings)
    both = [gates.Run(r.arm, r.size, r.seed, r.horizon, r.loss, curve(r.loss, r.horizon, 0.2)
                      if (r.size, r.horizon) == ("outer", e) else r.epoch_losses) for r in one]
    assert any(r.startswith("S3") for r in gates.decide("C", both).reasons)


def test_r_is_over_the_last_two_five_epoch_windows():
    losses = [1.0] * 20 + [2.0] * 5 + [1.0] * 5
    m1, m2, r = gates.r_of(losses)
    assert (m1, m2, r) == (2.0, 1.0, 0.5)
    with pytest.raises(ValueError):
        gates.r_of([1.0] * 9)


def test_missing_stage_1_runs_are_refused_and_duplicates_too():
    with pytest.raises(ValueError):
        gates.decide("C", runs()[:-1])
    with pytest.raises(ValueError):
        gates.decide("C", runs() + runs()[:1])


def test_the_16_hour_gate_and_the_policy_flip_safeguard():
    assert gates.policy(57_600) == "C" and gates.policy(57_600.001) == "D"
    assert gates.policy_with_safeguard(50_000, 60_000, 45) == ("D", gates.policy_with_safeguard(50_000, 60_000, 45)[1])
    assert gates.policy_with_safeguard(50_000, 60_000, 45)[1] is not None
    assert gates.policy_with_safeguard(60_000, 70_000, 45) == ("D", None)  # already D at E
    assert gates.policy_with_safeguard(50_000, 52_000, 45) == ("C", None)
    with pytest.raises(ValueError):
        gates.policy(float("nan"))


def test_h_proc_counts_policy_cs_c_models():
    m = timing.Measured(S={"inner": 10.0, "outer": 12.0, "final": 14.0}, d1={"inner": 5.0, "outer": 6.0, "final": 7.0},
                        k={"inner": 0.1, "outer": 0.1, "final": 0.1}, p=0.5)
    sizes = {"inner": [(100, 10)] * 95, "outer": [(120, 8)] * 19, "final": [(130, 0)]}
    jobs = timing.plan_jobs(sizes, {"inner": 2, "outer": 5, "final": 5}, 30)
    assert len(jobs) == 190 + 95 + 5
    assert gates.h_proc_c(sizes, m, 30, 100.0) == pytest.approx(timing.h_proc(jobs, m, 100.0))


def test_caps():
    assert gates.cap_seconds("C", 30) == 3 * 3600 and gates.cap_seconds("C", 68) == pytest.approx(6.8 * 3600)
    assert gates.cap_seconds("B1", 60) == 2 * 3600 and gates.cap_seconds("B1", 135) == pytest.approx(4.5 * 3600)
    assert gates.cap_seconds("C", 30, concurrent=True) == pytest.approx(4 * 3600)


def test_timing_stops_use_the_registered_thresholds():
    assert gates.timing_stops(cvs=[0.16]) and not gates.timing_stops(cvs=[0.15])
    assert gates.timing_stops(drifts=[0.051]) and not gates.timing_stops(drifts=[0.05])
    assert gates.timing_stops(ratios=[(3.1, 2)])  # above co-runners + 1
    assert gates.timing_stops(cross_checks=[(1.21, 1.0)]) and not gates.timing_stops(cross_checks=[(1.2, 1.0)])
