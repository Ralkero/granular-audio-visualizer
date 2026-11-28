"""A5.2 Stage 1, protocol §16 "Metrics and reporting": collars, the 2 s minimum, single-class
groups, event collars, the low-denominator rule, the screening classes (with Amendment 1)
and the bootstrap caveat. Plus the shared post-processing and the registered seeds.
Synthetic labels, intervals and outputs only."""

import dataclasses
import hashlib

import numpy as np
import pytest

from gmv.character.research.presence_model import bootstrap, metrics, postprocess, screening, seeds


def centres(duration=10.0):
    return np.round(np.arange(0.0232, duration, 0.02), 6)


# -- per-group counts, collars and the 2 s minimum ---------------------------------------------------


def test_counts_exclude_collars_and_score_detected_and_false_time():
    c = centres()
    confirmed = [(2.0, 6.0)]
    active = (c >= 1.0) & (c < 5.0)  # 1 s early, ends 1 s before the reference
    counts = metrics.excerpt_counts(c, active, 10.0, confirmed)
    keep = ((c < 1.7) | ((c > 2.3) & (c < 5.7)) | (c > 6.3))  # ±0.3 s collars around 2.0 and 6.0
    voice = (c >= 2.0) & (c < 6.0)
    assert counts.voiced == int((keep & voice).sum())
    assert counts.novoice == int((keep & ~voice).sum())
    assert counts.detected == int((keep & voice & active).sum())
    assert counts.false == int((keep & ~voice & active).sum())
    with pytest.raises(ValueError):
        metrics.excerpt_counts(c, active[:-1], 10.0, confirmed)


def test_the_two_second_minimum_is_inclusive_in_frames():
    assert metrics.Counts(voiced=100, novoice=100, detected=50, false=10).recall == 0.5
    assert metrics.Counts(voiced=99, novoice=100).recall is None
    assert metrics.Counts(voiced=100, novoice=99).fpa is None
    assert metrics.Counts(voiced=0, novoice=100, false=25).fpa == pytest.approx(15.0)  # 60 × 0.5 s / 2 s


def test_single_class_groups_count_only_in_their_class_and_are_never_dropped():
    groups = {
        "group_001": metrics.Counts(voiced=500, novoice=0, detected=400),            # voice only
        "group_002": metrics.Counts(voiced=0, novoice=1000, false=100),               # no voice only
        "group_003": metrics.Counts(voiced=200, novoice=300, detected=100, false=30),
    }
    summary = metrics.macro(groups)
    assert summary["groups_recall"] == 2 and summary["groups_fpa"] == 2
    assert summary["recall"] == pytest.approx((0.8 + 0.5) / 2)
    assert summary["fpa"] == pytest.approx((6.0 + 6.0) / 2)  # 60×100/1000 and 60×30/300
    pooled = metrics.pooled(groups)
    assert pooled["recall"] == pytest.approx(500 / 700) and pooled["fpa"] == pytest.approx(60 * 130 / 1300)
    assert metrics.objective(groups) == pytest.approx(0.5 * (0.65 + (0.9 + 0.9) / 2))


def test_the_objective_needs_both_classes():
    with pytest.raises(ValueError):
        metrics.objective({"group_001": metrics.Counts(voiced=0, novoice=500)})


# -- events ---------------------------------------------------------------------------------------------


def test_reference_events_merge_gaps_shorter_than_0_25_s_only():
    assert metrics.reference_events([(1.0, 2.0), (2.249, 3.0)], 10.0) == [(1.0, 3.0)]
    assert metrics.reference_events([(1.0, 2.0), (2.25, 3.0)], 10.0) == [(1.0, 2.0), (2.25, 3.0)]


def test_onset_and_offset_collars_are_inclusive_and_the_offset_collar_scales():
    ref = [(2.0, 7.0)]  # 5 s long: offset collar max(0.5, 1.0) = 1.0
    assert metrics.match_events(ref, [(2.5, 8.0)], 10.0)["detected"] == 1
    assert metrics.match_events(ref, [(2.5001, 7.0)], 10.0)["detected"] == 0
    assert metrics.match_events(ref, [(2.0, 8.0001)], 10.0)["detected"] == 0
    short = [(2.0, 3.0)]  # 1 s long: offset collar max(0.5, 0.2) = 0.5
    assert metrics.match_events(short, [(2.0, 3.5)], 10.0)["detected"] == 1
    assert metrics.match_events(short, [(2.0, 3.51)], 10.0)["detected"] == 0


def test_matching_is_one_to_one_and_counts_misses_and_false_events():
    ref = [(1.0, 2.0), (5.0, 6.0)]
    pred = [(1.1, 2.1), (1.2, 2.0), (8.0, 9.0)]
    result = metrics.match_events(ref, pred, 10.0)
    assert result == {**result, "references": 2, "predictions": 3, "detected": 1, "missed": 1, "false": 2}
    assert result["onset_errors_s"] == [pytest.approx(0.1)]  # the closer onset wins


def test_timing_errors_only_for_interior_boundaries_and_predictions_are_clipped():
    result = metrics.match_events([(0.0, 4.0)], [(-2.0, 4.2)], 10.0)  # starts before the excerpt
    assert result["detected"] == 1
    assert result["onset_errors_s"] == [] and result["offset_errors_s"] == [pytest.approx(0.2)]
    whole = metrics.match_events([(0.0, 10.0)], [(0.0, 12.0)], 10.0)
    assert whole["detected"] == 1 and whole["onset_errors_s"] == [] and whole["offset_errors_s"] == []
    assert metrics.match_events([], [(11.0, 12.0)], 10.0)["predictions"] == 0  # wholly outside


# -- the low-denominator rule and the screening classes -------------------------------------------------

GROUPS = [f"group_{i:03d}" for i in range(1, 20)]  # synthetic keys; the count comes from the caller
SEEDS_OK = [(i, -1.0, 0.0) for i in range(5)]  # (seed_index, D_FA, Δ_R) for the registered indices 0-4


def seed_rows(indices=range(5), d_fa=-1.0, d_recall=0.0):
    return [(i, d_fa, d_recall) for i in indices]


def leave_one_out(groups=GROUPS, d_fa=-1.0, d_recall=0.0):
    return [(g, d_fa, d_recall) for g in groups]


def complete():
    """A complete, passing robustness set: 5 seeds and every expected group once."""
    return {"seed_recomputations": SEEDS_OK, "group_recomputations": leave_one_out(), "expected_groups": GROUPS}


def check(seeds=SEEDS_OK, rows=None, groups=GROUPS):
    return screening.check_robustness(seeds, leave_one_out() if rows is None else rows, groups)


def test_relative_mode_from_exactly_3_s_per_minute_with_no_constant_added():
    s = screening.screen(1.5, 3.0, 0.8, 0.8, **complete())
    assert s.mode == "relative" and s.ratio == 0.5 and s.fifty_percent == "met" and s.result == "promising"
    s = screening.screen(1.51, 3.0, 0.8, 0.8, **complete())
    assert s.ratio == pytest.approx(1.51 / 3.0) and s.fifty_percent == "not met" and s.result == "mixed"


def test_absolute_mode_below_3_s_per_minute_including_zero():
    for baseline in (2.999, 1.0, 0.0):
        s = screening.screen(baseline, baseline, 0.8, 0.8, **complete())
        assert s.mode == "absolute" and s.ratio is None and s.fifty_percent == "not applicable"
        assert s.d_fa == 0.0


def test_amendment_1_classes_improved_maintained_worsened_mixed_and_recall():
    assert screening.screen(1.0, 2.0, 0.8, 0.8, **complete()).result == "promising"           # improved
    maintained = screening.screen(2.0, 2.0, 0.9, 0.8, **complete())
    assert maintained.result == "not promising" and maintained.reason.startswith("maintained")
    worsened = screening.screen(2.5, 2.0, 0.9, 0.8, **complete())
    assert worsened.result == "not promising" and worsened.reason.startswith("worsened")
    fragile = {**complete(), "seed_recomputations": SEEDS_OK[:4] + [(4, 0.1, 0.0)]}
    assert screening.screen(1.0, 2.0, 0.8, 0.8, **fragile).result == "mixed"                  # not robust
    assert screening.screen(1.0, 2.0, 0.8, 0.8).result == "mixed"                              # nothing supplied
    below = screening.screen(1.0, 2.0, 0.7499, 0.8, **complete())
    assert below.result == "not promising" and below.reason == "recall below margin"


def test_relative_mode_equal_false_activity_is_not_promising():
    s = screening.screen(4.0, 4.0, 0.9, 0.8, **complete())
    assert s.result == "not promising" and s.d_fa == 0.0


def test_the_recall_margin_boundary_is_inclusive():
    assert screening.screen(1.0, 4.0, 0.75, 0.80, **complete()).result == "promising"
    assert screening.screen(1.0, 4.0, 0.7499, 0.80, **complete()).result == "not promising"
    assert check(seeds=seed_rows(d_fa=-0.1, d_recall=-0.05)).passed
    assert not check(seeds=seed_rows(range(4), -0.1, -0.05) + [(4, -0.1, -0.0501)]).passed


def test_invalid_screening_inputs_are_refused():
    for args in ((-0.1, 3.0, 0.5, 0.5), (1.0, float("nan"), 0.5, 0.5), (1.0, 3.0, 1.2, 0.5)):
        with pytest.raises(ValueError):
            screening.screen(*args)


# -- robustness needs both complete sets (§13 (c), Amendment 1) -------------------------------------


def test_five_distinct_passing_seeds_and_every_group_once_are_robust():
    result = check()
    assert result.passed and result.seeds_expected == result.seeds_supplied == 5
    assert result.groups_expected == result.groups_supplied == 19
    assert screening.robust(**complete())
    assert check(seeds=list(reversed(SEEDS_OK))).passed  # order does not matter, identity does
    assert screening.SEED_INDICES == frozenset({0, 1, 2, 3, 4})


def test_a_failed_result_at_any_seed_index_is_not_robust():
    for index in range(5):
        for d_fa, d_recall in ((0.0, 0.0), (0.2, 0.0), (-1.0, -0.06)):  # D_FA must be < 0 and Δ_R ≥ −0.05
            rows = [(i, d_fa, d_recall) if i == index else (i, -1.0, 0.0) for i in range(5)]
            result = check(seeds=rows)
            assert not result.passed and "1 of 5 seed" in result.reason


def test_a_missing_seed_index_is_not_robust():
    result = check(seeds=seed_rows([0, 1, 2, 4]))
    assert not result.passed and "1 missing" in result.reason


def test_a_duplicated_seed_index_is_not_robust_even_with_five_rows():
    repeated = check(seeds=seed_rows([0, 1, 2, 3, 3]))  # five rows: one seed's result twice, seed 4 absent
    assert not repeated.passed and "1 duplicated" in repeated.reason and "1 missing" in repeated.reason
    same = check(seeds=seed_rows([2, 2, 2, 2, 2]))  # the length alone must not satisfy the check
    assert not same.passed and "4 missing" in same.reason
    extra = check(seeds=seed_rows([0, 1, 2, 3, 4, 2]))
    assert not extra.passed and "1 duplicated" in extra.reason


def test_an_unexpected_seed_index_is_not_robust():
    swapped = check(seeds=seed_rows([0, 1, 2, 3, 5]))  # five rows, index 5 instead of 4
    assert not swapped.passed and "1 unexpected" in swapped.reason and "1 missing" in swapped.reason
    negative = check(seeds=seed_rows([-1, 0, 1, 2, 3]))
    assert not negative.passed and "1 unexpected" in negative.reason


def test_one_failed_leave_one_out_group_is_not_robust():
    rows = leave_one_out()
    rows[7] = (rows[7][0], -1.0, -0.2)
    result = check(rows=rows)
    assert not result.passed and "1 of 19 leave-one-group-out" in result.reason


def test_four_seeds_are_not_enough_and_six_are_too_many():
    assert not check(seeds=seed_rows(range(4))).passed
    six = check(seeds=seed_rows(range(6)))
    assert not six.passed and "6 of 5 rows" in six.reason and "1 unexpected" in six.reason


def test_malformed_or_non_finite_seed_rows_are_refused():
    for bad in ([(0, -1.0)], [(0, float("nan"), 0.0)], [(0, -1.0, float("inf"))], [("0", -1.0, 0.0)],
                [(0.0, -1.0, 0.0)], [(True, -1.0, 0.0)], [(-1.0, 0.0)]):
        with pytest.raises(ValueError):
            check(seeds=SEEDS_OK[:4] + bad)


def test_a_missing_duplicated_or_unexpected_group_is_not_robust():
    missing = check(rows=leave_one_out()[:-1])
    assert not missing.passed and "1 missing" in missing.reason
    duplicated = check(rows=leave_one_out() + [leave_one_out()[3]])
    assert not duplicated.passed and "1 duplicated" in duplicated.reason
    swapped = check(rows=leave_one_out()[:-1] + [("group_999", -1.0, 0.0)])  # right count, wrong group
    assert not swapped.passed and "1 missing" in swapped.reason and "1 unexpected" in swapped.reason
    extra = check(rows=leave_one_out() + [("group_999", -1.0, 0.0)])
    assert not extra.passed and "1 unexpected" in extra.reason


def test_empty_or_incomplete_inputs_never_pass():
    assert not screening.robust((), (), ())
    assert not screening.robust(SEEDS_OK, (), GROUPS)
    assert not screening.robust((), leave_one_out(), GROUPS)
    assert not screening.robust(SEEDS_OK, (), ())  # no expected groups: nothing to be complete against
    assert not screening.robust(SEEDS_OK, leave_one_out(), ())


def test_both_modes_use_the_same_complete_identity_aware_requirement():
    relative_ok = screening.screen(1.0, 4.0, 0.8, 0.8, **complete())
    absolute_ok = screening.screen(1.0, 2.0, 0.8, 0.8, **complete())
    assert relative_ok.mode == "relative" and relative_ok.result == "promising" and relative_ok.robust
    assert absolute_ok.mode == "absolute" and absolute_ok.result == "promising" and absolute_ok.robust
    for broken in ({"group_recomputations": leave_one_out()[:-1]},       # a group missing
                   {"seed_recomputations": seed_rows([0, 1, 2, 3, 3])},      # a seed repeated, five rows
                   {"seed_recomputations": seed_rows([0, 1, 2, 3, 9])}):     # an unexpected seed index
        incomplete = {**complete(), **broken}
        relative_bad = screening.screen(1.0, 4.0, 0.8, 0.8, **incomplete)
        absolute_bad = screening.screen(1.0, 2.0, 0.8, 0.8, **incomplete)
        assert relative_bad.result == absolute_bad.result == "mixed"
        assert not relative_bad.robust and not absolute_bad.robust
        assert relative_bad.robustness == absolute_bad.robustness


def test_results_and_messages_never_carry_group_keys_or_seed_indices():
    rows = leave_one_out()[:-1] + [("group_999", -1.0, 0.0), leave_one_out()[2]]
    result = screening.screen(1.0, 2.0, 0.8, 0.8, SEEDS_OK, rows, GROUPS)
    assert "group_" not in repr(result) and "group_" not in result.robustness
    assert "group_" not in repr(check(rows=rows))
    odd = check(seeds=seed_rows([0, 1, 2, 3, 777]))  # an unexpected index is counted, never named
    assert "777" not in repr(odd)
    with pytest.raises(ValueError) as caught:
        check(seeds=SEEDS_OK[:4] + [("seed-four", -1.0, 0.0)])
    assert "seed-four" not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        check(rows=leave_one_out()[:-1] + [("group_019", float("nan"), 0.0)])
    assert "group_" not in str(caught.value)
    with pytest.raises(ValueError) as caught:
        check(rows=leave_one_out()[:-1] + [("group_019", -1.0)])  # a malformed row
    assert "group_" not in str(caught.value)
    with pytest.raises(ValueError):
        check(groups=GROUPS + ["group_001"])  # the expected set itself has a duplicate


# -- the descriptive bootstrap always carries its caveat -------------------------------------------------


def values():
    rng = np.random.default_rng(2)
    return {f"group_{i:03d}": (float(rng.uniform(1, 6)), float(rng.uniform(4, 12))) for i in range(19)}


def test_every_interval_carries_the_caveat_in_every_rendering():
    interval = bootstrap.difference(values(), draws=500)
    assert interval.caveat == bootstrap.CAVEAT
    assert interval.to_dict()["caveat"] == bootstrap.CAVEAT
    assert bootstrap.CAVEAT in str(interval)
    with pytest.raises(TypeError):
        bootstrap.DescriptiveInterval("x", 0.0, 0.0, 0.0, 1, 0, caveat="")  # cannot be passed in
    with pytest.raises(dataclasses.FrozenInstanceError):
        interval.caveat = ""  # cannot be changed afterwards


def test_the_difference_interval_is_paired_deterministic_and_brackets_the_estimate():
    first, second = bootstrap.difference(values(), draws=2000), bootstrap.difference(values(), draws=2000)
    assert first == second and first.undefined_draws == 0
    assert first.low <= first.estimate <= first.high
    point = np.mean([c for c, _ in values().values()]) - np.mean([b for _, b in values().values()])
    assert first.estimate == pytest.approx(point)


def test_the_ratio_interval_is_withheld_below_3_s_per_minute():
    low = {f"group_{i:03d}": (0.5, 2.0) for i in range(10)}
    withheld = bootstrap.ratio(low, draws=200)
    assert withheld.estimate is None and withheld.low is None and "below 3 s/min" in withheld.note
    assert withheld.caveat == bootstrap.CAVEAT
    shown = bootstrap.ratio(values(), draws=500)
    assert shown.estimate is not None and "of draws had a baseline below 3 s/min" in shown.note


# -- shared post-processing -----------------------------------------------------------------------------


def reference_hysteresis(scores, on, off):
    state, out = False, []
    for s in scores:
        state = (s >= off) if state else (s >= on)
        out.append(state)
    return out


def test_hysteresis_matches_the_engine_rule_and_off_is_t_minus_0_2():
    rng = np.random.default_rng(1)
    scores = rng.uniform(0, 1, 500)
    for t in postprocess.GRID:
        assert postprocess.decide(scores, t).tolist() == reference_hysteresis(scores, t, t - 0.2)
    boundary = np.array([0.6, 0.4, 0.3999, 0.6])
    assert postprocess.decide(boundary, 0.6).tolist() == [True, True, False, True]
    assert postprocess.GRID[0] == 0.30 and postprocess.GRID[-1] == 0.80 and len(postprocess.GRID) == 11


def test_silent_frames_are_forced_to_no_voice_before_the_hysteresis():
    scores = np.array([0.9, 0.9, 0.9, 0.9])
    silent = np.array([False, True, False, False])
    assert postprocess.decide(scores, 0.6, silent).tolist() == [True, False, True, True]
    with pytest.raises(ValueError):
        postprocess.decide(scores, 0.6, silent[:-1])
    with pytest.raises(ValueError):
        postprocess.hysteresis(scores, on=0.3, off=0.5)


# -- registered seeds -------------------------------------------------------------------------------------


def test_training_seeds_follow_the_registered_formula():
    text = "gmv-a5.2:C:group_007:2:4"
    expected = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    assert seeds.training_seed("C", "group_007", 2, 4) == expected
    assert seeds.training_seed("C", "group_007", 2, 4) != seeds.training_seed("B1", "group_007", 2, 4)
    assert seeds.FOLD_SEED == "gmv-a5.2-inner-folds" and seeds.BOOTSTRAP_SEED == "gmv-a5.2-bootstrap"
