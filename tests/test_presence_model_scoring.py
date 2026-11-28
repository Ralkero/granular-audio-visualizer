"""A5.2 Step 4, S4.4: single-model scoring, operating points, aggregates, effect sizes and
policy D in screening (protocol §8, §11-§13; Amendment 3, A3.1). Generated scores only."""

from fractions import Fraction

import numpy as np
import pytest

from gmv.character.research.presence_model import metrics, postprocess, scoring, screening

CENTRES = np.arange(-50, 500) * 0.02 + 0.0232  # 50 pre-roll frames, then 500 excerpt frames


def scored(group, scores, confirmed=((2.0, 5.0),), silent=None):
    scores = np.asarray(scores, dtype=float)
    return scoring.Scored(group, CENTRES, 10.0, tuple(confirmed), (), np.zeros(len(CENTRES), bool) if silent is None
                          else silent, scores)


def voice_scores(level_voice=0.9, level_none=0.1, confirmed=((2.0, 5.0),)):
    s = np.full(len(CENTRES), level_none)
    for a, b in confirmed:
        s[(CENTRES >= a) & (CENTRES < b)] = level_voice
    return s


def test_hysteresis_runs_over_the_whole_segment_from_an_inactive_start():
    s = np.full(len(CENTRES), 0.45)  # between off (0.3) and on (0.5) at t = 0.5
    s[:10] = 0.9  # pre-roll switches the detector on; it stays on into the excerpt
    on = scoring.counts_at([scored("g", s, confirmed=())], 0.5)["g"]
    assert on.false == on.novoice
    s[:10] = 0.45
    off = scoring.counts_at([scored("g", s, confirmed=())], 0.5)["g"]
    assert off.false == 0


def test_quiet_frames_are_forced_to_zero_and_still_counted():
    silent = np.zeros(len(CENTRES), bool)
    silent[100:200] = True
    c = scoring.counts_at([scored("g", np.full(len(CENTRES), 0.9), confirmed=(), silent=silent)], 0.5)["g"]
    assert c.novoice == 500 and c.false == 400


def test_j_is_exact_and_equals_the_float_objective():
    groups = scoring.counts_at([scored("a", voice_scores()), scored("b", voice_scores(0.6, 0.55))], 0.5)
    exact = scoring.j_exact(groups)
    assert isinstance(exact, Fraction) and float(exact) == pytest.approx(metrics.objective(groups), rel=1e-15)


@pytest.mark.parametrize("values, expected", (
    ({0.3: 1, 0.5: 3, 0.8: 2}, 0.5),
    ({0.4: 3, 0.45: 3, 0.7: 3}, 0.45),  # closest to 0.5
    ({0.45: 3, 0.55: 3}, 0.55),           # equally close: the higher one
    ({0.3: Fraction(1, 3), 0.35: Fraction(2, 6)}, 0.35),
))
def test_the_tie_rule(values, expected):
    assert scoring.choose(values) == expected


def test_j_bar_averages_the_inner_seeds_and_the_policy_fixes_which_seeds():
    a = scoring.counts_grid([scored("a", voice_scores(0.7, 0.2)), scored("b", voice_scores(0.9, 0.42))])
    b = scoring.counts_grid([scored("a", voice_scores(0.52, 0.2)), scored("b", voice_scores(0.9, 0.2))])
    bar = scoring.j_bar({0: a, 1: b})
    assert all(bar[t] == (scoring.j_exact(a[t]) + scoring.j_exact(b[t])) / 2 for t in postprocess.GRID)
    assert scoring.operating_point({0: a, 1: b}, "C") == scoring.choose(bar)
    assert scoring.operating_point({0: a}, "D") == scoring.choose({t: scoring.j_exact(a[t]) for t in a})
    with pytest.raises(ValueError):
        scoring.operating_point({0: a}, "C")
    with pytest.raises(ValueError):
        scoring.operating_point({0: a, 1: b}, "D")
    assert scoring.inner_agreement({0: a, 1: a}) is True


def outer(policy="C", candidate_none=0.05, baseline_none=0.5, seed_spread=0.01):
    """Per-seed outer counts for 6 groups (one voice-only, one no-voice-only)."""
    seeds = screening.seed_indices(policy)
    confirmed = {"g0": ((2.0, 5.0),), "g1": ((1.0, 9.0),), "g2": (), "g3": ((0.0, 10.0),), "g4": ((3.0, 4.0),),
                 "g5": ((2.0, 3.0),)}
    candidate = {}
    for s in sorted(seeds):
        rows = [scored(g, voice_scores(0.9, candidate_none + seed_spread * s + 0.1 * (k == 2), c), c)
                for k, (g, c) in enumerate(confirmed.items())]
        candidate[s] = scoring.counts_at(rows, 0.5)
    baseline = scoring.counts_at([scored(g, voice_scores(0.95, baseline_none + 0.02 * k, c), c)
                                  for k, (g, c) in enumerate(confirmed.items())], 0.5)
    return candidate, baseline


def test_the_primary_effects_are_means_of_the_seed_index_aggregates():
    candidate, baseline = outer()
    e = scoring.effects(candidate, baseline, "C")
    aggregates = scoring.seed_aggregates(candidate)
    assert e.fpa_candidate == pytest.approx(np.mean([aggregates[s]["fpa"] for s in range(5)]), rel=1e-15)
    assert e.d_fa == e.fpa_candidate - e.fpa_baseline
    assert [r[0] for r in e.seed_rows] == [0, 1, 2, 3, 4]
    assert e.seed_rows[2][1] == pytest.approx(aggregates[2]["fpa"] - e.fpa_baseline)


def test_the_macro_of_group_means_equals_the_mean_of_the_aggregates():
    candidate, _ = outer()
    means = scoring._means_over_seeds(candidate, "fpa")
    macro_of_means = np.mean([v for v in means.values() if v is not None])
    mean_of_macros = np.mean([a["fpa"] for a in scoring.seed_aggregates(candidate).values()])
    assert macro_of_means == pytest.approx(mean_of_macros, rel=1e-12)


def test_leave_one_group_out_rows_remove_each_groups_test_result_with_models_fixed():
    candidate, baseline = outer()
    e = scoring.effects(candidate, baseline, "C")
    assert [g for g, _d, _r in e.group_rows] == sorted(baseline)
    means = scoring._means_over_seeds(candidate, "fpa")
    rest = [g for g in baseline if g != "g0" and means[g] is not None]
    expected = np.mean([means[g] for g in rest]) - np.mean([baseline[g].fpa for g in rest])
    assert dict((g, d) for g, d, _r in e.group_rows)["g0"] == pytest.approx(expected, rel=1e-12)


def test_the_ratio_is_used_only_from_3_seconds_per_minute():
    candidate, baseline = outer(baseline_none=0.6)
    high = scoring.effects(candidate, baseline, "C")
    assert high.fpa_baseline >= 3 and high.ratio == pytest.approx(high.fpa_candidate / high.fpa_baseline)
    low_c, low_b = outer(baseline_none=0.0, candidate_none=0.0)
    assert scoring.effects(low_c, low_b, "C").ratio is None


def test_policy_d_screens_on_three_seed_aggregates_and_says_so():
    candidate, baseline = outer("D")
    e = scoring.effects(candidate, baseline, "D")
    result = e.screen("D")
    assert result.policy == "D" and "less demanding" in result.policy_note
    assert "3 seed" in result.robustness or result.robust
    with pytest.raises(ValueError):
        scoring.effects(candidate, baseline, "C")  # three aggregates are not policy C's five
    c5, b5 = outer("C")
    assert not scoring.effects(c5, b5, "C").screen("D").robust  # five rows are not policy D's set


def test_screening_robustness_counts_follow_the_policy():
    rows3 = [(s, -1.0, 0.0) for s in range(3)]
    groups = [("g", -1.0, 0.0)]
    assert screening.check_robustness(rows3, groups, ["g"], "D").passed
    assert not screening.check_robustness(rows3, groups, ["g"], "C").passed
    assert screening.check_robustness(rows3, groups, ["g"], "D").seeds_expected == 3
    with pytest.raises(ValueError):
        screening.check_robustness(rows3, groups, ["g"], "E")


def test_t_final_maximises_the_mean_j_over_the_outer_seeds():
    grids = {s: scoring.counts_grid([scored("a", voice_scores(0.62 + 0.01 * s, 0.3)),
                                     scored("b", voice_scores(0.9, 0.36))]) for s in range(3)}
    expected = scoring.choose({t: sum(scoring.j_exact(grids[s][t]) for s in grids) / 3 for t in postprocess.GRID})
    assert scoring.t_final(grids) == expected


def test_shifted_points_and_exact_group_values():
    assert scoring.shifted({"a": 0.3, "b": 0.8}, 0.05) == {"a": 0.35, "b": 0.85}
    c = metrics.Counts(voiced=150, novoice=100, detected=75, false=5)
    v = scoring.group_values({"g": c, "h": metrics.Counts(voiced=99, novoice=0)})
    assert v["g"] == {"fpa": Fraction(3), "recall": Fraction(1, 2)} and v["h"] == {"fpa": None, "recall": None}
