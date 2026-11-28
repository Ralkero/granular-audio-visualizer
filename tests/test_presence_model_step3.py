"""A5.2 Step 2a, S2a.5: step 3's split, end codes, rerun trigger, feasibility over every
removed set, and the sensitivity folds' commitments (protocol Amendment 3, A3.6, A3.7 and
A3.10), on generated inputs only."""

import itertools

import pytest

from gmv.character.research.presence_model import commitment, foldgen, folds, integrity, step3

RATE = {44100: "supported", 48000: "supported", 22050: "not_constructible", 30000: "constructible_not_computable"}


def rec(group, rate, i=0):
    return {"id": f"{group}-{i}", "group": group, "sample_rate": rate}


# -- the split -----------------------------------------------------------------------------------------------


def test_the_split_follows_b0s_rate_check():
    records = [rec("a", 44100), rec("a", 48000, 1), rec("b", 44100), rec("low", 22050)]
    assert step3.classify(records, RATE.get) == {"a": "paired", "b": "paired", "low": "low-rate"}


@pytest.mark.parametrize(("records", "words"), (
    ([rec("a", 44100), rec("a", 22050, 1)], "both classes"),
    ([rec("x", 22050), rec("y", 22050)], "more than one group"),
    ([rec("a", 30000)], "neither supported"),
))
def test_split_faults_stop_step_3(records, words):
    with pytest.raises(step3.Step3Error) as caught:
        step3.classify(records, RATE.get)
    assert words in str(caught.value)


def test_the_registry_and_manifest_must_agree_on_development_groups():
    registry = {"entries": {"a": {"partition": "development"}, "b": {"partition": "development"},
                            "v": {"partition": "validation"}}}
    step3.check_groups(registry, {"a": "paired", "b": "low-rate"})
    with pytest.raises(step3.Step3Error):
        step3.check_groups(registry, {"a": "paired"})


# -- end codes and the trigger -------------------------------------------------------------------------------


def test_group_counts_are_coded_at_both_ends():
    codes = [step3.count_code(k, 19) for k in range(20)]
    assert codes[:3] == ["0–2"] * 3 and codes[-3:] == ["17–19"] * 3
    assert codes[3:17] == [str(k) for k in range(3, 17)]
    with pytest.raises(ValueError):
        step3.count_code(20, 19)


def test_a_rerun_is_considered_only_when_the_published_code_shows_three_or_more():
    assert [step3.rerun_triggered(step3.count_code(k, 19), 19) for k in range(20)] == [False] * 3 + [True] * 17
    assert not step3.rerun_triggered(None, 19)  # a withheld size never triggers


# -- feasibility ---------------------------------------------------------------------------------------------


def stats_world(n=19, voiced=None):
    """Paired-group frame counts: every group has both classes unless ``voiced`` limits voice."""
    out = {}
    for i in range(n):
        has_voice = voiced is None or i < voiced
        out[f"g{i:02d}"] = (200 + 13 * i if has_voice else 0, 300 + 7 * i, 220 + 11 * i if has_voice else 0, 310 + 5 * i)
    return out


def test_fourteen_or_more_removed_fails_f1_from_the_size_alone():
    for code in ("14", "16", "17–19"):
        result = step3.feasibility(code, stats_world())
        assert result == step3.Feasibility(False, "F1", 0)


def test_every_removed_set_must_pass_not_just_the_actual_one():
    """Only 4 groups have voice. Removing 3 of them leaves one voiced group, so F5 fails for some
    removed sets of size 3, and the rerun is infeasible even if the actual set kept voice."""
    world = stats_world(n=12, voiced=4)
    result = step3.feasibility("3", world)
    assert not result.feasible and result.failed in ("F4", "F5", "F2")
    world = stats_world(n=12, voiced=8)
    assert step3.feasibility("3", world) == step3.Feasibility(True, None, len(list(itertools.combinations(range(12), 3))))


def test_feasibility_never_reads_membership():
    """The function takes only the published code and the paired groups' counts."""
    world = stats_world(n=10)
    assert step3.feasibility("3", world) == step3.feasibility("3", dict(reversed(list(world.items()))))


def test_the_compute_check_prices_the_most_expensive_removed_set():
    counts = {f"g{i}": c for i, c in enumerate([3, 9, 1, 7, 5, 8])}
    assert step3.worst_case_excerpts("3", counts) == 33 - (1 + 3 + 5)


# -- proposed Amendment 3A: the decision from the public item D figure ---------------------------------------


def test_the_item_d_figure_bounds_the_removed_set_size():
    assert step3.possible_sizes(0) == [0]
    assert step3.possible_sizes(5) == [0, 1, 2, 3, 4, 5]
    assert step3.possible_sizes(10) == [3, 4, 5, 6, 7, 8, 9, 10]
    assert step3.possible_sizes(26) == [19]


def test_a_rerun_is_considered_from_the_item_d_figure_alone():
    assert [step3.rerun_considered_from_item_d(d) for d in range(27)] == [False] * 3 + [True] * 24


def test_item_d_feasibility_checks_every_size_and_every_set():
    world = stats_world(n=12)
    result = step3.feasibility_from_item_d(3, world)  # sizes 1-3 (0 repeats the main run)
    assert result.feasible
    assert result.removed_sets_checked == sum(len(list(itertools.combinations(range(12), k))) for k in (1, 2, 3))


def test_an_item_d_figure_of_14_or_more_fails_f1_with_19_paired_groups():
    world = stats_world()
    for d in (14, 20, 26):
        assert step3.feasibility_from_item_d(d, world) == step3.Feasibility(False, "F1", 0)


# -- sensitivity folds ---------------------------------------------------------------------------------------


def test_sensitivity_folds_use_the_restricted_hashed_order_and_publish_only_commitments():
    world = stats_world(n=10)
    removed = ["g01", "g04", "g07"]
    private, public, opening = step3.sensitivity_folds(world, removed)
    main_order = foldgen.outer_order(world)
    assert private["outer_order"] == [g for g in main_order if g not in removed]  # restricted, renumbered
    assert set(public) == {"folds_commitment", "outer_folds"} and public["outer_folds"] == 7
    hashes = opening.to_private_record()["hashes"]
    assert public["folds_commitment"] not in hashes.values()
    assert commitment.verify(public["folds_commitment"], opening)


def test_trying_every_removed_set_never_reproduces_the_fold_commitment():
    world = stats_world(n=9)
    _private, public, _opening = step3.sensitivity_folds(world, ["g00", "g03", "g05"])
    for removed in itertools.combinations(sorted(world), 3):
        remaining = {g: s for g, s in world.items() if g not in removed}
        counts = {g: (s[0], s[1]) for g, s in remaining.items()}
        seconds = {g: (v / 50, n / 50) for g, (v, n) in counts.items()}
        order = foldgen.outer_order(counts)
        candidate = [integrity.sha256_bytes(folds.serialise(folds.for_outer(seconds, o))) for o in order]
        for h in [*candidate, integrity.sha256_bytes(integrity.canonical_json(order))]:
            assert h != public["folds_commitment"]
