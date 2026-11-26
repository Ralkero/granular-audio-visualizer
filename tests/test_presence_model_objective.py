"""A5.2 Stage 1, protocol §16 "Loss" (and the labels it depends on, §6 and §10). Synthetic
units and groups only.

Floating-point tolerance: weights are sums of at most a few thousand terms of size about
1e-4, so float64 rounding stays near 1e-15; assertions use rel=1e-12."""

import numpy as np
import pytest

from gmv.character.research.presence_model import labels
from gmv.character.research.presence_model import weights as w

REL = 1e-12


def unit(group, voiced, novoice, excluded=0, seed=0):
    """A unit with ``voiced`` eligible voice frames, ``novoice`` eligible no-voice frames and
    ``excluded`` frames of either class that are not eligible."""
    rng = np.random.default_rng(seed)
    y = np.array([1] * voiced + [0] * novoice + list(rng.integers(0, 2, excluded)), dtype=np.int8)
    eligible = np.array([True] * (voiced + novoice) + [False] * excluded)
    order = rng.permutation(len(y))
    return w.Unit(group, y[order], eligible[order])


def project():
    """Unequal groups on purpose: a long group, short groups, single-class groups, a group
    below the 2 s minimum in one class, and excluded frames."""
    return [
        unit("group_001", 3000, 2000, excluded=400, seed=1),  # long, both classes
        unit("group_001", 1500, 0, seed=2),                   # the same group, a second unit
        unit("group_002", 150, 300, excluded=50, seed=3),     # short, both classes
        unit("group_003", 0, 900, seed=4),                    # no-voice only
        unit("group_004", 700, 0, seed=5),                    # voice only
        unit("group_005", 99, 400, seed=6),                   # voice below 100 frames (2 s)
        unit("group_006", 100, 100, seed=7),                  # exactly at the minimum
    ]


def logits_for(units, seed=11):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, 2, len(u.labels)) for u in units]


def totals(units, weights):
    out = {}
    for u, ws in zip(units, weights, strict=True):
        for c in (0, 1):
            key = (u.group, c)
            out[key] = out.get(key, 0.0) + float(ws[np.asarray(u.labels) == c].sum())
    return out


def test_every_member_group_has_exactly_its_equal_share_in_each_class():
    units = project()
    member = w.members(units)
    assert member[1] == ["group_001", "group_002", "group_004", "group_006"]
    assert member[0] == ["group_001", "group_002", "group_003", "group_005", "group_006"]
    t = totals(units, w.frame_weights(units))
    for c in (0, 1):
        for group in member[c]:
            assert t[(group, c)] == pytest.approx(1 / (2 * len(member[c])), rel=REL)
    assert sum(t.values()) == pytest.approx(1.0, rel=REL)


def test_the_weighted_sum_reproduces_the_registered_loss():
    units = project()
    for seed in range(5):
        z = logits_for(units, seed)
        assert w.objective(units, z) == pytest.approx(w.registered_loss(units, z), rel=REL)


def test_a_longer_group_gains_no_weight_but_pooled_weighting_would():
    units = project()
    longer = units + [unit("group_002", 6000, 6000, seed=20 + i) for i in range(3)]
    t_before, t_after = totals(units, w.frame_weights(units)), totals(longer, w.frame_weights(longer))
    for c in (0, 1):
        assert t_after[("group_002", c)] == pytest.approx(t_before[("group_002", c)], rel=REL)
    # Ordinary pooled class balance (each class's frames sharing 1/2 equally) would hand the
    # longer group far more weight: the registered objective must not behave like it.
    pooled = {}
    for c in (0, 1):
        n = sum(int(np.count_nonzero(u.eligible & (np.asarray(u.labels) == c))) for u in longer)
        mine = sum(int(np.count_nonzero(u.eligible & (np.asarray(u.labels) == c))) for u in longer if u.group == "group_002")
        pooled[c] = 0.5 * mine / n
    assert pooled[1] > 3 * t_after[("group_002", 1)]


def test_single_class_groups_count_only_in_their_class():
    units = project()
    t = totals(units, w.frame_weights(units))
    assert t[("group_003", 1)] == 0 and t[("group_003", 0)] > 0
    assert t[("group_004", 0)] == 0 and t[("group_004", 1)] > 0


def test_the_two_second_minimum_is_inclusive_and_per_class():
    units = project()
    t = totals(units, w.frame_weights(units))
    assert t[("group_005", 1)] == 0  # 99 voice frames: not a member for voice ...
    assert t[("group_005", 0)] == pytest.approx(1 / (2 * 5), rel=REL)  # ... still a member for no voice
    assert t[("group_006", 1)] == pytest.approx(1 / (2 * 4), rel=REL)  # exactly 100 frames: a member


def test_excluded_frames_weigh_nothing_and_do_not_change_the_denominators():
    units = project()
    base = w.frame_weights(units)
    padded = [w.Unit(u.group, np.concatenate([u.labels, np.ones(300, np.int8)]),
                     np.concatenate([u.eligible, np.zeros(300, bool)])) for u in units]
    more = w.frame_weights(padded)
    for b, m, u in zip(base, more, units, strict=True):
        assert np.array_equal(m[: len(u.labels)], b)  # the same weights, bit for bit
        assert not m[len(u.labels):].any()


def test_one_epoch_of_batches_reproduces_the_objective_for_any_order_and_size():
    units = project()
    z = logits_for(units)
    target = w.objective(units, z)
    rng = np.random.default_rng(3)
    for batch_size in (1, 2, 3, 7):
        order = list(rng.permutation(len(units)))
        losses = w.batch_losses(units, z, order, batch_size)
        assert len(losses) == -(-len(units) // batch_size)
        assert float(np.mean(losses)) == pytest.approx(target, rel=REL)


def test_normalising_each_batch_by_its_own_weight_would_not_reproduce_the_objective():
    """Regression: per-batch normalisation reweights groups by batch composition."""
    units = project()
    z = logits_for(units)
    weights = w.frame_weights(units)
    order, size = list(range(len(units))), 2
    normalised = []
    for i in range(0, len(order), size):
        batch = order[i : i + size]
        mass = sum(weights[j].sum() for j in batch)
        normalised.append(sum(float(np.dot(weights[j], w.bce_with_logits(z[j], units[j].labels))) for j in batch) / mass)
    assert float(np.mean(normalised)) != pytest.approx(w.objective(units, z), rel=1e-3)


def test_invalid_inputs_are_refused():
    with pytest.raises(ValueError):
        w.Unit("group_001", np.array([0, 2]), np.array([True, True]))
    with pytest.raises(ValueError):
        w.Unit("group_001", np.array([0, 1]), np.array([True]))
    with pytest.raises(ValueError):
        w.Unit("group_001", np.array([0, 1]), np.array([1, 1]))  # eligibility must be boolean
    with pytest.raises(ValueError):  # nobody has 2 s of voice: the objective is undefined
        w.frame_weights([unit("group_001", 50, 500)])
    units = project()
    with pytest.raises(ValueError):
        w.batch_losses(units, logits_for(units), [0, 0, 1, 2, 3, 4, 5], 2)  # a unit twice, one missing
    with pytest.raises(ValueError):
        w.batch_losses(units, logits_for(units), list(range(len(units))), 0)


def test_the_cross_entropy_is_stable_for_large_logits():
    z = np.array([-800.0, -30.0, 0.0, 30.0, 800.0])
    loss1, loss0 = w.bce_with_logits(z, np.ones(5)), w.bce_with_logits(z, np.zeros(5))
    assert np.isfinite(loss1).all() and np.isfinite(loss0).all()
    assert loss1[-1] == pytest.approx(0.0, abs=1e-300) and loss0[0] == pytest.approx(0.0, abs=1e-300)
    assert loss1[2] == pytest.approx(np.log(2)) and loss1[0] == pytest.approx(800.0)


# -- labels, collars and the frames each use leaves out --------------------------------------------


def grid(duration, start=-1.0):
    return np.round(np.arange(start + 0.0232, duration, 0.02), 6)


def test_labels_are_half_open_and_follow_the_union_of_confirmed_intervals():
    centres = np.array([0.999, 1.0, 1.5, 1.999, 2.0, 2.5, 3.0])
    y = labels.frame_labels(centres, [(1.0, 2.0), (1.8, 2.5)], 10.0)  # overlapping: one union [1, 2.5)
    assert y.tolist() == [0, 1, 1, 1, 1, 0, 0]


def test_collars_are_inclusive_at_0_3_s_and_skip_excerpt_edges_and_touching_intervals():
    centres = np.array([4.69, 4.7, 5.0, 5.3, 5.31, 0.1, 9.9])
    mask = labels.collar_mask(centres, [(0.0, 5.0)], 10.0)  # start at the edge: no boundary there
    assert mask.tolist() == [False, True, True, True, False, False, False]
    assert labels.interior_boundaries([(2.0, 4.0), (4.0, 10.0)], 10.0) == [2.0]  # touching: merged
    assert labels.interior_boundaries([(0.0, 10.0)], 10.0) == []


def test_uncertain_only_time_preroll_and_silence_are_left_out_where_specified():
    duration = 10.0
    centres = grid(duration)
    confirmed, uncertain = [(2.0, 4.0)], [(3.0, 6.0)]
    silent = np.zeros(len(centres), bool)
    silent[(centres > 8.0) & (centres < 8.5)] = True
    evaluation = labels.evaluation_mask(centres, duration, confirmed, uncertain)
    training = labels.training_mask(centres, duration, confirmed, uncertain, silent)
    preroll = centres < 0
    assert not evaluation[preroll].any() and not training[preroll].any()
    inside_uncertain_only = (centres >= 4.3) & (centres < 6.0)  # after the end collar of 4.0
    assert not evaluation[inside_uncertain_only].any()
    assert evaluation[(centres > 2.3) & (centres < 3.7)].all()  # confirmed wins over uncertain
    assert evaluation[silent].all() and not training[silent].any()  # silence: training only
    with pytest.raises(ValueError):
        labels.training_mask(centres, duration, confirmed, uncertain, silent[:-1])


def test_intervals_outside_the_excerpt_are_refused():
    with pytest.raises(ValueError):
        labels.frame_labels([1.0], [(2.0, 1.0)], 10.0)
    with pytest.raises(ValueError):
        labels.frame_labels([1.0], [(9.0, 10.5)], 10.0)
