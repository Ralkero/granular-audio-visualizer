"""A5.2 Step 1, S1.8: normalisation interfaces and augmentation identities (generated arrays).

The options and values under test are pending decisions; these tests check the mathematics
and the isolation properties, not which choice is right.
"""

import numpy as np
import pytest

from gmv.character.research.presence_model import augment, grid, normalise, seedplan

RNG = np.random.default_rng(11)


def power(frames=120, bands=64, scale=1.0):
    return scale * RNG.gamma(2.0, 1.0, (frames, bands))


# -- normalisation -------------------------------------------------------------------------------


def units(n=4):
    return [(f"g{i}", RNG.normal(i, 1 + i, (50, 8)), np.arange(50) % 3 != 0) for i in range(n)]


def test_pooled_statistics_follow_their_definition_and_ignore_input_order():
    u = units()
    stats = normalise.pooled(u)
    x = np.concatenate([f[m] for _k, f, m in sorted(u, key=lambda t: t[0])])
    assert np.array_equal(stats.mean, x.mean(axis=0)) and np.array_equal(stats.scale, x.std(axis=0) + 1e-6)
    assert stats.frames == len(x)
    assert normalise.pooled(list(reversed(u))).sha256 == stats.sha256


def test_frames_outside_the_selection_and_other_groups_do_not_affect_statistics():
    u = units()
    stats = normalise.pooled(u)
    changed = [(k, np.where(m[:, None], f, 1e6), m) for k, f, m in u]  # unselected frames altered
    assert normalise.pooled(changed).sha256 == stats.sha256
    # statistics depend only on the units passed in: a held-out group changes nothing unless it
    # is (wrongly) included, which the next line shows would be visible in the digest
    held_out = ("g9", RNG.normal(0, 100, (50, 8)), np.ones(50, dtype=bool))
    assert normalise.pooled(u).sha256 != normalise.pooled([*u, held_out]).sha256


def test_the_statistics_mask_keeps_excerpt_frames_that_are_not_quiet():
    seg = grid.canonical(20 * 44100, 44100, 7.0, 17.0)
    quiet = np.zeros(seg.frames, dtype=bool)
    quiet[seg.f0 + 10 : seg.f0 + 20] = True
    mask = normalise.statistics_mask(seg, quiet)
    assert not mask[: seg.f0].any() and not mask[seg.f1 :].any()
    assert mask.sum() == (seg.f1 - seg.f0) - 10


def test_fixed_constants_are_validated_and_applied_as_float32():
    stats = normalise.fixed(np.zeros(548), np.full(548, 2.0))
    out = normalise.apply(np.full((3, 548), 4.0), stats)
    assert out.dtype == np.float32 and np.all(out == 2.0)
    for bad in ((np.zeros(547), np.ones(547)), (np.zeros(548), np.zeros(548)), (np.full(548, np.nan), np.ones(548))):
        with pytest.raises(ValueError):
            normalise.fixed(*bad)


# -- M1 mixing ---------------------------------------------------------------------------------------


@pytest.mark.parametrize("level_db", (-6.0, -3.0, 0.0))
def test_m1_keeps_the_primary_power_and_sets_the_partner_ratio(level_db):
    a, b = power(), power(scale=7.0)
    reference = np.arange(len(a)) % 5 != 0
    mixed = augment.mix(a, b, reference, level_db)
    e_a = a[reference].sum()
    assert mixed[reference].sum() == pytest.approx(e_a, rel=1e-12)
    g = 10 ** (level_db / 10)
    partner_part = g * (e_a / b[reference].sum()) * b[reference].sum() / (1 + g)
    primary_part = e_a / (1 + g)
    ratio_db = 10 * np.log10(partner_part / primary_part)
    assert ratio_db == pytest.approx(level_db) and -6 <= ratio_db <= 0
    unscaled = a + g * (e_a / b[reference].sum()) * b
    assert np.allclose(mixed * (1 + g), unscaled)  # rescaling multiplies both parts alike


def test_m1_skips_silent_inputs_and_refuses_non_finite_ones():
    a, ref = power(), np.ones(120, dtype=bool)
    assert augment.mix(np.zeros_like(a), a, ref, -3) is None
    assert augment.mix(a, np.full_like(a, 1e-30), ref, -3) is None
    bad = a.copy()
    bad[0, 0] = np.inf
    with pytest.raises(ValueError):
        augment.mix(bad, a, ref, -3)


def test_m0_is_a_plain_sum_and_m2_is_not_available():
    a, b = power(), power()
    assert np.allclose(augment.mix(a, b, None, -3, mode="M0"), a + 10 ** -0.3 * b)
    with pytest.raises(NotImplementedError):
        augment.mix(a, b, None, -3, mode="M2")


# -- gain, tilt, log, mask and their order -----------------------------------------------------------


def test_gain_and_tilt_are_multiplicative_and_commute():
    a = power()
    assert np.allclose(augment.tilt(augment.gain(a, 4), 2), augment.gain(augment.tilt(a, 2), 4))
    tilted = augment.tilt(np.ones((1, 64)), 3.0)
    assert 10 * np.log10(tilted[0, 0]) == pytest.approx(-3.0) and 10 * np.log10(tilted[0, -1]) == pytest.approx(3.0)


def test_gain_acts_before_the_log_floor():
    silent = np.zeros((2, 64))
    assert np.array_equal(augment.log_power(augment.gain(silent, 6)), augment.log_power(silent))


def test_the_full_order_masks_after_standardisation():
    a, b = power(), power(scale=3.0)
    stats = normalise.Stats(np.full(64, 1.0), np.full(64, 2.0), 0)
    d = {"mix": True, "level_db": -2.0, "gain_db": 1.5, "tilt_db": -1.0, "mask_start": 10, "mask_width": 4}
    out = augment.augment(a, stats, d, partner=b, reference=np.ones(120, dtype=bool))
    assert out.dtype == np.float32 and np.all(out[:, 10:14] == 0)
    expected = normalise.apply(augment.log_power(augment.tilt(augment.gain(
        augment.mix(a, b, np.ones(120, dtype=bool), -2.0), 1.5), -1.0)), stats)
    assert np.array_equal(np.delete(out, range(10, 14), axis=1), np.delete(expected, range(10, 14), axis=1))


def test_draws_come_from_the_identity_sub_streams_and_are_reproducible():
    identity = "gmv-a5.2-gate:C:outer-0:conv:0"
    n = seedplan.integer_of(identity)

    def streams():
        return {name.split("-")[1]: seedplan.numpy_generator(n, seedplan.consumer_key("C", name))
                for name in ("augment-mix", "augment-gain", "augment-tilt", "augment-mask")}

    args = dict(mix_probability=0.25, level_db=(-6, 0), gain_db=(-6, 6), tilt_db=(-3, 3), max_mask=6)
    s1, s2 = streams(), streams()
    first = [augment.draw(s1, **args) for _ in range(20)]
    assert first == [augment.draw(s2, **args) for _ in range(20)]
    assert all(-6 <= d["level_db"] <= 0 and 0 <= d["mask_width"] <= 6 and d["mask_start"] + d["mask_width"] <= 64 for d in first)
