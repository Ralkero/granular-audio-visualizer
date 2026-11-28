"""A5.2 Step 4, S4.2: model C's training inputs (protocol §6; Amendment 3, A3.2 B15, B16, V7,
V8; A3.3 R21, R22), on generated excerpts only."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import stage2_synth  # noqa: E402

from gmv.character.research.presence_model import augment, cbatches, normalise, seedplan  # noqa: E402

IDENTITY = "gmv-a5.2:C:0:outer:0"


def generators(identity=IDENTITY):
    n = seedplan.integer_of(identity)
    return {s: seedplan.numpy_generator(n, seedplan.consumer_key("C", f"augment-{s}")) for s in cbatches.STREAMS}


def test_statistics_come_from_the_selected_rows_before_augmentation_and_ignore_order():
    excerpts = stage2_synth.training_set()
    stats = cbatches.statistics(excerpts)
    rows = np.concatenate([augment.log_power(e.power)[e.selected] for e in excerpts])
    assert np.array_equal(stats.mean, rows.mean(axis=0)) or np.allclose(stats.mean, rows.mean(axis=0), rtol=1e-15)
    assert np.allclose(stats.scale, rows.std(axis=0) + normalise.SPREAD_EPSILON, rtol=1e-13)
    assert stats.frames == len(rows)
    shuffled = cbatches.statistics(list(reversed(excerpts)))
    assert np.array_equal(stats.mean, shuffled.mean) and np.array_equal(stats.scale, shuffled.scale)


def test_context_and_quiet_rows_never_enter_the_statistics():
    excerpts = stage2_synth.training_set()
    changed = []
    for e in excerpts:
        power = e.power.copy()
        power[~e.selected] *= 1e6  # context and quiet rows only
        changed.append(cbatches.Excerpt(e.key, e.group, e.voiced, power, e.labels, e.eligible, e.selected, e.excerpt_row))
    a, b = cbatches.statistics(excerpts), cbatches.statistics(changed)
    assert np.array_equal(a.mean, b.mean) and np.array_equal(a.scale, b.scale)


def test_context_rows_cannot_be_trained_on_or_used_for_statistics():
    e = stage2_synth.excerpt("x", "g", 50)
    eligible = e.eligible.copy()
    eligible[0] = True
    with pytest.raises(ValueError):
        cbatches.Excerpt(e.key, e.group, e.voiced, e.power, e.labels, eligible, e.selected, e.excerpt_row)
    with pytest.raises(ValueError):  # voiced must match the labels
        cbatches.Excerpt(e.key, e.group, False, e.power, e.labels, e.eligible, e.selected, e.excerpt_row)


def test_the_weights_are_the_section_6_objective_on_these_rows():
    excerpts = stage2_synth.training_set()
    w = cbatches.frame_weights(excerpts)
    assert abs(sum(x.sum() for x in w) - 1.0) < 1e-12
    for e, x in zip(excerpts, w, strict=True):
        assert not x[: e.excerpt_row].any() and not x[~e.eligible].any()


def test_partners_are_no_voice_excerpts_of_the_same_group_and_never_the_excerpt_itself():
    excerpts = stage2_synth.training_set()
    keys = [e.key for e in excerpts]
    got = {keys[i]: sorted(keys[j] for j in js) for i, js in cbatches.partners(excerpts).items()}
    assert got == {"a-2": [], "b-1": ["b-2", "b-3"], "b-2": ["b-1", "b-3"], "b-3": ["b-1", "b-2"]}
    assert "a-1" not in got and "c-1" not in got  # voiced excerpts are never mixed


def test_the_draws_follow_v7_and_each_stream_is_separate():
    order = list(range(400))
    d = cbatches.epoch_draws(generators(), order)
    assert 0.18 < np.mean([x["mix"] for x in d]) < 0.32
    assert all(-6 <= x["level_db"] <= 0 and -6 <= x["gain_db"] <= 6 and -3 <= x["tilt_db"] <= 3 for x in d)
    assert {x["mask_width"] for x in d} == set(range(7))
    assert all(0 <= x["mask_start"] <= 64 - x["mask_width"] for x in d)
    g = generators()
    g["mix"] = np.random.default_rng(99)  # a different mixing stream changes nothing else
    other = cbatches.epoch_draws(g, order)
    assert [x["gain_db"] for x in d] == [x["gain_db"] for x in other]
    assert [(x["mask_width"], x["mask_start"]) for x in d] == [(x["mask_width"], x["mask_start"]) for x in other]
    assert cbatches.draws_sha256(d) == cbatches.draws_sha256(cbatches.epoch_draws(generators(), order))
    assert cbatches.draws_sha256(d) != cbatches.draws_sha256(other)


def test_the_partner_is_aligned_by_excerpt_start():
    a = stage2_synth.excerpt("p", "g", 0, rows=100, context=20, seed=1)
    b = stage2_synth.excerpt("q", "g", 0, rows=90, context=5, seed=2)
    out = cbatches.aligned(a, b)
    assert np.array_equal(out[20], b.power[5]) and np.array_equal(out[99], b.power[84])  # excerpt starts meet
    assert np.array_equal(out[15], b.power[0])  # the partner's own context, where it has some
    assert not out[:15].any()  # no partner frame there: no power


def test_inputs_follow_the_registered_order_and_keep_the_primarys_labels():
    excerpts = stage2_synth.training_set()
    stats = cbatches.statistics(excerpts)
    partner_map = cbatches.partners(excerpts)
    draw = {"mix": True, "level_db": -3.0, "partner_u": 0.7, "gain_db": 2.0, "tilt_db": -1.5,
            "mask_width": 4, "mask_start": 10}
    b1 = 2  # "b-1": partners b-2, b-3; u = 0.7 picks b-3
    x = cbatches.inputs(excerpts, b1, draw, partner_map, stats)
    primary, partner = excerpts[b1], excerpts[4]
    mixed = augment.mix(primary.power, cbatches.aligned(primary, partner), primary.selected, -3.0, "M1")
    expected = augment.mask_bands(normalise.apply(augment.log_power(augment.tilt(augment.gain(mixed, 2.0), -1.5)), stats),
                                  10, 4)
    assert np.array_equal(x, expected.astype(np.float32))
    assert not x[:, 10:14].any() and x.dtype == np.float32


def test_a_voiced_excerpt_is_never_mixed_even_when_the_draw_says_mix():
    excerpts = stage2_synth.training_set()
    stats = cbatches.statistics(excerpts)
    draw = {"mix": True, "level_db": -1.0, "partner_u": 0.0, "gain_db": 0.0, "tilt_db": 0.0, "mask_width": 0,
            "mask_start": 0}
    x = cbatches.inputs(excerpts, 0, draw, cbatches.partners(excerpts), stats)
    assert np.array_equal(x, cbatches.plain(excerpts[0].power, stats))
    lonely = cbatches.inputs(excerpts, 1, draw, cbatches.partners(excerpts), stats)  # a-2 has no partner
    assert np.array_equal(lonely, cbatches.plain(excerpts[1].power, stats))


def test_duplicate_keys_are_refused():
    e = stage2_synth.excerpt("same", "g", 120)
    with pytest.raises(ValueError):
        cbatches.statistics([e, e])
