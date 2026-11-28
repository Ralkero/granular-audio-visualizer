"""A5.2 Step 4, S4.2: model C's training loop, inference and gate loss (protocol §5, §6;
Amendment 3, A3.2 B15-B17, V7, V8, A3.3). Needs PyTorch: runs in the experiments environment and
is skipped elsewhere. Generated excerpts only; short recipes (the registered 30 epochs are the
default, tests pass fewer explicitly)."""

import hashlib
import math
import stat
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "presence"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import ctrain  # noqa: E402
import stage2_synth  # noqa: E402

from gmv.character.research.presence_model import cbatches  # noqa: E402

IDENTITY = "gmv-a5.2:C:0:outer:0"
OTHER = "gmv-a5.2:C:0:outer:1"


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture(scope="module")
def excerpts():
    return stage2_synth.training_set(rows=120)


def test_the_registered_recipe_values():
    assert (ctrain.EPOCHS, ctrain.BATCH, ctrain.RATE, ctrain.FINAL_RATE) == (30, 8, 1e-3, 1e-5)
    assert (ctrain.BETAS, ctrain.EPSILON, ctrain.DECAY) == ((0.9, 0.999), 1e-8, 1e-4)


def test_the_cosine_starts_at_1e3_and_reaches_1e5_and_falls_every_step():
    total = 600
    rates = [ctrain.rate_at(k, total) for k in range(total + 1)]
    assert rates[0] == 1e-3 and math.isclose(rates[-1], 1e-5, rel_tol=1e-12)
    assert all(a > b for a, b in zip(rates, rates[1:], strict=False))
    assert ctrain.rate_at(0, 300) == ctrain.rate_at(0, 450)  # step 1 is the same whatever the horizon
    assert ctrain.rate_at(1, 300) != ctrain.rate_at(1, 450)  # step 2 is not


def test_the_same_seed_gives_identical_weights_and_records(excerpts):
    a = ctrain.train(IDENTITY, excerpts, epochs=2, batch=4)
    b = ctrain.train(IDENTITY, excerpts, epochs=2, batch=4)
    assert ctrain.to_bytes(a) == ctrain.to_bytes(b)
    assert (a.order_sha256, a.draws_sha256, a.dropout_sha256) == (b.order_sha256, b.draws_sha256, b.dropout_sha256)
    c = ctrain.train(OTHER, excerpts, epochs=2, batch=4)
    assert ctrain.to_bytes(a) != ctrain.to_bytes(c) and a.order_sha256 != c.order_sha256


def test_records_cover_every_epoch_and_step(excerpts):
    r = ctrain.train(IDENTITY, excerpts, epochs=3, batch=4)
    j = -(-len(excerpts) // 4)
    assert r.steps == 3 * j and len(r.dropout_sha256) == r.steps
    assert len(r.order_sha256) == len(r.draws_sha256) == 3
    assert r.initial_sha256 != r.after_step1_sha256


def test_the_statistics_are_the_training_excerpts_before_augmentation(excerpts):
    r = ctrain.train(IDENTITY, excerpts, epochs=1, batch=4)
    expected = cbatches.statistics(excerpts)
    assert np.array_equal(r.stats.mean, expected.mean) and np.array_equal(r.stats.scale, expected.scale)


def test_a_batch_carries_j_times_the_weights_and_padding_weighs_nothing(excerpts):
    stats = cbatches.statistics(excerpts)
    weights = cbatches.frame_weights(excerpts)
    rows = [cbatches.plain(excerpts[p].power, stats) for p in (3, 5)]  # 100 and 160 rows
    x, y, w = ctrain._batch(excerpts, [3, 5], rows, weights, j=2)
    assert x.shape == (2, 64, 160) and not x[0, :, 100:].any() and not w[0, 100:].any()
    assert math.isclose(float(w.double().sum()), 2 * float(weights[3].sum() + weights[5].sum()), rel_tol=1e-6)


def test_padding_after_an_excerpt_does_not_reach_its_frames(excerpts):
    r = ctrain.train(IDENTITY, excerpts, epochs=1, batch=4)
    e = excerpts[3]
    alone = ctrain.logits(r.model, e.power, r.stats)
    padded = np.concatenate([e.power, np.full((60, 64), 5.0)])
    assert np.allclose(ctrain.logits(r.model, padded, r.stats)[: e.rows], alone, rtol=0, atol=1e-5)


def test_one_epochs_batch_losses_reproduce_the_objective_without_augmentation(excerpts):
    """§6: with augmentation and dropout off and the parameters fixed, the batch losses of one
    epoch average to L."""
    r = ctrain.train(IDENTITY, excerpts, epochs=1, batch=4)
    model, stats = r.model.eval(), r.stats
    weights = cbatches.frame_weights(excerpts)
    j = -(-len(excerpts) // 4)
    order = list(range(len(excerpts)))
    losses = []
    with torch.no_grad():
        for begin in range(0, len(order), 4):
            positions = order[begin : begin + 4]
            rows = [cbatches.plain(excerpts[p].power, stats) for p in positions]
            x, y, w = ctrain._batch(excerpts, positions, rows, weights, j)
            losses.append(float(torch.nn.functional.binary_cross_entropy_with_logits(model(x).double(), y.double(),
                                                                                     weight=w.double(), reduction="sum")))
    assert np.mean(losses) == pytest.approx(ctrain.gate_loss(model, excerpts, stats), rel=1e-5)


def test_training_lowers_the_gate_loss_and_records_it_when_asked(excerpts):
    start = ctrain.gate_loss(ctrain.rng.build_for(IDENTITY), excerpts, cbatches.statistics(excerpts))
    r = ctrain.train(IDENTITY, excerpts, epochs=4, batch=4, evaluate_each_epoch=True)
    assert len(r.epoch_losses) == 4 and r.epoch_losses[-1] < start
    assert r.epoch_losses[-1] == ctrain.gate_loss(r.model, excerpts, r.stats)


def test_probabilities_are_float64_from_the_logit(excerpts):
    r = ctrain.train(IDENTITY, excerpts, epochs=1, batch=4)
    e = excerpts[0]
    p = ctrain.probabilities(r.model, e.power, r.stats)
    z = ctrain.logits(r.model, e.power, r.stats).astype(np.float64)
    assert p.dtype == np.float64 and np.array_equal(p, 1 / (1 + np.exp(-z)))


def test_the_trainer_leaves_global_random_state_alone(excerpts):
    before = torch.random.get_rng_state().clone()
    np_before = np.random.get_state()[1].copy()
    ctrain.train(IDENTITY, excerpts, epochs=1, batch=4)
    assert torch.equal(before, torch.random.get_rng_state())
    assert np.array_equal(np_before, np.random.get_state()[1])


def test_a_non_finite_loss_stops_the_run(excerpts):
    bad = list(excerpts)
    e = bad[0]
    power = e.power.copy()
    power[e.excerpt_row + 5, 3] = np.inf
    bad[0] = cbatches.Excerpt(e.key, e.group, e.voiced, power, e.labels, e.eligible, e.selected, e.excerpt_row)
    with np.errstate(all="ignore"), pytest.raises(ctrain.NonFinite):
        ctrain.train(IDENTITY, bad, epochs=1, batch=4)


def test_the_model_file_is_deterministic_and_created_new(tmp_path, excerpts):
    r = ctrain.train(IDENTITY, excerpts, epochs=1, batch=4)
    digest = ctrain.save(tmp_path / "m.c", r)
    raw = (tmp_path / "m.c").read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest() and stat.S_IMODE((tmp_path / "m.c").stat().st_mode) == 0o600
    with pytest.raises(FileExistsError):
        ctrain.save(tmp_path / "m.c", r)
    head = raw.partition(b"\n")[0]
    assert b"a5.2-c-model-v1" in head and IDENTITY.encode() in head


# -- model C's matched-start check (S4.3) -------------------------------------------------------------


def c_pair(excerpts, short=2, long=3, identity="gmv-a5.2-gate:C:outer-0:conv:0"):
    return ctrain.train(identity, excerpts, epochs=short, batch=4), ctrain.train(identity, excerpts, epochs=long, batch=4)


def test_a_genuine_c_pair_passes_and_parts_at_step_2(excerpts):
    from gmv.character.research.presence_model import matched

    a, b = c_pair(excerpts)
    matched.check_c(a, b)
    assert ctrain.to_bytes(a) != ctrain.to_bytes(b)


@pytest.mark.parametrize("field", ("initial_sha256", "order_sha256", "draws_sha256", "dropout_sha256",
                                   "after_step1_sha256"))
def test_a_perturbed_c_pair_fails(excerpts, field):
    from gmv.character.research.presence_model import matched

    a, b = c_pair(excerpts)
    value = getattr(b, field)
    if isinstance(value, list):
        value = list(value)
        value[0] = "0" * 64
    else:
        value = "0" * 64
    setattr(b, field, value)
    with pytest.raises(matched.MatchedStartError):
        matched.check_c(a, b)


def test_a_c_pair_with_another_seed_fails(excerpts):
    from gmv.character.research.presence_model import matched

    a, _ = c_pair(excerpts)
    _, b = c_pair(excerpts, identity="gmv-a5.2-gate:C:outer-0:conv:1")
    with pytest.raises(matched.MatchedStartError):
        matched.check_c(a, b)
