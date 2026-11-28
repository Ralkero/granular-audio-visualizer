"""A5.2 Step 4, S4.1: B1's trainer, inference and gate loss (protocol §3, §6; Amendment 3,
A3.2 and A3.3), on generated frames and generated stand-ins for B0's constants only."""

import hashlib
import stat

import numpy as np
import pytest

from gmv.character.research.presence_model import b1train, seedplan, weights

IDENTITY = "gmv-a5.2:B1:0:outer:0"
OTHER = "gmv-a5.2:B1:0:outer:1"


def constants(seed=3):
    rng = np.random.default_rng(seed)
    return b1train.Constants(rng.normal(0, 2, 548).astype(np.float32), rng.uniform(0.5, 3, 548).astype(np.float32),
                             hashlib.sha256(b"generated stand-in for B0").hexdigest())


def world(seed=1, scale=1.0):
    """Generated training excerpts: four groups, voice frames shifted in a few features."""
    rng = np.random.default_rng(seed)
    units, feats = [], []
    for g, (voiced, novoice) in enumerate(((300, 200), (0, 400), (250, 0), (120, 180))):
        n = voiced + novoice + 40
        y = np.array([1] * voiced + [0] * (novoice + 40), dtype=np.int8)
        eligible = np.array([True] * (voiced + novoice) + [False] * 40)
        order = rng.permutation(n)
        units.append(weights.Unit(f"group_{g}", y[order], eligible[order]))
        x = rng.normal(0, 1, (n, 548)) * scale
        x[:, :8] += 1.5 * y[order][:, None]
        feats.append(x.astype(np.float32))
    return units, feats


def data(seed=1):
    units, feats = world(seed)
    return b1train.frames(units, feats, constants())


def test_the_registered_size_is_70401_parameters():
    params = b1train.initialise(IDENTITY)
    assert b1train.PARAMETERS == 70_401 == b1train.parameter_count(params)
    assert {k: v.dtype for k, v in params.items()} == {k: np.float32 for k in b1train.NAMES}
    assert not params["b1"].any() and not params["b2"].any()


def test_initialisation_follows_b0s_scheme_from_the_init_stream():
    params = b1train.initialise(IDENTITY)
    assert abs(params["w1"].std() - np.sqrt(2 / 548)) < 0.002 and abs(params["w2"].std() - np.sqrt(1 / 128)) < 0.02
    again = b1train.initialise(IDENTITY)
    assert all(np.array_equal(params[k], again[k]) for k in b1train.NAMES)
    assert not np.array_equal(params["w1"], b1train.initialise(OTHER)["w1"])
    with pytest.raises(seedplan.SeedPlanError):
        b1train.initialise("gmv-a5.2:C:0:outer:0")


def test_only_frames_with_weight_above_zero_are_trained_and_their_weights_are_the_objective():
    units, feats = world()
    d = b1train.frames(units, feats, constants())
    total = sum(len(u.labels) for u in units)
    assert d.count < total  # the 40 ineligible frames per unit are dropped
    assert (d.w > 0).all()
    assert abs(d.w.sum() - 1.0) < 1e-12  # ½ per class, every member group 1/(2|Aᶜ|)


def test_the_gate_loss_equals_the_protocol_objective_on_the_training_frames():
    units, feats = world()
    c = constants()
    d = b1train.frames(units, feats, c)
    params = b1train.initialise(IDENTITY)
    per_unit = [b1train.logits(params, b1train.standardise(f, c)) for f in feats]
    expected = weights.registered_loss(units, per_unit)
    assert b1train.gate_loss(params, d) == pytest.approx(expected, rel=1e-6)  # float32 logits, float64 sum


def test_the_gradient_matches_finite_differences_of_the_batch_loss_plus_l2():
    rng = np.random.default_rng(5)
    params = {k: v.astype(np.float64) for k, v in b1train.initialise(IDENTITY).items()}
    x = rng.normal(0, 1, (30, 548))
    y = (rng.random(30) < 0.5).astype(np.float64)
    w = rng.uniform(0.01, 0.1, 30)

    def loss(p):
        h = np.maximum(x @ p["w1"] + p["b1"], 0)
        z = (h @ p["w2"] + p["b2"])[:, 0]
        return float(np.dot(w, weights.bce_with_logits(z, y))) + 0.5 * b1train.DECAY * (
            float((p["w1"] ** 2).sum()) + float((p["w2"] ** 2).sum()))

    g = b1train.gradient(params, x, y, w)
    for name, index in (("w1", (3, 7)), ("w1", (100, 0)), ("b1", (5,)), ("w2", (9, 0)), ("b2", (0,))):
        bumped = {k: v.copy() for k, v in params.items()}
        bumped[name][index] += 1e-6
        lowered = {k: v.copy() for k, v in params.items()}
        lowered[name][index] -= 1e-6
        numeric = (loss(bumped) - loss(lowered)) / 2e-6
        assert g[name][index] == pytest.approx(numeric, rel=1e-4, abs=1e-9)


def test_l2_applies_to_the_weight_matrices_only():
    params = b1train.initialise(IDENTITY)
    g = b1train.gradient(params, np.zeros((4, 548), np.float32), np.zeros(4, np.float32), np.zeros(4, np.float32))
    assert np.allclose(g["w1"], b1train.DECAY * params["w1"]) and np.allclose(g["w2"], b1train.DECAY * params["w2"])
    assert not g["b1"].any() and not g["b2"].any()


def test_the_learning_rate_drops_for_the_second_half():
    assert [b1train.rate_at(e, 60) for e in (0, 29, 30, 59)] == [1e-3, 1e-3, 3e-4, 3e-4]
    assert b1train.first_half(90) == 45 and b1train.first_half(135) == 67


def reference_train(identity, d, epochs, batch):
    """B0's script's loop (scripts/train_activity.py, ``train``), restated for one output and
    the §6 weights, with B1's schedule."""
    params = b1train.initialise(identity)
    params = [params[k] for k in b1train.NAMES]
    first = [np.zeros_like(p) for p in params]
    second = [np.zeros_like(p) for p in params]
    rng = b1train._stream(identity, "order")
    batches = -(-d.count // batch)
    step, rate = 0, 1e-3
    for epoch in range(epochs):
        order = rng.permutation(d.count)
        for begin in range(0, len(order), batch):
            index = order[begin : begin + batch]
            xb, target = d.x[index], d.y[index][:, None]
            weight = (batches * d.w[index]).astype(np.float32)[:, None]
            hidden_out = np.maximum(xb @ params[0] + params[1], 0)
            scores = 1 / (1 + np.exp(-(hidden_out @ params[2] + params[3])))
            error = (scores - target) * weight
            back = (error @ params[2].T) * (hidden_out > 0)
            gradients = [xb.T @ back + 1e-2 * params[0], back.sum(0), hidden_out.T @ error + 1e-2 * params[2],
                         error.sum(0)]
            step += 1
            for k, gradient in enumerate(gradients):
                first[k] = 0.9 * first[k] + 0.1 * gradient
                second[k] = 0.999 * second[k] + 0.001 * gradient**2
                momentum = first[k] / (1 - 0.9**step)
                update = momentum / (np.sqrt(second[k] / (1 - 0.999**step)) + 1e-8)
                params[k] -= (rate * update).astype(np.float32)
        if epoch + 1 == epochs // 2:
            rate = 3e-4
    return dict(zip(b1train.NAMES, params, strict=True))


def test_training_is_b0s_hand_written_adam_bit_for_bit():
    d = data()
    result = b1train.train(IDENTITY, d, epochs=4, batch=256)
    expected = reference_train(IDENTITY, d, 4, 256)
    assert all(np.array_equal(result.params[k], expected[k]) for k in b1train.NAMES)
    assert result.steps == 4 * -(-d.count // 256)


def test_the_same_seed_gives_identical_weights_and_another_seed_does_not():
    d = data()
    a = b1train.train(IDENTITY, d, epochs=3, batch=128)
    b = b1train.train(IDENTITY, d, epochs=3, batch=128)
    c = b1train.train(OTHER, d, epochs=3, batch=128)
    assert b1train.to_bytes(a, constants()) == b1train.to_bytes(b, constants())
    assert a.order_sha256 == b.order_sha256 and a.order_sha256 != c.order_sha256
    assert not np.array_equal(a.params["w1"], c.params["w1"])


def test_training_lowers_the_loss_and_records_each_epoch_when_asked():
    d = data()
    before = b1train.gate_loss(b1train.initialise(IDENTITY), d)
    result = b1train.train(IDENTITY, d, epochs=6, batch=128, evaluate_each_epoch=True)
    assert len(result.epoch_losses) == 6 and result.epoch_losses[-1] < before
    assert result.epoch_losses[-1] == b1train.gate_loss(result.params, d)
    assert b1train.train(IDENTITY, d, epochs=6, batch=128).epoch_losses == []


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
def test_a_non_finite_gradient_stops_the_run():
    units, feats = world()
    feats[0][0, 0] = np.inf
    d = b1train.frames(units, feats, constants())
    with pytest.raises(b1train.NonFinite):
        b1train.train(IDENTITY, d, epochs=1, batch=64)


def test_inference_is_the_engines_activity_model_formula(tmp_path):
    from gmv.activity import FEATURE_VERSION, ActivityModel

    d = data()
    c = constants()
    result = b1train.train(IDENTITY, d, epochs=2, batch=256)
    path = tmp_path / "b1.npz"
    np.savez(path, feature_version=FEATURE_VERSION, name="b1", labels=np.array(["voice"]), mean=c.mean, scale=c.scale,
             on=np.array([0.6]), off=np.array([0.4]), **result.params)
    features = np.random.default_rng(8).normal(0, 2, (50, 548))
    engine = ActivityModel(path).scores(features)[:, 0]
    assert np.array_equal(b1train.scores(result.params, features, c), engine)


def test_the_model_file_is_deterministic_round_trips_and_is_created_new(tmp_path):
    d = data()
    c = constants()
    result = b1train.train(IDENTITY, d, epochs=2, batch=256)
    digest = b1train.save(tmp_path / "m.b1", result, c)
    raw = (tmp_path / "m.b1").read_bytes()
    assert digest == hashlib.sha256(raw).hexdigest() == hashlib.sha256(b1train.to_bytes(result, c)).hexdigest()
    assert stat.S_IMODE((tmp_path / "m.b1").stat().st_mode) == 0o600
    header, params = b1train.from_bytes(raw)
    assert header["identity"] == IDENTITY and header["b0_sha256"] == c.b0_sha256
    assert all(np.array_equal(params[k], result.params[k]) for k in b1train.NAMES)
    with pytest.raises(FileExistsError):
        b1train.save(tmp_path / "m.b1", result, c)
    with pytest.raises(ValueError):
        b1train.from_bytes(raw[:-4])


@pytest.mark.parametrize("bad", ("shape", "zero", "nan", "hash"))
def test_malformed_b0_constants_are_refused(bad):
    c = constants()
    mean, scale, sha = c.mean, c.scale, c.b0_sha256
    if bad == "shape":
        mean = mean[:10]
    elif bad == "zero":
        scale = scale.copy()
        scale[3] = 0
    elif bad == "nan":
        mean = mean.copy()
        mean[0] = np.nan
    else:
        sha = "x" * 64
    with pytest.raises(ValueError):
        b1train.Constants(mean, scale, sha)
