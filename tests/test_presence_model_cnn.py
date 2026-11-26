"""A5.2 Stage 1, protocol §16 "Causality and normalisation": model C (forward pass only).

Needs PyTorch, so it runs in the separate experiments environment and is skipped elsewhere.
Inputs are generated tensors. No weights are trained or loaded.

What is compared, precisely: the registered network ends with dropout 0.2 before its output
layer. Dropout is meant to differ between training and inference, so whole-network outputs
are only compared across modes up to the representation *before* dropout
(``PresenceCNN.features``); FrameNorm is compared across modes on its own; and
training-mode causality is checked with the random generator reset before each forward
pass, so both runs draw the same dropout mask.
"""

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "presence"))

import cnn  # noqa: E402

from gmv.character.research.presence_model import seeds  # noqa: E402

FRAMES = 160


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def mel(batch=2, frames=FRAMES, seed=0, dtype=torch.float32):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(batch, cnn.BANDS, frames, generator=generator, dtype=dtype)


def changed_after(x, t, seed=99):
    y = x.clone()
    generator = torch.Generator().manual_seed(seed)
    y[..., t + 1 :] = torch.randn(y[..., t + 1 :].shape, generator=generator, dtype=x.dtype)
    return y


# -- size and receptive field -----------------------------------------------------------------------


def test_the_registered_architecture_has_154641_parameters():
    model = cnn.build(1)
    assert cnn.parameter_count(model) == cnn.PARAMETERS == 154_641
    assert sum(c.weight.numel() for c in model.convs) == 71_568
    assert sum(n.weight.numel() + n.bias.numel() for n in model.norms) == 448
    assert model.project.weight.numel() + 128 == 32_896
    assert sum(c.weight.numel() for c in model.temporal) + 4 * 128 == 49_664
    assert model.out.weight.numel() + model.out.bias.numel() == 65
    assert all(c.bias is None for c in [*model.convs, model.project, *model.temporal]) and model.out.bias is not None
    assert model.dropout.p == 0.2 and [c.dilation[0] for c in model.temporal] == [2, 4, 8, 16]


def test_the_receptive_field_is_exactly_73_frames():
    model = cnn.build(2).eval()
    x = mel(1, 220)
    t0 = 60
    y = x.clone()
    y[..., t0] += 3.0
    with torch.no_grad():
        a, b = model(x)[0], model(y)[0]
    assert torch.equal(a[:t0], b[:t0])  # nothing before the change moves
    assert not torch.equal(a[t0 + 72], b[t0 + 72])  # 72 frames later still sees it
    assert torch.equal(a[t0 + 73 :], b[t0 + 73 :])  # 73 frames later no longer does
    assert cnn.RECEPTIVE_FIELD == 73


# -- causality --------------------------------------------------------------------------------------


@pytest.mark.parametrize("t", (0, 5, 72, 100, FRAMES - 2))
def test_changing_later_input_leaves_earlier_outputs_bit_identical_in_inference(t):
    model = cnn.build(3).eval()
    x = mel()
    with torch.no_grad():
        assert torch.equal(model(x)[:, : t + 1], model(changed_after(x, t))[:, : t + 1])


@pytest.mark.parametrize("t", (10, 90))
def test_changing_later_input_leaves_earlier_outputs_bit_identical_in_training(t):
    model = cnn.build(4).train()
    x = mel()
    torch.manual_seed(1234)  # the same dropout mask in both runs
    a = model(x)
    torch.manual_seed(1234)
    b = model(changed_after(x, t))
    assert torch.equal(a[:, : t + 1], b[:, : t + 1])
    assert torch.equal(model.features(x)[..., : t + 1], model.features(changed_after(x, t))[..., : t + 1])


class SymmetricPadding(cnn.PresenceCNN):
    """A deliberately wrong variant: time padding (1, 1) lets frame t see frame t+1."""

    def features(self, mel):
        x = mel.unsqueeze(1)
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms, strict=True)):
            x = torch.relu(norm(conv(torch.nn.functional.pad(x, (1, 1, 1, 1)))))
            if i in cnn.POOL_AFTER:
                x = self.pool(x)
        h = torch.relu(self.project_norm(self.project(x.flatten(1, 2))))
        for conv, norm, d in zip(self.temporal, self.temporal_norms, cnn.DILATIONS, strict=True):
            h = h + torch.relu(norm(conv(torch.nn.functional.pad(h, (2 * d, 0)))))
        return h


class TimeBatchNorm(cnn.PresenceCNN):
    """A deliberately wrong variant: BatchNorm statistics span the batch and every frame."""

    def __init__(self):
        super().__init__()
        self.norms = torch.nn.ModuleList(torch.nn.BatchNorm2d(c) for c in cnn.CHANNELS)


def test_the_causality_test_catches_future_leakage_from_padding():
    torch.manual_seed(5)
    leaky = SymmetricPadding().eval()
    x = mel()
    with torch.no_grad():
        assert not torch.equal(leaky(x)[:, :51], leaky(changed_after(x, 50))[:, :51])


def test_the_causality_test_catches_future_leakage_from_batchnorm_in_training():
    torch.manual_seed(6)
    leaky = TimeBatchNorm().train()
    x = mel()
    assert not torch.equal(leaky.features(x)[..., :51], leaky.features(changed_after(x, 50))[..., :51])


# -- FrameNorm and modes ---------------------------------------------------------------------------


def test_framenorm_is_identical_in_training_and_inference_and_has_no_running_statistics():
    torch.manual_seed(7)
    norm2d, norm1d = cnn.FrameNorm2d(16), cnn.FrameNorm1d(64)
    with torch.no_grad():
        norm2d.weight.uniform_(0.5, 2.0)
        norm2d.bias.uniform_(-1, 1)
    x2, x1 = torch.randn(3, 16, 32, 50), torch.randn(3, 64, 50)
    assert torch.equal(norm2d.train()(x2), norm2d.eval()(x2))
    assert torch.equal(norm1d.train()(x1), norm1d.eval()(x1))
    assert not list(norm2d.buffers()) and not list(norm1d.buffers())
    # each frame is normalised by its own statistics: per (item, frame), zero mean and unit variance before scale/shift
    plain = cnn.FrameNorm2d(16)(x2)
    assert torch.allclose(plain.mean(dim=(1, 2)), torch.zeros(3, 50), atol=1e-5)
    assert torch.allclose(plain.var(dim=(1, 2), unbiased=False), torch.ones(3, 50), atol=1e-3)


def test_the_representation_before_dropout_is_identical_in_both_modes():
    model = cnn.build(8)
    x = mel()
    assert torch.equal(model.train().features(x), model.eval().features(x))
    # the registered dropout makes training-mode outputs differ, as intended; it is not removed
    torch.manual_seed(0)
    assert not torch.equal(model.train()(x), model.eval()(x))


# -- batch composition ------------------------------------------------------------------------------


def test_an_items_output_does_not_depend_on_its_companions():
    model = cnn.build(9).eval()
    x, others = mel(3, seed=1), mel(3, seed=2)
    mixed = torch.cat((x[:1], others[1:]))
    with torch.no_grad():
        assert torch.equal(model(x)[0], model(mixed)[0])  # same batch size: bit-identical
    model.train()
    assert torch.equal(model.features(x)[0], model.features(mixed)[0])


def test_batch_size_changes_only_rounding_not_information():
    """A different batch size can select a different convolution kernel, so float32 results
    may differ in the last bits (about 1e-6); float64 shows they agree to rounding."""
    for dtype, tolerance in ((torch.float32, 1e-5), (torch.float64, 1e-12)):
        model = cnn.build(10, dtype).eval()
        x = mel(4, dtype=dtype)
        with torch.no_grad():
            assert torch.allclose(model(x[:1])[0], model(x)[0], atol=tolerance, rtol=0)


def test_the_batch_test_catches_batchnorm_mixing_items():
    torch.manual_seed(11)
    leaky = TimeBatchNorm().train()
    x, others = mel(3, seed=1), mel(3, seed=2)
    mixed = torch.cat((x[:1], others[1:]))
    assert not torch.allclose(leaky.features(x)[0], leaky.features(mixed)[0])


# -- streaming ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(("dtype", "tolerance"), ((torch.float64, 1e-10), (torch.float32, 1e-4)))
def test_streaming_agrees_with_batch_inference(dtype, tolerance):
    model = cnn.build(12, dtype).eval()
    x = mel(1, 140, dtype=dtype)
    streamer = cnn.Streamer(model)
    streamed = torch.tensor([streamer.step(x[0, :, t]) for t in range(140)], dtype=dtype)
    with torch.no_grad():
        batched = model(x)[0]
    assert torch.allclose(streamed, batched, atol=tolerance, rtol=0)


def test_streaming_state_starts_empty_and_is_not_shared_between_streams():
    model = cnn.build(13, torch.float64).eval()
    x = mel(1, 30, dtype=torch.float64)
    one, two = cnn.Streamer(model), cnn.Streamer(model)
    first = [one.step(x[0, :, t]) for t in range(30)]
    assert [two.step(x[0, :, t]) for t in range(30)] == first


# -- seeds -------------------------------------------------------------------------------------------


def test_the_same_registered_seed_gives_identical_weights():
    seed = seeds.training_seed("C", "group_001", 0, 0)
    a, b = cnn.build(seed).state_dict(), cnn.build(seed).state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    c = cnn.build(seeds.training_seed("C", "group_001", 0, 1)).state_dict()
    assert any(not torch.equal(a[k], c[k]) for k in a)


def test_invalid_input_shapes_are_refused():
    model = cnn.build(14)
    with pytest.raises(ValueError):
        model(torch.randn(1, 63, 20))
    with pytest.raises(ValueError):
        model(torch.randn(64, 20))
