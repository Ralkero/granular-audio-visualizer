"""A5.2 Step 1, S1.1 (PyTorch part): SeededDropout, random-state isolation and replays.

Runs in the experiments environment (skipped elsewhere). Generated tensors only; the two
optimiser steps below train on random numbers to check determinism, not on project data.
"""

import io
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "presence"))

import cnn  # noqa: E402
import rng  # noqa: E402

from gmv.character.research.presence_model import seedplan  # noqa: E402

SOURCE = seedplan.REPLAYS["G-C3"]


@pytest.fixture(autouse=True)
def deterministic():
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(4)
    yield
    torch.use_deterministic_algorithms(False)


def test_seeded_dropout_is_inverted_dropout_from_its_own_generator():
    layer = cnn.SeededDropout(0.2).train()
    x = torch.ones(1000, 1000)
    with pytest.raises(RuntimeError):
        layer(x)
    layer.generator = torch.Generator().manual_seed(7)
    before = torch.get_rng_state()
    y = layer(x)
    assert torch.equal(torch.get_rng_state(), before)  # the global state is untouched
    values = set(torch.unique(y).tolist())
    assert values <= {0.0, 1.25}
    kept = float((y > 0).float().mean())
    assert abs(kept - 0.8) < 0.002
    layer.generator.manual_seed(7)
    assert torch.equal(layer(x), y)  # same generator state, same mask
    assert torch.equal(layer.eval()(x), x)  # identity in inference
    assert not list(layer.parameters()) and not list(layer.buffers())


def test_build_for_leaves_the_global_state_alone_and_is_deterministic():
    before = torch.get_rng_state()
    a = rng.build_for(SOURCE)
    b = rng.build_for(SOURCE)
    assert torch.equal(torch.get_rng_state(), before)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())
    other = rng.build_for("gmv-a5.2-gate:C:outer-0:conv:1")
    assert any(not torch.equal(a.state_dict()[k], other.state_dict()[k]) for k in a.state_dict())
    assert cnn.parameter_count(a) == cnn.PARAMETERS
    with pytest.raises(seedplan.SeedPlanError):
        rng.build_for("gmv-a5.2-gate:B1:outer-0:conv:0")


def two_steps(text):
    """Two optimiser steps on generated data, with the identity's order and dropout streams."""
    model = rng.build_for(text).train()
    order = rng.numpy_generator(text, "order")
    data = torch.randn(4, cnn.BANDS, 90, generator=torch.Generator().manual_seed(0))
    target = (torch.randn(4, 90, generator=torch.Generator().manual_seed(1)) > 0).float()
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(2):
        batch = torch.as_tensor(order.permutation(4)[:2])
        loss = torch.nn.functional.binary_cross_entropy_with_logits(model(data[batch]), target[batch])
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()
    buffer = io.BytesIO()
    torch.save(model.state_dict(), buffer)
    return buffer.getvalue()


def test_a_declared_replay_reproduces_its_source_bit_for_bit_and_keeps_global_state():
    plan = seedplan.plan("C")
    ledger = seedplan.RunLedger(plan)
    ledger.claim_production("G-C1", SOURCE)
    before = torch.get_rng_state()
    source = two_steps(SOURCE)
    ledger.claim_replay("G-C3", SOURCE)
    replay = two_steps(SOURCE)
    assert torch.equal(torch.get_rng_state(), before)
    seedplan.verify_replay(source, replay)  # RP1
    with pytest.raises(seedplan.ReplayError):  # RP5: a different identity is not a replay
        seedplan.verify_replay(source, two_steps("gmv-a5.2-gate:C:outer-0:conv:1"))
