"""Step 4, S4.2: model C's training loop, inference and gate loss (protocol §5, §6; Amendment 3,
A3.2 B15-B17, V7, V8 and A3.3). Needs PyTorch from the experiments environment. Built and tested
on generated data only. The inputs come from ``presence_model.cbatches``.

* **Seeds** (A3.3): the model is built by ``rng.build_for`` (initialisation from stream 0, dropout
  from stream 3); the epoch order is the ``order`` stream (1), and the four augmentation streams
  (2, 0…3) are NumPy ``PCG64`` generators.
* **Normalisation** (B15): ``cbatches.statistics`` over the model's own training excerpts,
  before augmentation, computed once before training.
* **Batches** (§6): each epoch visits every training excerpt once in a seeded order, in batches
  of 8 whole excerpts (context rows at weight 0). Excerpts shorter than the longest in their batch
  are padded at the end with zero input and weight 0; the network is causal, so padding after an
  excerpt never reaches its frames. With J batches per epoch the batch loss is J · Σ w·ℓ, not
  normalised per batch.
* **Optimiser** (B17): AdamW, β = (0.9, 0.999), ε = 1e-8, no AMSGrad, weight decay 1e-4 on all
  parameters, no warm-up and no clipping. The learning rate of optimiser step k (0-based) of the
  model's K steps is 1e-5 + ½(1e-3 − 1e-5)(1 + cos(π·k/K)), set in closed form before each step.
  *Engineering choice:* the convention of PyTorch's ``CosineAnnealingLR`` with T_max = K (step 1
  uses 1e-3; the rate reaches 1e-5 as the schedule ends), without its recursive update.
* **Gate loss** L(H) (A3.2): the §6 objective on the training excerpts in inference mode, without
  augmentation, accumulated in float64 (``gate_loss``).
* **Inference** (A3.5 B20): logits in float32, the voice probability in float64 from the logit.
* **Records** for the matched-start check (A3.3): initial parameters, parameters after optimiser
  step 1, each epoch's order and augmentation-draw digests, and each step's dropout-mask digest.
* **Stops** (A3.2, S1): any non-finite loss, parameter or gradient raises ``NonFinite``.
* The job entry point calls ``configure`` (deterministic algorithms; 4 intra-op and 1 inter-op
  threads, A3.3). ``train`` itself does not change global settings.
"""

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import cnn
import numpy as np
import rng
import torch
from torch.nn import functional

from gmv.character.research.presence_model import cbatches

EPOCHS = 30
BATCH = 8
RATE, FINAL_RATE = 1e-3, 1e-5
BETAS, EPSILON, DECAY = (0.9, 0.999), 1e-8, 1e-4
FORMAT = "a5.2-c-model-v1"


class NonFinite(RuntimeError):
    """S1: a non-finite loss, parameter or gradient; the run stops for the owner's review."""


def configure(threads=4):
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:  # already set in this process
        if torch.get_num_interop_threads() != 1:
            raise


def rate_at(step, total):
    """The learning rate of 0-based optimiser step ``step`` of ``total``."""
    return FINAL_RATE + 0.5 * (RATE - FINAL_RATE) * (1 + math.cos(math.pi * step / total))


def _generators(identity):
    return {s: rng.numpy_generator(identity, f"augment-{s}") for s in cbatches.STREAMS}


def _batch(excerpts, positions, rows, weights, j):
    longest = max(excerpts[p].rows for p in positions)
    x = torch.zeros(len(positions), cnn.BANDS, longest)
    y = torch.zeros(len(positions), longest)
    w = torch.zeros(len(positions), longest)
    for b, p in enumerate(positions):
        n = excerpts[p].rows
        x[b, :, :n] = torch.from_numpy(np.ascontiguousarray(rows[b].T))
        y[b, :n] = torch.from_numpy(np.asarray(excerpts[p].labels, dtype=np.float32))
        w[b, :n] = torch.from_numpy((j * weights[p]).astype(np.float32))
    return x, y, w


def _sha_state(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode())
        digest.update(value.detach().contiguous().numpy().tobytes())
    return digest.hexdigest()


@dataclass
class Result:
    identity: str
    epochs: int
    model: object
    stats: object
    steps: int
    order_sha256: list = field(default_factory=list)
    draws_sha256: list = field(default_factory=list)
    dropout_sha256: list = field(default_factory=list)
    epoch_losses: list = field(default_factory=list)
    initial_sha256: str = ""
    after_step1_sha256: str = ""


def train(identity, excerpts, epochs=EPOCHS, batch=BATCH, evaluate_each_epoch=False, marks=None):
    """Train one model C on ``excerpts`` (``cbatches.Excerpt``, the model's training set only)."""
    if epochs < 1 or batch < 1 or not excerpts:
        raise ValueError("a recipe needs at least one epoch, one excerpt and one excerpt per batch")
    excerpts = list(excerpts)
    stats = cbatches.statistics(excerpts)
    weights = cbatches.frame_weights(excerpts)
    partner_map = cbatches.partners(excerpts)
    model = rng.build_for(identity)
    model.dropout.record = []
    order_rng = rng.numpy_generator(identity, "order")
    generators = _generators(identity)
    j = -(-len(excerpts) // batch)
    total = epochs * j
    optimiser = torch.optim.AdamW(model.parameters(), lr=RATE, betas=BETAS, eps=EPSILON, weight_decay=DECAY,
                                  amsgrad=False)
    result = Result(identity, epochs, model, stats, 0, initial_sha256=_sha_state(model))
    step = 0
    for _epoch in range(epochs):
        if marks is not None:
            marks.begin("epoch")
        model.train()
        order = order_rng.permutation(len(excerpts))
        draws = cbatches.epoch_draws(generators, order)
        result.order_sha256.append(hashlib.sha256(order.astype("<i8").tobytes()).hexdigest())
        result.draws_sha256.append(cbatches.draws_sha256(draws))
        for begin in range(0, len(order), batch):
            positions = [int(p) for p in order[begin : begin + batch]]
            rows = [cbatches.inputs(excerpts, p, draws[begin + b], partner_map, stats) for b, p in enumerate(positions)]
            x, y, w = _batch(excerpts, positions, rows, weights, j)
            for group in optimiser.param_groups:
                group["lr"] = rate_at(step, total)
            optimiser.zero_grad(set_to_none=True)
            loss = functional.binary_cross_entropy_with_logits(model(x), y, weight=w, reduction="sum")
            if not torch.isfinite(loss):
                raise NonFinite("a batch loss is not finite")
            loss.backward()
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    raise NonFinite("a gradient is not finite")
            optimiser.step()
            step += 1
            for p in model.parameters():
                if not torch.isfinite(p).all():
                    raise NonFinite("a parameter is not finite")
            if step == 1:
                result.after_step1_sha256 = _sha_state(model)
        if marks is not None:
            marks.end("epoch")
        if evaluate_each_epoch:
            if marks is not None:
                marks.begin("gate_eval")
            result.epoch_losses.append(gate_loss(model, excerpts, stats, weights))
            if marks is not None:
                marks.end("gate_eval")
    result.steps = step
    result.dropout_sha256 = list(model.dropout.record)
    model.dropout.record = None
    model.eval()
    return result


@torch.no_grad()
def logits(model, power, stats):
    """Inference-mode logits (float32) for one excerpt's rows of linear mel power."""
    was = model.training
    model.eval()
    x = torch.from_numpy(np.ascontiguousarray(cbatches.plain(power, stats).T))[None]
    out = model(x)[0].numpy()
    model.train(was)
    return out


def probabilities(model, power, stats):
    """The voice probability per frame, in float64 from the float32 logit."""
    z = logits(model, power, stats).astype(np.float64)
    return 1 / (1 + np.exp(-z))


def gate_loss(model, excerpts, stats, weights=None):
    """L(H): Σ w·ℓ over the training excerpts, in inference mode, without augmentation, in float64."""
    from gmv.character.research.presence_model import weights as objective

    weights = cbatches.frame_weights(excerpts) if weights is None else weights
    total = 0.0
    for e, w in zip(excerpts, weights, strict=True):
        z = logits(model, e.power, stats)
        total += float(np.dot(w, objective.bce_with_logits(z, e.labels)))
    if not math.isfinite(total):
        raise NonFinite("a gate loss is not finite")
    return total


def to_bytes(result):
    """A deterministic model file: one canonical JSON header line, then the float32 tensors and
    the float64 statistics."""
    state = result.model.state_dict()
    header = {"format": FORMAT, "identity": result.identity, "epochs": result.epochs, "steps": result.steps,
              "tensors": [[k, list(v.shape)] for k, v in sorted(state.items())],
              "statistics": {"frames": result.stats.frames, "bands": len(result.stats.mean)}}
    body = b"".join(state[k].detach().contiguous().to(torch.float32).numpy().astype("<f4").tobytes()
                    for k in sorted(state))
    body += np.ascontiguousarray(result.stats.mean, dtype="<f8").tobytes()
    body += np.ascontiguousarray(result.stats.scale, dtype="<f8").tobytes()
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n" + body


def save(path, result):
    data = to_bytes(result)
    fd = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
    return hashlib.sha256(data).hexdigest()
