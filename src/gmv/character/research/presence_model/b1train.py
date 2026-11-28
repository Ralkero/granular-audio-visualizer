"""Step 4, S4.1: B1's trainer, its inference and its gate loss (protocol §3, §6; Amendment 3,
A3.2 and A3.3). NumPy only. Built and tested on generated data only.

**The model.** The 548 activity features, standardised with B0's stored ``mean`` and ``scale``
(B14: read only, the B0 file's SHA-256 recorded; the weights are trained from scratch) →
128 ReLU → 1 sigmoid, "voice present": 548·128 + 128 + 128 + 1 = **70,401** parameters.

**Training** follows B0's training script (``scripts/train_activity.py``) where the protocol
says "the existing hand-written Adam" and "the existing scheme", and the protocol where they
differ:

* initialisation: ``w1 ~ N(0, 2/548)``, ``w2 ~ N(0, 1/128)``, biases 0, all float32, drawn from
  the seed identity's ``init`` stream (k = 0);
* each epoch is a permutation, from the ``order`` stream (k = 1), of every frame with weight
  > 0, in batches of 1024; with J batches per epoch the batch loss is J · Σ w·ℓ (§6), so the
  gradient at a logit is J·w·(p − y);
* Adam with β = (0.9, 0.999), ε = 1e-8 and bias correction, written out as in B0's script;
  L2 1e-2 added to the gradients of the two weight matrices only (biases are not decayed);
* learning rate 1e-3, then 3e-4 for the second half (§6): the drop happens after
  ``first_half(E) = E // 2`` epochs (30 of 60, 45 of 90, 67 of 135). B0's own script dropped
  three epochs before the end; B1 follows the protocol. *Engineering choice:* for the odd
  E⁺⁺ = 135 the first half is 67 epochs;
* the float32 arithmetic of B0's script (V8): features, parameters, moments and updates are
  float32.

**Inference** is the engine's activity-model formula (``gmv.activity.ActivityModel.scores``):
``max((features − mean)/scale @ w1 + b1, 0)``, then the logistic, with B0's constants as
stored.

**The gate loss** L(H) (A3.2) is the §6 objective on the run's own training frames after the
last epoch, in inference mode, accumulated in float64 (``gate_loss``). It is the gate-only
end-of-epoch evaluation that A3.1 excludes from process-time; ``train`` computes it after each
epoch only when asked.

**Stops** (A3.2, S1): any non-finite loss, parameter or gradient raises ``NonFinite``.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import seedplan, weights

FEATURES = 548
HIDDEN = 128
PARAMETERS = FEATURES * HIDDEN + HIDDEN + HIDDEN + 1  # 70,401
EPOCHS = 60
BATCH = 1024
RATE, LATE_RATE = 1e-3, 3e-4
DECAY = 1e-2
BETAS, EPSILON = (0.9, 0.999), 1e-8  # written out literally in train(), as in B0's script
FORMAT = "a5.2-b1-model-v1"
NAMES = ("w1", "b1", "w2", "b2")


class NonFinite(RuntimeError):
    """S1: a non-finite loss, parameter or gradient; the run stops for the owner's review."""


def first_half(epochs):
    return epochs // 2


def rate_at(epoch, epochs):
    """The learning rate of zero-based ``epoch`` in a recipe of ``epochs`` epochs."""
    return RATE if epoch < first_half(epochs) else LATE_RATE


@dataclass(frozen=True)
class Constants:
    """B0's stored normalisation, as stored (dtype kept), and the B0 file's SHA-256."""

    mean: np.ndarray
    scale: np.ndarray
    b0_sha256: str

    def __post_init__(self):
        mean, scale = np.asarray(self.mean), np.asarray(self.scale)
        if mean.shape != (FEATURES,) or scale.shape != (FEATURES,):
            raise ValueError("B0's normalisation constants have the wrong shape")
        if not (np.isfinite(mean).all() and np.isfinite(scale).all()) or (scale <= 0).any():
            raise ValueError("B0's normalisation constants must be finite with positive spreads")
        if len(self.b0_sha256) != 64 or any(c not in "0123456789abcdef" for c in self.b0_sha256):
            raise ValueError("the B0 file's SHA-256 is malformed")


def standardise(features, constants):
    """Training input: B0's script's float32 path, (x − mean) / scale in float32."""
    x = np.asarray(features, dtype=np.float32)
    return (x - np.asarray(constants.mean, dtype=np.float32)) / np.asarray(constants.scale, dtype=np.float32)


@dataclass(frozen=True)
class Frames:
    """The training frames with weight > 0: standardised features, labels and weights."""

    x: np.ndarray  # (n, 548) float32
    y: np.ndarray  # (n,) float32, 0 or 1
    w: np.ndarray  # (n,) float64

    @property
    def count(self):
        return len(self.y)


def frames(units, features, constants, min_frames=weights.MIN_FRAMES):
    """``units``: ``weights.Unit`` per training excerpt (its frames on the canonical grid with
    the training mask as eligibility); ``features``: the matching (frames, 548) arrays. Only
    frames with weight > 0 are kept (§6: "a seeded permutation of all frames with weight > 0")."""
    units, features = list(units), list(features)
    if len(units) != len(features):
        raise ValueError("one feature array per training unit")
    w = weights.frame_weights(units, min_frames)
    xs, ys, ws = [], [], []
    for unit, f, wu in zip(units, features, w, strict=True):
        f = np.asarray(f)
        if f.shape != (len(unit.labels), FEATURES):
            raise ValueError("a feature array does not match its unit's frames")
        keep = wu > 0
        xs.append(standardise(f[keep], constants))
        ys.append(np.asarray(unit.labels)[keep].astype(np.float32))
        ws.append(wu[keep])
    return Frames(np.concatenate(xs), np.concatenate(ys), np.concatenate(ws))


def initialise(identity):
    """Fresh parameters for a B1 seed identity, by B0's scheme, from its ``init`` stream."""
    rng = _stream(identity, "init")
    return {
        "w1": (rng.standard_normal((FEATURES, HIDDEN)) * np.sqrt(2 / FEATURES)).astype(np.float32),
        "b1": np.zeros(HIDDEN, np.float32),
        "w2": (rng.standard_normal((HIDDEN, 1)) * np.sqrt(1 / HIDDEN)).astype(np.float32),
        "b2": np.zeros(1, np.float32),
    }


def _stream(identity, consumer):
    if seedplan.arm_of(identity) != "B1":
        raise seedplan.SeedPlanError("B1 needs a B1 seed identity")
    return seedplan.numpy_generator(seedplan.integer_of(identity), seedplan.consumer_key("B1", consumer))


def logits(params, x):
    """Training-path logits (float32)."""
    hidden = np.maximum(x @ params["w1"] + params["b1"], 0)
    return (hidden @ params["w2"] + params["b2"])[:, 0]


def gate_loss(params, data):
    """L(H): Σ w·ℓ over the training frames, ℓ and the sum in float64."""
    z = logits(params, data.x)
    loss = float(np.dot(data.w, weights.bce_with_logits(z, data.y)))
    if not np.isfinite(loss):
        raise NonFinite("a gate loss is not finite")
    return loss


def scores(params, features, constants):
    """Inference: the engine's activity-model formula with B0's constants as stored."""
    hidden = np.maximum((np.asarray(features) - constants.mean) / constants.scale @ params["w1"] + params["b1"], 0)
    return (1 / (1 + np.exp(-(hidden @ params["w2"] + params["b2"]))))[:, 0]


def gradient(params, x, y, w):
    """The gradient of Σ w·ℓ (``w`` already multiplied by J) plus the L2 term ½·1e-2·‖W‖² on
    the two weight matrices, by B0's script's formulas."""
    hidden = np.maximum(x @ params["w1"] + params["b1"], 0)
    p = 1 / (1 + np.exp(-(hidden @ params["w2"] + params["b2"])))
    error = ((p[:, 0] - y) * w)[:, None]
    back = (error @ params["w2"].T) * (hidden > 0)
    return {"w1": x.T @ back + DECAY * params["w1"], "b1": back.sum(0),
            "w2": hidden.T @ error + DECAY * params["w2"], "b2": error.sum(0)}


def _sha(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


@dataclass
class Result:
    identity: str
    epochs: int
    params: dict
    steps: int
    order_sha256: list  # one per epoch: the epoch's permutation
    epoch_losses: list = field(default_factory=list)  # L after each epoch, when evaluated
    snapshots: dict = field(default_factory=dict)  # {epoch count: params after that many epochs}
    initial: dict = field(default_factory=dict)


def train(identity, data, epochs=EPOCHS, batch=BATCH, evaluate_each_epoch=False, snapshot_after=(),
          marks=None):
    """Train one B1 model. ``snapshot_after``: epoch counts after which a copy of the
    parameters is kept (the matched-start check). ``marks``: an optional
    ``timing.Marks``-like object with ``begin(phase)`` and ``end(phase)``."""
    if epochs < 1 or batch < 1:
        raise ValueError("a recipe needs at least one epoch and one frame per batch")
    if data.count == 0:
        raise ValueError("no training frames")
    params = initialise(identity)
    initial = {k: v.copy() for k, v in params.items()}
    first = {k: np.zeros_like(v) for k, v in params.items()}
    second = {k: np.zeros_like(v) for k, v in params.items()}
    order_rng = _stream(identity, "order")
    batches = -(-data.count // batch)
    step, orders, losses, snaps = 0, [], [], {}
    for epoch in range(epochs):
        if marks is not None:
            marks.begin("epoch")
        rate = rate_at(epoch, epochs)
        order = order_rng.permutation(data.count)
        orders.append(_sha(order.astype("<i8")))
        for begin in range(0, data.count, batch):
            index = order[begin : begin + batch]
            x, y = data.x[index], data.y[index]
            w = (batches * data.w[index]).astype(np.float32)
            gradients = gradient(params, x, y, w)
            step += 1
            for k in NAMES:
                g = gradients[k]
                if not np.isfinite(g).all():
                    raise NonFinite("a gradient is not finite")
                first[k] = 0.9 * first[k] + 0.1 * g  # B0's script, literally
                second[k] = 0.999 * second[k] + 0.001 * g**2
                momentum = first[k] / (1 - 0.9**step)
                update = momentum / (np.sqrt(second[k] / (1 - 0.999**step)) + 1e-8)
                params[k] -= (rate * update).astype(np.float32)
                if not np.isfinite(params[k]).all():
                    raise NonFinite("a parameter is not finite")
        if marks is not None:
            marks.end("epoch")
        if epoch + 1 in snapshot_after:
            snaps[epoch + 1] = {k: v.copy() for k, v in params.items()}
        if evaluate_each_epoch:
            if marks is not None:
                marks.begin("gate_eval")
            losses.append(gate_loss(params, data))
            if marks is not None:
                marks.end("gate_eval")
    return Result(identity, epochs, params, step, orders, losses, snaps, initial)


def parameter_count(params):
    return sum(int(np.asarray(params[k]).size) for k in NAMES)


def to_bytes(result, constants):
    """A deterministic model file: one canonical JSON header line, then the float32 arrays."""
    header = {"format": FORMAT, "identity": result.identity, "epochs": result.epochs, "steps": result.steps,
              "b0_sha256": constants.b0_sha256,
              "arrays": [[k, list(np.asarray(result.params[k]).shape)] for k in NAMES]}
    body = b"".join(np.ascontiguousarray(result.params[k], dtype="<f4").tobytes() for k in NAMES)
    return json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\n" + body


def from_bytes(data):
    head, _, body = data.partition(b"\n")
    header = json.loads(head)
    if header.get("format") != FORMAT:
        raise ValueError("not a B1 model file")
    params, offset = {}, 0
    for name, shape in header["arrays"]:
        size = int(np.prod(shape)) * 4
        params[name] = np.frombuffer(body[offset : offset + size], dtype="<f4").reshape(shape).copy()
        offset += size
    if offset != len(body) or [n for n, _s in header["arrays"]] != list(NAMES):
        raise ValueError("a B1 model file is malformed")
    return header, params


def save(path, result, constants):
    """Write the model file, created new and owner-only; returns its SHA-256."""
    data = to_bytes(result, constants)
    fd = os.open(Path(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as file:
        file.write(data)
    return hashlib.sha256(data).hexdigest()
