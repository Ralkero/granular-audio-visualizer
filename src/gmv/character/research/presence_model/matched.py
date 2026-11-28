"""Step 4, S4.3: the matched-start check for the paired gate runs (protocol Amendment 3, A3.3;
a failure is S5 of A3.2, never the replay check). NumPy only; it reads the trainers' records.

A ``horizon`` pair is the same gate identity trained as two separate complete recipes, the
shorter (E, or E⁺) and the longer (E⁺, or E⁺⁺). Before their schedules part, the two must agree:

* **B1** (``b1train.Result``): identical initial parameters; identical data order for the shared
  epochs; identical parameters at the end of the shorter recipe's first half, before its learning
  rate drops (epoch 30 for 60 against 90, epoch 45 for 90 against 135). Both runs must have kept a
  snapshot there (``snapshot_after``). B1 has no augmentation and no dropout.
* **C** (``ctrain.Result``): identical initial parameters; identical data order and augmentation
  draws for the shared epochs; identical dropout masks for the shared steps; identical parameters
  after optimiser step 1. The cosine schedule depends on the horizon, so step 2 already differs.
"""

from . import b1train


class MatchedStartError(RuntimeError):
    """S5: a paired gate run did not start as its partner did; the run stops for review."""


def _pair(short, long):
    if short.identity != long.identity:
        raise MatchedStartError("a horizon pair must share its seed identity")
    if not short.epochs < long.epochs:
        raise MatchedStartError("a horizon pair needs a shorter and a longer recipe")


def b1_epoch(short_epochs):
    """The epoch count at which a B1 pair is compared."""
    return b1train.first_half(short_epochs)


def check_b1(short, long):
    _pair(short, long)
    if any(not (short.initial[k] == long.initial[k]).all() for k in b1train.NAMES):
        raise MatchedStartError("the initial parameters differ")
    if short.order_sha256[: short.epochs] != long.order_sha256[: short.epochs]:
        raise MatchedStartError("the data order differs in a shared epoch")
    at = b1_epoch(short.epochs)
    if at not in short.snapshots or at not in long.snapshots:
        raise MatchedStartError("a run kept no snapshot at the end of the shorter recipe's first half")
    if any(not (short.snapshots[at][k] == long.snapshots[at][k]).all() for k in b1train.NAMES):
        raise MatchedStartError("the parameters differ at the end of the shorter recipe's first half")


def check_c(short, long):
    _pair(short, long)
    if short.initial_sha256 != long.initial_sha256:
        raise MatchedStartError("the initial parameters differ")
    shared = short.epochs
    if short.order_sha256[:shared] != long.order_sha256[:shared]:
        raise MatchedStartError("the data order differs in a shared epoch")
    if short.draws_sha256[:shared] != long.draws_sha256[:shared]:
        raise MatchedStartError("the augmentation draws differ in a shared epoch")
    steps = short.steps
    if not short.dropout_sha256 or short.dropout_sha256[:steps] != long.dropout_sha256[:steps]:
        raise MatchedStartError("the dropout masks differ in a shared step")
    if not short.after_step1_sha256 or short.after_step1_sha256 != long.after_step1_sha256:
        raise MatchedStartError("the parameters differ after optimiser step 1")
