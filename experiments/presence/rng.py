"""Seeding model C from the seed plan (readiness proposal, revision 4.4, section 11).

Each consumer gets its own stream from ``seedplan``: initialisation and dropout are PyTorch
generators seeded with their 64-bit values; epoch order and the augmentation sub-streams are
NumPy ``PCG64`` generators built from their ``SeedSequence``. Nothing here reads or writes
PyTorch's global random state outside ``fork_rng``.
"""

import cnn
import torch

from gmv.character.research.presence_model import seedplan


def _integer(text):
    return seedplan.integer_of(text)


def torch_seed(text, consumer):
    key = seedplan.consumer_key(seedplan.arm_of(text), consumer)
    return seedplan.torch_seed_of(_integer(text), key)


def numpy_generator(text, consumer):
    key = seedplan.consumer_key(seedplan.arm_of(text), consumer)
    return seedplan.numpy_generator(_integer(text), key)


def build_for(text, dtype=torch.float32):
    """Model C for the seed identity ``text`` (its final string in the seed plan)."""
    if seedplan.arm_of(text) != "C":
        raise seedplan.SeedPlanError("model C needs a C seed identity")
    return cnn.build(torch_seed(text, "init"), dtype, dropout_seed=torch_seed(text, "dropout"))
