"""The registered seeds (protocol §7)."""

import hashlib

FOLD_SEED = "gmv-a5.2-inner-folds"
BOOTSTRAP_SEED = "gmv-a5.2-bootstrap"


def sha256_hex(text):
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def training_seed(arm, outer, inner, seed_index):
    """The first 8 hex digits of SHA-256(``gmv-a5.2:{arm}:{outer}:{inner}:{seed_index}``), as an
    integer. How final models (which have no outer or inner fold) fill those two fields is a
    Stage 2 detail the protocol leaves open."""
    return int(sha256_hex(f"gmv-a5.2:{arm}:{outer}:{inner}:{seed_index}")[:8], 16)


def numpy_seed(text):
    """A 64-bit integer seed for NumPy's generator, from a registered seed string."""
    return int(sha256_hex(text)[:16], 16)
