"""Step 2a, S2a.4: hiding commitments in place of public hashes of private inputs (protocol
Amendment 3, A3.6).

A plain public hash of private inputs can be matched by anyone who can guess them: with every
other input known, trying each possible review-decisions file (or each possible removed set of
groups) against a deterministic digest finds the real one. A commitment instead hashes a fixed
label, a fresh 32-byte secret from the operating system and the canonical list of the private
SHA-256 values:

* **hiding:** without the secret, no guess can be tested against the public value, and two
  runs never share a value, even with identical inputs;
* **binding:** the private opening (the secret and the hashes, kept only in the private run
  record) later shows which inputs were used; any change fails ``verify``.

The secret is never derived from a seed, and never printed: ``Opening`` withholds itself.
"""

import hashlib
import hmac
import json
import re
import secrets

LABEL = "gmv-a5.2-commitment-v1"
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class CommitmentError(ValueError):
    """The hashes to commit to, or an opening, are malformed."""


class Opening:
    """The private opening of a commitment: the secret and the committed hashes. Kept only in
    the private run record; it never prints its contents and is not JSON-serialisable (use
    ``to_private_record``)."""

    __slots__ = ("_secret", "_hashes", "label")

    def __init__(self, secret, hashes, label=LABEL):
        self._secret, self._hashes, self.label = bytes(secret), dict(sorted(hashes.items())), label

    def __repr__(self):
        return "<commitment opening: withheld>"

    __str__ = __repr__

    def to_private_record(self):
        """The opening as a dict, for the private run record only."""
        return {"label": self.label, "secret": self._secret.hex(), "hashes": dict(self._hashes)}

    @classmethod
    def from_private_record(cls, record):
        try:
            return cls(bytes.fromhex(record["secret"]), record["hashes"], record["label"])
        except (KeyError, TypeError, ValueError):
            raise CommitmentError("the commitment opening is malformed") from None


def _canonical(hashes):
    if not isinstance(hashes, dict) or not hashes or not all(
            isinstance(k, str) and k and isinstance(v, str) and SHA256.match(v) for k, v in hashes.items()):
        raise CommitmentError("a commitment needs named SHA-256 values")
    return json.dumps(sorted(hashes.items()), separators=(",", ":")).encode("utf-8")


def _value(label, secret, hashes):
    return hashlib.sha256(label.encode("utf-8") + b"\x00" + secret + b"\x00" + _canonical(hashes)).hexdigest()


def commit(hashes):
    """(public value, private opening) for named SHA-256 values, with a fresh secret."""
    secret = secrets.token_bytes(32)
    return _value(LABEL, secret, hashes), Opening(secret, hashes)


def verify(public, opening):
    """True only if ``opening`` opens ``public`` exactly."""
    if not isinstance(opening, Opening) or not isinstance(public, str):
        return False
    try:
        expected = _value(opening.label, opening._secret, opening._hashes)
    except CommitmentError:
        return False
    return hmac.compare_digest(expected, public)
