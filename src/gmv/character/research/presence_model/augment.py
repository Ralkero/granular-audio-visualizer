"""Model C's augmentation (readiness proposal, revision 4.4, section 9). The options (M0, M1,
M2) and every probability and range are pending owner decisions, so all are parameters.

Order: mix, gain and tilt on linear mel power; then log with the floor; then standardise;
then band masking (masked bands are 0, the band's training mean).

* M1 (proposed): P_mix = (P_a + g·(E_a/E_b)·P_b) / (1 + g), with E_a and E_b summed over the
  same reference frames F (the primary's non-quiet excerpt frames) and g = 10^(G/10). Over F
  the total power stays E_a and the partner-to-primary power ratio is g, so G in [-6, 0] dB
  keeps the partner at or below the primary. Skipped when E_a is 0 or E_b <= 1e-20·E_a.
* M0: P_a + g·P_b, no rescaling. M2 (voiced primaries) is not implemented: it would need a
  protocol amendment.

Augmentation changes inputs only: labels, masks and frame weights are never passed in.
"""

import numpy as np

from .frontend import FLOOR

SILENT_PARTNER = 1e-20


def mix(primary, partner, reference, level_db, mode="M1"):
    primary, partner = np.asarray(primary, dtype=np.float64), np.asarray(partner, dtype=np.float64)
    if primary.shape != partner.shape:
        raise ValueError("primary and partner power must have the same shape")
    if not (np.isfinite(primary).all() and np.isfinite(partner).all()):
        raise ValueError("mel power must be finite")
    g = 10 ** (level_db / 10)
    if mode == "M0":
        return primary + g * partner
    if mode == "M2":
        raise NotImplementedError("M2 mixes voiced primaries and needs a protocol amendment")
    if mode != "M1":
        raise ValueError("unknown mixing option")
    reference = np.asarray(reference, dtype=bool)
    e_a, e_b = primary[reference].sum(), partner[reference].sum()
    if e_a == 0 or e_b <= SILENT_PARTNER * e_a:
        return None
    return (primary + g * (e_a / e_b) * partner) / (1 + g)


def gain(power, gain_db):
    return np.asarray(power, dtype=np.float64) * 10 ** (gain_db / 10)


def tilt(power, tau_db):
    power = np.asarray(power, dtype=np.float64)
    bands = power.shape[-1]
    t = tau_db * (2 * np.arange(bands) / (bands - 1) - 1)
    return power * 10 ** (t / 10)


def log_power(power):
    return np.log(np.asarray(power, dtype=np.float64) + FLOOR)


def mask_bands(x, start, width):
    out = np.array(x, copy=True)
    out[..., start : start + width] = 0
    return out


def augment(power, stats, draw, partner=None, reference=None, mode="M1"):
    """One training input in the registered order; ``draw``: mix (bool), level_db, gain_db,
    tilt_db, mask_start, mask_width."""
    from . import normalise

    p = power
    if draw["mix"] and partner is not None:
        mixed = mix(p, partner, reference, draw["level_db"], mode)
        p = p if mixed is None else mixed
    p = tilt(gain(p, draw["gain_db"]), draw["tilt_db"])
    x = normalise.apply(log_power(p), stats)
    return mask_bands(x, draw["mask_start"], draw["mask_width"])


def draw(generators, mix_probability, level_db, gain_db, tilt_db, max_mask, bands=64):
    """One excerpt's parameters from its four augmentation streams (mix, gain, tilt, mask);
    probabilities and ranges are explicit because they are not yet decided."""
    width = int(generators["mask"].integers(0, max_mask + 1))
    return {"mix": bool(generators["mix"].random() < mix_probability),
            "level_db": float(generators["mix"].uniform(*level_db)),
            "gain_db": float(generators["gain"].uniform(*gain_db)),
            "tilt_db": float(generators["tilt"].uniform(*tilt_db)),
            "mask_width": width, "mask_start": int(generators["mask"].integers(0, bands - width + 1))}
