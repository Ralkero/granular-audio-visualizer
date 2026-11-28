"""Step 4, S4.2: model C's training inputs, in NumPy (protocol §6; Amendment 3, A3.2 B15, B16,
V7 and V8; A3.3 R21 and R22). The PyTorch loop is ``experiments/presence/ctrain.py``. Built and
tested on generated data only.

**An excerpt** carries the rows model C trains on, ``grid.Segment.c_rows()`` = [s0, f1): up to
1.5 s of context, then the excerpt's own frames. Per row: linear mel power (64 bands, float64),
the label, the training eligibility (context rows are pre-roll, so never eligible: their weight
is 0) and the statistics selection (B15: centred inside the excerpt and not quiet; collar and
uncertain frames included).

* ``statistics``: each model's band statistics from its own training excerpts only, before
  augmentation: ``normalise.pooled`` over log(mel power + 1e-10) on the selected rows (float64,
  population spread + 1e-6).
* ``frame_weights``: the §6 weights on these rows.
* ``partners``: for each no-voice excerpt, the no-voice excerpts of the same group in the same
  training set, itself excluded (B16). Voiced excerpts are never mixed.
* ``epoch_draws``: per visited excerpt, in the epoch's order, the V7 draws from four separate
  streams: mix (u < 0.25, G ~ U[−6, 0] dB and a partner position, always drawn so the stream
  advances the same way whatever happens), gain U[−6, +6] dB, tilt U[−3, +3] dB, and one masked
  block of width U{0…6} starting at U{0…64 − width}.
* ``inputs``: the registered order mix → gain → tilt → log → standardise → mask
  (``augment.augment``, M1). The partner is aligned on the primary's rows by excerpt start
  (frame f0 to frame f0); where it has no frame, it adds no power. *Engineering choice:* the
  mixing reference frames F are the primary's statistics rows (its non-quiet excerpt frames),
  as ``augment.mix`` defines them. R22: labels, eligibility and weights stay the primary's.
* ``plain``: the input without augmentation (the gate loss and inference).
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from . import augment, normalise, weights
from .frontend import BANDS

MIX_PROBABILITY = 0.25
MIX_LEVEL_DB = (-6.0, 0.0)
GAIN_DB = (-6.0, 6.0)
TILT_DB = (-3.0, 3.0)
MAX_MASK = 6
STREAMS = ("mix", "gain", "tilt", "mask")


@dataclass(frozen=True)
class Excerpt:
    key: str
    group: str
    voiced: bool
    power: np.ndarray  # (rows, 64) float64, linear mel power on rows [s0, f1)
    labels: np.ndarray  # (rows,) 0 or 1
    eligible: np.ndarray  # (rows,) bool, the training mask
    selected: np.ndarray  # (rows,) bool, the statistics selection
    excerpt_row: int  # f0 − s0: the row of the excerpt's first frame

    def __post_init__(self):
        rows = len(self.labels)
        if np.asarray(self.power).shape != (rows, BANDS):
            raise ValueError("an excerpt's power must be (rows, 64)")
        if np.asarray(self.eligible).shape != (rows,) or np.asarray(self.selected).shape != (rows,):
            raise ValueError("an excerpt's masks must have one value per row")
        if not 0 <= self.excerpt_row <= rows:
            raise ValueError("an excerpt's first row is outside its rows")
        if np.asarray(self.eligible)[: self.excerpt_row].any() or np.asarray(self.selected)[: self.excerpt_row].any():
            raise ValueError("context rows can be neither trained on nor used for statistics")
        if self.voiced != bool(np.asarray(self.labels).any()):
            raise ValueError("an excerpt is voiced exactly when it has a voice frame")

    @property
    def rows(self):
        return len(self.labels)


def _keys(excerpts):
    keys = [e.key for e in excerpts]
    if len(set(keys)) != len(keys):
        raise ValueError("excerpt keys must be unique")
    return keys


def statistics(excerpts):
    """B15: pooled band statistics over the training excerpts' selected rows, before augmentation."""
    _keys(excerpts)
    return normalise.pooled([(e.key, augment.log_power(e.power), e.selected) for e in excerpts])


def frame_weights(excerpts, min_frames=weights.MIN_FRAMES):
    return weights.frame_weights([weights.Unit(e.group, np.asarray(e.labels), np.asarray(e.eligible, dtype=bool))
                                  for e in excerpts], min_frames)


def partners(excerpts):
    """{position: [positions of same-group no-voice excerpts]} for each no-voice excerpt."""
    _keys(excerpts)
    out = {}
    for i, e in enumerate(excerpts):
        if not e.voiced:
            out[i] = [j for j, f in enumerate(excerpts) if j != i and not f.voiced and f.group == e.group]
    return out


def epoch_draws(generators, order):
    """One epoch's draws, one per visited excerpt, in ``order``."""
    out = []
    for _i in order:
        mix = generators["mix"]
        width = int(generators["mask"].integers(0, MAX_MASK + 1))
        out.append({"mix": bool(mix.random() < MIX_PROBABILITY), "level_db": float(mix.uniform(*MIX_LEVEL_DB)),
                    "partner_u": float(mix.random()),
                    "gain_db": float(generators["gain"].uniform(*GAIN_DB)),
                    "tilt_db": float(generators["tilt"].uniform(*TILT_DB)),
                    "mask_width": width, "mask_start": int(generators["mask"].integers(0, BANDS - width + 1))})
    return out


def draws_sha256(draws):
    text = ";".join(f"{d['mix']:d},{d['level_db']!r},{d['partner_u']!r},{d['gain_db']!r},{d['tilt_db']!r},"
                    f"{d['mask_width']},{d['mask_start']}" for d in draws)
    return hashlib.sha256(text.encode()).hexdigest()


def aligned(primary, partner):
    """The partner's power on the primary's rows, aligned by excerpt start; 0 where it has no frame."""
    out = np.zeros_like(np.asarray(primary.power, dtype=np.float64))
    shift = partner.excerpt_row - primary.excerpt_row
    for row in range(primary.rows):
        source = row + shift
        if 0 <= source < partner.rows:
            out[row] = partner.power[source]
    return out


def chosen_partner(position, draw, partner_map):
    """The partner's position if this draw mixes, else None."""
    candidates = partner_map.get(position, [])
    if not draw["mix"] or not candidates:
        return None
    return candidates[min(int(draw["partner_u"] * len(candidates)), len(candidates) - 1)]


def inputs(excerpts, position, draw, partner_map, stats):
    """One augmented, standardised input (rows, 64) float32 for ``excerpts[position]``."""
    primary = excerpts[position]
    partner = chosen_partner(position, draw, partner_map)
    other = None if partner is None else aligned(primary, excerpts[partner])
    return augment.augment(primary.power, stats, draw, other, primary.selected, mode="M1").astype(np.float32)


def plain(power, stats):
    """The input without augmentation: log, then standardise (float32)."""
    return normalise.apply(augment.log_power(power), stats)
