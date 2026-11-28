"""Step 2a, S2a.6 (first part): the disclosure check for step 2 and 3's MUSDB count figures
(protocol Amendment 3, A3.5 N1 and A3.6 L6), exact and from public information only.

**The figures checked here**, in the registered release order: the item D figures for MUSDB-train
and MUSDB-validation (exempt, published as they are), then the size of G_T, the size of G_V and
their overlap, each shown with the end code ("0–2", exact, "17–19"). The breakdowns by partition
and match type and the matched-record counts are not modelled by this checker; under N1.7 item 7
they form one registered fixed unit, withheld whole.

**The model.** Assessed groups are exchangeable within their public category: 19 paired, 1
low-rate, 6 validation. Each group has two bits, a name or file match to MUSDB-train (a) and to
MUSDB-validation (b); for paired groups these are G_T and G_V membership. A world is the
number of paired groups in each of the four (a, b) states, the low-rate group's two bits, and
the validation groups' state counts: 517,440 worlds, enumerated exactly.

**Protected quantities** (N1.3): every group's a and b, and the sum and the difference of any
two groups' bits of the same kind, must keep at least two possible values. A quantity that an
item D figure already fixes completely (0 or all 26 assessed groups) counts as already known
(the Fix 1 extension).

**Release** (N1.7 item 5): a figure is released only if every value it could show, given what is
already public, keeps every protected quantity safe. The decision reads only public information
(the released figures and the structure), never the true world; the true world decides only
what a released figure shows.
"""

import itertools
from dataclasses import dataclass

import numpy as np

from .step3 import count_code

PAIRED, LOW, VALIDATION = 19, 1, 6
ASSESSED = PAIRED + LOW + VALIDATION
STATES = ((0, 0), (1, 0), (0, 1), (1, 1))  # (train match, validation match)
ORDER = ("D_train", "D_validation", "G_T", "G_V", "overlap")


def _compositions(total, parts):
    for cut in itertools.combinations(range(total + parts - 1), parts - 1):
        bounds = (-1, *cut, total + parts - 1)
        yield tuple(bounds[i + 1] - bounds[i] - 1 for i in range(parts))


def _worlds():
    paired = np.array(list(_compositions(PAIRED, 4)), dtype=np.int16)  # n00 n10 n01 n11
    validation = np.array(list(_compositions(VALIDATION, 4)), dtype=np.int16)
    low = np.array(STATES, dtype=np.int16)  # (aL, bL)
    p, v, lo = np.meshgrid(np.arange(len(paired)), np.arange(len(validation)), np.arange(len(low)), indexing="ij")
    p, v, lo = p.ravel(), v.ravel(), lo.ravel()
    return {
        "p_a": paired[p, 1] + paired[p, 3], "p_b": paired[p, 2] + paired[p, 3], "p_ab": paired[p, 3],
        "v_a": validation[v, 1] + validation[v, 3], "v_b": validation[v, 2] + validation[v, 3],
        "l_a": low[lo, 0], "l_b": low[lo, 1],
    }


_WORLDS = None


def worlds():
    global _WORLDS
    if _WORLDS is None:
        w = _worlds()
        w["D_train"] = w["p_a"] + w["l_a"] + w["v_a"]
        w["D_validation"] = w["p_b"] + w["l_b"] + w["v_b"]
        _WORLDS = w
    return _WORLDS


def shown(figure, w):
    """What each world shows for a figure: exact item D values, coded group counts."""
    if figure in ("D_train", "D_validation"):
        return w[figure].astype(np.int64)
    k = {"G_T": w["p_a"], "G_V": w["p_b"], "overlap": w["p_ab"]}[figure]
    codes = np.array([count_code(i, PAIRED) for i in range(PAIRED + 1)], dtype=object)
    return codes[k]


def _single(count, size):
    """Possible values of one group's bit in a category with ``count`` of ``size`` set."""
    return {v for v, ok in ((1, count >= 1), (0, count <= size - 1)) if ok}


def _pair_values(count, size):
    sums, diffs = set(), set()
    if size - count >= 2:
        sums.add(0)
        diffs.add(0)
    if count >= 2:
        sums.add(2)
        diffs.add(0)
    if count >= 1 and size - count >= 1:
        sums.add(1)
        diffs.update((1, -1))
    return sums, diffs


def unsafe(mask, fixed=frozenset()):
    """The protected quantities that fail (fewer than two possible values) over the worlds in
    ``mask``. ``fixed``: the bit kinds ("a", "b") an item D figure already fixes completely."""
    w = worlds()
    failures = []
    combos = set()
    for i in np.flatnonzero(mask):
        combos.add(tuple(int(w[k][i]) for k in ("p_a", "p_b", "v_a", "v_b", "l_a", "l_b")))
    for bit in ("a", "b"):
        if bit in fixed:
            continue
        idx = {"a": (0, 2, 4), "b": (1, 3, 5)}[bit]
        cats = {"paired": PAIRED, "validation": VALIDATION, "low-rate": LOW}
        singles = {c: set() for c in cats}
        sums = {}
        diffs = {}
        for combo in combos:
            per = {"paired": (combo[idx[0]], PAIRED), "validation": (combo[idx[1]], VALIDATION),
                   "low-rate": (combo[idx[2]], LOW)}
            values = {c: _single(*per[c]) for c in cats}
            for c in cats:
                singles[c] |= values[c]
                if cats[c] >= 2:
                    s, d = _pair_values(*per[c])
                    sums.setdefault((c, c), set()).update(s)
                    diffs.setdefault((c, c), set()).update(d)
            for c1, c2 in itertools.combinations(cats, 2):
                sums.setdefault((c1, c2), set()).update(x + y for x in values[c1] for y in values[c2])
                diffs.setdefault((c1, c2), set()).update(x - y for x in values[c1] for y in values[c2])
        failures += [f"one {c} group's {bit}" for c, v in singles.items() if len(v) < 2]
        failures += [f"a pair ({c1}, {c2})'s {bit} sum" for (c1, c2), v in sums.items() if len(v) < 2]
        failures += [f"a pair ({c1}, {c2})'s {bit} difference" for (c1, c2), v in diffs.items() if len(v) < 2]
    return failures


@dataclass(frozen=True)
class Decision:
    figure: str
    released: bool
    shown: object  # the value shown when released, else None
    reason: str  # public terms only


def decide(true_world):
    """Run the release procedure for a true world (a dict of the world fields, as ``worlds()``
    holds them per world, plus nothing else). Returns one ``Decision`` per figure, in order. The
    decision to release depends only on what is already public; ``true_world`` decides only
    the values shown."""
    w = worlds()
    feasible = np.ones(len(w["p_a"]), dtype=bool)
    fixed = set()
    decisions = []
    for figure in ORDER:
        exempt = figure.startswith("D_")
        values = shown(figure, w)
        if exempt:
            outcome_ok, reason = True, "an item D figure: exempt"
        else:
            outcome_ok, reason = True, "every value it could show keeps every protected quantity safe"
            for value in sorted(set(values[feasible].tolist()), key=str):
                failures = unsafe(feasible & (values == value), frozenset(fixed))
                if failures:
                    outcome_ok, reason = False, "some value it could show would narrow a protected quantity"
                    break
        if not outcome_ok:
            decisions.append(Decision(figure, False, None, reason))
            continue
        truth = _shown_for(figure, true_world)
        feasible &= values == truth
        if exempt:
            d = int(truth)
            if d in (0, ASSESSED):  # the Fix 1 extension: this figure fixes every group's bit of its kind
                fixed.add("a" if figure == "D_train" else "b")
        decisions.append(Decision(figure, True, truth, reason))
    return decisions


def _shown_for(figure, t):
    if figure == "D_train":
        return int(t["p_a"] + t["l_a"] + t["v_a"])
    if figure == "D_validation":
        return int(t["p_b"] + t["l_b"] + t["v_b"])
    k = {"G_T": t["p_a"], "G_V": t["p_b"], "overlap": t["p_ab"]}[figure]
    return count_code(int(k), PAIRED)
