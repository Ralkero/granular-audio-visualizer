"""Step 2a, S2a.6 (first part): the disclosure check for step 2 and 3's MUSDB count figures
(protocol Amendment 3, A3.5 N1 and A3.6 L6), exact and from public information only.

**The figures checked here**, in the registered release order: the item D figures for MUSDB-train
and MUSDB-validation (exempt, published as they are), then the size of G_T, the size of G_V and
their overlap, each shown with the end code ("0–2", exact, "(n−2)–n"). The breakdowns by
partition and match type and the matched-record counts are not modelled by this checker; under
N1.7 item 7 they form one registered fixed unit, withheld whole.

**The model.** Assessed groups are exchangeable within their public category: by registration 19
paired, 1 low-rate and 6 validation (``Model`` takes other sizes for tests). Each group has two
bits, a name or file match to MUSDB-train (a) and to MUSDB-validation (b); for paired groups these
are G_T and G_V membership. A world is the number of groups of each category in each of the four
(a, b) states: 517,440 worlds at the registered sizes, enumerated exactly.

**Protected quantities** (N1.3): every group's a and b, and the sum and the difference of any
two groups' bits of the same kind, must keep at least two possible values. A quantity that an
item D figure already fixes completely (0 or all assessed groups) counts as already known (the
Fix 1 extension).

**Release** (N1.7 item 5): a figure is released only if every value it could show, given what is
already public, keeps every protected quantity safe. The decision reads only public information
(the released figures and the structure), never the true world; the true world decides only
what a released figure shows.
"""

import itertools
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .step3 import count_code

PAIRED, LOW, VALIDATION = 19, 1, 6
ORDER = ("D_train", "D_validation", "G_T", "G_V", "overlap")
CATEGORIES = ("paired", "low-rate", "validation")


def _compositions(total, parts):
    for cut in itertools.combinations(range(total + parts - 1), parts - 1):
        bounds = (-1, *cut, total + parts - 1)
        yield tuple(bounds[i + 1] - bounds[i] - 1 for i in range(parts))


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


@dataclass(frozen=True)
class Decision:
    figure: str
    released: bool
    shown: object  # the value shown when released, else None
    reason: str  # public terms only


class Model:
    """The exact count model for given category sizes."""

    def __init__(self, paired=PAIRED, low=LOW, validation=VALIDATION):
        self.sizes = {"paired": paired, "low-rate": low, "validation": validation}
        self.assessed = paired + low + validation
        states = {c: np.array(list(_compositions(n, 4)), dtype=np.int16) for c, n in self.sizes.items()}
        grids = np.meshgrid(*(np.arange(len(states[c])) for c in CATEGORIES), indexing="ij")
        idx = {c: g.ravel() for c, g in zip(CATEGORIES, grids, strict=True)}
        w = {}
        for c, short in zip(CATEGORIES, ("p", "l", "v"), strict=True):
            s = states[c][idx[c]]  # n00 n10 n01 n11
            w[f"{short}_a"], w[f"{short}_b"], w[f"{short}_ab"] = s[:, 1] + s[:, 3], s[:, 2] + s[:, 3], s[:, 3]
        w["D_train"] = w["p_a"] + w["l_a"] + w["v_a"]
        w["D_validation"] = w["p_b"] + w["l_b"] + w["v_b"]
        self.w = w

    def shown(self, figure, w=None):
        """What each world shows for a figure: exact item D values, coded group counts."""
        w = self.w if w is None else w
        if figure in ("D_train", "D_validation"):
            return w[figure].astype(np.int64)
        k = {"G_T": w["p_a"], "G_V": w["p_b"], "overlap": w["p_ab"]}[figure]
        n = self.sizes["paired"]
        codes = np.array([count_code(i, n) for i in range(n + 1)], dtype=object)
        return codes[k]

    def unsafe(self, mask, fixed=frozenset()):
        """The protected quantities that fail (fewer than two possible values) over the worlds
        in ``mask``. ``fixed``: the bit kinds ("a", "b") an item D figure already fixes."""
        w = self.w
        keys = ("p_a", "p_b", "l_a", "l_b", "v_a", "v_b")
        combos = {tuple(int(w[k][i]) for k in keys) for i in np.flatnonzero(mask)}
        failures = []
        for bit in ("a", "b"):
            if bit in fixed:
                continue
            positions = {"paired": keys.index(f"p_{bit}"), "low-rate": keys.index(f"l_{bit}"),
                         "validation": keys.index(f"v_{bit}")}
            cats = [c for c in CATEGORIES if self.sizes[c] > 0]
            singles = {c: set() for c in cats}
            sums, diffs = {}, {}
            for combo in combos:
                per = {c: (combo[positions[c]], self.sizes[c]) for c in cats}
                values = {c: _single(*per[c]) for c in cats}
                for c in cats:
                    singles[c] |= values[c]
                    if self.sizes[c] >= 2:
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

    def truth(self, figure, t):
        if figure == "D_train":
            return int(t["p_a"] + t["l_a"] + t["v_a"])
        if figure == "D_validation":
            return int(t["p_b"] + t["l_b"] + t["v_b"])
        k = {"G_T": t["p_a"], "G_V": t["p_b"], "overlap": t["p_ab"]}[figure]
        return count_code(int(k), self.sizes["paired"])

    def decide(self, true_world):
        """The release procedure for a true world (``p_a``, ``p_b``, ``p_ab``, ``l_a``, ``l_b``,
        ``v_a``, ``v_b``). Returns one ``Decision`` per figure, in order. Release decisions
        depend only on what is already public; the true world decides only the values shown."""
        feasible = np.ones(len(self.w["p_a"]), dtype=bool)
        fixed, decisions = set(), []
        for figure in ORDER:
            exempt = figure.startswith("D_")
            values = self.shown(figure)
            ok, reason = True, ("an item D figure: exempt" if exempt
                                else "every value it could show keeps every protected quantity safe")
            if not exempt:
                for value in sorted(set(values[feasible].tolist()), key=str):
                    if self.unsafe(feasible & (values == value), frozenset(fixed)):
                        ok, reason = False, "some value it could show would narrow a protected quantity"
                        break
            if not ok:
                decisions.append(Decision(figure, False, None, reason))
                continue
            shown_value = self.truth(figure, true_world)
            feasible &= values == shown_value
            if exempt and shown_value in (0, self.assessed):  # the Fix 1 extension
                fixed.add("a" if figure == "D_train" else "b")
            decisions.append(Decision(figure, True, shown_value, reason))
        return decisions


@lru_cache(maxsize=4)
def model(paired=PAIRED, low=LOW, validation=VALIDATION):
    return Model(paired, low, validation)


# -- the registered sizes ------------------------------------------------------------------------------------


def worlds():
    return model().w


def shown(figure, w=None):
    return model().shown(figure, w)


def unsafe(mask, fixed=frozenset()):
    return model().unsafe(mask, fixed)


def decide(true_world):
    return model().decide(true_world)
