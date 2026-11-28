"""Step 2a, S2a.5: step 3's MUSDB and sensitivity-rerun part (protocol Amendment 3, A3.6 and
A3.7). Built and tested on generated inputs only.

* ``classify``: the paired/low-rate split (option C, L1), from the development manifest's
  sample rates with B0's rate check, the rule that defines the training set. G_T and G_V then
  come from ``musdb.exposure`` with the owner's review decisions.
* ``count_code``: the end code for published group counts (0, 1 or 2 shown as "0–2"; the top
  three values as "(n−2)–n"), and ``rerun_triggered``: a sensitivity rerun is considered only
  when the published code shows at least three groups removed. Neither ever reads membership.
* ``feasibility``: F1-F5 must pass for **every** possible removed set of the published size
  among the paired groups, not only the actual one, so the decision does not depend on who the
  members are. F1 fails from the size alone when fewer than 6 groups would remain.
* ``sensitivity_folds``: the folds of P′ (the registered algorithm, the hashed outer order
  restricted to P′ and renumbered), recorded publicly only as hiding commitments.
"""

import itertools
from dataclasses import dataclass

from . import commitment, foldgen, folds, integrity

DEVELOPMENT_CATEGORIES = ("paired", "low-rate")
MIN_REMAINING = 6  # F1: 5 non-empty inner folds need n′ − 1 ≥ 5
MIN_FRAMES = foldgen.MIN_FRAMES  # 2 s at 20 ms


class Step3Error(RuntimeError):
    """A registered step 3 check failed; the step stops. Messages carry counts only."""


# -- the paired/low-rate split ------------------------------------------------------------------------------


def classify(records, rate_class):
    """{group: "paired" | "low-rate"} from manifest records (``group``, ``sample_rate``) and
    ``rate_class`` (rate -> B0 class). Low-rate means B0 cannot be constructed at the rate.
    Stops if the low-rate excerpts span more than one group, a group has excerpts in both
    classes, or a paired rate is not fully supported."""
    classes = {}
    for r in records:
        kind = rate_class(r["sample_rate"])
        if kind == "not_constructible":
            category = "low-rate"
        elif kind == "supported":
            category = "paired"
        else:
            raise Step3Error("a development excerpt's rate is neither supported by B0 nor below its construction limit")
        classes.setdefault(r["group"], set()).add(category)
    if any(len(c) > 1 for c in classes.values()):
        raise Step3Error(f"{sum(len(c) > 1 for c in classes.values())} group(s) have excerpts in both classes")
    categories = {g: c.pop() for g, c in classes.items()}
    if sum(1 for c in categories.values() if c == "low-rate") > 1:
        raise Step3Error("excerpts B0 cannot construct come from more than one group")
    return dict(sorted(categories.items()))


def check_groups(registry, categories):
    """The registry's development groups must equal the manifest's groups."""
    development = {k for k, e in (registry.get("entries") or {}).items() if e.get("partition") == "development"}
    if development != set(categories):
        raise Step3Error(f"the registry and the development manifest disagree on "
                         f"{len(development ^ set(categories))} development group(s)")


def group_stats(records, excerpt_records):
    """{group: (voiced training, no-voice training, voiced evaluation, no-voice evaluation)}
    frames, from manifest records and their integrity excerpt records (R-A, by excerpt id)."""
    stats = {}
    for r in records:
        e = excerpt_records[r["id"]]
        a = stats.get(r["group"], (0, 0, 0, 0))
        stats[r["group"]] = (a[0] + e["n_voiced_training"], a[1] + e["n_novoice_training"],
                             a[2] + e["n_voiced_evaluation"], a[3] + e["n_novoice_evaluation"])
    return stats


# -- codes and the rerun trigger ----------------------------------------------------------------------------


def count_code(k, n):
    """How a group count k out of n is shown: "0–2", "(n−2)–n", or the exact number."""
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n):
        raise ValueError("a count must be a whole number between 0 and its total")
    if k <= 2:
        return "0–2"
    if k >= n - 2:
        return f"{n - 2}–{n}"
    return str(k)


def rerun_triggered(code, n):
    """True only if the published code establishes at least three groups removed: an exact
    count of 3 or more, or the top end code. "0–2" or a withheld size (None) never triggers."""
    if code is None or code == "0–2":
        return False
    if code == f"{n - 2}–{n}":
        return True
    return code.isdigit() and int(code) >= 3


def code_sizes(code, n):
    """The removed-set sizes a published code allows."""
    if code == "0–2":
        return [0, 1, 2]
    if code == f"{n - 2}–{n}":
        return [n - 2, n - 1, n]
    return [int(code)]


# -- feasibility over every removed set ----------------------------------------------------------------------


@dataclass(frozen=True)
class Feasibility:
    feasible: bool
    failed: str | None  # "F1" … "F5" when infeasible
    removed_sets_checked: int


def _counts_ok(remaining, stats):
    """F3-F5 for P′ (evaluation frames). Returns the first failed condition or None."""
    voice = [g for g in remaining if stats[g][2] >= MIN_FRAMES]
    novoice = [g for g in remaining if stats[g][3] >= MIN_FRAMES]
    if not voice or not novoice:
        return "F4"  # R_macro or FPA_macro undefined
    if len(voice) < 2 or len(novoice) < 2:
        return "F5"  # some leave-one-group-out recomputation undefined
    for held_out in remaining:  # F3: J(t) is computed on the other groups' inner predictions
        if not any(g != held_out for g in voice) or not any(g != held_out for g in novoice):
            return "F3"
    return None


def _folds_ok(remaining, stats):
    """F1 and F2 for P′: every outer fold has a registered inner assignment, and its checks
    (including a member group of each class in every inner training set) pass."""
    counts = {g: (stats[g][0], stats[g][1]) for g in remaining}
    seconds = {g: (v / 50, n / 50) for g, (v, n) in counts.items()}
    try:
        for g in counts:
            foldgen.group_type(*counts[g])
        for outer in foldgen.outer_order(counts):
            foldgen.verify(counts, outer, folds.for_outer(seconds, outer))
    except (folds.FoldAssignmentError, foldgen.FoldGenerationError, ValueError):
        return "F2"
    return None


def feasibility(code, stats):
    """F1-F5 for every possible removed set of the sizes the published ``code`` allows, among
    the paired groups in ``stats``. Never reads which groups are actually removed."""
    paired = sorted(stats)
    n = len(paired)
    if not rerun_triggered(code, n):
        raise ValueError("no rerun is considered for this published size")
    checked = 0
    for k in code_sizes(code, n):
        if n - k < MIN_REMAINING:
            return Feasibility(False, "F1", checked)
        for removed in itertools.combinations(paired, k):
            remaining = [g for g in paired if g not in removed]
            checked += 1
            failed = _counts_ok(remaining, stats) or _folds_ok(remaining, stats)
            if failed:
                return Feasibility(False, failed, checked)
    return Feasibility(True, None, checked)


def worst_case_excerpts(code, excerpt_counts):
    """For F6: the largest number of excerpts P′ can keep for the published size (removing
    the groups with the fewest excerpts), so the compute check prices the most expensive set."""
    n = len(excerpt_counts)
    k = min(code_sizes(code, n))
    return sum(excerpt_counts.values()) - sum(sorted(excerpt_counts.values())[:k])


# -- the sensitivity folds, as commitments -------------------------------------------------------------------


def sensitivity_folds(stats, removed):
    """The private folds of P′ = paired groups minus ``removed`` (the registered algorithm, the
    hashed outer order restricted to P′), with public commitments in place of their hashes.
    Returns (private, public, openings)."""
    remaining = {g: s for g, s in stats.items() if g not in set(removed)}
    counts = {g: (s[0], s[1]) for g, s in remaining.items()}
    seconds = {g: (v / 50, n / 50) for g, (v, n) in counts.items()}
    order = foldgen.outer_order(counts)
    files, private = {}, {"outer_order": order, "folds": []}
    for o, outer in enumerate(order):
        assignment = folds.for_outer(seconds, outer)
        foldgen.verify(counts, outer, assignment)
        data = folds.serialise(assignment)
        private["folds"].append(data)
        files[f"folds/{o:02d}.json"] = integrity.sha256_bytes(data)
    files["outer_order.json"] = integrity.sha256_bytes(integrity.canonical_json(order))
    public, opening = commitment.commit(files)
    return private, {"folds_commitment": public, "outer_folds": len(order)}, opening
