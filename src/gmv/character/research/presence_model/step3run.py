"""Step 2a, S2a.8: step 3's MUSDB part and the public count report of steps 2 and 3 (protocol
Amendment 3, A3.6 and A3.7, with the proposed Amendment 3A for the reruns). Built and tested on
generated data only.

From step 2's private match list, the owner's review decisions, the development manifest's
group and rate fields (B0's rate check decides paired or low-rate), the registry and the groups'
frame counts, it derives privately G_T, G_V and each group's match status, then builds the public
report (L6) **only through the disclosure check**:

* the item D figures (exempt) and the sizes of G_T and G_V and their overlap, released or withheld
  by ``disclosure_counts``;
* the MUSDB list's own figures (record, unparsed and reading counts) and the public totals, which
  describe no group;
* the breakdowns by partition and match type, the matched-record counts and step 3's voiced and
  no-voice durations: not modelled by the Step 2a checker, so one fixed unit withheld whole
  (N1.7 item 7); the low-rate cells always private;
* whether each sensitivity rerun is considered, decided from the item D figure, and the outcome of
  its data checks over every possible removed set (proposed 3A). Rerun sizes, sets and results
  stay private; the sensitivity folds appear only as hiding commitments.

Nothing is published before RG2.
"""

import json
import os
from pathlib import Path

from . import commitment, disclosure_counts, integrity, musdb, runlock, step3

FORMAT = "a5.2-step3-musdb-v1"
WITHHELD_UNIT = "withheld: a fixed unit the Step 2a disclosure check does not model"


def load_step2(path):
    """Step 2's private match list as a deferred ``musdb.Result``, with the test rejections
    already applied there."""
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    matches = [(t, s, k, tuple(types)) for t, s, k, types in body["matches"]]
    sources = {(t, k): tuple(v) for t, k, v in body["sources"]}
    result = musdb.Result(counts={}, musdb_records=body["musdb_records"], unparsed={s: 0 for s in musdb.SPLITS},
                          exposed=None, private=musdb.Private(None, None, sorted(matches), sources),
                          readings_used=body["readings_used"])
    return result, frozenset(tuple(p) for p in body.get("test_rejections_applied", []))


def _matched(result, split):
    return {k for _t, s, k, types in result.matches if s == split and {"exact", "checksum"} & set(types)}


def compute(development_records, registry, step2_result, applied_test_rejections, decisions, rate_class, stats,
            step2_public, commit):
    """Returns (private, public). ``decisions``: every rejection in the review-decisions file;
    those applied at step 2 (test pairs) are not applied again. ``stats``: paired group ->
    (voiced training, no-voice training, voiced evaluation, no-voice evaluation) frames."""
    categories = step3.classify(development_records, rate_class)
    step3.check_groups(registry, categories)
    later = frozenset(decisions) - frozenset(applied_test_rejections)
    result = musdb.exposure(step2_result, registry, categories, rejections=later)
    entries = registry["entries"]
    paired = sorted(k for k, c in categories.items() if c == "paired")
    low = [k for k, c in categories.items() if c == "low-rate"]
    validation = sorted(k for k, e in entries.items() if e.get("partition") == "validation")
    train, val, test = _matched(result, "train"), _matched(result, "validation"), _matched(result, "test")
    truth = {"p_a": len(result.g_t), "p_b": len(result.g_v), "p_ab": len(result.g_t & result.g_v),
             "l_a": len(train & set(low)), "l_b": len(val & set(low)),
             "v_a": len(train & set(validation)), "v_b": len(val & set(validation))}
    model = disclosure_counts.model(len(paired), len(low), len(validation))
    decided = {d.figure: d for d in model.decide(truth)}

    def figure(name):
        d = decided[name]
        return d.shown if d.released else "withheld"

    paired_stats = {g: stats[g] for g in paired}
    reruns, private_reruns = {}, {}
    for label, item_d, removed in (("training", decided["D_train"].shown, result.g_t),
                                   ("validation", decided["D_validation"].shown, result.g_v)):
        considered = step3.rerun_considered_from_item_d(item_d)
        entry = {"considered": considered}
        if considered:
            feasible = step3.feasibility_from_item_d(item_d, paired_stats, non_paired=len(low) + len(validation))
            entry["data_checks"] = "passed" if feasible.feasible else f"failed {feasible.failed}"
            if feasible.feasible:
                fold_private, fold_public, opening = step3.sensitivity_folds(paired_stats, sorted(removed))
                entry["folds_commitment"] = fold_public["folds_commitment"]
                private_reruns[label] = {"removed": sorted(removed), "folds": [f.decode() for f in fold_private["folds"]],
                                         "outer_order": fold_private["outer_order"],
                                         "folds_opening": opening.to_private_record()}
        reruns[label] = entry
    inputs = {"step2_matches": integrity.sha256_bytes(json.dumps(
        [[t, s, k, list(x)] for t, s, k, x in step2_result.matches]).encode()),
              "review_decisions": integrity.sha256_bytes(json.dumps(sorted(map(list, decisions))).encode()),
              "registry": integrity.sha256_bytes(integrity.canonical_json(registry)),
              "categories": integrity.sha256_bytes(integrity.canonical_json(categories))}
    public_value, opening = commitment.commit(inputs)
    public = {
        "format": FORMAT, "commit": commit, "inputs_commitment": public_value,
        "item_d": {"train": decided["D_train"].shown, "validation": decided["D_validation"].shown, "test": len(test)},
        "g_t_size": figure("G_T"), "g_v_size": figure("G_V"), "g_t_g_v_overlap": figure("overlap"),
        "musdb_records": step2_public["musdb_records"], "unparsed_records": step2_public["unparsed_records"],
        "credits_read_by_reading": step2_public["credits_read_by_reading"],
        "totals": {"assessed_groups": len(paired) + len(low) + len(validation), "paired_groups": len(paired),
                   "low_rate_groups": len(low), "validation_groups": len(validation)},
        "breakdowns_by_partition_and_match_type": WITHHELD_UNIT, "matched_record_counts": WITHHELD_UNIT,
        "voiced_and_no_voice_durations": WITHHELD_UNIT, "low_rate_cells": "withheld: one group",
        "sensitivity_reruns": reruns,
    }
    private = {"format": FORMAT, "categories": categories, "g_t": sorted(result.g_t), "g_v": sorted(result.g_v),
               "matched": {"train": sorted(train), "validation": sorted(val), "test": sorted(test)},
               "truth_for_the_check": truth, "reruns": private_reruns, "inputs_sha256": inputs,
               "commitment_opening": opening.to_private_record(),
               "check": {k: {"released": d.released, "reason": d.reason} for k, d in decided.items()}}
    if public["item_d"]["test"]:
        raise step3.Step3Error("a MUSDB-test match remains after the review; step 2 should have stopped")
    return private, public


def run(out_dir, lock_path=None, **inputs):
    """Run step 3's MUSDB part under the exclusive lock and write its private and public files
    (owner-only permissions, never overwritten)."""
    with runlock.exclusive(lock_path):
        private, public = compute(**inputs)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=False)
        os.chmod(out_dir, 0o700)
        for name, body in (("private.json", private), ("public.json", public)):
            fd = os.open(out_dir / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(body, file, indent=1, sort_keys=True)
        return public
