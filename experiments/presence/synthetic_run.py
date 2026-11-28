"""Step 4, S4.10: the synthetic end-to-end run of B3 (protocol Amendment 3, A3.4 B3 and A3.10 "B3 at
RG1b"). Generated data only; needs PyTorch (the experiments environment). It shows the output
classes on a whole development run: models, normalisation statistics, predictions, thresholds,
metrics, screening, AUC, the seal, and the run and gate records.

**The world** is generated: 6 groups of 2 excerpts each (one voiced, one without voice), 5 s long
with 0.5 s of pre-roll, with generated B1 features, generated mel power for C, generated stand-ins
for B0's normalisation constants and for B0's scores, and a metadata field per excerpt (its
``selection`` text) that must never influence computation.

**The run** follows §8 under policy D (I = {0}, O = {0, 1, 2}) with the registered seed identities,
the hashed outer order and the registered inner-fold algorithm: inner models choose each outer
fold's t (J̄, tie rule), outer models are scored at it, final models are trained on every group and
t_final is chosen from the outer predictions; B0ₜ's t comes from B0's scores; effect sizes,
robustness rows and the §13 screening follow, and AUC per B20. **Short recipes** (B1 3 epochs, C 1
epoch) keep it fast; the gate records are B1's registered 60/90-epoch pairs on outer fold 0's and
inner fold 0's training sets, with the ledger and the matched-start check (C's gate rule is covered
by unit tests).

**Output classes** (``integrity.write_artifact``): ``computational`` (model files with C's
statistics, predictions, thresholds, metrics, screening, AUC, gate decisions), ``provenance`` (the
run record, the inputs record, the seal), ``runlog`` (not compared). The sealed folder holds exactly
the model and prediction files the seal lists.
"""

import hashlib
import json
import sys
from pathlib import Path

import ctrain
import numpy as np

from gmv.character.research.presence_model import (
    auc, b1train, cbatches, folds, foldgen, gates, integrity, labels, matched, metrics, scoring, seal,
    seedplan, weights)

PRE, FRAMES, DURATION = 25, 274, 5.0
CENTRES = (np.arange(FRAMES) - PRE) * 0.02 + 0.0232
F0 = int((CENTRES < 0).sum())  # the first excerpt frame
POLICY, INNER, OUTER = "D", (0,), (0, 1, 2)
EPOCHS = {"B1": 3, "C": 1}
GROUPS = [f"group_{c}" for c in "abcdef"]


def world(seed=0, selection="phase A3 development excerpt"):
    """The generated development partition: {group: [excerpt dicts]}."""
    rng = np.random.default_rng(seed)
    out = {}
    for gi, g in enumerate(GROUPS):
        rows = []
        for e, voiced in enumerate((True, False)):
            confirmed = ((1.0 + 0.1 * gi, 4.0),) if voiced else ()
            lab = labels.frame_labels(CENTRES, confirmed, DURATION)
            feats = rng.normal(0, 1, (FRAMES, 548)) + 0.2 * gi
            feats[:, :8] += 1.2 * lab[:, None]
            power = rng.gamma(2.0, 1e-3, (FRAMES, 64))
            power[lab == 1, 8:20] *= 20
            b0 = 1 / (1 + np.exp(-(feats[:, :8].mean(axis=1) * 1.5 - 0.4)))
            rows.append({"key": f"{g}-{e}", "confirmed": confirmed, "labels": lab, "features": feats.astype(np.float32),
                         "power": power, "b0": b0, "quiet": np.zeros(FRAMES, bool), "selection": f"{selection} {gi}"})
        out[g] = rows
    return out


def constants():
    rng = np.random.default_rng(99)
    return b1train.Constants(rng.normal(0.1, 0.05, 548).astype(np.float32), rng.uniform(0.9, 1.1, 548).astype(np.float32),
                             hashlib.sha256(b"generated stand-in for B0's file").hexdigest())


def _training(excerpt):
    return labels.training_mask(CENTRES, DURATION, excerpt["confirmed"], (), excerpt["quiet"])


def b1_data(w, groups, c):
    units = [weights.Unit(g, e["labels"], _training(e)) for g in groups for e in w[g]]
    return b1train.frames(units, [e["features"] for g in groups for e in w[g]], c)


def c_excerpts(w, groups):
    out = []
    for g in groups:
        for e in w[g]:
            selected = labels.in_excerpt(CENTRES, DURATION) & ~e["quiet"]
            out.append(cbatches.Excerpt(e["key"], g, bool(e["labels"].any()), e["power"], e["labels"], _training(e),
                                        selected, F0))
    return out


def scored(w, g, scores_of):
    return [scoring.Scored(g, CENTRES, DURATION, e["confirmed"], (), e["quiet"], scores_of(e)) for e in w[g]]


class Run:
    def __init__(self, root, w, name, commit):
        self.root, self.w, self.name, self.commit = Path(root), w, name, commit
        self.const = constants()
        self.sealed = self.root / "computational" / "sealed"
        self.inventory, self.log = [], []
        self.plan = seedplan.plan(POLICY)
        self.ledger = seedplan.RunLedger(self.plan)

    def _model(self, arm, role, o, i, s, groups):
        head = f"gmv-a5.2:{arm}:{o}:{i}:{s}"
        self.ledger.claim_production(f"{role}-{arm}-{o}-{i}-{s}", head, EPOCHS[arm])
        if arm == "B1":
            result = b1train.train(head, b1_data(self.w, groups, self.const), epochs=EPOCHS["B1"], batch=256)
            data = b1train.to_bytes(result, self.const)
            score = lambda e: b1train.scores(result.params, e["features"], self.const)  # noqa: E731
        else:
            result = ctrain.train(head, c_excerpts(self.w, groups), epochs=EPOCHS["C"], batch=8)
            data = ctrain.to_bytes(result)
            score = lambda e: ctrain.probabilities(result.model, e["power"], result.stats)  # noqa: E731
        entry = {"namespace": "main", "arm": arm, "role": role, "outer": str(o), "inner": str(i), "seed": s,
                 "identity": head, "model_file": f"models/main/{arm}/{o}-{i}-{s}.model",
                 "predictions_file": None if role == "final" else f"predictions/main/{arm}/{o}-{i}-{s}.json"}
        path = self.sealed / entry["model_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entry["model_sha256"] = hashlib.sha256(data).hexdigest()
        entry["predictions_sha256"] = None
        self.inventory.append(entry)
        return entry, score

    def _predict(self, entry, score, groups):
        out = {g: [score(e) for e in self.w[g]] for g in groups}
        body = integrity.canonical_json({g: [[float(v) for v in s] for s in out[g]] for g in sorted(out)})
        path = self.sealed / entry["predictions_file"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        entry["predictions_sha256"] = hashlib.sha256(body).hexdigest()
        return {g: [scoring.Scored(g, CENTRES, DURATION, e["confirmed"], (), e["quiet"], s)
                    for e, s in zip(self.w[g], out[g], strict=True)] for g in groups}

    def execute(self):
        w = self.w
        counts = {}
        for g in GROUPS:
            c = sum((metrics.excerpt_counts(CENTRES, np.zeros(FRAMES, bool), DURATION, e["confirmed"]) for e in w[g]),
                    metrics.Counts())
            counts[g] = (c.voiced_s, c.novoice_s)
        order = foldgen.outer_order(GROUPS)
        index = {"main": {}}
        thresholds = {"C": {}, "B1": {}, "B0t": {}}
        outer_counts = {"C": {s: {} for s in OUTER}, "B1": {s: {} for s in OUTER}}
        outer_grids = {"C": {s: {} for s in OUTER}, "B1": {s: {} for s in OUTER}}
        b0t_counts, auc_values = {}, {"C": {s: {} for s in OUTER}, "B1": {s: {} for s in OUTER}}
        for o, g in enumerate(order):
            assignment = folds.for_outer(counts, g).folds
            index["main"][str(o)] = sorted(set(assignment.values()))
            rest = [k for k in GROUPS if k != g]
            b0_scored = [x for k in rest for x in scored(w, k, lambda e: e["b0"])]
            thresholds["B0t"][g] = scoring.choose({t: scoring.j_exact(cg)
                                                   for t, cg in scoring.counts_grid(b0_scored).items()})
            for arm in ("B1", "C"):
                grids = {}
                for s in INNER:
                    predictions = []
                    for i in sorted(set(assignment.values())):
                        held = [k for k in rest if assignment[k] == i]
                        train = [k for k in rest if assignment[k] != i]
                        entry, score = self._model(arm, "inner", o, i, s, train)
                        predictions += [x for k in held for x in self._predict(entry, score, held)[k]]
                    grids[s] = scoring.counts_grid(predictions)
                thresholds[arm][g] = scoring.operating_point(grids, POLICY)
                for s in OUTER:
                    entry, score = self._model(arm, "outer", o, "outer", s, rest)
                    held = self._predict(entry, score, [g])[g]
                    outer_counts[arm][s][g] = scoring.counts_at(held, thresholds[arm][g])[g]
                    for t, cg in scoring.counts_grid(held).items():
                        outer_grids[arm][s].setdefault(t, {})[g] = cg[g]
                    auc_values[arm][s].update(auc.group_values(held))
            b0t_counts[g] = scoring.counts_at(scored(w, g, lambda e: e["b0"]), thresholds["B0t"][g])[g]
        final = {}
        for arm in ("B1", "C"):
            for s in OUTER:
                self._model(arm, "final", "final", "final", s, GROUPS)
            final[arm] = scoring.t_final(outer_grids[arm])
        results = {}
        for arm in ("B1", "C"):
            e = scoring.effects(outer_counts[arm], b0t_counts, POLICY)
            sc = e.screen(POLICY)
            m = auc.macro(auc_values[arm])
            results[arm] = {"fpa": e.fpa_candidate, "recall": e.recall_candidate, "d_fa": e.d_fa,
                            "delta_recall": e.delta_recall, "ratio": e.ratio, "screening": sc.result,
                            "robust": sc.robust, "policy_note": sc.policy_note, "auc": auc.public(m)}
        results["B0t"] = {"fpa": metrics.macro(b0t_counts)["fpa"], "recall": metrics.macro(b0t_counts)["recall"]}
        gate = self._gates(order, counts)
        self._write(index, thresholds, final, outer_counts, b0t_counts, results, gate)
        return results

    def _gates(self, order, counts):
        """B1's registered gate pairs (60 and 90 epochs) at outer and inner size."""
        g0 = order[0]
        rest = [k for k in GROUPS if k != g0]
        assignment = folds.for_outer(counts, g0).folds
        inner = [k for k in rest if assignment[k] != 0]
        runs = []
        for size, groups in (("outer", rest), ("inner", inner)):
            data = b1_data(self.w, groups, self.const)
            for s in (0, 1):
                text = self.plan_identity(f"gmv-a5.2-gate:B1:{size}-0:conv:{s}")
                self.ledger.claim_production(f"gate-{size}-{s}", text, 60)
                short = b1train.train(text, data, epochs=60, batch=256, evaluate_each_epoch=True, snapshot_after=(30,))
                self.ledger.claim_horizon(f"H-B1-{size}-0:conv:{s}-90", text, 90)
                long = b1train.train(text, data, epochs=90, batch=256, evaluate_each_epoch=True, snapshot_after=(30,))
                ok = True
                try:
                    matched.check_b1(short, long)
                except matched.MatchedStartError:
                    ok = False
                for r in (short, long):
                    runs.append(gates.Run("B1", size, s, r.epochs, r.epoch_losses[-1], tuple(r.epoch_losses),
                                          checks_passed=ok))
        verdict = gates.decide("B1", runs)
        return {"outcome": verdict.outcome, "epochs": verdict.epochs, "reasons": list(verdict.reasons),
                "warnings": list(verdict.warnings),
                "losses": [[r.size, r.seed, r.horizon, r.loss] for r in runs]}

    def plan_identity(self, base):
        return {b: f for b, f, _n in self.plan.identities}[base]

    def _write(self, index, thresholds, final, outer_counts, b0t_counts, results, gate):
        root = self.root
        integrity.write_artifact(root, "computational", "results/thresholds.json",
                                 {a: {g: t for g, t in v.items()} for a, v in thresholds.items()} | {"t_final": final})
        integrity.write_artifact(root, "computational", "results/metrics.json", {
            a: {str(s): {g: vars(c) for g, c in sorted(v.items())} for s, v in per.items()} for a, per in outer_counts.items()}
            | {"B0t": {g: vars(c) for g, c in sorted(b0t_counts.items())}})
        integrity.write_artifact(root, "computational", "results/screening.json", results)
        integrity.write_artifact(root, "computational", "results/gates.json", gate)
        fold_bytes = integrity.canonical_json(index)
        integrity.write_artifact(root, "computational", "results/fold_index.json", fold_bytes)
        amendment5 = {"fold_index_sha256": hashlib.sha256(fold_bytes).hexdigest(), "sensitivity_sets": []}
        amendment6 = {"policy": POLICY, "epochs": dict(EPOCHS)}
        declaration = seal.Declaration(self.name, "step 6 (synthetic)", self.commit, POLICY, dict(EPOCHS), ())
        expected = seal.expectation(fold_bytes, amendment5, amendment6, declaration, self.plan)
        provenance = root / "provenance"
        provenance.mkdir(parents=True, exist_ok=True)
        record = provenance / "expectation.json"
        seal.record_expectation(record, expected)
        header = {"format": seal.FORMAT, "version": seal.VERSION, "run": self.name, "step": declaration.step,
                  "commit": self.commit, "integrity_sha256": self._inputs_digest(), "fold_index_sha256":
                  amendment5["fold_index_sha256"], **seedplan.configuration(POLICY), "policy": POLICY,
                  "epochs": dict(EPOCHS), "thresholds_sha256": integrity.file_sha256(
                      root / "computational" / "results" / "thresholds.json")}
        seal_sha = seal.write(provenance / "seal.json", header, self.inventory)
        verified = seal.verify(provenance / "seal.json", self.sealed, expected, record, declaration, amendment5)
        integrity.write_artifact(root, "provenance", "run_record.json", {
            "run": self.name, "commit": self.commit, "policy": POLICY, "epochs": dict(EPOCHS),
            "seal_sha256": seal_sha, "expectation_sha256": verified.expectation_sha256, "models": verified.entries,
            "inputs": {"computational_sha256": self._inputs_digest(), "metadata_sha256": self._metadata_digest()}})
        integrity.write_artifact(root, "runlog", "log.json", {"entries": self.log})

    def _inputs_digest(self):
        h = hashlib.sha256()
        for g in GROUPS:
            for e in self.w[g]:
                for k in ("labels", "features", "power", "b0", "quiet"):
                    h.update(np.ascontiguousarray(e[k]).tobytes())
                h.update(repr(e["confirmed"]).encode())
        h.update(self.const.b0_sha256.encode())
        return h.hexdigest()

    def _metadata_digest(self):
        return hashlib.sha256("\n".join(e["selection"] for g in GROUPS for e in self.w[g]).encode()).hexdigest()


def run(root, w=None, name="synthetic-run", commit="0" * 40):
    ctrain.configure(4)
    return Run(root, w or world(), name, commit).execute()


if __name__ == "__main__":
    print(json.dumps(run(sys.argv[1]), indent=1, default=str))
