"""A5.2 Step 4, S4.10: B3's synthetic end-to-end run (protocol Amendment 3, A3.4 B3; A3.10 "B3 at
RG1b"). Needs PyTorch: runs in the experiments environment and is skipped elsewhere. Generated
data only. The run is repeated identically and with only metadata and the run's name changed;
computational outputs must be byte-identical and provenance may differ only where allowed."""

import json
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments" / "presence"))

import synthetic_run  # noqa: E402

from gmv.character.research.presence_model import integrity, seal  # noqa: E402

# DU-style allowed changes when only the run's name and excerpt metadata change
ALLOWED = ("run", "inputs.metadata_sha256", "seal_sha256", "expectation_sha256", "header.run")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    base = tmp_path_factory.mktemp("e2e")
    a = synthetic_run.run(base / "a")
    b = synthetic_run.run(base / "b")
    variant = synthetic_run.world(selection="negative candidate, bass_focused: another reason")
    c = synthetic_run.run(base / "c", variant, name="synthetic-run-renamed")
    return base, (a, b, c)


def test_the_run_writes_every_output_class(runs):
    base, _ = runs
    root = base / "a"
    for rel in ("computational/results/thresholds.json", "computational/results/metrics.json",
                "computational/results/screening.json", "computational/results/gates.json",
                "computational/results/fold_index.json", "provenance/run_record.json", "provenance/seal.json",
                "provenance/expectation.json", "runlog/log.json"):
        assert (root / rel).is_file(), rel
    models = list((root / "computational" / "sealed" / "models").rglob("*.model"))
    assert len(models) == 2 * (6 * 5 * 1 + 6 * 3 + 3)  # policy D: inner, outer and final, per arm


def test_an_identical_rerun_is_byte_identical(runs):
    base, (a, b, _c) = runs
    report = integrity.compare_trees(base / "a", base / "b")
    assert report == {"missing_or_extra_files": 0, "computational": 0, "provenance_unexpected": 0,
                      "provenance_allowed": 0, "breakdown": 0}
    assert a == b


def test_metadata_and_the_run_name_change_only_allowed_provenance(runs):
    base, (a, _b, c) = runs
    report = integrity.compare_trees(base / "a", base / "c", ALLOWED)
    assert report["computational"] == 0 and report["missing_or_extra_files"] == 0
    assert report["provenance_unexpected"] == 0 and report["provenance_allowed"] > 0
    assert a == c  # the screening and every figure behind it are unchanged


def test_the_seal_verifies_and_lists_exactly_the_sealed_files(runs):
    base, _ = runs
    record = json.loads((base / "a" / "provenance" / "run_record.json").read_text())
    body = json.loads((base / "a" / "provenance" / "seal.json").read_text())
    assert record["models"] == len(body["inventory"]) == 102
    assert record["seal_sha256"] == seal.sha256((base / "a" / "provenance" / "seal.json").read_bytes())
    assert body["header"]["policy"] == "D" and body["header"]["seed_format"] == "a5.2-seed-format-v1"


def test_gate_records_come_from_registered_horizon_pairs(runs):
    base, _ = runs
    gates = json.loads((base / "a" / "computational" / "results" / "gates.json").read_text())
    assert sorted({h for _z, _s, h, _l in gates["losses"]}) == [60, 90]
    assert gates["outcome"] in ("adopt", "stage 2", "stop")


def test_screening_uses_policy_d_and_reports_it(runs):
    _base, (a, _b, _c) = runs
    for arm in ("B1", "C"):
        assert "Policy D" in a[arm]["policy_note"]
        assert set(a[arm]["auc"]) == {"roc_auc_macro", "pr_auc_macro", "eligible_groups", "mean_voiced_share"}
