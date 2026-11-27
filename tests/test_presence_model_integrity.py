"""A5.2 Step 1, S1.4: integrity records, mismatch detection and artifact-tree comparison.

Synthetic records only. Every R-A field and every other record field is mutated in turn and
must be detected, and reports must carry kinds and counts, never identifiers or values.
"""

import copy
from pathlib import Path

import numpy as np
import pytest

from gmv.character.research.presence_model import frontend, grid, integrity

REPOSITORY = Path(__file__).resolve().parents[1]


def excerpt(start=7.0, confirmed=((2.0, 4.0),), first=None):
    rate = 44100
    seg = grid.canonical(20 * rate, rate, start, start + 10.0)
    if first is not None:  # a segment cut elsewhere, as a mistaken data layer might
        end = min(seg.total_samples, round((start + 10.0) * rate) + frontend.engine_window(rate))
        n = frontend.frame_count(end - first, rate)
        centres = frontend.frame_centres(n, rate, first / rate) - start
        seg = grid.Segment(rate, seg.total_samples, start, start + 10.0, first, end, n, centres,
                           int((centres < 0).sum()), int((centres < 10.0).sum()))
    quiet = np.zeros(seg.frames, dtype=bool)
    quiet[seg.f0 + 300 : seg.f0 + 305] = True
    arrays = grid.frame_arrays(seg, list(confirmed), quiet=quiet)
    return integrity.excerpt_record("a" * 64, 1000, rate, 2, 20 * rate, seg, arrays, quiet)


def base_record():
    return integrity.record(
        {"CANARY-excerpt-1": excerpt(), "CANARY-excerpt-2": excerpt(3.013)},
        history_sha256="b" * 64, folded_state_sha256="c" * 64, development_manifest_sha256="d" * 64,
        private_inputs={"registry_sha256": "CANARY-registry", "fingerprints_file_sha256": "f" * 64},
        environment={"python": "3.x", "numpy": "2.x"}, configuration={"preroll_s": 5.0, "collar_s": 0.3},
        caches={"mel": "e" * 64},
    )


def test_identical_records_have_no_difference():
    assert integrity.compare(base_record(), base_record()) == {}
    integrity.require_equal(base_record(), base_record())


@pytest.mark.parametrize("field", integrity.EXCERPT_FIELDS)
def test_every_excerpt_field_mutation_is_detected(field):
    a, b = base_record(), base_record()
    value = b["R-A"]["CANARY-excerpt-1"][field]
    b["R-A"]["CANARY-excerpt-1"][field] = value + 1 if isinstance(value, int) else "0" * 64
    assert integrity.compare(a, b) == {f"R-A.{field}": 1}
    with pytest.raises(integrity.IntegrityMismatch):
        integrity.require_equal(a, b)


@pytest.mark.parametrize("path", (("R-B", "history_sha256"), ("R-B", "folded_state_sha256"),
                                  ("R-C", "development_manifest_sha256"), ("R-D", "numpy"),
                                  ("R-E", "configuration_sha256"), ("R-F", "mel")))
def test_every_record_field_mutation_is_detected(path):
    a, b = base_record(), base_record()
    b[path[0]][path[1]] = "changed"
    assert integrity.compare(a, b) == {".".join(path): 1}


def test_missing_extra_and_private_differences_are_reported_without_identifiers():
    a, b = base_record(), base_record()
    del b["R-A"]["CANARY-excerpt-2"]
    b["R-A"]["CANARY-excerpt-3"] = excerpt(0.5)
    b["R-C"]["private"]["registry_sha256"] = "CANARY-other"
    report = integrity.compare(a, b)
    assert report == {"R-A.extra_excerpt": 1, "R-A.missing_excerpt": 1, "R-C.private": 1}
    with pytest.raises(integrity.IntegrityMismatch) as caught:
        integrity.require_equal(a, b)
    assert "CANARY" not in str(caught.value) and "CANARY" not in repr(report)


def test_a_segment_cut_on_another_grid_fails_the_contract():
    a = base_record()
    b = copy.deepcopy(a)
    b["R-A"]["CANARY-excerpt-2"] = excerpt(3.013, first=66723)  # the naive 1.5 s cut
    report = integrity.compare(a, b)
    assert report.get("R-A.first") == 1 and report.get("R-A.training_sha256") == 1


def test_the_public_digest_has_no_identifiers_or_private_values():
    digest = integrity.public_digest(base_record())
    text = integrity.canonical_json(digest).decode()
    assert "CANARY" not in text and "registry" not in text
    assert set(digest) == {"development_manifest_sha256", "history_sha256", "folded_state_sha256",
                           "excerpts_sha256", "configuration_sha256", "environment"}


def test_array_hashes_depend_on_dtype_shape_and_content():
    x = np.zeros(4, dtype=bool)
    assert integrity.array_sha256(x) != integrity.array_sha256(x.astype(np.int8))
    assert integrity.array_sha256(x) != integrity.array_sha256(np.zeros(5, dtype=bool))
    assert integrity.array_sha256(x) == integrity.array_sha256(np.zeros(4, dtype=bool))


def test_the_environment_record_has_versions_commit_and_no_timestamps():
    env = integrity.environment(REPOSITORY, REPOSITORY / "experiments" / "presence" / "requirements-lock.txt")
    assert {"python", "numpy", "cpu", "threads", "lock_sha256", "commit", "clean"} <= set(env)
    assert len(env["commit"]) == 40 and isinstance(env["clean"], bool)
    assert not any("time" in key or "date" in key for key in integrity.flatten(env))


def make_tree(root, provenance, computational=b"folds", breakdown=b"table", runlog=b"t=1"):
    integrity.write_artifact(root, "computational", "folds.bin", computational)
    integrity.write_artifact(root, "provenance", "index.json", provenance)
    integrity.write_artifact(root, "breakdown", "table.json", breakdown)
    integrity.write_artifact(root, "runlog", "log.txt", runlog)


PROVENANCE = {"inputs": {"development_manifest_sha256": "1", "history_sha256": "2"},
              "chain": {"integrity_record_sha256": "3"}}


def test_tree_comparison_separates_computational_and_provenance(tmp_path):
    changed = copy.deepcopy(PROVENANCE)
    changed["inputs"]["development_manifest_sha256"] = "9"
    changed["chain"]["integrity_record_sha256"] = "8"
    make_tree(tmp_path / "a", PROVENANCE)
    make_tree(tmp_path / "b", changed, runlog=b"t=2")
    allowed = ("inputs.development_manifest_sha256", "chain.integrity_record_sha256")
    report = integrity.compare_trees(tmp_path / "a", tmp_path / "b", allowed)
    assert report == {"missing_or_extra_files": 0, "computational": 0, "provenance_unexpected": 0,
                      "provenance_allowed": 2, "breakdown": 0}
    # a change outside the allowed list, a computational change and a breakdown change all count
    changed["inputs"]["history_sha256"] = "7"
    make_tree(tmp_path / "c", changed, computational=b"other", breakdown=b"other")
    report = integrity.compare_trees(tmp_path / "a", tmp_path / "c", allowed)
    assert report["provenance_unexpected"] == 1 and report["computational"] == 1 and report["breakdown"] == 1
    assert integrity.compare_trees(tmp_path / "a", tmp_path / "c", allowed, breakdown_may_differ=True)["breakdown"] == 0
