"""A5.2 Step 1b: the MUSDB check's access profile on a synthetic project (spy test).

Step 2 may read only the corpus manifest, the split registry, ``identity/aliases.json`` and
``identity/titles.json`` (readiness proposal revision 4, §6 and §7). A synthetic world is built
with ``presence_synth``, plus a generated corpus manifest and both identity files. Canaries go
into group keys, names, titles and track identifiers. The target (``privacy_targets.musdb_check``)
runs under all three recorders, and each recorder's detections are checked separately, as in
PC3-PC6. The strace tests fail while strace is not installed.
"""

import hashlib
import json
import secrets
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import presence_synth  # noqa: E402
import privacy_harness as harness  # noqa: E402

from gmv.character.research.presence import identity, project  # noqa: E402

INPUTS = ("step2:corpus-manifest", "tier1:registry", "step2:aliases", "step2:titles")


def token():
    return "CANARY-" + secrets.token_hex(10)


@pytest.fixture(scope="module")
def canary():
    return token()


@pytest.fixture(scope="module")
def world(tmp_path_factory, canary):
    base = tmp_path_factory.mktemp("musdb-spy")
    spec = [(f"{g}-{canary}", p, r, c) for g, p, r, c in presence_synth.default_spec(groups=3)]
    built = presence_synth.build(base / "world" / "presence", spec)
    root = built["root"]
    registry = project.read_registry(root)
    names = {}
    for key, entry in registry["entries"].items():
        names[key] = identity.normalise(f"artist {key}")  # name nodes are stored normalised, as identity.build does
        entry["nodes"].append(f"name:{names[key]}")
    registry["entries"][f"reserved-{canary}"] = {"partition": "reserved",
                                                 "nodes": [f"name:{identity.normalise(f'reserved {canary}')}"]}
    project.write_registry(root, registry)
    recs = {key: next(n[4:] for n in e["nodes"] if n.startswith("rec:")) for key, e in registry["entries"].items()
            if e["partition"] != "reserved"}
    (root / "identity" / "aliases.json").write_text(json.dumps(
        {"aliases": {f"spelling {canary}": names[spec[0][0]]}, "links": []}))
    (root / "identity" / "titles.json").write_text(json.dumps({sha: f"Title {canary}" for sha in recs.values()}))
    (root / "identity" / "groups.json").write_text(json.dumps({"note": canary}))  # Tier 3: never read

    def musdb_record(track_id, split, digest=None):
        return {"dataset": "musdb18hq", "dataset_track_id": track_id, "split": split,
                "sha256": digest or hashlib.sha256(track_id.encode()).hexdigest()}

    first, second = spec[0][0], spec[2][0]
    corpus = base / "world" / "corpus" / "manifest.json"
    corpus.parent.mkdir(parents=True)
    corpus.write_text(json.dumps({"recordings": [
        musdb_record(f"Spelling {canary} - Song", "train"),  # alias -> first group, exact
        musdb_record(f"Somebody {canary} - Song", "validation", digest=recs[second]),  # checksum -> second group
        musdb_record(f"Reserved {canary} - Song", "train"),  # a reserved group: not assessed
        musdb_record(f"Nobody {canary} - Song", "test"),
    ]}))
    categories = {k: ("low-rate" if k.startswith("group-low") else "paired")
                  for k, e in registry["entries"].items() if e["partition"] == "development"}
    return {"world": base / "world", "root": root, "corpus": corpus, "categories": categories,
            "first": first, "second": second}


def run(world, tmp_path_factory, name, also=None):
    evidence = tmp_path_factory.mktemp("musdb-evidence", numbered=True) / name
    out = evidence.parent / "out"
    report = harness.run_target(world["world"], evidence, "musdb_check",
                                {"root": str(world["root"]), "corpus": str(world["corpus"]),
                                 "categories": world["categories"], "out": str(out), "also": also}, set())
    return report, out, evidence


@pytest.fixture(scope="module")
def permitted(world, tmp_path_factory):
    return run(world, tmp_path_factory, "permitted")


def test_the_check_reads_only_the_four_step_2_inputs(permitted):
    report, out, _evidence = permitted
    assert report["exit"] == 0, "the synthetic MUSDB check failed under recording"
    assert harness.violations(report, harness.ALLOWED_STEP2) == {}, "a prohibited item was accessed"
    assert all(n == 0 for n in report["unresolved"].values()), "records were left unresolved"
    assert (out / "public.json").is_file()


@pytest.mark.parametrize("category", INPUTS)
def test_the_audit_hook_records_each_permitted_input(permitted, category):
    harness.require(permitted[0], "audit-hook", category, "open", "main")


@pytest.mark.parametrize("category", INPUTS)
def test_strace_records_each_permitted_input(permitted, category):
    harness.require(permitted[0], "strace", category, "open", "main")


def test_the_step_2_inputs_stay_prohibited_at_step_3(permitted):
    """The new categories widen nothing: the corpus manifest and both identity files are still
    violations for fold generation."""
    found = harness.violations(permitted[0], harness.ALLOWED_STEP3)
    assert {"step2:corpus-manifest", "step2:aliases", "step2:titles"} <= set(found)


@pytest.mark.parametrize(("extra", "category"), (
    ("identity/groups.json", "tier3:identity"),
    ("manifests/development.json", "development-manifest"),
    ("validation/annotations.jsonl", "tier3:validation"),
))
def test_an_access_beyond_the_step_2_inputs_is_detected_by_the_audit_hook_and_strace(world, tmp_path_factory, extra,
                                                                                     category):
    report, _out, _evidence = run(world, tmp_path_factory, f"extra-{category.replace(':', '-')}",
                                  also=str(world["root"] / extra))
    assert report["exit"] == 0
    assert category in harness.violations(report, harness.ALLOWED_STEP2)
    harness.require(report, "audit-hook", category, "open", "main")
    harness.require(report, "strace", category, "open", "main")


def test_public_output_holds_counts_only_and_no_canary(permitted, canary):
    report, out, evidence = permitted
    public = json.loads((out / "public.json").read_text())
    assert public["counts"]["train"]["paired"]["exact"] == {"groups": 1, "records": 1}  # the alias match
    assert public["counts"]["validation"]["paired"]["checksum"] == {"groups": 1, "records": 1}
    assert public["exposed_paired_groups"] == {"training": 1, "validation": 1, "both": 0}
    assert public["musdb_records"] == {"train": 2, "validation": 1, "test": 1}
    files = [out / "public.json", evidence / "stdout.txt", evidence / "stderr.txt",
             evidence.parent / f"{evidence.name}.report.json"]
    assert harness.canary_hits([canary, canary.lower()], files) == 0
