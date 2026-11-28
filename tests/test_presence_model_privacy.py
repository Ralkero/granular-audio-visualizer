"""A5.2 Step 1, S1.6: the gateway, the selection parser and the spy test (PC1-PC9, DU1, DU4,
DU5). Synthetic worlds only, built under pytest's temporary directory.

Canaries are fresh random tokens planted in Tier 1 and Tier 3 items, in development keys and
paths, in ``source`` and in ``selection``. They are synthetic, not private data, but PC7 is
applied in full.

Access is recorded by three recorders whose evidence is kept apart (``privacy_harness``):
strace (layer 1, when installed), the audit hook (layer 2) and the stat probes (layer 3).
Each PC3-PC6 control is split by recorder: a test naming a recorder passes only if that
recorder itself classified the injected access. The strace tests fail while strace is not
installed. Relative paths are resolved against the calling process's directory; anything that
cannot be resolved is reported as ``unresolved``, which is a violation. Negative controls show
that one recorder cannot stand in for another, that a non-empty trace proves nothing by
itself, and that canaries reach neither reports nor pytest's failure output.
"""

import json
import secrets
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import presence_synth  # noqa: E402
import privacy_harness as harness  # noqa: E402

from gmv.character.research.presence import project, split  # noqa: E402
from gmv.character.research.presence_model import b0check, enginepass, foldgen, gateway, integrity  # noqa: E402

VOCAB = {"a3": ["cleans", "gutturals"], "candidate": ["drums_only"]}


def token():
    return "CANARY-" + secrets.token_hex(12)


def make_world(base, canaries, final_test=False, groups=6):
    """A synthetic world with canaries in every class of item."""
    dev, tier1, tier3, sel, src = (canaries[k] for k in ("dev", "tier1", "tier3", "selection", "source"))
    spec = presence_synth.default_spec(groups=groups)
    spec = [(f"{g}-{dev}" if p == "development" else f"val-{tier1}", p, r, c) for g, p, r, c in spec]
    n = len(spec)

    def selection(i):
        if i == 0:
            return "phase A3 development excerpt (delivery answer: cleans)"  # PC8a: allow-listed, no canary
        if i == 1:
            return f"negative candidate, {sel}: harmless"  # PC8c: canary where a category belongs
        if i == 2:
            return ""
        return f"negative candidate, drums_only: {sel}"  # PC8b: canary in the reason

    world = base / "world"
    built = presence_synth.build(
        world / "presence", spec, selection=selection, source=lambda i: f"src-{src}",
        audio_name=lambda i: f"{'val-' + tier3 if spec[i][1] == 'validation' else 'dev-' + dev}-{i:02d}.wav",
        extra_registry={f"reserved-{tier1}": {"partition": "reserved", "nodes": [f"name:{tier1}"]}},
        validation_note=tier3)
    root = built["root"]
    (root / "final-test" / "annotations.jsonl").write_text(json.dumps({"note": tier3}) + "\n")
    (root / "development" / "session.json").write_text(json.dumps({"note": tier3}))
    if final_test:
        freeze(root, base, tier1, tier3)
    paired = {str(Path(p).resolve()) for p, s in zip(built["paths"], spec, strict=True)
              if s[1] == "development" and s[2] >= 23700}
    assert n == len(built["paths"])
    return {"world": world, "root": root, "expected": built["expected"], "paired": paired, "paths": built["paths"]}


def freeze(root, base, tier1, tier3):
    """Add a frozen final-test cohort (manifest in manifests/, audio never to be opened)."""
    audio = base / "world" / "audio" / f"final-{tier3}.wav"
    sha = presence_synth.write_audio(audio, 44100, seed=99)
    key = f"final-{tier1}"
    registry = project.read_registry(root)
    registry["entries"][key] = {"partition": "reserved", "nodes": [f"rec:{sha}"]}
    project.write_registry(root, registry)
    frozen = split.freeze_final_test(project.read_registry(root), {key}, 1)
    excerpt = {"id": project.excerpt_id(sha, 7.0), "source": "new", "recording": sha, "path": str(audio),
               "size": audio.stat().st_size, "duration_s": 20.0, "sample_rate": 44100, "channels": 2,
               "start_s": 7.0, "end_s": 17.0, "group": key, "selection": f"final {tier3}"}
    project.write_frozen_cohort(root, [excerpt], frozen, {"frozen_utc": frozen["final_test_frozen_utc"], "groups": [key]})


@pytest.fixture(scope="module")
def canaries():
    return {k: token() for k in ("dev", "tier1", "tier3", "selection", "source")}


@pytest.fixture(scope="module")
def weights(tmp_path_factory):
    return str(b0check.write_weights(tmp_path_factory.mktemp("w") / "w.npz"))


@pytest.fixture(scope="module")
def world(tmp_path_factory, canaries):
    return make_world(tmp_path_factory.mktemp("spy"), canaries)


def evidence_dir(tmp_path_factory, name):
    return tmp_path_factory.mktemp("evidence", numbered=True) / name


def python_only(report):
    """The same report with every strace record removed: what the Python recorders alone saw."""
    return {**report, "recorders": [r for r in report["recorders"] if r != "strace"],
            "detections": {k: n for k, n in report["detections"].items() if not k.startswith("strace|")}}


@pytest.fixture(scope="module")
def step3(world, weights, tmp_path_factory):
    out = tmp_path_factory.mktemp("out3")
    evidence = evidence_dir(tmp_path_factory, "step3")
    report = harness.run_target(world["world"], evidence, "fold_generation",
                                {"root": str(world["root"]), "out": str(out), "weights": weights,
                                 "expected": world["expected"]}, world["paired"])
    return report, out, evidence


# -- PC1: the permitted run -----------------------------------------------------------------------


def test_pc1_fold_generation_touches_only_permitted_items(step3):
    report, out, _evidence = step3
    assert report["exit"] == 0, "fold generation failed under recording"
    assert harness.violations(report) == {}, "a prohibited item was accessed"
    assert all(n == 0 for n in report["unresolved"].values()), "records were left unresolved"
    for needed in ("development-manifest", "development-history", "tier1:validation-manifest",
                   "tier1:registry", "tier1:fingerprints", "paired-audio"):
        assert report["counts"].get(needed, 0) > 0
    assert (out / "computational" / "outer_order.json").is_file()


def test_pc1_a_frozen_final_test_manifest_is_read_but_its_folder_and_audio_are_not(tmp_path_factory, canaries, weights):
    world = make_world(tmp_path_factory.mktemp("frozen"), canaries, final_test=True)
    out = tmp_path_factory.mktemp("out-frozen")
    report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "frozen"), "fold_generation",
                                {"root": str(world["root"]), "out": str(out), "weights": weights,
                                 "expected": world["expected"]}, world["paired"])
    assert report["exit"] == 0
    assert report["counts"].get("tier1:final-test-manifest", 0) > 0
    assert harness.violations(report) == {}


# -- PC2: corrupted Tier 1 metadata ---------------------------------------------------------------


def test_pc2_a_corrupted_tier1_file_gives_a_generic_integrity_failure(world, weights, tmp_path_factory, canaries):
    copy = tmp_path_factory.mktemp("corrupt") / "world"
    shutil.copytree(world["world"], copy)
    target = copy / "presence" / "manifests" / "validation.json"
    data = bytearray(target.read_bytes())
    data[10] ^= 1
    target.write_bytes(bytes(data))
    evidence = evidence_dir(tmp_path_factory, "corrupt")
    report = harness.run_target(copy, evidence, "fold_generation",
                                {"root": str(copy / "presence"), "out": str(evidence.parent / "o"), "weights": weights,
                                 "expected": world["expected"]}, set())
    assert report["exit"] == 3 and report["exception_type"] == "IntegrityError"
    assert harness.canary_hits(list(canaries.values()), [evidence / "exception.txt"]) == 0


# -- PC3-PC6: prohibited accesses are detected, by the recorders revision 4.4 requires ---------------
#
# Each control is split by recorder. A test that names a recorder passes only if that recorder
# itself classified the injected access (``harness.require``); another recorder's detection
# never counts. The strace tests fail while strace is not installed.


def validation_audio(world):
    return next(str(p) for p in world["paths"] if "val-" in p.name)


PC3_FILES = (("validation/annotations.jsonl", "tier3:validation"),
             ("final-test/annotations.jsonl", "tier3:final-test"),
             ("development/session.json", "tier3:session"))


@pytest.fixture(scope="module")
def pc3(world, tmp_path_factory):
    return [(category, harness.run_target(world["world"], evidence_dir(tmp_path_factory, "open"), "inject_open",
                                          {"root": str(world["root"]), "path": str(world["root"] / rel)},
                                          world["paired"]))
            for rel, category in PC3_FILES]


@pytest.fixture(scope="module")
def pc4(world, tmp_path_factory):
    return harness.run_target(world["world"], evidence_dir(tmp_path_factory, "probe"), "inject_probes",
                              {"root": str(world["root"]), "audio": validation_audio(world)}, world["paired"])


@pytest.fixture(scope="module")
def pc5(world, tmp_path_factory):
    return harness.run_target(world["world"], evidence_dir(tmp_path_factory, "native"), "inject_native",
                              {"root": str(world["root"]), "path": validation_audio(world)}, world["paired"])


@pytest.fixture(scope="module")
def pc6(world, tmp_path_factory):
    return harness.run_target(world["world"], evidence_dir(tmp_path_factory, "child"), "inject_child",
                              {"root": str(world["root"]),
                               "path": str(world["root"] / "validation" / "annotations.jsonl")}, world["paired"])


def test_pc3_the_audit_hook_detects_each_tier3_open(pc3):
    """PC3, layer 2 part (revision 4.4: flagged by layer 1 and layer 2)."""
    for category, report in pc3:
        assert report["exit"] == 0
        harness.require(report, "audit-hook", category, "open", "main")


def test_pc3_strace_detects_each_tier3_open(pc3):
    """PC3, layer 1 part."""
    for category, report in pc3:
        harness.require(report, "strace", category, "open", "main")


def test_pc4_the_stat_probes_detect_each_metadata_probe(pc4):
    """PC4, layer 3 part: ``os.stat("final-test/annotations.jsonl")`` (relative, from the
    project root) and ``Path(<validation audio>).exists()``."""
    assert pc4["exit"] == 0
    harness.require(pc4, "stat-probes", "tier3:final-test", "stat", "main")
    harness.require(pc4, "stat-probes", "tier3:audio", "stat", "main")


def test_pc4_the_audit_hook_detects_the_listing(pc4):
    """``os.listdir("final-test")`` is an audit event, not a stat."""
    harness.require(pc4, "audit-hook", "tier3:final-test", "list", "main")


def test_pc4_strace_detects_each_probe_and_the_listing(pc4):
    """PC4, layer 1 part (revision 4.4: flagged by layer 1, and layer 3)."""
    harness.require(pc4, "strace", "tier3:final-test", "stat", "main")
    harness.require(pc4, "strace", "tier3:final-test", "list", "main")
    harness.require(pc4, "strace", "tier3:audio", "stat", "main")


def test_pc5_the_python_recorders_cannot_see_the_native_open(pc5):
    """Negative control: Python sees only soundfile's own existence check (a stat), never the
    open made inside libsndfile. So the Python recorders cannot satisfy PC5."""
    assert pc5["exit"] == 0
    assert harness.detected(pc5, "stat-probes", "tier3:audio", "stat") > 0
    assert harness.detected(pc5, "audit-hook", "tier3:audio") == 0
    assert harness.detected(pc5, "stat-probes", "tier3:audio", "open") == 0
    with pytest.raises(harness.RecorderRequirement):
        harness.require(python_only(pc5), "strace", "tier3:audio", "open")


def test_pc5_strace_detects_the_native_open(pc5):
    """PC5 (revision 4.4: flagged by layer 1): the open inside libsndfile itself."""
    harness.require(pc5, "strace", "tier3:audio", "open", "main")


def test_pc6_the_audit_hook_detects_the_launch(pc6):
    """PC6, layer 2 part: the process start and its arguments."""
    assert pc6["exit"] == 0
    harness.require(pc6, "audit-hook", "tier3:validation", "launch", "main")


def test_pc6_strace_detects_the_child_processes_own_open(pc6):
    """PC6 (revision 4.4: flagged by layer 1): the file opened by the child process itself."""
    harness.require(pc6, "strace", "tier3:validation", "open", "child")


def test_layer1_strace_classifies_the_permitted_run(step3):
    """A non-empty trace is not enough: strace must itself classify the run's permitted
    reads, with nothing left unresolved."""
    report, _out, _evidence = step3
    harness.require(report, "strace", "development-manifest", "open", "main")
    harness.require(report, "strace", "paired-audio", "open", "main")
    assert report["unresolved"]["strace"] == 0, "strace records were left unresolved"


# -- PC7: no canary leaks ---------------------------------------------------------------------------


def test_pc7_no_canary_appears_outside_inputs_and_raw_evidence(step3, canaries):
    report, out, evidence = step3
    never = [canaries[k] for k in ("tier1", "tier3", "selection", "source")]
    assert harness.canary_hits(never, [out]) == 0, "a Tier 1, Tier 3, selection or source canary reached an output"
    assert harness.canary_hits(never + [canaries["dev"]], [evidence / "stdout.txt", evidence / "stderr.txt",
                                                           evidence.parent / f"{evidence.name}.report.json",
                                                           out / "runlog", out / "computational" / "summary.json"]) == 0
    # development identifiers may appear only in the designated private outputs
    assert harness.canary_hits([canaries["dev"]], [out / "computational" / "outer_order.json"]) == 1


# -- PC8: the parser and the field-level gateway ----------------------------------------------------


@pytest.mark.parametrize(("value", "expected"), (
    ("phase A3 development excerpt (delivery answer: cleans)", "A3:cleans"),
    ("negative candidate, drums_only: any reason at all", "candidate:drums_only"),
    ("phase A3 development excerpt (delivery answer: cleans)\n", "unrecognised"),
    (" phase A3 development excerpt (delivery answer: cleans)", "unrecognised"),
    ("Phase A3 development excerpt (delivery answer: cleans)", "unrecognised"),
    ("negative candidate, drums_only: two\nlines", "unrecognised"),
    ("negative candidate, DRUMS_ONLY: reason", "unrecognised"),
    ("phase A3 development excerpt (delivery answer: high_screams)", "unrecognised"),  # not in this vocabulary
    ("", "missing"), (None, "missing"), (7, "unrecognised"), (b"bytes", "unrecognised"),
))
def test_pc8_the_parser_maps_values_to_fixed_tokens(value, expected):
    vocabulary = gateway.Vocabulary(tuple(VOCAB["a3"]), tuple(VOCAB["candidate"]))
    assert gateway.parse(value, vocabulary) == expected


def test_pc8_canaries_in_reason_or_category_never_reach_a_token(canaries):
    vocabulary = gateway.Vocabulary(tuple(VOCAB["a3"]), tuple(VOCAB["candidate"]))
    c = canaries["selection"]
    tokens = [gateway.parse(f"negative candidate, drums_only: {c}", vocabulary),
              gateway.parse(f"negative candidate, {c}: x", vocabulary),
              gateway.parse(f"phase A3 development excerpt (delivery answer: {c})", vocabulary)]
    assert tokens == ["candidate:drums_only", "unrecognised", "unrecognised"]
    assert all(c not in t for t in tokens)


def test_pc8_the_tracked_code_vocabulary_is_only_a_candidate():
    candidate = gateway.tracked_code_candidate()
    assert not candidate.approved
    assert "cleans" in candidate.a3 and "other_song_intro" in candidate.candidate


def test_pc8_source_and_raw_selection_never_leave_the_gateway(world):
    for step in (3, 5, 6):
        records = gateway.Gateway(world["root"], step).records()
        assert records and all(set(r) == set(gateway.FIELDS) for r in records)
    assert gateway.Gateway(world["root"], 3)._selection is None


def test_pc8_the_report_stage_releases_tokens_only_and_suppresses_small_categories(world, tmp_path_factory, canaries):
    out = tmp_path_factory.mktemp("report")
    seal = out / "seal.json"
    seal.write_text(json.dumps({"sealed": True, "predictions_sha256": "0" * 64}))
    evidence = evidence_dir(tmp_path_factory, "report")
    report = harness.run_target(world["world"], evidence, "report_stage",
                                {"root": str(world["root"]), "out": str(out / "o"), "seal": str(seal),
                                 "vocabulary": VOCAB}, world["paired"])
    assert report["exit"] == 0 and harness.violations(report) == {}
    table = json.loads((out / "o" / "breakdown" / "selection.json").read_text())
    assert set(table["categories"]) == {"candidate:drums_only"}
    assert table["suppressed"]["suppressed_categories"] == 3  # A3:cleans, unrecognised, missing (one group each)
    assert harness.canary_hits(list(canaries.values()), [out / "o", evidence / "stdout.txt", evidence / "stderr.txt"]) == 0


def test_pc8_error_paths_leak_nothing(world, tmp_path_factory, canaries):
    evidence = evidence_dir(tmp_path_factory, "error")
    report = harness.run_target(world["world"], evidence, "raise_after_reading", {"root": str(world["root"])},
                                world["paired"])
    assert report["exit"] == 3 and report["exception_type"] == "RuntimeError"
    assert harness.canary_hits(list(canaries.values()), [evidence / "exception.txt", evidence / "stdout.txt",
                                                         evidence / "stderr.txt"]) == 0
    vocabulary = gateway.Vocabulary(("cleans",), ())
    for value in (object(), 3.5, ["x"], {"a": 1}, "\x00", "é" * 10_000):
        assert gateway.parse(value, vocabulary) in ("missing", "unrecognised")


# -- PC9: the seal gate ------------------------------------------------------------------------------


def test_pc9_tokens_need_step_6_a_seal_and_an_approved_vocabulary(world, tmp_path):
    vocabulary = gateway.Vocabulary(tuple(VOCAB["a3"]), tuple(VOCAB["candidate"]))
    with pytest.raises(gateway.GatewayError):
        gateway.Gateway(world["root"], 3).release_tokens(tmp_path / "seal.json", vocabulary, require_approved=False)
    six = gateway.Gateway(world["root"], 6)
    with pytest.raises(gateway.GatewayError):
        six.release_tokens(tmp_path / "missing.json", vocabulary, require_approved=False)
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"sealed": True, "predictions_sha256": "0" * 64}))
    with pytest.raises(gateway.GatewayError):
        six.release_tokens(seal, vocabulary)  # not approved
    assert six.release_tokens(seal, vocabulary, require_approved=False)


# -- DU1, DU4, DU5: metadata that must not influence computation ----------------------------------


def du_world(base):
    spec = [(g, p, r, c) for g, p, r, c in presence_synth.default_spec(groups=6)]
    return presence_synth.build(base / "presence", spec)


def du_variant(base, target, development=None, validation=None, reserved=None):
    """A copy of ``base`` (same histories and audio) with only the named metadata changed,
    rewritten through the project API so the fingerprints stay consistent."""
    root = target / "presence"
    shutil.copytree(base["root"], root)
    for partition, change in (("development", development), ("validation", validation)):
        if change is not None:
            manifest = project.read_manifest(root, partition)
            project.write_manifest(root, partition, [change(dict(e)) for e in manifest["excerpts"]])
    if reserved is not None:
        registry = project.read_registry(root)
        registry["entries"].update(reserved)
        project.write_registry(root, registry)
    return {"root": root, "expected": base["expected"]}


def run_folds(built, out, rate_class):
    records = gateway.Gateway(built["root"], 3).records()
    foldgen.generate(records, built["root"], out, rate_class, enginepass.quiet_mask, built["expected"])


@pytest.fixture(scope="module")
def rate_class(weights):
    cache = {}

    def classify(rate):
        if rate not in cache:
            cache[rate] = b0check.classify(rate, weights)["classification"]
        return cache[rate]

    return classify


@pytest.fixture(scope="module")
def du_base(tmp_path_factory, rate_class):
    base = tmp_path_factory.mktemp("du-base")
    built = du_world(base)
    run_folds(built, base / "out", rate_class)
    return built, base / "out"


# Revision 4.3's DU4/DU5 list: integrity record (R-C development manifest, private fingerprints);
# fold index (inputs and public development manifest, both chain hashes); public digest (its
# development manifest field).
DEV_MANIFEST_CHANGES = ("R-C.development_manifest_sha256", "R-C.private.fingerprints_file_sha256",
                        "inputs.development_manifest_sha256", "public.development_manifest_sha256",
                        "chain.integrity_record_sha256", "chain.public_digest_sha256", "development_manifest_sha256")


def check_du(base_out, built, out, allowed, expected_allowed):
    report = integrity.compare_trees(base_out, out, allowed)
    assert report["computational"] == 0 and report["missing_or_extra_files"] == 0
    assert report["provenance_unexpected"] == 0 and report["provenance_allowed"] == expected_allowed
    # every provenance value equals a fresh recomputation from the actual files (DU4 and DU5, D)
    root = built["root"]
    recorded = json.loads((out / "provenance" / "integrity.json").read_text())
    public = json.loads((out / "provenance" / "public_digest.json").read_text())
    index = json.loads((out / "provenance" / "fold_index.json").read_text())
    development = integrity.file_sha256(root / "manifests" / "development.json")
    assert recorded["R-C"]["development_manifest_sha256"] == development
    for name, rel in (("registry_sha256", "manifests/split-registry.json"),
                      ("fingerprints_file_sha256", "fingerprints/manifests.json"),
                      ("validation_manifest_sha256", "manifests/validation.json")):
        assert recorded["R-C"]["private"][name] == integrity.file_sha256(root / rel)
    assert index["inputs"]["development_manifest_sha256"] == index["public"]["development_manifest_sha256"] == development
    assert public == integrity.public_digest(recorded) and public["development_manifest_sha256"] == development
    assert index["chain"]["integrity_record_sha256"] == integrity.sha256_bytes(integrity.canonical_json(recorded))
    assert index["chain"]["public_digest_sha256"] == integrity.sha256_bytes(integrity.canonical_json(public))


def public_provenance(out):
    index = json.loads((out / "provenance" / "fold_index.json").read_text())
    return (out / "provenance" / "public_digest.json").read_bytes(), index["public"], index["chain"]["public_digest_sha256"]


def retag(excerpt):
    excerpt["selection"] = "negative candidate, bass_focused: another reason"
    return excerpt


def test_du4_changing_selection_changes_only_expected_provenance(du_base, tmp_path, rate_class):
    base, base_out = du_base
    built = du_variant(base, tmp_path, development=retag)
    run_folds(built, tmp_path / "out", rate_class)
    check_du(base_out, built, tmp_path / "out", DEV_MANIFEST_CHANGES, 7)


def test_du5_changing_source_changes_only_expected_provenance(du_base, tmp_path, rate_class):
    base, base_out = du_base
    built = du_variant(base, tmp_path, development=lambda e: e | {"source": "another-source"})
    run_folds(built, tmp_path / "out", rate_class)
    check_du(base_out, built, tmp_path / "out", DEV_MANIFEST_CHANGES, 7)


def test_du1_changing_tier1_metadata_leaves_development_provenance_alone(du_base, tmp_path, rate_class):
    base, base_out = du_base
    built = du_variant(base, tmp_path, validation=lambda e: e | {"selection": "another", "source": "another"},
                       reserved={"reserved-x": {"partition": "reserved", "nodes": ["name:x"]}})
    run_folds(built, tmp_path / "out", rate_class)
    allowed = ("R-C.private.validation_manifest_sha256", "R-C.private.registry_sha256",
               "R-C.private.fingerprints_file_sha256", "chain.integrity_record_sha256")
    check_du(base_out, built, tmp_path / "out", allowed, 4)
    # public provenance stays development-only: byte-identical under Tier 1 changes
    assert public_provenance(tmp_path / "out") == public_provenance(base_out)


def test_du4_the_breakdown_follows_the_changed_selection_values(du_base, tmp_path):
    base, _out = du_base
    vocabulary = gateway.Vocabulary((), ("drums_only", "bass_focused"))
    seal = tmp_path / "seal.json"
    seal.write_text(json.dumps({"sealed": True, "predictions_sha256": "0" * 64}))
    changed = du_variant(base, tmp_path, development=retag)
    tables = []
    for built in (base, changed):
        gw = gateway.Gateway(built["root"], 6)
        groups = {r["id"]: r["group"] for r in gw.records()}
        tables.append(gateway.breakdown(gw.release_tokens(seal, vocabulary, require_approved=False), groups))
    assert set(tables[0]["categories"]) == {"candidate:drums_only"}
    assert set(tables[1]["categories"]) == {"candidate:bass_focused"}


# -- relative paths, directory changes and normalisation -------------------------------------------
#
# Each case: (directory to change into, relative to the project root; path; operation; recorder
# and kind the Python layers must report; kind strace must report). Every case reaches a Tier 3
# item.

RELATIVE = {
    "plain": (".", "validation/annotations.jsonl", "open", "audit-hook", "open", "open"),
    "dot-segments": (".", "./validation/../validation/annotations.jsonl", "open", "audit-hook", "open", "open"),
    "double-separator": (".", "validation//annotations.jsonl", "open", "audit-hook", "open", "open"),
    "parent-segment": ("development", "../validation/annotations.jsonl", "open", "audit-hook", "open", "open"),
    "inside-tier3": ("validation", "annotations.jsonl", "open", "audit-hook", "open", "open"),
    "stat": (".", "final-test/annotations.jsonl", "stat", "stat-probes", "stat", "stat"),
    "listing": (".", "final-test", "list", "audit-hook", "list", "list"),
    "glob": (".", "validation/*", "glob", "audit-hook", "list", "list"),
    "child": (".", "validation/annotations.jsonl", "child", "audit-hook", "launch", "open"),
    # a symbolic link outside the project, opened by a relative path from its own folder
    "symlink": ("../..", "outside-link.jsonl", "open", "audit-hook", "open", "open"),
}
RELATIVE_CATEGORY = {"stat": "tier3:final-test", "listing": "tier3:final-test"}


@pytest.fixture(scope="module")
def relative(world, tmp_path_factory):
    link = world["world"].parent / "outside-link.jsonl"
    if not link.exists():
        link.symlink_to(world["root"] / "validation" / "annotations.jsonl")
    return {name: harness.run_target(world["world"], evidence_dir(tmp_path_factory, f"rel-{name}"), "inject_relative",
                                     {"root": str(world["root"]), "rel": rel, "operation": operation,
                                      "directory": directory}, world["paired"])
            for name, (directory, rel, operation, *_kinds) in RELATIVE.items()}


@pytest.mark.parametrize("case", RELATIVE)
def test_relative_tier3_accesses_are_detected_by_the_python_recorders(relative, case):
    """Negative controls 3 and 4: a relative path, after a change of directory, still names the
    Tier 3 item it reaches."""
    _directory, _rel, _operation, recorder, kind, _strace_kind = RELATIVE[case]
    report = relative[case]
    assert report["exit"] == 0
    harness.require(report, recorder, RELATIVE_CATEGORY.get(case, "tier3:validation"), kind, "main")
    assert report["unresolved"][recorder] == 0


def test_changing_into_a_tier3_folder_is_itself_detected(relative):
    harness.require(relative["inside-tier3"], "audit-hook", "tier3:validation", "chdir", "main")


@pytest.mark.parametrize("case", RELATIVE)
def test_relative_tier3_accesses_are_detected_by_strace(relative, case):
    """The same accesses in the trace, resolved through the process's own directory (for the
    child, the directory it inherited)."""
    origin = "child" if case == "child" else "main"
    harness.require(relative[case], "strace", RELATIVE_CATEGORY.get(case, "tier3:validation"), RELATIVE[case][5], origin)


def test_a_pattern_spanning_project_folders_fails_closed(world, tmp_path_factory):
    report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "rel-pattern"), "inject_relative",
                                {"root": str(world["root"]), "rel": "*/annotations.jsonl", "operation": "glob"},
                                world["paired"])
    assert report["exit"] == 0 and "unclassified" in harness.violations(report)


# -- negative controls -------------------------------------------------------------------------------


def test_negative_python_only_detection_cannot_satisfy_a_strace_requirement(pc3):
    """Control 1: the Python recorders caught the open, yet the strace requirement still fails."""
    _category, report = pc3[0]
    harness.require(report, "audit-hook", "tier3:validation", "open")
    with pytest.raises(harness.RecorderRequirement):
        harness.require(python_only(report), "strace", "tier3:validation", "open")


def synthetic_trace(project_root, cwd):
    """Hand-written lines in strace's ``-f -y`` format: a non-empty trace of permitted reads."""
    return "\n".join((
        '300 execve("/usr/bin/python3", ["python3", "bootstrap.py"], 0x7ffd5 /* 30 vars */) = 0',
        '300 openat(AT_FDCWD, "/usr/lib/python3.12/os.py", O_RDONLY|O_CLOEXEC) = 3</usr/lib/python3.12/os.py>',
        f'300 newfstatat(AT_FDCWD, "{cwd}", {{st_mode=S_IFDIR|0700, st_size=0, ...}}, 0) = 0',
        f'300 openat(AT_FDCWD, "{project_root}/manifests/development.json", O_RDONLY|O_CLOEXEC) = '
        f'3<{project_root}/manifests/development.json>',
        '300 exit_group(0)                       = ?',
    )) + "\n"


def test_negative_a_non_empty_strace_log_without_the_event_does_not_count(world, tmp_path):
    """Control 2: a trace that is non-empty and classified, but lacks the injected access, does
    not satisfy the strace requirement, although the Python recorders caught the access."""
    fresh = tmp_path / "fresh"
    report = harness.run_target(world["world"], fresh, "inject_open",
                                {"root": str(world["root"]), "path": str(world["root"] / "validation" / "annotations.jsonl")},
                                world["paired"])
    assert report["exit"] == 0
    evidence = tmp_path / "with-trace"
    evidence.mkdir(mode=0o700)
    for name in ("audit.jsonl", "probes.jsonl"):
        shutil.copy(fresh / name, evidence / name)
    (evidence / "strace.log").write_text(synthetic_trace(world["root"], fresh))
    analysed = harness.analyse(evidence, world["world"], world["paired"], fresh)
    harness.require(analysed, "strace", "development-manifest", "open")  # the trace was read and classified
    harness.require(analysed, "audit-hook", "tier3:validation", "open")
    with pytest.raises(harness.RecorderRequirement):
        harness.require(analysed, "strace", "tier3:validation", "open")


def test_negative_unclassifiable_records_are_reported_and_fail_closed(world, tmp_path_factory):
    """Control 5: a listing and a stat through a directory descriptor cannot be resolved by the
    Python recorders; they are counted as unresolved, and unresolved is a violation."""
    report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "unresolvable"), "inject_unresolvable",
                                {"root": str(world["root"])}, world["paired"])
    assert report["exit"] == 0
    assert report["unresolved"]["audit-hook"] >= 1 and report["unresolved"]["stat-probes"] >= 1
    assert "unresolved" in harness.violations(report)
    lines = ["not json", '{"event": "unrecordable"}', '{"event": "os.system", "args": [{"s": "cat x"}], "cwd": "/"}',
             '{"event": "open", "args": [{"fd": 5}, {"s": "r"}, {"fd": 0}], "cwd": "/"}',
             '{"event": "os.stat", "args": [{"s": "x"}], "cwd": null}']
    assert [a.path for a in harness.python_accesses(lines, "audit-hook")] == [None] * 5


NESTED = '''
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, {tests!r})
import presence_synth
import privacy_harness as harness

OUT = Path({out!r})


def world(tmp_path):
    canary = "CANARY-" + secrets.token_hex(12)
    known = json.loads((OUT / "canaries.json").read_text()) if (OUT / "canaries.json").exists() else []
    (OUT / "canaries.json").write_text(json.dumps([*known, canary]))
    built = presence_synth.build(tmp_path / "world" / "presence", presence_synth.default_spec(groups=2),
                                 selection=lambda i: "negative candidate, drums_only: " + canary,
                                 source=lambda i: "src-" + canary, validation_note=canary)
    target = built["root"] / "validation" / (canary + ".jsonl")
    target.write_text(canary)
    return harness.run_target(tmp_path / "world", tmp_path / "evidence", "inject_open",
                              {{"root": str(built["root"]), "path": str(target)}}, set())


def test_a_violation_assertion_fails(tmp_path):
    assert harness.violations(world(tmp_path)) == {{}}


def test_a_recorder_requirement_fails(tmp_path):
    harness.require(world(tmp_path), "strace", "tier3:selection", "open")
'''


def test_negative_canaries_do_not_reach_reports_or_failure_output(tmp_path):
    """Control 6: two tests fail on purpose, on worlds whose Tier 3 file names and contents,
    sources and selections carry canaries. Neither pytest's failure output nor the sanitised
    reports contain any canary."""
    out = tmp_path / "out"
    out.mkdir()
    test_file = tmp_path / "test_nested_failure.py"
    test_file.write_text(NESTED.format(tests=str(Path(__file__).resolve().parent), out=str(out)))
    basetemp = tmp_path / "basetemp"
    done = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "--tb=short",
                           f"--basetemp={basetemp}", str(test_file)], cwd=tmp_path, capture_output=True, check=False)
    (out / "pytest-output.txt").write_bytes(done.stdout + done.stderr)
    assert done.returncode == 1 and b"2 failed" in done.stdout
    canaries = json.loads((out / "canaries.json").read_text())
    reports = list(basetemp.rglob("*.report.json"))
    assert len(canaries) == 2 and len(reports) == 2
    assert harness.canary_hits(canaries, [out / "pytest-output.txt", *reports]) == 0


# -- the strace reader, on hand-written lines in strace's format ---------------------------------------
#
# These check the reader's logic on lines written to strace's documented ``-f -y`` format. Whether
# the installed strace prints exactly this format is established only by the strace tests above, on
# real traces.


def trace_report(world, lines, cwd):
    evidence = world["world"].parent / f"trace-{secrets.token_hex(4)}"
    evidence.mkdir(mode=0o700)
    (evidence / "strace.log").write_text("\n".join(lines) + "\n")
    return harness.analyse(evidence, world["world"], world["paired"], cwd)


def test_the_strace_reader_follows_directories_through_chdir_threads_and_children(world):
    r = world["root"]
    report = trace_report(world, [
        '100 execve("/usr/bin/python3", ["python3", "x.py"], 0x7ffd /* 30 vars */) = 0',
        f'100 chdir("{r}") = 0',
        '100 openat(AT_FDCWD, "validation/annotations.jsonl", O_RDONLY|O_CLOEXEC) = -1 ENOENT (No such file)',
        '100 newfstatat(AT_FDCWD, "final-test/annotations.jsonl", 0x7ffc5, 0) = 0',
        '100 clone3({flags=CLONE_VM|CLONE_FS|CLONE_FILES|CLONE_SIGHAND|CLONE_THREAD|CLONE_SYSVSEM, exit_signal=0,'
        ' stack=0x7f, stack_size=0x7fff00} => {parent_tid=[101]}, 88) = 101',
        '101 chdir("validation") = 0',  # a thread shares the directory
        '100 newfstatat(AT_FDCWD, "annotations.jsonl", 0x7ffc5, 0) = 0',
        '100 clone(child_stack=NULL, flags=CLONE_CHILD_CLEARTID|CLONE_CHILD_SETTID|SIGCHLD, child_tidptr=0x7f) = 103',
        '103 chdir("/") = 0',  # a new process changes only its own copy
        '100 access("annotations.jsonl", R_OK) = 0',
        '100 vfork() = 102',
        '102 execve("/usr/bin/cat", ["cat", "../final-test/annotations.jsonl"], 0x7ffd /* 30 vars */) = 0',
        '102 openat(AT_FDCWD, "../final-test/annotations.jsonl", O_RDONLY) = -1 ENOENT (No such file)',
        f'100 getdents64(5<{r}/final-test>, 0x5591, 32768) = 48',
        '102 +++ exited with 1 +++',
        '100 --- SIGCHLD {si_signo=SIGCHLD, si_code=CLD_EXITED, si_pid=102} ---',
    ], "/evidence")
    assert report["unresolved"]["strace"] == 0
    assert harness.detected(report, "strace", "tier3:validation", "open", "main") == 1
    assert harness.detected(report, "strace", "tier3:final-test", "stat", "main") == 1
    assert harness.detected(report, "strace", "tier3:validation", "stat", "main") == 2  # after the thread's chdir
    assert harness.detected(report, "strace", "tier3:validation", "chdir", "main") == 1  # a thread is the main process
    assert harness.detected(report, "strace", "tier3:final-test", "open", "child") == 1  # the inherited directory
    assert harness.detected(report, "strace", "tier3:final-test", "list", "main") == 1


def test_the_strace_reader_uses_kernel_paths_and_joins_interrupted_calls(world):
    r = world["root"]
    report = trace_report(world, [
        f'200 openat(AT_FDCWD</elsewhere>, "x", O_RDONLY) = 3<{r}/validation/annotations.jsonl>',
        f'200 openat(AT_FDCWD, "{r}/final-test/annotations.jsonl", O_RDONLY <unfinished ...>',
        '201 newfstatat(AT_FDCWD, "/usr", 0x7ff, 0) = 0',
        f'200 <... openat resumed>) = 4<{r}/final-test/annotations.jsonl>',
        f'200 openat(AT_FDCWD, "{r}/validation/\\303\\251.jsonl", O_RDONLY) = -1 ENOENT (No such file)',
        f'200 openat(AT_FDCWD, "y", O_RDONLY) = 5<{r}/validation/gone.jsonl (deleted)>',
        f'200 fchdir(6<{r}/development>) = 0',
        '200 openat(AT_FDCWD, "session.json", O_RDONLY) = -1 ENOENT (No such file)',
        f'200 statx(7<{r}/final-test>, "", AT_EMPTY_PATH, STATX_ALL, 0x7ff) = 0',
    ], "/evidence")
    assert report["unresolved"]["strace"] == 0
    assert harness.detected(report, "strace", "tier3:validation", "open") == 3
    assert harness.detected(report, "strace", "tier3:final-test", "open") == 1
    assert harness.detected(report, "strace", "tier3:session", "open") == 1
    assert harness.detected(report, "strace", "tier3:final-test", "stat") == 1


def test_the_strace_reader_reports_records_it_cannot_resolve(world):
    r = world["root"]
    report = trace_report(world, [
        '400 execve("/usr/bin/python3", ["python3"], 0x7ffd /* 30 vars */) = 0',
        '400 openat(7, "annotations.jsonl", O_RDONLY) = -1 EACCES (Permission denied)',  # a bare directory descriptor
        '404 openat(AT_FDCWD, "validation/annotations.jsonl", O_RDONLY) = -1 ENOENT (x)',  # a process of unknown origin
        f'400 openat(AT_FDCWD, "{r}/validation/annot"..., O_RDONLY) = -1 ENOENT (x)',  # a truncated path
        '400 this is not a system call',
        'a line without a process id',
        '400 <... openat resumed>) = 3',  # resumed without a start
        '400 fchdir(9) = 0',  # an undecorated descriptor
        '400 newfstatat(AT_FDCWD, "x", 0x7ff, 0) = 0',  # the directory is unknown after fchdir(9)
    ], "/evidence")
    assert report["unresolved"]["strace"] == 8
    assert "unresolved" in harness.violations(report)


def test_the_strace_reader_reads_input_paths_not_returned_buffers(world):
    """Regression (found in the experiments environment): ``readlink`` returns a link's target
    in an output buffer, which strace prints truncated at 32 characters; ``getcwd`` returns the
    working directory. Neither is an access. Input paths of the same calls are still
    classified, and a truncated input path still fails closed."""
    r = world["root"]
    report = trace_report(world, [
        '500 execve("/usr/bin/python3", ["python3"], 0x7ffd /* 30 vars */) = 0',
        # returned buffers and data: not accesses
        '500 readlink("/usr/bin/python3", "/usr/lib/a/very/long/link/target"..., 4096) = 69',  # truncated output
        f'500 readlinkat(AT_FDCWD</evidence>, "/proc/self/exe", "{r}/validation/annotations.jsonl", 4096) = 84',
        f'500 getcwd("{r}/validation", 4096) = 50',
        f'500 getcwd("{r}/final-test/a/long/directory/na"..., 4096) = 90',  # truncated output
        f'500 getxattr("/usr/lib", "user.{r}", "{r}/final-test"..., 64) = -1 ENODATA (No data available)',
        f'500 symlink("{r}/validation/annotations.jsonl", "/evidence/link") = 0',  # the target is data
        # input paths: still classified
        f'500 readlink("{r}/validation/link", 0x7ffd, 4096) = -1 EINVAL (Invalid argument)',
        f'500 readlinkat(AT_FDCWD</evidence>, "{r}/final-test/x", "short", 4096) = 5',
        f'500 lgetxattr("{r}/identity/x", "user.y", 0x7ffd, 64) = -1 ENODATA (No data available)',
        f'500 symlinkat("target", AT_FDCWD<{r}>, "selection/link") = 0',
        # a truncated input path: unresolved, a violation
        f'500 readlink("{r}/validation/annot"..., 0x7ffd, 4096) = -1 EINVAL (Invalid argument)',
    ], "/evidence")
    assert report["unresolved"]["strace"] == 1
    assert "unresolved" in harness.violations(report)
    assert harness.detected(report, "strace", "tier3:validation") == 1  # the readlink input only
    assert harness.detected(report, "strace", "tier3:validation", "stat", "main") == 1
    assert harness.detected(report, "strace", "tier3:final-test") == 1  # the readlinkat input only
    assert harness.detected(report, "strace", "tier3:identity", "stat") == 1
    assert harness.detected(report, "strace", "tier3:selection") == 1  # the new link's own path
