"""A5.2 Step 1, S1.6: the gateway, the selection parser and the spy test (PC1-PC9, DU1, DU4,
DU5). Synthetic worlds only, built under pytest's temporary directory.

Canaries are fresh random tokens planted in Tier 1 and Tier 3 items, in development keys and
paths, in ``source`` and in ``selection``. They are synthetic, not private data, but PC7 is
applied in full. Access is recorded by an audit hook and stat probes (layers 2 and 3) and,
when installed, by strace (layer 1). The tests that need strace fail without it.
"""

import json
import secrets
import shutil
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


# -- PC3-PC6: prohibited accesses are detected ------------------------------------------------------


def test_pc3_opening_tier3_files_is_detected(world, tmp_path_factory):
    for rel, category in (("validation/annotations.jsonl", "tier3:validation"),
                          ("final-test/annotations.jsonl", "tier3:final-test"),
                          ("development/session.json", "tier3:session")):
        report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "open"), "inject_open",
                                    {"root": str(world["root"]), "path": str(world["root"] / rel)}, world["paired"])
        assert category in harness.violations(report), "an injected open was not detected"


def test_pc4_metadata_probes_and_listings_are_detected(world, tmp_path_factory):
    validation_audio = next(str(p) for p in world["paths"] if "val-" in p.name)
    report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "probe"), "inject_probes",
                                {"root": str(world["root"]),
                                 "paths": [str(world["root"] / "final-test"), validation_audio]}, world["paired"])
    found = harness.violations(report)
    assert "tier3:final-test" in found and "tier3:audio" in found


def test_pc5_native_library_access_is_detected(world, tmp_path_factory):
    if not harness.strace_available():
        pytest.fail("PC5 needs strace (layer 1); it is not installed, so native access is not recorded")
    validation_audio = next(str(p) for p in world["paths"] if "val-" in p.name)
    report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "native"), "inject_native",
                                {"root": str(world["root"]), "path": validation_audio}, world["paired"])
    assert "tier3:audio" in harness.violations(report)


def test_pc6_child_process_access_is_detected(world, tmp_path_factory):
    report = harness.run_target(world["world"], evidence_dir(tmp_path_factory, "child"), "inject_child",
                                {"root": str(world["root"]),
                                 "path": str(world["root"] / "validation" / "annotations.jsonl")}, world["paired"])
    assert "tier3:validation" in harness.violations(report)


def test_layer1_strace_is_recording(step3):
    report, _out, evidence = step3
    if not harness.strace_available():
        pytest.fail("strace (layer 1) is not installed; native and child-process access is only partly recorded")
    assert report["layers"][0] == "strace" and (evidence / "strace.log").stat().st_size > 0


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


DEV_MANIFEST_CHANGES = ("R-C.development_manifest_sha256", "R-C.private.fingerprints_file_sha256",
                        "inputs.development_manifest_sha256", "chain.integrity_record_sha256")


def check_du(base_out, built, out, allowed, expected_allowed):
    report = integrity.compare_trees(base_out, out, allowed)
    assert report["computational"] == 0 and report["missing_or_extra_files"] == 0
    assert report["provenance_unexpected"] == 0 and report["provenance_allowed"] == expected_allowed
    recorded = json.loads((out / "provenance" / "integrity.json").read_text())
    assert recorded["R-C"]["development_manifest_sha256"] == integrity.file_sha256(
        built["root"] / "manifests" / "development.json")


def retag(excerpt):
    excerpt["selection"] = "negative candidate, bass_focused: another reason"
    return excerpt


def test_du4_changing_selection_changes_only_expected_provenance(du_base, tmp_path, rate_class):
    base, base_out = du_base
    built = du_variant(base, tmp_path, development=retag)
    run_folds(built, tmp_path / "out", rate_class)
    check_du(base_out, built, tmp_path / "out", DEV_MANIFEST_CHANGES, 4)


def test_du5_changing_source_changes_only_expected_provenance(du_base, tmp_path, rate_class):
    base, base_out = du_base
    built = du_variant(base, tmp_path, development=lambda e: e | {"source": "another-source"})
    run_folds(built, tmp_path / "out", rate_class)
    check_du(base_out, built, tmp_path / "out", DEV_MANIFEST_CHANGES, 4)


def test_du1_changing_tier1_metadata_leaves_development_provenance_alone(du_base, tmp_path, rate_class):
    base, base_out = du_base
    built = du_variant(base, tmp_path, validation=lambda e: e | {"selection": "another", "source": "another"},
                       reserved={"reserved-x": {"partition": "reserved", "nodes": ["name:x"]}})
    run_folds(built, tmp_path / "out", rate_class)
    allowed = ("R-C.private.validation_manifest_sha256", "R-C.private.registry_sha256",
               "R-C.private.fingerprints_file_sha256", "chain.integrity_record_sha256")
    check_du(base_out, built, tmp_path / "out", allowed, 4)


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
