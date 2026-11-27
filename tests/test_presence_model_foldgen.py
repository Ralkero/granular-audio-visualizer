"""A5.2 Step 1, S1.5: fold generation on a synthetic presence project (application env).

The project is generated under a temporary folder (``presence_synth``): 8 paired groups at
44.1 kHz, a low-rate group at 22.05 kHz and one validation group. Its expected aggregates
replace the registered ones, which apply only to the real run.
"""

import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent))

import presence_synth  # noqa: E402

from gmv.character.research.presence import loaders  # noqa: E402
from gmv.character.research.presence_model import b0check, enginepass, folds, foldgen, integrity, seedplan  # noqa: E402


def gateway_like(root):
    return [{k: v for k, v in r.items() if k in foldgen.GATEWAY_FIELDS}
            for r in loaders.development_examples(root, ("development",))]


@pytest.fixture(scope="module")
def rate_class(tmp_path_factory):
    weights = b0check.write_weights(tmp_path_factory.mktemp("w") / "w.npz")
    cache = {}

    def classify(rate):
        if rate not in cache:
            cache[rate] = b0check.classify(rate, weights)["classification"]
        return cache[rate]

    return classify


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    return presence_synth.build(tmp_path_factory.mktemp("synthetic") / "presence")


@pytest.fixture(scope="module")
def generated(project, rate_class, tmp_path_factory):
    out = tmp_path_factory.mktemp("folds")
    record = foldgen.generate(gateway_like(project["root"]), project["root"], out, rate_class,
                              enginepass.quiet_mask, project["expected"])
    return out, record


def test_folds_are_generated_for_every_paired_group(generated):
    out, record = generated
    assert len(record["R-A"]) == 16  # the low-rate and validation excerpts are not read
    assert sorted(p.name for p in (out / "computational" / "folds").iterdir()) == [f"{o:02d}.json" for o in range(8)]
    order = integrity.json.loads((out / "computational" / "outer_order.json").read_text())
    assert len(order) == 8 and "group-low" not in order and "group-val" not in order


def test_the_quiet_gap_is_excluded_from_training_but_kept_for_evaluation(generated):
    _out, record = generated
    negatives = [r for r in record["R-A"].values() if r["n_voiced_evaluation"] == 0]
    assert negatives
    for r in negatives:
        assert r["n_novoice_evaluation"] == 500 and 470 < r["n_novoice_training"] < 500


def test_two_runs_are_byte_identical(generated, project, rate_class, tmp_path):
    out, _record = generated
    foldgen.generate(gateway_like(project["root"]), project["root"], tmp_path, rate_class,
                     enginepass.quiet_mask, project["expected"])
    report = integrity.compare_trees(out, tmp_path)
    assert report == {"missing_or_extra_files": 0, "computational": 0, "provenance_unexpected": 0,
                      "provenance_allowed": 0, "breakdown": 0}


def test_provenance_carries_the_public_digest_and_the_seed_fields(generated):
    out, record = generated
    index = integrity.json.loads((out / "provenance" / "fold_index.json").read_text())
    public = integrity.json.loads((out / "provenance" / "public_digest.json").read_text())
    assert public == integrity.public_digest(record)
    assert index["public"] == {"development_manifest_sha256": record["R-C"]["development_manifest_sha256"]}
    assert index["chain"] == {"integrity_record_sha256": integrity.sha256_bytes(integrity.canonical_json(record)),
                              "public_digest_sha256": integrity.sha256_bytes(integrity.canonical_json(public))}
    configuration = record["R-E"]["configuration"]
    assert {k: configuration[k] for k in integrity.REQUIRED_CONFIGURATION} == seedplan.configuration()
    assert record["R-E"]["configuration_sha256"] == integrity.sha256_bytes(integrity.canonical_json(configuration))


def test_a_configuration_with_other_seed_fields_stops(project, rate_class, tmp_path):
    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.generate(gateway_like(project["root"]), project["root"], tmp_path, rate_class, enginepass.quiet_mask,
                         project["expected"], configuration={"seed_enumeration_sha256": "0" * 64})
    assert not any(tmp_path.iterdir())


def test_an_aggregate_mismatch_stops(project, rate_class, tmp_path):
    expected = dict(project["expected"], paired_excerpts=project["expected"]["paired_excerpts"] + 1)
    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.generate(gateway_like(project["root"]), project["root"], tmp_path, rate_class,
                         enginepass.quiet_mask, expected)
    assert not any(tmp_path.iterdir())


def test_an_unsupported_paired_rate_stops(project, tmp_path):
    def rate_class(rate):
        return "not_constructible" if rate == 22050 else "not_computable"

    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.generate(gateway_like(project["root"]), project["root"], tmp_path, rate_class,
                         enginepass.quiet_mask, project["expected"])


def test_records_with_fields_the_gateway_drops_are_refused(project, rate_class, tmp_path):
    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.generate(loaders.development_examples(project["root"], ("development",)), project["root"],
                         tmp_path, rate_class, enginepass.quiet_mask, project["expected"])


def test_a_changed_audio_file_stops(project, rate_class, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(project["root"].parent, copy)
    root = copy / "presence"
    records = gateway_like(root)
    for r in records:
        r["path"] = str(copy / "audio" / Path(r["path"]).name)
    target = Path(sorted(records, key=lambda r: r["id"])[0]["path"])
    data = bytearray(target.read_bytes())
    data[-1] ^= 1
    target.write_bytes(bytes(data))
    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.generate(records, root, tmp_path / "out", rate_class, enginepass.quiet_mask, project["expected"])


@pytest.mark.parametrize("fmt", ("WAV", "FLAC"))
def test_seek_and_read_equals_decode_and_slice(tmp_path, fmt):
    path = tmp_path / f"x.{fmt.lower()}"
    audio = np.random.default_rng(0).normal(0, 0.1, (44100 * 3, 2))
    sf.write(path, audio, 44100, format=fmt, subtype="PCM_16")
    full = sf.read(path, dtype="float32", always_2d=True)[0]
    assert np.array_equal(foldgen.read_segment(path, 12345, 98765), full[12345:98765])


def test_the_independent_verifier_catches_broken_assignments():
    counts = {f"g{i}": (400, 400) for i in range(8)}
    good = folds.for_outer({k: (v / 50, n / 50) for k, (v, n) in counts.items()}, "g0")
    foldgen.verify(counts, "g0", good)
    crowded = folds.Assignment({k: 0 for k in counts if k != "g0"}, 0, {})
    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.verify(counts, "g0", crowded)
    missing = folds.Assignment({k: f for k, f in good.folds.items() if k != "g1"}, 0, {})
    with pytest.raises(foldgen.FoldGenerationError):
        foldgen.verify(counts, "g0", missing)
