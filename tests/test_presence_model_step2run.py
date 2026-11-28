"""A5.2 Step 2a, S2a.8: the step 2 runner end to end, sandboxed and traced for real, on a
synthetic world laid out like the real one (protocol Amendment 3, A3.4, A3.6 and A3.10). Nothing
here reads project data. Canaries go into names, titles and track identifiers."""

import json
import secrets
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import step2_synth  # noqa: E402

from gmv.character.research.presence import identity  # noqa: E402
from gmv.character.research.presence_model import commitment, runlock, sandbox, step2run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SITE = Path(sys.prefix) / "lib" / "python3.14" / "site-packages"
CANARY = "CANARY" + secrets.token_hex(6)

pytestmark = pytest.mark.skipif(
    not (shutil.which("bwrap") and shutil.which("strace") and Path(sandbox.INTERPRETER).exists()
         and all((SITE / p).exists() for p in sandbox.PACKAGES)),
    reason="bubblewrap, strace, the system interpreter or the application environment's packages are missing")


def world(tmp_path, extra_records=(), readings=None, decisions=()):
    records = [step2_synth.corpus_record(f"Artist group-00 {CANARY} - Song", "train"),
               step2_synth.corpus_record(f"Clara And group-01 {CANARY} - Air", "validation"),
               step2_synth.corpus_record("Nobody - Tune", "test"), *extra_records]
    names = {"group-00": [f"Artist group-00 {CANARY}"], "group-01": [f"group-01 {CANARY}"]}
    return step2_synth.build(tmp_path / "world", names=names, records=records,
                             readings=readings if readings is not None else
                             {f"Clara And group-01 {CANARY}": "whole-and-pieces"},
                             titles={"group-00": f"Title {CANARY}"}, decisions=decisions)


def declaration(w, tmp_path, name, phase="matching", follows_stop=False):
    return step2run.Declaration(name=name, commit="0" * 40, phase=phase, corpus=w["corpus"],
                                presence_root=w["root"], readings=w["readings"], decisions=w["decisions"],
                                out_root=tmp_path / "runs", follows_stop=follows_stop)


def run(w, tmp_path, name, **kwargs):
    return step2run.run(declaration(w, tmp_path, name, **kwargs), REPO / "src", SITE,
                        lock_path=tmp_path / "experiment.lock", check_commit=False, timeout=300)


def private_text(outcome):
    return "".join(p.read_text() for p in (outcome.run_dir / "out").iterdir())


def test_the_listing_phase_reads_only_the_corpus_and_lists_ambiguous_credits(tmp_path):
    w = world(tmp_path)
    outcome = run(w, tmp_path, "listing", phase="listing")
    assert outcome.status == "listed" and outcome.accepted and outcome.public_sentence is None
    listing = json.loads((outcome.run_dir / "out" / "listing.json").read_text())
    assert listing["ambiguous_credits"] == [f"Clara And group-01 {CANARY}"]
    assert outcome.trace["counts"]["input:open"] == 1  # the corpus manifest only


def test_the_matching_phase_writes_exactly_its_three_files_and_is_accepted(tmp_path):
    w = world(tmp_path)
    outcome = run(w, tmp_path, "match")
    out = outcome.run_dir / "out"
    assert outcome.status == "completed" and outcome.accepted, outcome
    assert {p.name for p in out.iterdir()} == {"matches.json", "review.json", "public.json"}
    assert all(stat.S_IMODE(p.stat().st_mode) == 0o600 for p in out.iterdir())
    matches = json.loads((out / "matches.json").read_text())
    groups = {k for _t, _s, k, _x in matches["matches"]}
    assert groups == {"group-00", "group-01"}
    sources = {(t, k): v for t, k, v in matches["sources"]}
    assert sources[(f"Clara And group-01 {CANARY} - Air", "group-01")] == ["piece"]
    public = json.loads((out / "public.json").read_text())
    text = json.dumps(public)
    assert CANARY not in text and CANARY.lower() not in text
    assert not any(v in text for v in matches["input_sha256"].values())
    opening = commitment.Opening.from_private_record(matches["commitment_opening"])
    assert commitment.verify(public["inputs_commitment"], opening)
    assert outcome.trace["violations"] == {} and outcome.trace["unresolved"] == 0


def test_the_sandboxed_outputs_equal_a_reference_run_outside_the_sandbox(tmp_path):
    w = world(tmp_path)
    outcome = run(w, tmp_path, "ref")
    inside = {p.name: json.loads(p.read_text()) for p in (outcome.run_dir / "out").iterdir()}
    host_paths = {"corpus": w["corpus"], "registry": w["registry"], "aliases": w["aliases"], "titles": w["titles"],
                  "readings": w["readings"], "decisions": w["decisions"]}
    status, outside = step2run.compute("matching", host_paths, "0" * 40)
    assert status == "completed"
    outside = json.loads(json.dumps(outside))
    for files in (inside, outside):  # the commitment's secret is fresh each run, by design
        files["public.json"].pop("inputs_commitment")
        files["matches.json"].pop("commitment_opening")
    assert inside == outside


def test_a_test_split_match_stops_with_a_private_review_of_the_test_pairs(tmp_path):
    w = world(tmp_path, extra_records=[step2_synth.corpus_record(f"Artist group-00 {CANARY} - Leak", "test")])
    outcome = run(w, tmp_path, "test-stop")
    assert outcome.status == "test-split stop" and outcome.accepted
    assert outcome.public_sentence is None  # Amendment 3A: only the final outcome is public
    body = json.loads((outcome.run_dir / "out" / "review.json").read_text())
    assert [(i["group"], i["corpus_split"]) for i in body["pairs"]] == [("group-00", "test")]


def test_rejecting_the_test_pair_lets_a_re_run_complete_with_the_neutral_sentence(tmp_path):
    track = f"Artist group-00 {CANARY} - Leak"
    w = world(tmp_path, extra_records=[step2_synth.corpus_record(track, "test")], decisions=[(track, "group-00")])
    outcome = run(w, tmp_path, "rerun", follows_stop=True)
    assert outcome.status == "completed" and outcome.public_sentence is None
    assert json.loads((outcome.run_dir / "record.json").read_text())["follows_stop"] is True  # private only
    matches = json.loads((outcome.run_dir / "out" / "matches.json").read_text())
    assert not any(t == track for t, *_x in matches["matches"])


@pytest.mark.parametrize("fault", ("no reading", "no alias file", "unparsed"))
def test_faults_stop_with_the_same_neutral_sentence_and_counts_only(tmp_path, fault):
    extra = [step2_synth.corpus_record(f"WholeName{CANARY}", "train")] if fault == "unparsed" else []
    w = world(tmp_path, extra_records=extra, readings={} if fault == "no reading" else None)
    if fault == "no alias file":
        w["aliases"].unlink()
    outcome = run(w, tmp_path, f"fault-{fault.replace(' ', '-')}")
    assert outcome.status == "stop" and outcome.public_sentence is None
    if fault != "no alias file":
        stop = json.loads((outcome.run_dir / "out" / "stop.json").read_text())
        assert CANARY not in stop["message"] and identity.normalise(CANARY) not in stop["message"]


def test_a_held_lock_stops_the_runner_before_anything_starts(tmp_path):
    w = world(tmp_path)
    with runlock.exclusive(tmp_path / "experiment.lock"):
        with pytest.raises(runlock.LockHeld):
            run(w, tmp_path, "locked")
    assert not (tmp_path / "runs" / "locked").exists()


def test_the_source_tree_must_be_clean_at_the_authorised_commit(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    git = ["git", "-C", str(repo), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    subprocess.run([*git, "init", "-q"], check=True)
    (repo / "f").write_text("x")
    subprocess.run([*git, "add", "f"], check=True)
    subprocess.run([*git, "commit", "-q", "-m", "x"], check=True)
    head = subprocess.run([*git, "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    step2run.require_clean(repo, head)
    with pytest.raises(sandbox.SandboxError):
        step2run.require_clean(repo, "0" * 40)
    (repo / "f").write_text("changed")
    with pytest.raises(sandbox.SandboxError):
        step2run.require_clean(repo, head)


def test_nothing_private_reaches_the_conversation_side(tmp_path, capsys):
    w = world(tmp_path)
    outcome = run(w, tmp_path, "quiet")
    captured = capsys.readouterr()
    assert CANARY not in captured.out + captured.err
    assert CANARY not in repr(outcome)
    assert CANARY in private_text(outcome)  # the private files do hold it
    record = json.loads((outcome.run_dir / "record.json").read_text())
    assert stat.S_IMODE((outcome.run_dir / "record.json").stat().st_mode) == 0o600
    assert set(record["trace_verdict"]) == {"verdict", "complete", "violations", "counts", "unresolved"}
