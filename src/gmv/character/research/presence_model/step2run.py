"""Step 2a, S2a.8: the step 2 runner, the MUSDB18-HQ overlap check (protocol Amendment 3, A3.4,
A3.6 and A3.9). Built and tested on generated data only; a real run needs its own
authorisation after RG1c, a run declaration naming the input files, and the owner's
attestations.

**Two phases**, each one sandboxed, traced run:

* ``listing``: reads only the corpus manifest's MUSDB records and lists the ambiguous credits
  (public MUSDB names, nothing from the library). It runs every time.
* ``matching``: reads the corpus manifest, the registry (through its fingerprint check), the
  mandatory alias file, the titles, the credit-readings file and the review-decisions file;
  matches with the split deferred to step 3 (option C) and the owner's test-split rejections
  applied first; writes exactly the private match list, the owner's review file and the staged
  public figures to ``/out``.

**Stops.** Unparsed MUSDB records, any input fault and a test-split match stop the step (a
test-split stop also writes the review file of the test pairs). Publicly, every stop reads
``review.STOP`` and every run after a stop ``review.RERUN``; nothing else distinguishes them.

**Outside the sandbox** the launcher takes the exclusive experiment lock, checks the source
tree is clean at the authorised commit, launches, reads the trace, and accepts the run only if
the trace is complete, holds no violation, and ``/out`` holds exactly the expected files.
"""

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from . import commitment, musdb, readers, review, runlock, sandbox, trace

FORMAT = "a5.2-step2-v1"
MODULE = "gmv.character.research.presence_model.step2run"
IN = {
    "corpus": "/in/corpus/manifest.json",
    "registry": "/in/presence/manifests/split-registry.json",
    "fingerprints": "/in/presence/fingerprints/manifests.json",
    "aliases": "/in/presence/identity/aliases.json",
    "titles": "/in/presence/identity/titles.json",
    "readings": "/in/step2/credit-readings.json",
    "decisions": "/in/step2/review-decisions.json",
}
PRESENCE_ROOT = "/in/presence"
EXPECTED = {"listed": {"listing.json"}, "completed": {"matches.json", "review.json", "public.json"},
            "test-split stop": {"review.json", "stop.json"}, "stop": {"stop.json"}}
EXIT = {"listed": 0, "completed": 0, "stop": 3, "test-split stop": 4}
NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


@dataclass(frozen=True)
class Declaration:
    """A step 2 run declaration (private: it holds host paths). ``presence_root`` holds
    ``manifests/``, ``fingerprints/`` and ``identity/``; ``follows_stop`` marks a re-run after the
    owner corrected a private input."""

    name: str
    commit: str
    phase: str
    corpus: Path
    presence_root: Path
    readings: Path
    decisions: Path
    out_root: Path
    follows_stop: bool = False

    def inputs(self):
        if not NAME.match(self.name) or self.phase not in ("listing", "matching"):
            raise ValueError("the run declaration's name or phase is malformed")
        root = Path(self.presence_root)
        hosts = {"corpus": self.corpus, "registry": root / "manifests" / "split-registry.json",
                 "fingerprints": root / "fingerprints" / "manifests.json",
                 "aliases": root / "identity" / "aliases.json", "titles": root / "identity" / "titles.json",
                 "readings": self.readings, "decisions": self.decisions}
        names = ("corpus",) if self.phase == "listing" else tuple(IN)
        return {IN[n]: Path(hosts[n]) for n in names}


@dataclass(frozen=True)
class Outcome:
    status: str  # listed, completed, stop, test-split stop, rejected
    accepted: bool
    public_sentence: str | None  # the only public wording of a stop or re-run
    trace: dict = field(default_factory=dict)  # counts only
    run_dir: Path | None = None


# -- the step itself (inside the sandbox, or directly for the reference run) ---------------------------------


def _records(corpus):
    return [{"dataset": "musdb18hq", "dataset_track_id": t, "split": s, "sha256": h} for t, s, h in corpus["musdb"]]


def compute(phase, paths, commit):
    """The step's computation on the given input paths (sandbox or host). Returns (status,
    {file name: JSON-able value}). Stop messages carry counts only."""
    try:
        corpus = readers.read_corpus(paths["corpus"])
        if phase == "listing":
            return "listed", {"listing.json": {"format": FORMAT, "phase": "listing",
                                               "ambiguous_credits": musdb.list_ambiguous(_records(corpus.value))}}
        registry = readers.read_registry(Path(paths["registry"]).parents[1])
        aliases = readers.read_aliases(paths["aliases"])
        readings = readers.read_readings(paths["readings"])
        decisions = readers.read_decisions(paths["decisions"])
        assessed = [n[4:] for e in registry.value["entries"].values() if e["partition"] in ("development", "validation")
                    for n in e["nodes"] if n.startswith("rec:")]
        titles = readers.read_titles(paths["titles"], assessed)
        records = _records(corpus.value)
        tests = {t for t, s, _h in corpus.value["musdb"] if s == "test"}
        test_rejections = frozenset(p for p in decisions.value if p[0] in tests)
        try:
            result = musdb.match(records, registry.value, aliases.value, titles.value, None, readings.value,
                                 test_rejections)
        except musdb.TestSplitOverlap as stop:
            pairs = review.test_split_pairs(stop)
            return "test-split stop", {
                "review.json": _review_body(pairs, registry.value, test_split_only=True),
                "stop.json": {"format": FORMAT, "stop": "test-split", "summary": review.summary(pairs)}}
    except (readers.ReaderError, musdb.MusdbInputError) as error:
        return "stop", {"stop.json": {"format": FORMAT, "stop": type(error).__name__, "message": str(error)}}
    hashes = {"corpus_manifest": corpus.sha256, "registry": registry.sha256, "aliases": aliases.sha256,
              "titles": titles.sha256, "credit_readings": readings.sha256, "review_decisions": decisions.sha256}
    public_value, opening = commitment.commit(hashes)
    pairs = musdb.review_pairs(result)
    return "completed", {
        "matches.json": {"format": FORMAT, "matches": [[t, s, k, list(types)] for t, s, k, types in result.matches],
                         "sources": [[t, k, list(v)] for (t, k), v in sorted(result.sources.items())],
                         "musdb_records": result.musdb_records, "readings_used": result.readings_used,
                         "test_rejections_applied": sorted(list(p) for p in test_rejections),
                         "input_sha256": hashes, "commitment_opening": opening.to_private_record(),
                         "review_summary": review.summary(pairs)},
        "review.json": _review_body(pairs, registry.value),
        "public.json": {"format": FORMAT, "commit": commit, "inputs_commitment": public_value,
                        "musdb_records": result.musdb_records, "unparsed_records": result.unparsed,
                        "credits_read_by_reading": result.readings_used},
    }


def _review_body(pairs, registry, test_split_only=False):
    return {"format": review.FORMAT, "version": 1,
            "scope": "MUSDB-test pairs after a test-split stop" if test_split_only else "all name-based and file matches",
            "how_to_decide": ("Copy the decision of every pair that is not the same artist into the review-decisions "
                              "file. Leave the others. Nothing can be added here: a missed match is fixed in the "
                              "alias file and step 2 is run again."),
            "pairs": review.review_items(pairs, registry)}


def _write(folder, files):
    for name, value in files.items():
        fd = os.open(Path(folder) / name, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(value, file, indent=1, ensure_ascii=False, sort_keys=True)


def sandbox_main(phase, commit):
    """The registered entry point inside the sandbox."""
    sandbox.check_handles()
    sandbox.not_dumpable()
    status, files = compute(phase, IN, commit)
    _write("/out", files)
    sys.exit(EXIT[status])


# -- the launcher (outside the sandbox) ----------------------------------------------------------------------


def require_clean(repo, commit):
    """The source tree must be clean at the authorised commit (checked outside the sandbox)."""
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    status = subprocess.run(["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True,
                            check=False)
    if head.returncode or status.returncode:
        raise sandbox.SandboxError("the source tree is not a readable git checkout")
    if head.stdout.strip() != commit:
        raise sandbox.SandboxError("the source tree is not at the authorised commit")
    if status.stdout.strip():
        raise sandbox.SandboxError("the source tree has uncommitted changes")


def run(declaration, src, site_packages, lock_path=None, check_commit=True, timeout=1800):
    """Run one phase of step 2 under the lock, sandboxed and traced. Returns an ``Outcome``;
    the private run record is written under ``out_root/name``."""
    with runlock.exclusive(lock_path):
        if check_commit:
            require_clean(Path(src).parent, declaration.commit)
        inputs = declaration.inputs()
        run_dir = Path(declaration.out_root) / declaration.name
        if run_dir.exists():
            raise sandbox.SandboxError("this run's folder already exists")
        run_dir.mkdir(parents=True)
        os.chmod(run_dir, 0o700)
        missing = sum(1 for host in inputs.values() if not Path(host).is_file())
        if missing:
            _record(run_dir, declaration, "stop", None, f"{missing} declared input file(s) are missing", set())
            return Outcome("stop", True, review.STOP, {}, run_dir)
        layout = sandbox.Layout(inputs, run_dir / "out", Path(src), Path(site_packages))
        launched = sandbox.launch(layout, sandbox.startup(MODULE, "sandbox_main", declaration.phase, declaration.commit),
                                  run_dir / "trace" / "strace.log", timeout)
        status = {v: k for k, v in EXIT.items() if k != "listed"}.get(launched.strace_exit, "rejected")
        if status == "completed" and declaration.phase == "listing":
            status = "listed"
        verdict = trace.read(launched.trace.read_text() if launched.trace.exists() else "", list(inputs),
                             launched.strace_exit, allowed_exits=tuple(EXIT.values()))
        files = set(os.listdir(layout.out))
        accepted = status != "rejected" and verdict.accepted and files == EXPECTED.get(status)
        _record(run_dir, declaration, status, verdict.public(), None, files)
        if not accepted:
            status = "rejected"
        if status in ("completed", "listed"):
            sentence = review.RERUN if declaration.follows_stop else None
        else:
            sentence = review.STOP
        return Outcome(status, accepted, sentence, verdict.public(), run_dir)


def _record(run_dir, declaration, status, verdict, note, files):
    """The private run record (host paths included): never published."""
    body = {"format": FORMAT, "name": declaration.name, "commit": declaration.commit, "phase": declaration.phase,
            "status": status, "trace_verdict": verdict, "note": note, "outputs": sorted(files),
            "inputs": {k: str(v) for k, v in declaration.inputs().items()}, "follows_stop": declaration.follows_stop}
    fd = os.open(Path(run_dir) / "record.json", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as file:
        json.dump(body, file, indent=1, sort_keys=True)
