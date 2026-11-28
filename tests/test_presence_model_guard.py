"""A5.2 Step 1: a guard that no A5.2 test or test helper refers to the repository's private
evidence folder (the git-ignored folder that holds the owner's real runtime data).

It reads the sources themselves on every run: every ``test_presence_model_*.py`` file, the
synthetic-project builder, the spy-test helpers and the timing toy job. The folder name is
assembled below so that this file does not contain it.
"""

import re
from pathlib import Path

TESTS = Path(__file__).resolve().parent
NAME = "lo" + "cal" + "-" + "evid" + "ence"
# the name joined by "-", "_" or a space, or split across string pieces (see the spellings below)
FORBIDDEN = re.compile("lo" + "cal" + r"[\W_]{0,8}" + "evid" + "ence", re.IGNORECASE)


def a52_sources():
    return sorted({*TESTS.glob("test_presence_model_*.py"), *TESTS.glob("privacy_*.py"),
                   TESTS / "presence_synth.py", TESTS / "step2_synth.py", TESTS / "timing_toy_job.py"})


def test_the_guard_reads_every_a52_test_source():
    names = {p.name for p in a52_sources()}
    assert {Path(__file__).name, "test_presence_model_privacy.py", "privacy_harness.py", "privacy_targets.py",
            "privacy_bootstrap.py", "presence_synth.py", "timing_toy_job.py"} <= names
    assert len(names) >= 20
    assert all(p.is_file() and p.stat().st_size > 0 for p in a52_sources())


def test_no_a52_test_source_refers_to_the_private_evidence_folder():
    offending = sorted(p.name for p in a52_sources() if FORBIDDEN.search(p.read_text(encoding="utf-8")))
    assert offending == [], "an A5.2 test source refers to the private evidence folder"


def test_the_guard_recognises_the_usual_spellings():
    for text in (f"{NAME}/sonic-character", NAME.upper().replace("-", "_"), NAME.replace("-", " "),
                 'Path("lo' + 'cal") / "evid' + 'ence"', "'lo" + "cal' + '-evid" + "ence'"):
        assert FORBIDDEN.search(text), "a spelling of the folder name was not recognised"
    assert not FORBIDDEN.search("local variables and evidence hashes are separate words here")
