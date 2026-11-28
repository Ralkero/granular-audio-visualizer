"""A5.2 Step 2a, S2a.7: the bubblewrap sandbox and the live-trace reader, run for real on this
machine with generated files only (protocol Amendment 3, A3.4 and the Step 2a checks of A3.10).

These tests start bubblewrap under strace. They are skipped where either is missing; they never
change a system setting and read no project data. The sandbox is the proposed configuration for
synthetic Step 2a testing, not a proven security boundary.
"""

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gmv.character.research.presence_model import sandbox, trace

REPO = Path(__file__).resolve().parents[1]
SITE = Path(sys.prefix) / "lib" / "python3.14" / "site-packages"
CHECKS = "gmv.character.research.presence_model.sandbox_checks"
INPUT = "/in/corpus/manifest.json"

pytestmark = pytest.mark.skipif(
    not (shutil.which("bwrap") and shutil.which("strace") and Path(sandbox.INTERPRETER).exists()
         and all((SITE / p).exists() for p in sandbox.PACKAGES)),
    reason="bubblewrap, strace, the system interpreter or the application environment's packages are missing")


@pytest.fixture
def inputs(tmp_path):
    path = tmp_path / "host-inputs" / "manifest.json"
    path.parent.mkdir()
    path.write_text('{"generated": true}')
    return {INPUT: path}


def layout(tmp_path, inputs, name):
    return sandbox.Layout(inputs=inputs, out=tmp_path / f"out-{name}", src=REPO / "src", site_packages=SITE)


def run(tmp_path, inputs, probe, *args, extra_fds=()):
    lay = layout(tmp_path, inputs, probe)
    result = sandbox.launch(lay, sandbox.startup(CHECKS, probe, *args), tmp_path / f"trace-{probe}.txt",
                            timeout=120, extra_fds=extra_fds)
    verdict = trace.read(result.trace.read_text(), list(inputs), result.strace_exit)
    lines = [json.loads(x) for x in result.stdout.decode().splitlines() if x.strip()]
    return result, verdict, (lines[-1] if lines else None), lay


def test_a_full_walk_finds_exactly_the_mounted_paths(tmp_path, inputs):
    _result, _verdict, seen, _lay = run(tmp_path, inputs, "walk")
    assert seen["top"] == ["dev", "etc", "gmv", "in", "lib", "lib64", "out", "proc", "usr"]
    listing = seen["listing"]
    assert listing["/etc"] == ["ld.so.cache"]
    assert listing["/usr"] == ["bin", "lib"] and listing["/usr/bin"] == ["python3.14"]
    assert listing["/gmv"] == ["site-packages", "src"]
    assert listing["/gmv/site-packages"] == sorted(sandbox.PACKAGES)
    assert listing["/out"] == []
    assert seen["inputs"] == [INPUT]
    assert not {"tmp", "home", "root", "run", "sys", "var"} & set(seen["top"])


def test_writes_fail_everywhere_except_out_and_are_flagged(tmp_path, inputs):
    _result, verdict, seen, lay = run(tmp_path, inputs, "writes")
    assert [k for k, v in seen.items() if v == "written"] == ["/out/ok"]
    assert (lay.out / "ok").read_text() == "x"
    assert not verdict.accepted and verdict.violations["write outside the allowed classes"] >= 7
    assert not (tmp_path / "host-inputs" / "x").exists()


def test_every_connection_fails_and_is_flagged(tmp_path, inputs):
    _result, verdict, seen, _lay = run(tmp_path, inputs, "connections")
    assert set(seen) == {"loopback", "outside", "host-socket-file", "abstract-socket"}
    assert "connected" not in seen.values()
    assert verdict.violations["a network call"] >= 4 and not verdict.accepted


def test_proc_shows_only_the_sandbox_and_a_shell_cannot_start(tmp_path, inputs):
    _r, _v, seen, _lay = run(tmp_path, inputs, "processes")
    assert seen["pids"] == [1, 2]  # bubblewrap's init and the step
    _r, verdict, seen, _lay = run(tmp_path, inputs, "shell")
    assert seen["shell"] == "FileNotFoundError"
    assert verdict.violations["a child process"] >= 1 and not verdict.accepted


def test_a_clean_run_has_exactly_three_handles_and_is_accepted(tmp_path, inputs):
    result, verdict, seen, _lay = run(tmp_path, inputs, "handles")
    assert seen == {"handles": [0, 1, 2]}  # the trace file's handle is not inherited
    assert result.strace_exit == 0 and verdict.accepted and verdict.complete
    assert verdict.counts["step:start"] == 1 and verdict.unresolved == 0


def test_a_planted_extra_handle_stops_the_step(tmp_path, inputs):
    planted = os.open(tmp_path / "planted.txt", os.O_CREAT | os.O_WRONLY, 0o600)
    os.set_inheritable(planted, True)
    try:
        result, verdict, seen, _lay = run(tmp_path, inputs, "handles", extra_fds=(planted,))
    finally:
        os.close(planted)
    assert seen["handles"] == "refused" and result.strace_exit != 0 and not verdict.accepted


def test_a_forced_crash_stores_no_dump(tmp_path, inputs):
    _r, verdict, seen, _lay = run(tmp_path, inputs, "dumpable")
    assert seen == {"dumpable": 0}
    result, verdict, _seen, _lay = run(tmp_path, inputs, "crash")
    assert result.strace_exit != 0 and not verdict.core_dumped  # strace reports "(core dumped)" when one is written
    assert "killed by SIGABRT" in result.trace.read_text()


def test_an_io_uring_call_is_flagged(tmp_path, inputs):
    _r, verdict, _seen, _lay = run(tmp_path, inputs, "io_uring")
    assert verdict.violations["io_uring"] == 1 and not verdict.accepted


def _beats(path):
    try:
        return int(path.read_text() or 0)
    except (OSError, ValueError):
        return 0


def _wait_for_beats(path, minimum=3, timeout=30):
    end = time.monotonic() + timeout
    while time.monotonic() < end and _beats(path) < minimum:
        time.sleep(0.1)
    assert _beats(path) >= minimum, "the probe never started"


def _stopped(path):
    before = _beats(path)
    time.sleep(1.0)
    return _beats(path) == before


def test_killing_strace_stops_the_step(tmp_path, inputs):
    lay = layout(tmp_path, inputs, "kill-strace")
    argv = sandbox.prepare(lay, sandbox.startup(CHECKS, "heartbeat", "60"), tmp_path / "trace-kill.txt")
    process = sandbox.popen(argv)
    try:
        _wait_for_beats(lay.out / "beat")
        process.send_signal(signal.SIGKILL)
        process.wait(timeout=30)
        assert _stopped(lay.out / "beat")
    finally:
        process.kill()


def test_killing_the_launcher_stops_the_step(tmp_path, inputs):
    lay = layout(tmp_path, inputs, "kill-launcher")
    launcher = subprocess.Popen([sys.executable, "-c", (
        "import sys; from pathlib import Path; from gmv.character.research.presence_model import sandbox; "
        f"lay = sandbox.Layout(inputs={{'{INPUT}': Path('{inputs[INPUT]}')}}, out=Path('{lay.out}'), "
        f"src=Path('{REPO / 'src'}'), site_packages=Path('{SITE}')); "
        f"sandbox.launch(lay, sandbox.startup('{CHECKS}', 'heartbeat', '60'), Path('{tmp_path / 'trace-l.txt'}'), 120)")],
        cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    try:
        _wait_for_beats(lay.out / "beat")
        launcher.send_signal(signal.SIGKILL)
        launcher.wait(timeout=30)
        assert _stopped(lay.out / "beat")
    finally:
        launcher.kill()


def test_an_existing_staging_folder_and_an_input_outside_in_are_refused(tmp_path, inputs):
    lay = layout(tmp_path, inputs, "exists")
    lay.out.mkdir()
    with pytest.raises(sandbox.SandboxError):
        sandbox.prepare(lay, sandbox.startup(CHECKS, "walk"), tmp_path / "t.txt")
    bad = sandbox.Layout(inputs={"/etc/passwd": inputs[INPUT]}, out=tmp_path / "o", src=REPO / "src", site_packages=SITE)
    with pytest.raises(sandbox.SandboxError):
        sandbox.bwrap_argv(bad, "pass")
    with pytest.raises(sandbox.SandboxError):
        sandbox.startup(CHECKS, "walk; import os")
