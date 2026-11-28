"""A5.2 Step 4, S4.7: the recorders, the launcher and the synthetic-data guard (protocol
Amendment 3, A3.8 C1, V6, B9; A3.10 "Timing and resources"). Linux only. Generated jobs only:
small Python processes that allocate, share, fork, sleep or swap out their own memory."""

import builtins
import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from gmv.character.research.presence_model import launcher, recorders, runlock

pytestmark = pytest.mark.skipif(not sys.platform.startswith("linux"), reason="the recorders read /proc")
MIB = 1024**2


def start(script):
    """Start a Python helper that prints "ready" once its memory is in place."""
    process = subprocess.Popen([sys.executable, "-c", textwrap.dedent(script)], stdout=subprocess.PIPE, text=True)
    assert process.stdout.readline().strip() == "ready"
    return process


def stop(process):
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        process.kill()
    process.wait()


SHARED = """
    import mmap, os, sys, time
    size = {size}
    block = mmap.mmap(-1, size, flags=mmap.MAP_SHARED)
    block.write(b"x" * size)
    os.setpgid(0, 0)
    for _ in range({children}):
        if os.fork() == 0:
            total = sum(block[i] for i in range(0, size, 4096))  # every child touches every page
            time.sleep(60)
            os._exit(0)
    time.sleep(1.0)
    print("ready", flush=True)
    time.sleep(60)
"""


def test_a_block_shared_by_four_processes_is_counted_once():
    process = start(SHARED.format(size=128 * MIB, children=3))
    try:
        used, count = recorders.tree_memory(process.pid)
        assert count == 4
        assert 128 * MIB * 0.9 < used < 128 * MIB * 1.6  # once, plus the interpreters (not four times)
    finally:
        stop(process)


COPY_ON_WRITE = """
    import os, time
    data = bytearray(b"y" * ({size}))
    os.setpgid(0, 0)
    for _ in range(2):
        if os.fork() == 0:
            time.sleep(60)
            os._exit(0)
    time.sleep(1.0)
    print("ready", flush=True)
    time.sleep(60)
"""


def test_copy_on_write_children_are_found_and_not_double_counted():
    process = start(COPY_ON_WRITE.format(size=128 * MIB))
    try:
        used, count = recorders.tree_memory(process.pid)
        assert count == 3
        assert 128 * MIB * 0.9 < used < 128 * MIB * 1.6
    finally:
        stop(process)


SWAPPED = """
    import mmap, time
    size = {size}
    block = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE)
    block.write(bytes(range(256)) * (size // 256))
    block.madvise(getattr(mmap, "MADV_PAGEOUT", 21))  # 21 is Linux's MADV_PAGEOUT
    time.sleep(1.0)
    print("ready", flush=True)
    time.sleep(60)
"""


def swap_present():
    try:
        return any(int(line.split()[1]) > 0 for line in Path("/proc/meminfo").read_text().splitlines()
                   if line.startswith("SwapTotal:"))
    except OSError:
        return False


@pytest.mark.skipif(not swap_present(), reason="needs swap")
def test_swapped_out_memory_is_still_counted():
    process = start(SWAPPED.format(size=64 * MIB))
    try:
        pss, swap = recorders.memory_of(process.pid)
        if swap < 32 * MIB:
            pytest.skip("the kernel kept the pages in memory")
        assert pss + swap > 64 * MIB * 0.9  # the swapped-out pages still count
        assert recorders.tree_memory(process.pid)[0] == pss + swap
    finally:
        stop(process)


def test_the_memory_stop_fires_at_a_lowered_limit(tmp_path):
    job = [sys.executable, "-c", "import time; x = bytearray(200 * 1024 * 1024); time.sleep(30)"]
    outcome = launcher.run_job(job, cap_s=20, memory_limit=50 * MIB, interval=0.2, lock_path=tmp_path / "l")
    assert outcome.status == "memory" and outcome.peak_bytes > 50 * MIB and outcome.elapsed_s < 15


def test_a_cap_hit_is_a_stop(tmp_path):
    outcome = launcher.run_job([sys.executable, "-c", "import time; time.sleep(30)"], cap_s=1.0, interval=0.2,
                               lock_path=tmp_path / "l")
    assert outcome.status == "cap" and outcome.elapsed_s < 10


def test_a_completed_job_has_process_time_cpu_time_and_aggregate_samples_only(tmp_path):
    job = [sys.executable, "-c", "import time; t = time.time()\nwhile time.time() - t < 1.2: pass"]
    outcome = launcher.run_job(job, cap_s=20, interval=0.2, lock_path=tmp_path / "l")
    assert outcome.status == "completed" and outcome.returncode == 0
    assert 1.0 < outcome.elapsed_s < 10 and outcome.cpu_s is not None and outcome.cpu_s > 0.5
    assert outcome.samples and max(s.cpu_job for s in outcome.samples) > 0.5
    for s in outcome.samples:
        assert all(isinstance(v, (int, float)) or v is None for v in vars(s).values())  # numbers only


def test_the_recorder_reads_no_name_command_line_or_environment(monkeypatch, tmp_path):
    opened = []
    real = builtins.open

    def spy(path, *args, **kwargs):
        opened.append(str(path))
        return real(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", spy)
    process = start(SHARED.format(size=4 * MIB, children=1))
    try:
        r = recorders.Recorder(process.pid, interval=0.1)
        r.start()
        time.sleep(0.5)
        r.stop()
    finally:
        stop(process)
    mine = [p for p in opened if p.startswith(("/proc", "/sys"))]
    assert mine and not any(p.endswith(("/cmdline", "/comm", "/environ", "/status", "/maps")) for p in mine)
    assert all(p.startswith(("/proc/stat", "/proc/meminfo", "/proc/vmstat", "/sys/devices/system/cpu/"))
               or p.endswith(("/stat", "/smaps_rollup", "/children")) for p in mine)


def test_disturbance_thresholds_and_swap_as_its_own_reason():
    def s(other, swap=0):
        return recorders.Sample(0.0, other + 1, 1.0, other, None, swap, None, 0, 1)

    assert recorders.disturbed([s(0.5)] * 10) == (False, ())
    assert recorders.disturbed([s(1.2)] * 10)[0]
    burst = [s(4.5)] * 2 + [s(0.0)] * 8  # over a tenth of the samples above four
    assert "four" in recorders.disturbed(burst)[1][0]
    assert not recorders.disturbed([s(4.5)] + [s(0.0)] * 9)[0]  # exactly a tenth is not "over"
    swapped = recorders.disturbed([s(0.0)] * 9 + [s(0.0, swap=3)])
    assert swapped == (True, ("swap activity occurred during the run",))


def test_the_launcher_takes_the_lock_and_a_second_runner_refuses(tmp_path):
    lock = tmp_path / "experiment.lock"
    with runlock.exclusive(lock):
        with pytest.raises(runlock.LockHeld):
            launcher.run_job([sys.executable, "-c", "pass"], cap_s=5, lock_path=lock)


def test_jobs_die_with_their_launcher(tmp_path):
    pid_file = tmp_path / "job.pid"
    script = textwrap.dedent(f"""
        import sys
        from gmv.character.research.presence_model import launcher
        job = [sys.executable, "-c", "import os, time; open({str(pid_file)!r}, 'w').write(str(os.getpid())); time.sleep(60)"]
        launcher.run_job(job, cap_s=60, lock_path={str(tmp_path / 'l')!r})
    """)
    outer = subprocess.Popen([sys.executable, "-c", script])
    try:
        deadline = time.monotonic() + 10
        while not pid_file.exists() or not pid_file.read_text():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        job = int(pid_file.read_text())
        outer.kill()
        outer.wait()
        deadline = time.monotonic() + 5
        while Path(f"/proc/{job}").exists() and "Z" not in Path(f"/proc/{job}/stat").read_text().split(")")[-1][:3]:
            assert time.monotonic() < deadline, "the job outlived its launcher"
            time.sleep(0.05)
    finally:
        if outer.poll() is None:
            outer.kill()


def test_the_synthetic_runner_refuses_project_data(tmp_path):
    root = launcher.make_synthetic_root(tmp_path / "generated")
    (root / "audio.wav").write_bytes(b"generated")
    launcher.check_synthetic([root / "audio.wav"], root)  # accepted
    outside = tmp_path / "elsewhere.wav"
    outside.write_bytes(b"x")
    cases = [([outside], root),                                         # outside the generated folder
             ([root / "missing.wav"], root)]                            # does not exist
    (root / "link.wav").symlink_to(outside)
    cases.append(([root / "link.wav"], root))                           # a link leading out
    unmarked = tmp_path / "unmarked"
    unmarked.mkdir()
    (unmarked / "a.wav").write_bytes(b"x")
    cases.append(([unmarked / "a.wav"], unmarked))                      # not a marked folder
    private = launcher.make_synthetic_root(tmp_path / launcher.PRIVATE_FOLDERS[0] / "g")
    (private / "a.wav").write_bytes(b"x")
    cases.append(([private / "a.wav"], private))                        # inside a private folder
    for inputs, base in cases:
        with pytest.raises(launcher.SyntheticDataError):
            launcher.check_synthetic(inputs, base)
    with pytest.raises(launcher.SyntheticDataError):
        launcher.run_job([sys.executable, "-c", "raise SystemExit(9)"], cap_s=5, synthetic_root=root, inputs=[outside],
                         lock_path=tmp_path / "l")
