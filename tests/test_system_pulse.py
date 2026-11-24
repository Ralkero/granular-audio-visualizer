"""Linux capture: the default output's monitor through the PulseAudio service PipeWire provides.

Everything here runs against ``fake_pulse.py``, a stand-in for ``pactl`` and ``parec``: no
sound server is needed and nothing real is recorded (``conftest.py`` points the real client
programs at a server that does not exist). The stand-in's audio is a known signal, so what
arrives can be checked sample by sample.
"""

import json
import os
import re
import sys
import threading
from pathlib import Path
from threading import Event, Thread
from time import perf_counter, sleep

import numpy as np
import psutil
import pytest

from gmv import host, sources
from gmv.buffer import BoundedQueue
from gmv.sources import system_pulse
from gmv.sources.system_pulse import DeliveryClock, PulseProblem, PulseSystemAudioSource

FAKE = Path(__file__).with_name("fake_pulse.py")
SERVER = "PulseAudio (on PipeWire 1.6.11)"
CARD = "alsa_output.usb-Maker_Card_Name-00.analog-stereo"
# Names come from hardware and users; they are never code.
ODD = "bluez_output.Kopfhörer (Wohnzimmer) 'A&B' $HOME; rm -rf ~ .a2dp-sink"
MIC = {
    "name": "alsa_input.usb-Maker_Card_Name-00.analog-stereo",
    "index": 9,
    "monitor_source": "",
    "sample_specification": "s24le 2ch 48000Hz",
}


def sink(name, description, rate=48000):
    return {
        "name": name,
        "description": description,
        "monitor_source": f"{name}.monitor",
        "sample_specification": f"s24le 2ch {rate}Hz",
    }


def monitor(name, index, rate=48000):
    return {
        "name": f"{name}.monitor",
        "index": index,
        "monitor_source": name,  # pactl's JSON: the output this source monitors
        "sample_specification": f"s24le 2ch {rate}Hz",
    }


class FakePulse:
    """The server the stand-in pretends to be. Two outputs: a sound card (the default,
    48 kHz) and headphones with an awkward name (44.1 kHz); and a microphone."""

    def __init__(self, folder, monkeypatch):
        self.folder = folder
        folder.mkdir(parents=True, exist_ok=True)
        monkeypatch.setenv("FAKE_PULSE_DIR", str(folder))
        monkeypatch.setattr(
            system_pulse,
            "COMMANDS",
            {
                "pactl": (sys.executable, str(FAKE), "pactl"),
                "parec": (sys.executable, str(FAKE), "parec"),
            },
        )
        self.state = {
            "server": True,
            "server_name": SERVER,
            "default_sink": CARD,
            "sinks": [sink(CARD, "Sound Card Analog Stereo"), sink(ODD, "Headphones", 44100)],
            "sources": [monitor(CARD, 1), MIC, monitor(ODD, 2, 44100)],
        }
        self.write()

    def write(self):
        path = self.folder / "state.json"
        temporary = path.with_name("state.tmp")
        temporary.write_text(json.dumps(self.state), encoding="utf-8")
        os.replace(temporary, path)

    def set(self, **changes):
        self.state.update(changes)
        self.write()

    def records(self):
        """Every stream ever opened: (pid, what parec was asked and where it records)."""
        found = []
        for path in sorted((self.folder / "streams").glob("*.json")):
            found.append((int(path.stem), json.loads(path.read_text(encoding="utf-8"))))
        return found

    def alive(self):
        """The stand-in parec processes still running."""
        running = []
        for pid, record in self.records():
            try:
                if psutil.Process(pid).status() != psutil.STATUS_ZOMBIE:
                    running.append((pid, record))
            except psutil.Error:
                pass
        return running


@pytest.fixture
def pulse(tmp_path, monkeypatch):
    return FakePulse(tmp_path / "server", monkeypatch)


def wait_until(condition, timeout=10.0):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if condition():
            return True
        sleep(0.02)
    return False


class Running:
    """A source run in a thread of its own, as the engine runs it."""

    def __init__(self, source, capacity=1024):
        self.source, self.output, self.stop = source, BoundedQueue(capacity), Event()
        self.states, self.received = [], []
        self.origin = source.timeline_origin = perf_counter()
        self.thread = Thread(target=source.run, args=(self.output, self.stop, self.status))
        self.thread.start()

    def status(self, state, detail):
        self.states.append((state, detail))

    def said(self, text):
        return any(text in detail for _, detail in self.states)

    def collect(self, enough, timeout=15.0):
        deadline = perf_counter() + timeout
        while not enough(self.received):
            assert perf_counter() < deadline, f"not enough audio: {self.states[-5:]}"
            block = self.output.get(timeout=0.05)
            if block is not None:
                self.received.append(block)
        return self.received

    def close(self):
        self.stop.set()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive(), "the source did not stop"


def samples(blocks):
    """The audio of ``blocks``, and the stand-in's frame index (mod 1000) of every frame."""
    audio = np.concatenate(
        [np.frombuffer(b.data, "<f4").reshape(b.frames, b.channels) for b in blocks]
    )
    return audio, np.rint(audio[:, 0] * 2000 + 500).astype(int) % 1000


# -- the default output and its monitor ----------------------------------------------------------


def test_the_default_output_and_its_monitor_are_what_the_server_says(pulse):
    assert system_pulse.default_output() == system_pulse.Output(
        CARD, "Sound Card Analog Stereo", f"{CARD}.monitor", 48000, SERVER
    )
    pulse.set(default_sink=ODD)
    found = system_pulse.check()
    assert (found.sink, found.monitor, found.rate) == (ODD, f"{ODD}.monitor", 44100)


def without_monitor(state):
    state["sinks"][0]["monitor_source"] = ""


def monitor_not_listed(state):
    state["sources"] = [MIC, monitor(ODD, 2, 44100)]


def monitor_is_a_microphone(state):
    state["sinks"][0]["monitor_source"] = MIC["name"]


@pytest.mark.parametrize(
    "change, said",
    [
        ({"default_sink": ""}, f"{SERVER} has no default output"),
        ({"default_sink": "@DEFAULT_SINK@"}, f"{SERVER} has no default output"),
        ({"default_sink": "gone"}, "The default output gone is not among the outputs"),
        (without_monitor, "has no monitor source, so what it plays cannot be recorded"),
        (monitor_not_listed, "is not among the sources"),
        (monitor_is_a_microphone, "is not the monitor of the default output"),
        (
            {"server": False},
            "No PulseAudio-compatible sound server answered (Connection failure: Connection "
            "refused). On CachyOS that server is PipeWire's pipewire-pulse",
        ),
        ({"json": False}, "needs pactl from PulseAudio 16 or newer"),
    ],
)
def test_what_keeps_capture_from_starting_is_said_in_a_sentence(pulse, change, said):
    if callable(change):
        change(pulse.state)
        pulse.write()
    else:
        pulse.set(**change)
    with pytest.raises(PulseProblem, match=re.escape(said)):
        system_pulse.check()
    with pytest.raises(sources.Unavailable):  # the application's refusal, not a crash
        system_pulse.system_audio_source()
    assert not pulse.records()  # nothing was opened


@pytest.mark.parametrize("missing", [("pactl",), ("parec",), ("pactl", "parec")])
def test_missing_client_programs_are_named_with_the_package_that_has_them(
    pulse, monkeypatch, missing
):
    commands = dict(system_pulse.COMMANDS)
    for name in missing:
        commands[name] = (f"gmv-test-no-such-{name}",)
    monkeypatch.setattr(system_pulse, "COMMANDS", commands)
    assert system_pulse.missing_programs() == list(missing)
    with pytest.raises(PulseProblem) as refused:
        system_pulse.check()
    message = str(refused.value)
    assert " and ".join(missing) + (" is" if len(missing) == 1 else " are") in message
    assert "sudo pacman -S libpulse" in message
    assert system_pulse.programs_problem() == message


def test_sample_rates_are_read_from_the_server_s_specifications():
    assert system_pulse.sample_rate("s24le 2ch 48000Hz") == 48000
    assert system_pulse.sample_rate("float32le 6ch 44100Hz") == 44100
    for unclear in ("", None, "s16le 2ch", "s16le 2ch 1Hz", "s16le 2ch 999999Hz"):
        assert system_pulse.sample_rate(unclear) is None


# -- recording ---------------------------------------------------------------------------------


def test_the_monitor_is_recorded_as_float32_stereo_on_a_contiguous_sample_clock(pulse):
    source = system_pulse.system_audio_source()
    run = Running(source)
    try:
        blocks = run.collect(lambda got: len(got) >= 40)
    finally:
        run.close()
    assert run.output.closed
    assert {(b.sample_rate, b.channels, b.frames, b.segment) for b in blocks} == {
        (48000, 2, 480, 1)
    }
    assert {b.timestamp_quality for b in blocks} == {"pulse_delivery_anchored_sample_clock"}
    assert blocks[0].discontinuity and not any(b.discontinuity for b in blocks[1:])
    for a, b in zip(blocks, blocks[1:], strict=False):
        assert b.timestamp == pytest.approx(a.timestamp + a.frames / a.sample_rate, abs=1e-9)
    for block in blocks:
        assert block.capture_monotonic == pytest.approx(run.origin + block.timestamp, abs=1e-9)
        # not later than delivery: the block's end was captured before it was received
        assert block.capture_monotonic + block.frames / block.sample_rate <= (
            block.received_monotonic + system_pulse.AHEAD_S
        )
    # The server's samples, unchanged: left then right, no gain, none missing or doubled.
    audio, index = samples(blocks)
    expected = ((index - 500) / 2000).astype(np.float32)
    assert np.array_equal(audio[:, 0], expected) and np.array_equal(audio[:, 1], -expected)
    assert np.all(np.diff(index) % 1000 == 1)
    # The first 100 ms were measured to anchor the clock, not published.
    clock = source.describe()["clock"]
    assert clock["settle_discarded_frames"] == 4320 and index[0] == 4320 % 1000
    assert clock["runs"] == 1 and clock["behind_resyncs"] == clock["ahead_resyncs"] == 0


def test_parec_is_asked_for_the_monitor_by_name_and_told_not_to_move(pulse):
    pulse.set(default_sink=ODD)
    run = Running(system_pulse.system_audio_source())
    try:
        blocks = run.collect(lambda got: len(got) >= 3)
    finally:
        run.close()
    assert {b.sample_rate for b in blocks} == {44100}  # the monitor's own rate
    [(pid, record)] = pulse.records()
    arguments = record["arguments"]
    assert f"--device={ODD}.monitor" in arguments  # one argument, exactly as named
    for wanted in (
        "--raw",
        "--format=float32le",
        "--channels=2",
        "--rate=44100",
        "--latency-msec=10",
        "--property=node.dont-reconnect=true",
        "--property=node.dont-fallback=true",
        "--property=node.dont-move=true",
    ):
        assert wanted in arguments
    assert record["device"] != MIC["name"]


def test_stopping_ends_the_recording_process_and_its_threads_every_time(pulse):
    for cycle in range(6):
        source = PulseSystemAudioSource()
        run = Running(source)
        if cycle % 2:
            run.collect(lambda got: len(got) >= 2)  # some stop with audio flowing
        else:
            sleep(0.05 * cycle)  # some before any arrived
        began = perf_counter()
        run.close()
        assert perf_counter() - began < 3 and run.output.closed
        assert not pulse.alive()
    assert not [t for t in threading.enumerate() if t.name.startswith("pulse-capture")]


def test_a_stream_that_ends_by_itself_is_reopened_as_a_new_segment(pulse):
    pulse.set(die_after=0.6)
    source = PulseSystemAudioSource()
    run = Running(source)
    most = 0
    try:

        def two_segments(got):
            nonlocal most
            most = max(most, len(pulse.alive()))
            return {b.segment for b in got} >= {1, 2}

        blocks = run.collect(two_segments)
        pulse.set(die_after=None)
    finally:
        run.close()
    assert most <= 1  # never two streams at once
    assert run.said("ended (Stream error: Connection terminated)") and source.recoveries >= 1
    first_end = max(b.timestamp + b.frames / b.sample_rate for b in blocks if b.segment == 1)
    second = next(b for b in blocks if b.segment == 2)
    assert second.discontinuity and second.timestamp >= first_end - 1e-9  # never back


def test_capture_waits_for_a_sound_server_that_went_away_and_resumes(pulse):
    source = PulseSystemAudioSource()
    run = Running(source)
    try:
        run.collect(lambda got: len(got) >= 5)
        pulse.set(server=False)  # the stream ends and nothing answers
        assert wait_until(lambda: run.said("No PulseAudio-compatible sound server answered"))
        assert wait_until(lambda: not pulse.alive())
        sleep(1.0)  # a few attempts, spaced out
        pulse.set(server=True)
        run.collect(lambda got: any(b.segment == 2 for b in got))
    finally:
        run.close()
    retries = [detail for state, detail in run.states if state == "recovering"]
    assert retries and all(
        detail.endswith("s.") and "Trying again in" in detail for detail in retries
    )
    assert len(retries) <= 6  # spaced out, not a tight loop


def test_when_the_output_disappears_capture_moves_to_the_new_default(pulse):
    source = PulseSystemAudioSource()
    run = Running(source)
    try:
        run.collect(lambda got: len(got) >= 5)
        # The card goes away before the server has named another default.
        pulse.set(sinks=[sink(ODD, "Headphones", 44100)], sources=[MIC, monitor(ODD, 2, 44100)])
        assert wait_until(lambda: run.said("is not among the outputs"))
        pulse.set(default_sink=ODD)
        blocks = run.collect(lambda got: any(b.sample_rate == 44100 for b in got))
    finally:
        run.close()
    first_end = max(b.timestamp + b.frames / b.sample_rate for b in blocks if b.segment == 1)
    moved = next(b for b in blocks if b.sample_rate == 44100)
    assert moved.segment > 1 and moved.discontinuity and moved.timestamp >= first_end - 1e-9
    assert source.describe()["monitor"] == f"{ODD}.monitor"
    assert [r["device"] for _, r in pulse.records()][-1] == f"{ODD}.monitor"
    assert all(r["device"] != MIC["name"] for _, r in pulse.records())  # never the microphone


def test_a_new_default_output_is_followed_without_a_restart(pulse):
    source = PulseSystemAudioSource()
    run = Running(source)
    most = 0
    try:
        run.collect(lambda got: len(got) >= 5)
        pulse.set(default_sink=ODD)

        def switched(got):
            nonlocal most
            most = max(most, len(pulse.alive()))
            return any(b.sample_rate == 44100 for b in got)

        blocks = run.collect(switched)
    finally:
        run.close()
    assert most <= 1 and source.output_switches == 1 and source.recoveries == 0
    assert run.said("The default output changed")
    assert [b.segment for b in blocks if b.sample_rate == 44100][0] == 2
    assert [r["device"] for _, r in pulse.records()] == [f"{CARD}.monitor", f"{ODD}.monitor"]


def test_a_stream_the_server_moved_to_another_source_is_stopped(pulse):
    pulse.set(move_to=MIC["name"])  # as if the server had put it on the microphone
    source = PulseSystemAudioSource()
    run = Running(source)
    try:
        assert wait_until(lambda: run.said(f"was moved to {MIC['name']}"), timeout=10)
        assert wait_until(lambda: not pulse.alive() or source.streams_opened > 1)
    finally:
        run.close()
    assert source.recoveries >= 1


def test_a_slow_reader_loses_the_oldest_blocks_and_capture_never_waits(pulse):
    run = Running(PulseSystemAudioSource(), capacity=4)
    try:
        assert wait_until(lambda: run.output.dropped > 20)
        assert len(run.output) == 4  # bounded
        newest = run.collect(lambda got: len(got) >= 8)
    finally:
        run.close()
    # what is left is recent: the stand-in's frames still advance one by one within a block
    audio, index = samples(newest[-1:])
    assert np.all(np.diff(index) % 1000 == 1)


# -- the clock -----------------------------------------------------------------------------------


def run_clock(clock, deliveries, rate=48000, frames=480):
    """Feed ``clock`` blocks delivered at ``deliveries``; the published ones, with the
    delivery time of each."""
    published = []
    for now in deliveries:
        stamped = clock.stamp(frames, rate, now)
        if stamped is not None:
            published.append((*stamped, now))
    return published


def check_published(published, origin, frames=480, rate=48000):
    end = None
    for timestamp, mono, gap, now in published:
        assert mono == pytest.approx(origin + timestamp)
        # not after delivery, beyond the tolerance a new run starts at
        assert origin + timestamp + frames / rate <= now + system_pulse.AHEAD_S + 1e-9
        if end is not None:
            assert timestamp >= end - 1e-9  # never back
            if not gap:
                assert timestamp == pytest.approx(end, abs=1e-9)  # contiguous within a run
        end = timestamp + frames / rate


def burst_deliveries(start, blocks, rng, quantum=1024, frames=480, rate=48000):
    """When PipeWire delivers ``blocks`` 10 ms blocks: a block is there once the server
    period (``quantum`` frames, 21.3 ms by default) holding its last frame is complete, plus
    a little scheduling delay, and now and then a longer one. Measured on CachyOS: data
    arrives in whole periods, about 21.3 ms apart."""
    times, late = [], 0.0
    for i in range(blocks):
        period_end = -(-(i + 1) * frames // quantum) * quantum / rate
        delay = rng.uniform(0.0003, 0.002) + (rng.uniform(0, 0.008) if rng.random() < 0.03 else 0)
        late = max(late, start + period_end + delay)  # in order: a pipe
        times.append(late)
    return times


def test_the_clock_anchors_on_the_earliest_delivery_and_stays_contiguous():
    rng = np.random.default_rng(7)
    origin, start = 100.0, 100.5  # stream frame 0 captured half a second into the session
    clock = DeliveryClock(origin)
    published = run_clock(clock, burst_deliveries(start, 3000, rng))
    check_published(published, origin)
    assert published[0][2] and not any(gap for _, _, gap, _ in published[1:])
    first = round((published[0][0] - (start - origin)) / 0.01)  # its block number
    for number, (timestamp, *_) in enumerate(published, first):
        error = timestamp - (start - origin + number * 0.01)  # later than the truth by
        assert 0.0 <= error <= 0.004  # the period's own wait is not counted against it
    assert clock.runs == 1 and clock.behind == clock.ahead == 0
    # The settle sees the floor to within the misalignment of blocks and periods (about
    # 1.3 ms here) plus scheduling delay; well inside the tolerance either way.
    assert -system_pulse.AHEAD_S * 1000 < clock.floor_ms < 3 and clock.max_lateness_ms < 32


def test_lost_audio_starts_a_new_run_further_on():
    origin, start = 0.0, 1.0
    blocks = [start + (i + 1) * 0.01 + 0.002 for i in range(300)]
    # then 50 ms of audio never arrives: the same frames count for times 50 ms later
    blocks += [start + (i + 1) * 0.01 + 0.052 for i in range(300, 900)]
    clock = DeliveryClock(origin)
    published = run_clock(clock, blocks)
    check_published(published, origin)
    assert clock.behind == 1 and clock.runs == 2
    gaps = [i for i, (_, _, gap, _) in enumerate(published) if gap]
    assert len(gaps) == 2
    before, after = published[gaps[1] - 1], published[gaps[1]]
    # the gap left is the audio that was lost; nothing more is thrown away
    assert after[0] - (before[0] + 0.01) == pytest.approx(0.05, abs=0.002)
    assert clock.resync_discarded_frames == 0


def test_an_output_clock_running_fast_never_moves_the_timeline_back():
    origin = 0.0
    blocks = [1.0 + (i + 1) * 0.01 / 1.005 + 0.002 for i in range(800)]  # 0.5 % fast
    clock = DeliveryClock(origin)
    published = run_clock(clock, blocks)
    check_published(published, origin)
    assert clock.ahead >= 1 and clock.runs == clock.ahead + 1
    assert clock.resync_discarded_frames <= 2 * 480 * clock.ahead  # a block or two each


def test_the_drift_measured_on_cachyos_costs_one_block_in_half_an_hour():
    """The Sound BlasterX G6 ran about 11 ppm fast against CLOCK_MONOTONIC (a ten-minute
    capture on CachyOS). Over half an hour that is one correction of about a block."""
    origin = 0.0
    blocks = 1.0 + np.arange(1, 6000 * 30 + 1) * 0.01 / (1 + 11e-6) + 0.002
    clock = DeliveryClock(origin)
    published = run_clock(clock, blocks.tolist())
    assert clock.ahead == 1 and clock.behind == 0 and clock.runs == 2
    assert clock.resync_discarded_frames <= 2 * 480
    times = np.array([timestamp for timestamp, *_ in published])
    delivered = np.array([now for *_, now in published])
    assert np.all(np.diff(times) >= 0.01 - 1e-9)  # never back, never overlapping
    assert np.all(origin + times + 0.01 <= delivered + system_pulse.AHEAD_S + 1e-9)


def test_a_pause_in_delivery_and_a_suspend_start_new_runs(monkeypatch):
    origin = 0.0
    blocks = [1.0 + (i + 1) * 0.01 for i in range(100)]
    blocks += [3.0 + (i + 1) * 0.01 for i in range(100, 200)]  # nothing for a second
    clock = DeliveryClock(origin)
    published = run_clock(clock, blocks)
    check_published(published, origin)
    assert clock.delivery_gaps == 1 and clock.runs == 2

    offsets = iter([0.0] * 60 + [30.0] * 1000)  # the computer slept for 30 s
    monkeypatch.setattr(system_pulse, "_boot_offset", lambda: next(offsets))
    clock = DeliveryClock(origin)
    published = run_clock(clock, [1.0 + (i + 1) * 0.01 for i in range(200)])
    check_published(published, origin)
    assert clock.suspends == 1 and clock.runs == 2
    assert sum(gap for _, _, gap, _ in published) == 2


# -- in the application ------------------------------------------------------------------------


def test_captured_audio_is_analysed_logged_without_audio_and_replays(pulse, tmp_path):
    from gmv.engine import Engine
    from gmv.visualizer.replay import ReplaySession

    engine = Engine(system_pulse.system_audio_source(), session_root=tmp_path / "sessions")
    engine.start()
    try:
        assert wait_until(lambda: engine.frame_count > 40)
    finally:
        engine.stop()
    assert engine.done.wait(10) and engine.error is None
    metrics = engine.metrics()
    assert metrics["dropped_blocks"] == 0 and engine.discontinuities == 0
    folder = engine.writer.path
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    capture = json.loads((folder / "capture_info.json").read_text(encoding="utf-8"))
    assert manifest["complete"] and not manifest["raw_audio_saved"]
    assert capture["backend"] == "pulse_monitor_parec"
    assert capture["monitor"] == f"{CARD}.monitor" and capture["format"] == [48000, 2, "float32le"]
    audio = [p.name for p in folder.rglob("*") if p.suffix.lower() in (".wav", ".flac", ".raw")]
    assert audio == []
    assert not pulse.alive()
    # Replayed with nothing live: the logged analysis, not the audio.
    session = ReplaySession(folder, speed=1000)
    received = session.bus.subscribe(100000)
    session.start()
    session.thread.join(30)
    assert session.finished and session.error is None
    frames = [
        m for m in (received.get() for _ in range(len(received))) if m.type == "feature_frame"
    ]
    assert len(frames) == engine.frame_count and all(f.replayed for f in frames)


def test_linux_builds_system_capture_from_this_backend_and_nothing_else(pulse, monkeypatch):
    monkeypatch.setattr(host, "system", lambda: "linux")
    built = sources.system_audio_source()
    assert isinstance(built, PulseSystemAudioSource) and built.live
    for refused in (
        lambda: sources.system_audio_source("Speakers [Loopback]"),
        lambda: sources.application_audio_source(pid=os.getpid()),
    ):
        with pytest.raises(sources.Unavailable, match="is not available on Linux"):
            refused()
    with pytest.raises(sources.Unavailable, match="Windows one"):
        sources.system_audio_source(backend="native")
