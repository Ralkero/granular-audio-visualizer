"""Sonic Character, Phase B: the live analysis (``gmv.character.live``). Generated audio and
generated frames only; nothing records this machine's sound (see conftest)."""

import random
import threading
from threading import Event
from time import perf_counter, sleep

import numpy as np
import pytest
import soundfile as sf
from test_character import RATE, dense_audio, play, sound, sparse_audio
from test_visualizer import event, frame

from gmv.bus import EventBus
from gmv.character import DIMENSION_NAMES, WINDOWS, SonicCharacter, diagnose
from gmv.character import live as live_module
from gmv.character.live import LiveCharacter, series
from gmv.engine import Engine
from gmv.models import FeatureFrame
from gmv.sources.file import FileAudioSource
from gmv.visualizer.state import VisualState


def stream(seconds=12.0, start=0.0, session="s", hop=0.02, onsets_every=0.25, **changes):
    """Generated feature frames with onset events between them, in publication order."""
    out, next_onset = [], start
    for index in range(round(seconds / hop)):
        time = start + index * hop
        if onsets_every and time >= next_onset:
            out.append(event(time, "onset", session=session))
            next_onset += onsets_every
        out.append(sound(time, session=session, **changes))
    return out


def silence(seconds, start, session="s", hop=0.02):
    return [sound(start + i * hop, session=session, level=-90.0, silence=True) for i in range(round(seconds / hop))]


def offline(messages):
    """Phase A's way: one tracker fed every message in order."""
    tracker = SonicCharacter()
    for message in messages:
        tracker.consume(message)
    return tracker


def wait_until(condition, timeout=10.0):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        if condition():
            return True
        sleep(0.005)
    return condition()


# -- equivalence with Phase A ----------------------------------------------------------------------


def test_the_live_worker_gives_exactly_the_offline_profiles_for_the_same_messages():
    messages = stream(20.0) + silence(2.0, 20.0) + stream(10.0, start=22.0, spectral_flux=0.25)
    bus = EventBus()
    live = LiveCharacter()
    live.attach(bus)
    for message in messages:
        bus.publish(message)
    assert wait_until(lambda: live.stats.messages == len(messages))
    live.detach()
    reference = offline(messages)
    for window in WINDOWS:
        assert live.tracker.profile(window) == reference.profile(window)
    snap = live.snapshot()
    assert snap.time == pytest.approx(live.history()[-1].time)
    assert live.dropped == 0 and live.stats.errors == 0


def test_the_live_engine_stream_matches_phase_a_offline_analysis_of_the_same_file(tmp_path):
    path = tmp_path / "music.wav"
    sf.write(path, np.concatenate((dense_audio(8.0), sparse_audio(6.0))), RATE, subtype="FLOAT")
    engine = Engine(FileAudioSource(path, realtime=False), session_root=tmp_path / "sessions", logging=False)
    raw = engine.bus.subscribe(16384)
    live = LiveCharacter(capacity=16384)
    live.attach(engine.bus)
    engine.start()
    assert engine.done.wait(60) and not engine.error
    messages = []
    while len(raw):
        messages.append(raw.get())
    assert wait_until(lambda: live.stats.messages == len(messages))
    live.detach()
    reference = offline(messages)
    for window in WINDOWS:
        assert live.tracker.profile(window) == reference.profile(window)
    # Phase A's offline diagnostic runs the same file through its own engine: the same profile.
    result = diagnose.analyse(*diagnose.audio_messages(path))
    offline_values = [d["value"] for d in result["overall"]["dimensions"]]
    live_values = list(live.tracker.profile("whole").values())
    assert offline_values == pytest.approx(live_values, abs=1e-9)


@pytest.mark.parametrize("sizes", ("one", "seven", "random", "all"))
def test_batch_sizes_do_not_change_snapshots_history_or_profiles(sizes):
    messages = stream(15.0) + stream(6.0, start=15.0, spectral_centroid_hz=2500.0)
    whole = LiveCharacter()
    whole.process(messages)
    split = LiveCharacter()
    rng = random.Random(4)
    position = 0
    while position < len(messages):
        size = {"one": 1, "seven": 7, "random": rng.randint(1, 60), "all": len(messages)}[sizes]
        split.process(messages[position : position + size])
        position += size
    assert split.history() == whole.history() and split.snapshot() == whole.snapshot()
    reference = offline(messages)
    for window in WINDOWS:
        assert split.tracker.profile(window) == reference.profile(window)


def test_history_points_are_the_offline_profiles_at_each_sample_time():
    messages = stream(8.0)
    live = LiveCharacter(sample_s=0.5)
    live.process(messages)
    tracker, expected = SonicCharacter(), []
    sampled = None
    for message in messages:
        if tracker.consume(message) and isinstance(message, FeatureFrame):
            if sampled is None or tracker.time >= sampled + 0.5:
                sampled = tracker.time
                expected.append((tracker.time, tracker.profile("short").values()))
    got = [(s.time, s.readings["short"].values) for s in live.history()]
    assert got == expected and len(got) == 16


# -- playback states and sources ----------------------------------------------------------------


def test_pause_holds_the_values_and_resume_carries_on():
    live = LiveCharacter()
    live.process(stream(12.0))
    before = live_module.reading(live.tracker.profile("medium"))  # the state as the pause begins
    live.process(silence(4.0, 12.0))
    paused = live.snapshot().readings["medium"]
    assert not paused.active and paused.values == before.values
    assert paused.confidences == before.confidences  # a pause does not age the profile
    assert any(w.startswith("held") for w in paused.warnings)
    live.process(stream(4.0, start=16.0))
    resumed = live.snapshot().readings["medium"]
    assert resumed.active and resumed.heard_s > before.heard_s


def test_a_pause_with_no_frames_at_all_changes_nothing():
    live = LiveCharacter()
    live.process(stream(6.0))
    first = live.snapshot()
    live.process([])  # nothing arrives while the player is paused and the stream idles
    assert live.snapshot() is first


def test_stop_keeps_the_last_values_and_a_new_start_begins_afresh():
    bus = EventBus()
    live = LiveCharacter()
    live.attach(bus)
    for message in stream(6.0):
        bus.publish(message)
    assert wait_until(lambda: live.snapshot() is not None and live.backlog == 0)
    assert live.detach() and not live.attached
    last = live.snapshot()
    assert last is not None  # still readable after the stop
    live.attach(EventBus())
    assert live.snapshot() is None and live.history() == ()
    live.detach()


def test_a_new_session_starts_everything_afresh():
    live = LiveCharacter()
    live.process(stream(10.0, session="first"))
    assert live.history()
    live.process(stream(3.0, session="second"))
    snap = live.snapshot()
    assert snap.session_id == "second" and snap.readings["short"].heard_s < 3.5
    assert all(s.session_id == "second" for s in live.history())


def test_a_new_segment_or_a_restart_clears_the_history():
    live = LiveCharacter()
    live.process(stream(10.0))
    generation = live.snapshot().generation
    live.process([sound(10.0 + i * 0.02, segment=1) for i in range(100)])
    assert live.snapshot().generation > generation
    assert all(s.generation == live.snapshot().generation for s in live.history())
    restarted = LiveCharacter()
    restarted.process(stream(10.0))
    restarted.process(stream(2.0, start=0.0))  # the timeline started again
    assert restarted.snapshot().readings["short"].heard_s < 2.5


def test_empty_input_and_messages_without_frames_give_no_snapshot():
    live = LiveCharacter()
    live.process([])
    live.process([event(0.1, "beat_prediction"), event(0.2, "note_attack_candidate")])
    assert live.snapshot() is None and live.history() == ()
    bus = EventBus()
    live.attach(bus)
    sleep(0.2)
    assert live.snapshot() is None
    assert live.detach()


# -- confidence ----------------------------------------------------------------------------------


def test_too_little_signal_is_reported_as_not_usable_not_as_zero():
    live = LiveCharacter()
    live.process(stream(1.0))
    short = live.snapshot().readings["short"]
    dynamics = DIMENSION_NAMES.index("dynamics")
    regularity = DIMENSION_NAMES.index("regularity")
    assert not short.usable(dynamics) and not short.usable(regularity)
    assert any(w.startswith("little_music") for w in short.warnings)
    live.process(stream(40.0, start=1.0))
    assert live.snapshot().readings["medium"].usable(DIMENSION_NAMES.index("brightness"))


def test_before_any_music_values_are_unknown_with_zero_confidence():
    live = LiveCharacter()
    live.process(silence(3.0, 0.0))
    reading = live.snapshot().readings["short"]
    assert all(v is None for v in reading.values) and all(c == 0.0 for c in reading.confidences)
    assert not any(reading.usable(i) for i in range(len(DIMENSION_NAMES)))


def test_confidence_is_passed_through_from_phase_a_unchanged():
    messages = stream(9.0)
    live = LiveCharacter()
    live.process(messages)
    assert live.tracker.profile("long").confidences() == offline(messages).profile("long").confidences()
    snap = live.snapshot()  # the last sample point: the offline profile at that frame
    upto = [m for m in messages if m.timestamp <= snap.time]
    assert snap.readings["long"].confidences == offline(upto).profile("long").confidences()


# -- bounded queues, stale results, threads -------------------------------------------------------------


def test_the_queue_is_bounded_drops_are_counted_and_the_newest_state_wins(monkeypatch):
    release = Event()
    original = LiveCharacter.process

    def stalled(self, messages):
        release.wait(10)
        return original(self, messages)

    monkeypatch.setattr(LiveCharacter, "process", stalled)
    bus = EventBus()
    live = LiveCharacter(capacity=64)
    live.attach(bus)
    messages = stream(20.0, onsets_every=0)
    for message in messages:
        bus.publish(message)
    # The stalled worker may already hold one batch (at most 512 messages); the queue holds 64.
    assert live.backlog <= 64 and live.dropped >= len(messages) - 64 - 512 and live.dropped > 0
    release.set()
    assert wait_until(lambda: live.backlog == 0 and live.snapshot() is not None)
    assert wait_until(lambda: live.stats.messages + live.dropped == len(messages))  # every one accounted for
    sleep(0.1)
    newest = messages[-1].timestamp
    assert live.snapshot().time >= newest - live.sample_s  # no backlog of old results
    live.detach()


def test_history_never_grows_beyond_its_length():
    live = LiveCharacter(sample_s=0.5, history_s=10.0)
    live.process(stream(60.0, onsets_every=0))
    history = live.history()
    assert len(history) == 20 and history[-1].time - history[0].time == pytest.approx(9.5, abs=0.05)
    times, values, _ = series(history, "short", "pace")
    assert len(times) == len(values) == 20


def test_readers_on_other_threads_never_see_a_half_built_state():
    bus = EventBus()
    live = LiveCharacter(capacity=20000)
    live.attach(bus)
    problems, stop = [], Event()

    def read():
        while not stop.is_set():
            try:
                snap, history = live.snapshot(), live.history()
                if snap is not None:
                    assert set(snap.readings) == set(live.windows)
                for item in history:
                    assert len(item.readings["short"].values) == len(DIMENSION_NAMES)
            except Exception as exc:  # pragma: no cover - reported below
                problems.append(exc)

    readers = [threading.Thread(target=read) for _ in range(3)]
    for reader in readers:
        reader.start()
    messages = stream(40.0)
    for message in messages:
        bus.publish(message)
    assert wait_until(lambda: live.stats.messages == len(messages))
    stop.set()
    for reader in readers:
        reader.join(5)
    assert not problems and live.detach() and live.stats.errors == 0


def test_attach_and_detach_leave_no_threads_behind():
    baseline = threading.active_count()
    live = LiveCharacter()
    for _ in range(5):
        bus = EventBus()
        live.attach(bus)
        for message in stream(1.0):
            bus.publish(message)
        start = perf_counter()
        assert live.detach()
        assert perf_counter() - start < live_module.JOIN_S
    assert threading.active_count() == baseline
    assert live.detach()  # detaching twice is harmless


def test_a_failing_message_is_counted_and_the_worker_carries_on(monkeypatch):
    bus = EventBus()
    live = LiveCharacter()
    original = live.tracker.consume
    failed = {"done": False}

    def flaky(message):
        if not failed["done"] and isinstance(message, FeatureFrame):
            failed["done"] = True
            raise ValueError("generated failure")
        return original(message)

    monkeypatch.setattr(live.tracker, "consume", flaky)
    live.attach(bus)
    live.tracker.consume = flaky  # attach restarts the tracker, which keeps its methods
    messages = stream(4.0)
    for message in messages:
        bus.publish(message)
    assert wait_until(lambda: live.stats.messages >= len(messages) - 1)
    assert wait_until(lambda: live.snapshot() is not None)
    assert live.stats.errors == 1 and "generated failure" in live.stats.last_error
    live.detach()


def test_the_snapshot_holds_no_audio_or_per_frame_measurements():
    live = LiveCharacter()
    live.process(stream(3.0))
    snap = live.snapshot()
    reading = snap.readings["short"]
    assert set(vars(reading)) == {"window", "values", "confidences", "active", "heard_s", "evidence_s", "warnings"}
    assert set(vars(snap)) == {"session_id", "generation", "time", "readings", "frames", "music_frames"}


# -- the visualizer is not affected ------------------------------------------------------------------------


def test_the_visualizer_sees_the_same_messages_and_state_with_the_live_analysis_attached():
    messages = stream(10.0)

    def run(with_live):
        bus = EventBus()
        view_queue = bus.subscribe(8192)
        live = LiveCharacter()
        if with_live:
            live.attach(bus)
        for message in messages:
            bus.publish(message)
        if with_live:
            assert wait_until(lambda: live.stats.messages == len(messages))
            live.detach()
        state, seen = VisualState(), []
        while len(view_queue):
            message = view_queue.get()
            seen.append((type(message).__name__, message.timestamp, message.sequence))
            state.consume(message)
        return seen, state.musical.value, len(state.frames)

    assert run(False) == run(True)


def test_frames_of_another_shape_than_expected_never_stop_the_worker():
    live = LiveCharacter()
    odd = frame(0.0, spectrum_db=(), band_energy=(1.0, 2.0))
    live.process([odd] + stream(2.0, start=0.02))
    assert live.stats.errors == 0 and live.snapshot() is not None


def test_play_helper_still_drives_a_plain_tracker():  # keeps the shared helper honest
    tracker = SonicCharacter()
    assert play(tracker, 0.0, 1.0) == 1.0 and tracker.frames == 50
