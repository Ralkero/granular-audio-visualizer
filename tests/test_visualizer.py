"""The visualizer, driven without audio hardware: synthetic messages, a file and a replay."""

import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf

from gmv.bus import EventBus
from gmv.engine import Engine
from gmv.evaluation.detect import run_detector
from gmv.models import AnalysisConfig, FeatureFrame, MusicalEvent, Status
from gmv.sources.file import FileAudioSource
from gmv.visualizer.lanes import Lane, Track
from gmv.visualizer.replay import ReplaySession
from gmv.visualizer.state import SpectrumDisplay, VisualState

RATE = 48000


def frame(time, session="s", scores=(), active=(), method=None, **changes):
    values = dict(
        session_id=session,
        source="synthetic",
        timestamp=time,
        detected_monotonic=0,
        window_end=time + 0.023,
        sample_rate=RATE,
        channels=2,
        rms=0.1,
        peak=0.3,
        loudness_dbfs=-20,
        short_term_loudness_dbfs=-20,
        spectral_centroid_hz=1000,
        spectral_bandwidth_hz=1000,
        spectral_rolloff_hz=4000,
        spectral_contrast_db=(0,) * 7,
        zero_crossing_rate=0.1,
        band_energy=(0.01,) * 7,
        spectral_flux=0.1,
        transient_strength=0.1,
        onset_strength=0.2,
        dynamic_intensity=0.6,
        silence=False,
        spectrum_db=tuple(np.linspace(-30, -50, 48).tolist()),
        spectrum_low_hz=30,
        spectrum_high_hz=18000,
        instrument_scores=tuple(scores),
        instruments_active=tuple(active),
        activity_method=method,
    )
    return FeatureFrame(**{**values, **changes})


def event(time, kind, evidence="candidate", session="s", **changes):
    values = dict(
        session_id=session,
        source="synthetic",
        timestamp=time,
        detected_monotonic=0,
        event_id=f"{kind}-{time}",
        event_type=kind,
        intensity=0.6,
        confidence=0.4,
        method="test",
        evidence_kind=evidence,
    )
    return MusicalEvent(**{**values, **changes})


def feed(state, seconds, start=0.0, clock=100.0, **frame_changes):
    """Frames every 20 ms, each arriving 50 ms after its audio time on a local clock."""
    for index in range(round(seconds / 0.02)):
        time = start + index * 0.02
        state.consume(frame(time, **frame_changes), arrival=clock + time + 0.05)
    return clock + start + seconds + 0.03


def test_display_clock_follows_the_audio_timeline():
    state = VisualState()
    assert state.now(5.0) is None and state.condition(5.0) == "idle"
    clock = feed(state, 2.0)
    assert state.now(clock) == pytest.approx(clock - 100.05)
    # One late delivery must not drag the timeline backwards.
    state.consume(frame(2.0), arrival=clock + 0.2)
    assert state.now(clock + 0.2) == pytest.approx(clock + 0.2 - 100.05)
    assert state.condition(clock + 0.2) == "live"
    assert state.condition(clock + 3) == "waiting"


def test_events_keep_their_audio_time_whatever_the_arrival_order():
    state = VisualState()
    clock = feed(state, 2.0)
    now = state.now(clock)
    # A prediction for the future, an onset just heard, and a late report about older audio.
    state.consume(
        event(now + 0.3, "beat_prediction", "predicted", measurements=(("tempo_bpm", 120.0),))
    )
    state.consume(event(now - 0.04, "onset", "observed"))
    state.consume(
        event(now - 1.5, "harmonic_percussive_balance", "context", analysis_tier="advanced")
    )
    marks = {mark.kind: mark for mark in state.markers}
    assert marks["beat_prediction"].time == pytest.approx(now + 0.3)
    assert marks["onset"].time == pytest.approx(now - 0.04)
    assert marks["harmonic_percussive_balance"].time == pytest.approx(now - 1.5)
    assert [mark.kind for mark in state.due(now)] == ["onset"]  # the prediction is not due yet
    assert state.due(now + 0.1) == []
    assert [mark.kind for mark in state.due(now + 0.31)] == ["beat_prediction"]
    assert state.beat_tempo == 120.0


def test_observed_candidate_predicted_and_delayed_marks_stay_distinct():
    state = VisualState()
    feed(state, 0.2)
    for kind, evidence, changes in (
        ("onset", "observed", {}),
        ("bass_transient_candidate", "candidate", {}),
        ("beat_prediction", "predicted", {}),
        ("beat_observed", "candidate", {}),
        ("harmonic_percussive_balance", "candidate", {"analysis_tier": "advanced"}),
        ("some_future_event", "observed", {}),
    ):
        state.consume(event(0.1, kind, evidence, **changes))
    marks = {mark.kind: mark for mark in state.markers}
    assert (marks["onset"].row, marks["onset"].style) == ("onsets", "observed")
    low = marks["bass_transient_candidate"]
    # A low-frequency candidate is a frequency region, never a named instrument.
    assert (low.row, low.style, low.region, low.label) == (
        "regions",
        "candidate",
        0,
        "Low-frequency transient",
    )
    assert (marks["beat_prediction"].row, marks["beat_prediction"].style) == ("beats", "predicted")
    assert marks["beat_observed"].style == "candidate"
    assert marks["harmonic_percussive_balance"].row == "context"
    assert marks["harmonic_percussive_balance"].style == "context"
    assert marks["some_future_event"].row == "changes"


def test_lanes_show_only_published_detector_output():
    state = VisualState()
    feed(state, 0.2)
    lanes = {lane.key: live for lane, live in state.board.lanes()}
    assert lanes == dict.fromkeys(("drums", "bass", "vocals", "guitar", "other_harmonic"), False)
    feed(
        state,
        0.2,
        start=0.2,
        scores=(("drums", 0.9), ("bass", 0.2), ("vocals", 0.7)),
        active=("drums", "vocals"),
        method="activity_mlp_v2",
    )
    lanes = {lane.key: live for lane, live in state.board.lanes()}
    assert lanes["drums"] and lanes["bass"] and lanes["vocals"]
    assert not lanes["guitar"] and not lanes["other_harmonic"]
    assert state.board.history["bass"].last.tolist() == pytest.approx([0.38, 0.2, 0.0])
    assert state.board.history["drums"].last[2] == 1.0
    assert state.board.method == "activity_mlp_v2"


def test_a_new_detector_label_gets_its_own_lane_without_renderer_changes():
    state = VisualState()
    feed(state, 0.1, scores=(("kick", 0.8), ("theremin", 0.5)), active=("kick",), method="future")
    lanes = {lane.key: (lane, live) for lane, live in state.board.lanes()}
    assert lanes["kick"][1] and lanes["kick"][0].parent == "drums"
    assert lanes["theremin"][1] and lanes["theremin"][0].title == "Theremin"
    order = [lane.key for lane, _ in state.board.lanes()]
    assert order.index("drums") < order.index("kick") < order.index("bass")
    state.board.register(Lane("theremin", "Theremin (lead)", "#ffffff"))
    assert (
        dict((lane.key, lane.title) for lane, _ in state.board.lanes())["theremin"]
        == "Theremin (lead)"
    )


def test_history_is_bounded():
    state = VisualState(capacity=500)
    feed(state, 60.0, scores=(("drums", 0.5),), method="m")
    for index in range(5000):
        state.consume(event(index * 0.01, "onset", "observed"))
    assert len(state.frames) == 500 and state.frames.rows.shape == (1000, 4)
    assert len(state.board.history["drums"]) == 500
    assert len(state.markers) == 2048 and len(state._pending) <= 256
    newest = state.frames.since(59.0)
    assert newest[0, 0] >= 59.0 and newest[-1, 0] == pytest.approx(59.98)


def test_track_keeps_the_newest_rows_in_order():
    track = Track(4, 2)
    for value in range(11):
        track.append(float(value), value * 10.0)
    assert track.since()[:, 0].tolist() == [7.0, 8.0, 9.0, 10.0]
    assert track.since(9.0)[:, 1].tolist() == [90.0, 100.0]
    assert track.last.tolist() == [10.0, 100.0]


def test_new_session_restart_and_gaps():
    state = VisualState()
    clock = feed(state, 1.0, scores=(("drums", 0.9),), active=("drums",), method="m")
    state.consume(event(0.5, "onset", "observed"))
    generation = state.generation
    # A capture gap: audio time jumps ahead, history is kept and the hole is recorded.
    state.consume(frame(3.0, scores=(("drums", 0.9),), method="m", segment=1), arrival=clock + 2.1)
    assert len(state.markers) == 1 and state.generation == generation
    assert state.gaps[-1] == pytest.approx((1.003, 3.0))
    hole = state.board.history["drums"].since(1.0)
    assert hole[:, 1].tolist() == pytest.approx([0.0, 0.0, 0.9])
    assert state.now(clock + 2.1) == pytest.approx(3.0)
    # The timeline starting again (a new file) and a new session both clear the history.
    state.consume(frame(0.0), arrival=clock + 5)
    assert len(state.markers) == 0 and len(state.frames) == 1 and state.generation > generation
    state.consume(event(0.0, "onset", "observed", session="other"))
    assert state.session_id == "other" and len(state.frames) == 0


def test_status_and_replay_conditions():
    state = VisualState()
    clock = feed(state, 0.5, replayed=True)
    assert state.condition(clock) == "replay"
    state.consume(
        Status(
            session_id="s",
            source="x",
            timestamp=1,
            detected_monotonic=0,
            state="stopped",
            detail="Session complete",
        )
    )
    assert state.condition(clock) == "stopped"


def test_sessions_without_a_display_spectrum_fall_back_to_the_seven_bands():
    state = VisualState()
    state.consume(
        frame(0.0, spectrum_db=(), spectrum_low_hz=None, spectrum_high_hz=None), arrival=1.0
    )
    assert len(state.spectrum) == 7 and state.spectrum_range is None
    assert SpectrumDisplay().step(state.spectrum, 0.016, state.spectrum_range).shape == (7,)


def test_spectrum_display_rises_at_once_and_falls_gradually():
    display = SpectrumDisplay(release_s=0.2)
    quiet, loud = np.full(48, -70.0), np.full(48, -70.0)
    loud[10] = -15.0
    display.step(quiet, 0.016)
    risen = display.step(loud, 0.016)[10]
    assert risen > 0.5
    fallen = display.step(quiet, 0.016)[10]
    assert 0.8 * risen < fallen < risen  # one refresh later most of the bar is still there
    for _ in range(120):
        values = display.step(quiet, 0.016)
    assert values[10] < 0.02 and 0 <= values.min() and values.max() <= 1
    assert display.peaks[10] >= values[10]


def test_analyzer_publishes_a_display_spectrum_from_its_own_fft():
    seconds = np.arange(RATE) / RATE
    audio = np.repeat((0.5 * np.sin(2 * np.pi * 1000 * seconds))[:, None], 2, axis=1).astype(
        np.float32
    )
    frames, *_ = run_detector(audio, RATE, "candidate", AnalysisConfig())
    spectrum = np.array(frames[-1].spectrum_db)
    edges = np.geomspace(frames[-1].spectrum_low_hz, frames[-1].spectrum_high_hz, 49)
    assert len(spectrum) == 48
    assert edges[spectrum.argmax()] <= 1000 < edges[spectrum.argmax() + 1]
    assert spectrum.max() - np.median(spectrum) > 40
    frames, *_ = run_detector(audio, RATE, "candidate", AnalysisConfig(spectrum_bands=0))
    assert frames[-1].spectrum_db == () and frames[-1].spectrum_low_hz is None


# -- the window itself (Qt, offscreen) -------------------------------------------------------


@pytest.fixture(scope="module")
def app():
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def window(app):
    from gmv.visualizer.window import VisualizerWindow

    view = VisualizerWindow()
    view.timer.stop()  # the tests drive the refresh themselves
    view.resize(1280, 800)
    yield view
    view.close()


def publish_music(bus, seconds, start=0.0, **frame_changes):
    for index in range(round(seconds / 0.02)):
        time = start + index * 0.02
        bus.publish(frame(time, **frame_changes))
        if index % 10 == 0:
            bus.publish(event(time, "onset", "observed"))
            bus.publish(
                event(
                    time + 0.2, "beat_prediction", "predicted", measurements=(("tempo_bpm", 150.0),)
                )
            )


def colours(image, step=16):
    return {
        image.pixelColor(x, y).name()
        for x in range(0, image.width(), step)
        for y in range(0, image.height(), step)
    }


def test_window_has_its_own_subscription_and_releases_it(window):
    bus = EventBus()
    other = bus.subscribe(8)  # e.g. the control center
    window.attach(bus, source="test")
    assert len(bus._subscribers) == 2 and window.subscription is not other
    publish_music(bus, 0.2)
    window.tick()
    assert len(window.state.frames) == 10 and len(other) == 8  # each consumer has its own queue
    window.close()
    assert bus._subscribers == [other] and window.subscription is None


def test_window_draws_both_modes_and_every_state(window):
    idle = window.grab().toImage()
    assert window.canvas.condition == "idle"
    bus = EventBus()
    window.attach(bus, source="test", beat_tracker=True)
    publish_music(
        bus,
        3.0,
        scores=(("drums", 0.9), ("bass", 0.4), ("vocals", 0.1)),
        active=("drums",),
        method="activity_mlp_v2",
    )
    window.tick()
    assert window.canvas.condition == "live" and window.canvas.now == pytest.approx(2.98, abs=0.1)
    bars = window.grab().toImage()
    assert len(colours(bars)) > len(colours(idle)) + 20
    window.set_mode("radial")
    window.tick()
    radial = window.grab().toImage()
    assert window.canvas.mode == "radial" and radial != bars
    window.set_shown("drums", False)
    window.set_shown("onsets", False)
    window.set_beats(False)
    window.tick()
    assert window.grab().toImage() != radial
    bus.publish(
        Status(
            session_id="s",
            source="x",
            timestamp=3,
            detected_monotonic=0,
            state="stopped",
            detail="done",
        )
    )
    window.tick()
    assert window.canvas.condition == "stopped"
    assert max(window.canvas.paint_ms) < 250


def test_reset_clears_history_but_keeps_following_the_stream(window):
    bus = EventBus()
    window.attach(bus)
    publish_music(bus, 1.0)
    window.tick()
    assert len(window.state.markers)
    window.state.reset()
    assert not len(window.state.markers) and not len(window.state.frames)
    publish_music(bus, 0.2, start=1.0)
    window.tick()
    assert len(window.state.frames) == 10


def test_a_stalled_window_never_blocks_the_publisher(window):
    bus = EventBus()
    window.attach(bus)
    began = perf_counter()
    publish_music(bus, 120.0)  # 6000 frames plus events while the window is not refreshing
    assert perf_counter() - began < 5
    assert window.subscription.dropped > 0 and len(window.subscription) <= 2048
    window.tick()
    # Only the newest messages survived, and the display continues from them.
    assert window.state.latest.timestamp == pytest.approx(119.98)
    assert window.canvas.info["dropped"] == window.subscription.dropped


def test_the_refresh_timer_slows_down_when_nothing_is_arriving(window):
    window.timer.start(16)
    window.tick()
    assert window.timer.interval() == 250
    bus = EventBus()
    window.attach(bus)
    publish_music(bus, 0.5)
    window.tick()
    assert window.timer.interval() == 16
    window.timer.stop()


def write_clicks(path, seconds=4):
    audio = np.zeros((RATE * seconds, 2), dtype=np.float32)
    rng = np.random.default_rng(3)
    for start in range(RATE // 2, len(audio), RATE // 2):
        burst = rng.standard_normal((2400, 1)) * np.exp(-np.arange(2400) / 300)[:, None]
        audio[start : start + 2400] += 0.5 * burst.astype(np.float32)
    sf.write(path, audio, RATE, subtype="FLOAT")
    return path


def test_live_engine_feeds_the_window_and_the_session_replays_with_original_times(
    window, tmp_path, app
):
    source = FileAudioSource(write_clicks(tmp_path / "clicks.wav"), realtime=False)
    engine = Engine(source, session_root=tmp_path / "sessions")
    window.attach(engine.bus, source="file")
    engine.start()
    assert engine.done.wait(30) and engine.error is None
    window.tick()
    live_onsets = [mark.time for mark in window.state.markers if mark.kind == "onset"]
    assert len(live_onsets) >= 6 and window.state.spectrum is not None
    assert not window.state.replayed

    session = ReplaySession(engine.writer.path, speed=200)
    assert session.configuration["spectrum_bands"] == 48
    window.attach(session.bus, source=session.title)
    session.start()
    deadline = perf_counter() + 20
    while not session.finished and perf_counter() < deadline:
        window.tick()
        app.processEvents()
    window.tick()
    session.stop()
    assert session.finished and session.error is None
    assert window.state.replayed and window.canvas.condition == "stopped"
    replayed = [mark.time for mark in window.state.markers if mark.kind == "onset"]
    assert replayed == pytest.approx(live_onsets)  # audio timestamps, not replay arrival times

