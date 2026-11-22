"""The engine in its own process: the same messages in the same order, a clean lifecycle,
and a window that carries on when the worker does not. Also the display-aware frame rate."""

import os
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import psutil
import pytest
import soundfile as sf

import synth_instruments as synth
from gmv.engine import Engine
from gmv.models import AnalysisConfig, FeatureFrame, InstrumentActivity, Status
from gmv.remote import RemoteEngine
from gmv.sources.file import FileAudioSource
from gmv.visualizer.window import automatic_rate

from test_instrument_layers import guitar

RATE = synth.RATE
CONFIG = AnalysisConfig(instrument_analyzers=("guitar", "drums"))


@pytest.fixture(scope="module")
def band(tmp_path_factory):
    path = tmp_path_factory.mktemp("band") / "band.wav"
    hits = np.arange(0.3, 2.8, 0.25)
    sf.write(
        path, guitar(3.0, np.arange(0.3, 2.9, 0.1)) + synth.kick(3.0, hits), RATE, subtype="FLOAT"
    )
    return path


def run(engine, seconds=40, during=None):
    queue = engine.bus.subscribe(32768)
    engine.start()
    if during:
        during(engine)
    assert engine.done.wait(seconds), "the engine did not finish"
    found = []
    while (message := queue.get(timeout=0)) is not None:
        found.append(message)
    return found


def children():
    return [
        p
        for p in psutil.Process().children(recursive=True)
        if p.is_running() and p.name().lower().startswith("python")
    ]


def test_the_worker_delivers_the_same_analysis_in_order_with_its_own_stamps(band, tmp_path):
    here = run(Engine(FileAudioSource(band, realtime=True), CONFIG, logging=False))
    engine = RemoteEngine(
        FileAudioSource, (band,), {"realtime": True}, CONFIG, session_root=tmp_path, logging=True
    )
    there = run(engine)
    assert engine.error is None and engine.state == "stopped"
    metrics = engine.metrics()
    assert metrics["dropped_blocks"] == 0 and engine.forwarder_dropped == 0
    assert (
        metrics["frames"] == sum(isinstance(m, FeatureFrame) for m in there) == engine.frame_count
    )
    # Nothing was restamped on the way: sequence numbers are the worker's own, in order,
    # without holes, and each message was published before it arrived here.
    sequences = [m.sequence for m in there]
    assert sequences == sorted(sequences) and len(set(sequences)) == len(sequences)
    assert sequences[-1] - sequences[0] == len(sequences) - 1
    assert all(m.published_monotonic <= perf_counter() for m in there)
    assert abs(engine.clock_offset) < 0.05  # the two processes share one monotonic clock
    # The analysis itself is the same as in this process: same frames, same strokes.

    def strokes(messages):
        return [
            round(m.timestamp, 3)
            for m in messages
            if isinstance(m, InstrumentActivity) and m.instrument == "guitar" and m.attack > 0
        ]

    def frames(messages):
        return [round(m.timestamp, 4) for m in messages if isinstance(m, FeatureFrame)]

    assert (
        frames(there) == frames(here)
        and strokes(there) == strokes(here)
        and len(strokes(there)) > 15
    )
    # The session was logged by the worker, where the engine is.
    assert (
        engine.writer is not None and (engine.writer.path / "messages.jsonl").stat().st_size > 10000
    )
    sleep(0.3)
    assert not children()  # the worker has gone


def test_stop_sensitivity_and_repeated_sessions(band):
    def lower(engine):
        engine.instrument_sensitivity["guitar"] = 0.0

    normal = run(RemoteEngine(FileAudioSource, (band,), {"realtime": True}, CONFIG, logging=False))
    quiet = run(
        RemoteEngine(FileAudioSource, (band,), {"realtime": True}, CONFIG, logging=False),
        during=lower,
    )
    def flux(messages):
        return np.mean(
            [m.flux for m in messages if isinstance(m, InstrumentActivity) and m.instrument == "guitar"]
        )

    assert flux(quiet) < flux(normal)  # the change reached the running analyzer

    def stop_early(engine):  # once the worker is up and analysing
        deadline = perf_counter() + 20
        while engine.frame_count < 10 and perf_counter() < deadline:
            sleep(0.02)
        engine.stop()

    for _ in range(3):  # start, stop early, start again: no worker is left behind
        engine = RemoteEngine(FileAudioSource, (band,), {"realtime": True}, CONFIG, logging=False)
        began = perf_counter()
        found = run(engine, during=stop_early)
        assert perf_counter() - began < 12 and engine.error is None
        assert 0 < sum(isinstance(m, FeatureFrame) for m in found) < 140  # it stopped early
    sleep(0.3)
    assert not children()


def test_a_worker_that_fails_or_dies_ends_the_session_with_an_error_and_nothing_hangs(
    band, tmp_path
):
    missing = RemoteEngine(
        FileAudioSource, (tmp_path / "nothing.wav",), {"realtime": True}, CONFIG, logging=False
    )
    found = run(missing, seconds=30)
    assert missing.state == "error" and missing.error and missing.done.is_set()
    assert any(isinstance(m, Status) and m.state == "error" for m in found)

    def kill(engine):
        deadline = perf_counter() + 20
        while engine.frame_count < 20 and perf_counter() < deadline:
            sleep(0.05)
        psutil.Process(engine.process_id).kill()

    crashed = RemoteEngine(FileAudioSource, (band,), {"realtime": True}, CONFIG, logging=False)
    began = perf_counter()
    found = run(crashed, seconds=30, during=kill)
    assert perf_counter() - began < 25
    assert crashed.state == "error" and "ended unexpectedly" in crashed.error
    assert isinstance(found[-1], Status) and found[-1].state == "error"
    crashed.stop()  # stopping a dead engine is harmless
    # And the next session simply starts a new worker.
    again = RemoteEngine(FileAudioSource, (band,), {"realtime": True}, CONFIG, logging=False)
    assert sum(isinstance(m, FeatureFrame) for m in run(again)) > 100 and again.error is None
    sleep(0.3)
    assert not children()


def test_the_application_uses_the_worker_by_default_and_the_thread_when_asked(
    app, band, tmp_path, monkeypatch
):
    from gmv.app.controller import SessionController, Settings

    controller = SessionController(tmp_path)
    settings = Settings(source="file", file=str(band), logging=False, pitch_tracking=False)
    assert settings.analysis_process
    engine = controller.prepare(settings)
    assert isinstance(engine, RemoteEngine) and engine.source.path.endswith("band.wav")
    queue = engine.bus.subscribe(32768)
    engine.start()
    assert controller.running
    assert "Analysis" in controller.diagnostics()
    controller.set_sensitivity("guitar", 0.9)
    assert engine.done.wait(40) and not controller.running and engine.error is None
    assert engine.metrics()["frames"] > 100 and len(queue) > 300
    assert "separate process" in controller.diagnostics()["Analysis process"]
    assert controller.shutdown()
    in_process = SessionController(tmp_path).prepare(settings.changed(analysis_process=False))
    assert isinstance(in_process, Engine)
    monkeypatch.setenv("GMV_ANALYSIS", "thread")
    assert isinstance(SessionController(tmp_path).prepare(settings), Engine)


def test_the_frame_rate_is_a_whole_fraction_of_the_displays(window):
    assert automatic_rate(60) == 60 and automatic_rate(120) == 60 and automatic_rate(240) == 60
    assert automatic_rate(144) == 72 and automatic_rate(180) == 60 and automatic_rate(75) == 75
    assert automatic_rate(0) == 60 and automatic_rate(-1) == 60  # an unusable report
    window.set_frame_rate(144)
    assert window.frame_s == pytest.approx(1 / 144)
    window.set_frame_rate(None)
    refresh = window.screen().refreshRate()
    assert window.frame_s == pytest.approx(1 / automatic_rate(refresh))
    # A late wake-up draws at once and the grid is kept: no burst of catch-up frames.
    window.set_frame_rate(72)
    window.next_frame = perf_counter() - 1.0  # a whole second behind
    painted = window.canvas.perf.series.get("tick")
    before = painted.total if painted else 0
    window.poll()
    assert window.canvas.perf.series["tick"].total == before + 1
    assert 0 < window.next_frame - perf_counter() <= 1 / 72  # back on the grid from now
    window.timer.setInterval(10)  # as while frames are being drawn
    window.poll()  # too early for the next frame: nothing is drawn
    assert window.canvas.perf.series["tick"].total == before + 1
