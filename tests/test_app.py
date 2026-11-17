"""The unified application: one window, settings in a drawer, driven without audio hardware."""

import os
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from test_visualizer import write_clicks

from gmv.app.controller import Settings, analysis_config, audio_source
from gmv.app.window import MainWindow


@pytest.fixture(scope="module")
def app():
    from gmv.app.window import application

    return application()


@pytest.fixture
def window(app, tmp_path):
    main = MainWindow(session_root=tmp_path / "sessions", remember=False, discover_sources=False)
    main.view.timer.stop()
    main.poller.stop()
    main.resize(1280, 800)
    main.show()
    yield main
    main.close()


def pump(window, app, until, timeout=15):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        app.processEvents()
        window.poll()
        window.view.tick()
        if until():
            return True
        sleep(0.01)
    return False


def use_file(window, path, logging=False):
    drawer = window.drawer
    drawer.source.setCurrentIndex(drawer.source.findData("file"))
    drawer.file.setText(str(path))
    drawer.logging.setChecked(logging)


def test_opens_straight_into_the_visualizer_with_nothing_running(window):
    assert window.view.isVisible() and not window.drawer.isVisible()
    assert window.toggle.text() == "Start" and not window.apply.isVisible()
    assert window.view.canvas.condition == "idle" and window.view.bus is None
    assert "Default system output" in window.status.text()
    assert window.controller.engine is None


def test_old_entry_points_open_the_same_window():
    from gmv import ui

    assert ui.ControlCenter is MainWindow


def test_start_stop_and_settings_drawer_do_not_interfere(window, app, tmp_path):
    use_file(window, write_clicks(tmp_path / "clicks.wav", seconds=6))
    window.start_or_stop()
    controller = window.controller
    assert controller.running and window.problem == ""
    assert window.view.bus is controller.engine.bus
    assert pump(window, app, lambda: len(window.view.state.frames) > 20)
    assert window.toggle.text() == "Stop"
    # Opening, using and closing the drawer leaves the analysis alone.
    window.show_settings(True)
    assert pump(
        window,
        app,
        lambda: window.drawer.isVisible() and "DSP p95" in window.drawer.diagnostics.text(),
    )
    assert window.drawer.geometry().left() >= window.width() - 440  # it covers only the right edge
    window.drawer.mode.setCurrentIndex(1)
    window.drawer.gain.setValue(6)
    assert window.view.canvas.mode == "radial" and window.view.canvas.display.gain_db == 6.0
    window.show_settings(False)
    assert pump(window, app, lambda: not window.drawer.isVisible())
    assert controller.running and not window.apply.isVisible()
    window.start_or_stop()
    assert pump(window, app, lambda: not controller.running)
    assert window.toggle.text() == "Start" and controller.engine.error is None
    assert window.view.canvas.condition == "stopped"


def test_engine_settings_changed_while_running_are_applied_by_a_restart(window, app, tmp_path):
    use_file(window, write_clicks(tmp_path / "clicks.wav", seconds=8))
    window.start()
    first = window.controller.engine
    assert not first.config.experimental_beat_phase
    window.drawer.beat_tracker.setChecked(True)
    window.poll()
    assert window.apply.isVisible()  # offered, not forced
    assert window.controller.running and window.controller.engine is first
    window.apply_changes()
    assert pump(
        window, app, lambda: window.controller.engine is not first and window.controller.running
    )
    assert first.done.is_set()
    assert window.controller.engine.config.experimental_beat_phase
    assert window.view.bus is window.controller.engine.bus and not window.apply.isVisible()
    assert window.view.canvas.info["beat_tracker"] is True


def test_source_can_be_changed_between_runs_without_reopening(window, app, tmp_path):
    use_file(window, write_clicks(tmp_path / "one.wav", seconds=3))
    window.start()
    window.stop()
    assert pump(window, app, lambda: not window.controller.running)
    use_file(window, write_clicks(tmp_path / "two.wav", seconds=3))
    window.start()
    assert window.controller.running and window.controller.engine.source.path.endswith("two.wav")
    assert "two.wav" in window.status.text()


def test_problems_are_reported_and_leave_the_application_usable(window, tmp_path):
    use_file(window, tmp_path / "missing.wav")
    window.start()
    assert "Unable to start" in window.status.text() and window.toggle.text() == "Start"
    assert not window.controller.running and window.view.bus is None
    use_file(window, write_clicks(tmp_path / "clicks.wav", seconds=2))
    window.drawer.activity_model.setText(str(tmp_path / "nothing.npz"))
    window.start()
    assert "weights file not found" in window.status.text() and not window.controller.running
    window.drawer.activity_model.setText("")
    window.start()
    assert window.controller.running and window.problem == ""


def test_replay_runs_inside_the_same_window_with_original_times(window, app, tmp_path):
    use_file(window, write_clicks(tmp_path / "clicks.wav", seconds=3), logging=True)
    window.start()
    engine = window.controller.engine
    assert pump(window, app, lambda: engine.done.is_set(), timeout=20)
    window.view.tick()
    live = [mark.time for mark in window.view.state.markers if mark.kind == "onset"]
    assert len(live) >= 4 and window.controller.session_path.is_dir()
    window.replay(window.controller.session_path, speed=100)
    assert window.problem == "" and "Replaying" in window.status.text()
    assert window.toggle.text() == "Stop"
    session = window.controller.replaying
    assert pump(window, app, lambda: session.finished)
    window.view.tick()
    assert window.view.state.replayed
    assert [m.time for m in window.view.state.markers if m.kind == "onset"] == pytest.approx(live)
    window.stop()
    assert window.controller.replaying is None and window.view.bus is None


def test_closing_the_window_stops_workers_and_releases_the_subscription(app, tmp_path):
    main = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    use_file(main, write_clicks(tmp_path / "clicks.wav", seconds=10))
    main.show()
    main.start()
    engine, subscription = main.controller.engine, main.view.subscription
    assert main.controller.running
    main.close()
    assert engine.done.is_set() and engine.error is None
    assert main.view.subscription is None and subscription not in engine.bus._subscribers
    assert not main.view.timer.isActive() and not main.poller.isActive()
    assert all(not thread.is_alive() for thread in engine.threads)


def test_remembered_settings_round_trip_without_private_process_details(window, tmp_path):
    use_file(window, tmp_path / "song.flac")
    window.drawer.beat_tracker.setChecked(True)
    window.drawer.mode.setCurrentIndex(1)
    window.drawer.window.setCurrentIndex(3)
    saved = window.drawer.remembered()
    assert "application" not in saved and saved["source"] == "file" and saved["window"] == 30
    other = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    other.drawer.restore(saved)
    assert other.drawer.settings().file.endswith("song.flac")
    assert other.drawer.settings().beat_tracker and other.view.canvas.mode == "radial"
    assert other.view.state.window_s == 30
    other.close()


def test_settings_become_engine_configuration():
    settings = Settings(beat_tracker=True, hop_ms=10, delayed_analysis=True)
    config = analysis_config(settings)
    assert config.experimental_beat_phase and config.hop_ms == 10 and config.advanced
    assert config.activity_model_path is None and config.guitar_model_path is None
    assert Settings(source="device").source_label() == "No output device selected"
    with pytest.raises(ValueError):
        audio_source(Settings(source="application"))
    with pytest.raises(ValueError):
        audio_source(Settings(source="device"))
