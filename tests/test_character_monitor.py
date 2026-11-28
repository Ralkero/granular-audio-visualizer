"""Sonic Character, Phase B: the monitor window and its place in the application. Offscreen,
generated audio and generated frames only."""

import os
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
import soundfile as sf
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QKeyEvent
from PySide6.QtCore import QEvent
from test_app import pump, use_file
from test_character import RATE, dense_audio
from test_character_live import silence, stream

from gmv.app.character_monitor import CharacterMonitor
from gmv.app.window import MainWindow
from gmv.character.live import LiveCharacter
from gmv.character.profile import DIMENSION_NAMES


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


def render(widget):
    image = QImage(widget.size(), QImage.Format.Format_ARGB32)
    widget.render(image)
    return image


def test_the_monitor_shows_nothing_running_until_there_is_data(app):
    live = LiveCharacter()
    monitor = CharacterMonitor(live)
    monitor.show()
    app.processEvents()
    assert "Not running" in monitor.status.text()
    assert monitor.readout.describe() == "No values yet."
    render(monitor)  # draws without data
    monitor.close()


def test_the_monitor_shows_all_seven_values_their_confidence_and_low_signal(app):
    live = LiveCharacter()
    monitor = CharacterMonitor(live)
    monitor.show()
    live.process(stream(1.0))  # one second: some dimensions cannot be measured yet
    monitor.refresh()
    text = monitor.readout.describe()
    assert all(name.capitalize() in text for name in DIMENSION_NAMES)
    assert "not enough signal" in text and "Dynamics: not enough signal" in text
    assert "Under five seconds of music" in monitor.warnings.text()
    live.process(stream(30.0, start=1.0))
    monitor.window_choice.setCurrentIndex(monitor.window_choice.findData("medium"))
    monitor.refresh()
    text = monitor.readout.describe()
    assert "Brightness 0." in text and "confidence" in text
    assert "Measuring" in monitor.status.text()
    render(monitor)
    monitor.close()


def test_the_monitor_redraws_only_when_something_new_arrived(app):
    live = LiveCharacter()
    monitor = CharacterMonitor(live)
    monitor.show()
    live.process(stream(5.0))
    monitor.refresh()
    redraws = monitor.redraws
    for _ in range(5):
        monitor.refresh()
    assert monitor.redraws == redraws and monitor.refreshes >= 6
    live.process(stream(1.0, start=5.0))
    monitor.refresh()
    assert monitor.redraws == redraws + 1
    monitor.close()


def test_the_rolling_history_follows_the_stream_and_its_length_is_fixed(app):
    live = LiveCharacter(sample_s=0.5, history_s=20.0)
    monitor = CharacterMonitor(live)
    monitor.show()
    live.process(stream(30.0))
    monitor.refresh()
    assert monitor.plot.history and len(monitor.plot.history) == 40
    assert monitor.plot.window == "short"
    first_end = monitor.plot.history[-1].time
    live.process(stream(5.0, start=30.0, spectral_centroid_hz=3000.0))
    monitor.refresh()
    assert monitor.plot.history[-1].time > first_end and len(monitor.plot.history) == 40
    render(monitor)
    monitor.close()


def test_silence_reads_as_held_not_as_zero(app):
    live = LiveCharacter()
    monitor = CharacterMonitor(live)
    monitor.show()
    live.process(stream(8.0) + silence(3.0, 8.0))
    monitor.refresh()
    assert "No music at the moment" in monitor.status.text()
    assert "held" in monitor.warnings.text()
    monitor.close()


def test_the_timer_runs_only_while_the_monitor_is_open(app):
    monitor = CharacterMonitor(LiveCharacter())
    assert not monitor.timer.isActive()
    monitor.show()
    app.processEvents()
    assert monitor.timer.isActive() and monitor.timer.interval() == 250
    monitor.hide()
    assert not monitor.timer.isActive()


# -- in the application ------------------------------------------------------------------------------


def test_the_character_button_and_the_c_key_open_and_close_the_monitor(window, app):
    assert not window.monitor.isVisible() and not window.character_button.isChecked()
    window.character_button.click()
    app.processEvents()
    assert window.monitor.isVisible()
    window.monitor.close_button.click()
    app.processEvents()
    assert not window.monitor.isVisible() and not window.character_button.isChecked()
    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier))
    assert window.monitor.isVisible() and window.character_button.isChecked()
    window.keyPressEvent(QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.NoModifier))
    assert not window.monitor.isVisible()


def write_music(path, seconds=8.0):
    sf.write(path, dense_audio(seconds), RATE, subtype="FLOAT")
    return path


def test_a_running_session_feeds_the_monitor_and_stop_keeps_the_last_values(window, app, tmp_path):
    use_file(window, write_music(tmp_path / "music.wav"))
    window.character_button.click()
    window.start()
    assert window.controller.running and window.character.attached
    assert window.character.bus is window.view.bus  # the same bus, no second pipeline
    assert pump(window, app, lambda: window.character.snapshot() is not None, timeout=30)
    window.monitor.refresh()
    assert "Measuring" in window.monitor.status.text()
    window.stop()
    assert pump(window, app, lambda: not window.controller.running and not window.character.attached, timeout=30)
    window.monitor.refresh()
    assert "Stopped, last values" in window.monitor.status.text()
    assert window.character.snapshot() is not None


def test_a_new_start_begins_the_monitor_afresh(window, app, tmp_path):
    path = write_music(tmp_path / "music.wav", seconds=5.0)
    use_file(window, path)
    window.start()
    assert pump(window, app, lambda: window.character.snapshot() is not None, timeout=30)
    first = window.character.snapshot().session_id
    window.stop()
    assert pump(window, app, lambda: not window.controller.running, timeout=30)
    window.start()
    assert window.character.snapshot() is None  # afresh, before the new session's first frame
    assert pump(window, app, lambda: window.character.snapshot() is not None, timeout=30)
    assert window.character.snapshot().session_id != first
    window.stop()
    assert pump(window, app, lambda: not window.controller.running, timeout=30)


def test_the_visualizer_draws_the_same_with_the_monitor_open_or_closed(window, app, tmp_path):
    """The monitor is a separate window reading snapshots: the view's state is unaffected."""
    path = write_music(tmp_path / "music.wav", seconds=4.0)

    def run(open_monitor):
        window.monitor.setVisible(open_monitor)
        use_file(window, path)
        window.start()
        assert pump(window, app, lambda: not window.controller.running, timeout=40)
        deadline = perf_counter() + 5
        while len(window.view.subscription or ()) and perf_counter() < deadline:
            window.view.tick()
            app.processEvents()
            sleep(0.01)
        state = window.view.state
        result = (len(state.frames), round(state.musical.value, 9), window.view.canvas.mode)
        window.stop()
        return result

    closed, opened = run(False), run(True)
    assert closed[0] > 50 and closed == opened


def test_closing_the_application_stops_the_live_analysis(app, tmp_path):
    main = MainWindow(session_root=tmp_path / "sessions", remember=False, discover_sources=False)
    main.view.timer.stop()
    main.poller.stop()
    main.show()
    use_file(main, write_music(tmp_path / "music.wav", seconds=6.0))
    main.start()
    assert main.character.attached
    main.close()
    assert not main.character.attached and not main.monitor.isVisible()
    assert main.controller.engine.done.wait(10)


def test_values_drawn_are_numbers_between_zero_and_one(app):
    live = LiveCharacter()
    live.process(stream(20.0))
    reading = live.snapshot().readings["short"]
    assert all(v is None or 0.0 <= v <= 1.0 for v in reading.values)
    assert all(0.0 <= c <= 1.0 for c in reading.confidences)
    assert np.isfinite([c for c in reading.confidences]).all()
