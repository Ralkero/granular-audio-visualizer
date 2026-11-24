"""The application on each system: which sources its window offers, a file analysed and its
session recorded and replayed where Linux keeps them, and Qt started on the chosen platform.

The Linux tests run on any system: ``host`` is told it is Linux, and its folders are
temporary. Where a test needs a real Linux it says so and is skipped elsewhere. No test
touches the user's settings: the ones that save settings do it in a store of their own.
"""

import json
import os
import sys
from pathlib import Path
from time import perf_counter, sleep

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import soundfile as sf
from test_host import AS_LINUX, GAMESCOPE, WAYLAND_DESKTOP, X11_DESKTOP, run_python
from test_visualizer import write_clicks

from gmv import host, paths
from gmv.app import window as app_window
from gmv.app.controller import SOURCES, Settings, audio_source, available_sources
from gmv.app.window import MainWindow
from gmv.engine import Engine
from gmv.sources.base import AudioSource
from gmv.sources.file import FileAudioSource
from gmv.visualizer.replay import ReplaySession

NOT_YET = "Live system audio capture is not available on Linux yet."


@pytest.fixture(scope="module")
def app():
    return app_window.application()


@pytest.fixture
def linux(monkeypatch, tmp_path):
    """Linux, with a home, XDG folders and a program model folder of the test's own."""
    monkeypatch.setattr(host, "system", lambda: "linux")
    (tmp_path / "home").mkdir()
    for variable in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(variable, str(tmp_path / "home"))
    for variable, folder in (
        ("XDG_CONFIG_HOME", "config"),
        ("XDG_DATA_HOME", "data"),
        ("XDG_CACHE_HOME", "cache"),
    ):
        monkeypatch.setenv(variable, str(tmp_path / folder))
    for variable in ("GMV_ACTIVITY_MODEL", "GMV_GUITAR_MODEL"):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(paths, "models_folder", lambda: tmp_path / "program-models")
    return tmp_path


def opened(**arguments):
    main = MainWindow(remember=False, discover_sources=False, **arguments)
    main.view.timer.stop()
    main.poller.stop()
    main.resize(1280, 800)
    main.show()
    return main


@pytest.fixture
def window(app, linux):
    """The application as it opens on Linux, with the folders it chooses itself."""
    main = opened()
    yield main
    main.close()


def pump(window, app, until, timeout=25):
    deadline = perf_counter() + timeout
    while perf_counter() < deadline:
        app.processEvents()
        window.poll()
        window.view.tick()
        if until():
            return True
        sleep(0.01)
    return False


def onsets(window):
    return [mark.time for mark in window.view.state.markers if mark.kind == "onset"]


# -- the sources a window offers ---------------------------------------------------------------


def test_linux_offers_system_audio_through_pipewire_and_the_audio_file(window):
    drawer = window.drawer
    offered = [(drawer.source.itemData(i), drawer.source.itemText(i)) for i in range(2)]
    assert drawer.source.count() == 2
    assert offered == [("system", "System audio (PipeWire)"), ("file", "Audio file")]
    assert drawer.settings().source == "system" and Settings().source == "system"
    for hidden in (drawer.devices, drawer.applications, drawer.refresh):
        assert not drawer.form.isRowVisible(hidden)  # no device or application to choose
    note = drawer.source_note.text()
    assert "default output" in note and "monitor" in note and "Not the microphone" in note
    assert "capture follows it" in note and NOT_YET not in note
    assert "Default system output" in window.status.text()
    drawer.source.setCurrentIndex(drawer.source.findData("file"))
    assert drawer.file.isEnabled() and "not played aloud" in drawer.source_note.text()
    assert drawer.replay.isEnabled()  # a recorded session can always be replayed


def test_system_audio_that_cannot_start_is_explained_and_the_window_carries_on(window):
    window.start()  # Linux opens on System audio; in the tests no sound server answers
    assert "Unable to start" in window.status.text() and "pactl" in window.status.text()
    assert window.toggle.text() == "Start" and not window.controller.running
    assert window.controller.engine is None


def test_windows_offers_every_source_it_always_did(app, linux, monkeypatch):
    monkeypatch.setattr(host, "system", lambda: "windows")
    assert available_sources() == SOURCES
    assert [title for _, title in SOURCES] == [
        "Default system output",
        "Specific output device",
        "Application / process tree",
        "Audio file",
    ]
    main = opened(session_root=linux / "sessions")
    try:
        drawer = main.drawer
        assert [drawer.source.itemData(i) for i in range(drawer.source.count())] == [
            "system",
            "device",
            "application",
            "file",
        ]
        assert drawer.settings().source == "system" and Settings().source == "system"
        for shown in (drawer.devices, drawer.applications, drawer.refresh):
            assert drawer.form.isRowVisible(shown)
        assert "not available" not in drawer.source_note.text()
        assert "Default system output" in main.status.text()
    finally:
        main.close()


def test_a_source_this_system_does_not_have_is_refused_in_words_not_by_a_missing_library(window):
    drawer = window.drawer
    drawer.restore({"source": "device", "device": "Speakers [Loopback]"})  # saved on Windows
    assert drawer.settings().source == "system"  # the nearest Linux has: the default output
    for kind, name in (("device", "one output device"), ("application", "one application")):
        settings = Settings(source=kind, device="Speakers", application={"pid": os.getpid()})
        with pytest.raises(ValueError) as refused:
            audio_source(settings)
        assert str(refused.value).startswith(f"Capture of {name} is not available on Linux")
        with pytest.raises(ValueError, match="is not available on Linux"):
            window.controller.prepare(settings)
    assert window.controller.engine is None
    drawer.source.setCurrentIndex(drawer.source.findData("file"))
    window.start()  # no file chosen yet: the ordinary message, and the window carries on
    assert "Choose an existing audio file" in window.status.text()
    assert window.toggle.text() == "Start"


# -- a file, its session and its replay --------------------------------------------------------


def clicks(path, **how):
    """Two seconds of clicks in the format the file's name asks for."""
    path.parent.mkdir(parents=True, exist_ok=True)
    audio, rate = sf.read(write_clicks(path.with_suffix(".source.wav"), seconds=2))
    path.with_suffix(".source.wav").unlink()
    sf.write(path, audio, rate, **how)
    return path


FORMATS = {
    "wav": {"subtype": "PCM_16"},  # the ordinary file
    "flac": {"subtype": "PCM_24"},  # lossless
    "ogg": {"format": "OGG", "subtype": "VORBIS"},
    "mp3": {"format": "MP3"},
}


@pytest.mark.parametrize("extension", FORMATS)
def test_linux_analyses_a_file_records_its_session_in_the_data_folder_and_replays_it(
    window, app, linux, extension
):
    if FORMATS[extension].get("format", extension.upper()) not in sf.available_formats():
        pytest.skip(f"this libsndfile does not write {extension}")
    audio = clicks(linux / "music" / f"clicks.{extension}", **FORMATS[extension])
    drawer = window.drawer
    drawer.source.setCurrentIndex(drawer.source.findData("file"))
    drawer.file.setText(str(audio))
    drawer.logging.setChecked(True)
    window.start()
    engine = window.controller.engine
    assert window.problem == "" and window.controller.running
    assert isinstance(engine.source, AudioSource) and engine.source.kind == "file_audio"
    assert pump(window, app, lambda: engine.done.is_set())
    window.view.tick()
    assert engine.error is None and not window.controller.running
    live = onsets(window)
    assert len(live) >= 3 and len(window.view.state.frames) > 20
    # where it was recorded: the user's data folder, nowhere else
    session = window.controller.session_path
    assert session.parent == linux / "data" / "GranularMusicVisualizer" / "sessions"
    manifest = json.loads((session / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] and manifest["source"]["kind"] == "file_audio"
    assert not manifest["raw_audio_saved"] and (session / "messages.jsonl").stat().st_size > 0
    assert [p for p in linux.rglob("session_*") if p.is_dir()] == [session]
    assert not list(linux.rglob("*.jsonl")) or session in next(linux.rglob("*.jsonl")).parents
    assert not (linux / "home" / "GranularMusicVisualizer").exists()
    # replayed without its audio: the file is gone, the times are the recorded ones
    audio.unlink()
    window.replay(session, speed=100)
    assert window.problem == "" and "Replaying" in window.status.text()
    replaying = window.controller.replaying
    assert pump(window, app, lambda: replaying.finished)
    window.view.tick()
    assert replaying.error is None and window.view.state.replayed
    assert onsets(window) == pytest.approx(live)
    window.stop()
    assert window.controller.replaying is None and window.view.bus is None


def logged(folder):
    return [
        json.loads(line)
        for line in (folder / "messages.jsonl").read_text(encoding="utf-8").splitlines()
    ]


def test_a_session_recorded_on_windows_replays_as_it_is(window, app, linux):
    """Windows writes its logs with CRLF line ends and its manifest names Windows paths.
    Replay needs neither the paths nor the audio."""
    audio = write_clicks(linux / "clicks.wav", seconds=3)
    engine = Engine(FileAudioSource(audio, realtime=False), session_root=linux / "recorded")
    engine.start()
    assert engine.done.wait(30) and engine.error is None
    original = engine.writer.path
    recorded = logged(original)
    assert len(recorded) > 100
    copy = linux / "from-windows" / "session_2026-11-20_101500_0a1b2c3d"
    copy.mkdir(parents=True)
    text = (original / "messages.jsonl").read_bytes().replace(b"\r\n", b"\n")
    (copy / "messages.jsonl").write_bytes(text.replace(b"\n", b"\r\n"))
    manifest = json.loads((original / "manifest.json").read_text(encoding="utf-8"))
    manifest["source"]["path"] = r"D:\Music\Album\clicks.wav"
    manifest["configuration"]["activity_model_path"] = r"C:\Program Files\GMV\models\a.npz"
    (copy / "manifest.json").write_bytes(
        json.dumps(manifest, indent=2).replace("\n", "\r\n").encode("utf-8")
    )
    audio.unlink()

    session = ReplaySession(copy, speed=1000)
    assert session.title == copy.name
    assert session.configuration["activity_model_path"].startswith("C:\\")
    received = session.bus.subscribe(len(recorded) + 10)
    session.start()
    session.thread.join(60)
    assert session.finished and session.error is None
    replayed = [received.get() for _ in range(len(received))]
    assert [m.timestamp for m in replayed] == [m["timestamp"] for m in recorded]
    assert [m.type for m in replayed] == [m["type"] for m in recorded]
    assert all(m.replayed for m in replayed)
    assert [m.original_published_monotonic for m in replayed] == [
        m["published_monotonic"] for m in recorded
    ]

    window.replay(copy / "messages.jsonl", speed=200)  # the file, as the dialog gives it
    assert window.problem == "" and copy.name in window.status.text()
    replaying = window.controller.replaying
    assert pump(window, app, lambda: replaying.finished)
    window.view.tick()
    assert replaying.error is None and window.view.state.replayed
    expected = [m["timestamp"] for m in recorded if m.get("event_type") == "onset"]
    assert len(expected) >= 4 and onsets(window) == pytest.approx(expected)


def test_a_sessions_folder_that_cannot_be_written_is_explained_and_nothing_stops(
    app, linux, monkeypatch
):
    blocked = linux / "data"
    blocked.write_text("a file where XDG_DATA_HOME should be a folder", encoding="utf-8")
    main = opened()
    try:
        assert main.problem == ""  # opening the window needs no folder
        main.drawer.source.setCurrentIndex(main.drawer.source.findData("file"))
        main.drawer.file.setText(str(write_clicks(linux / "clicks.wav", seconds=2)))
        main.drawer.logging.setChecked(True)
        main.start()
        assert "Unable to start: Cannot save session logs in" in main.status.text()
        assert str(main.controller.session_root) in main.status.text()
        assert not main.controller.running and main.toggle.text() == "Start"
        main.drawer.logging.setChecked(False)  # without logs there is nothing to write
        main.start()
        assert main.problem == "" and main.controller.running
        main.stop()
        assert pump(main, app, lambda: not main.controller.running)
    finally:
        main.close()
    assert blocked.is_file()


# -- models ------------------------------------------------------------------------------------


def test_linux_starts_with_the_users_models_and_never_replaces_a_chosen_file(app, linux):
    user, program = paths.user_models_folder(), paths.models_folder()
    for folder in (user, program):
        folder.mkdir(parents=True)
        (folder / "activity_v2.npz").write_bytes(b"")
    (program / "guitar_v1.npz").write_bytes(b"")
    (user / "guitar_mn10_v1.json").write_text("{}", encoding="utf-8")  # experimental: stays off
    chosen = linux / "elsewhere" / "my_weights.npz"
    chosen.parent.mkdir()
    chosen.write_bytes(b"")
    main = opened()
    try:
        drawer = main.drawer
        assert drawer.activity_model.text() == str(user / "activity_v2.npz")
        assert drawer.guitar_model.text() == str(program / "guitar_v1.npz")
        drawer.restore({"activity_model": str(chosen), "guitar_model": ""})
        assert drawer.activity_model.text() == str(chosen)
        assert not drawer.guitar_model.text().endswith(".json")
    finally:
        main.close()
    for name in ("activity_v2.npz", "guitar_v1.npz"):
        (program / name).unlink()
    (user / "activity_v2.npz").unlink()
    main = opened()
    try:  # a recognition model lying in the folder is not switched on by being there
        assert main.drawer.activity_model.text() == "" and main.drawer.guitar_model.text() == ""
    finally:
        main.close()


# -- diagnostics -------------------------------------------------------------------------------


def test_diagnostics_name_the_system_the_qt_platform_and_the_folders(window, app, linux):
    lines = window.describe_system()
    assert lines["System"].startswith("Linux · Qt platform " + app.platformName())
    assert lines["Sessions folder"] == str(linux / "data/GranularMusicVisualizer/sessions")
    assert lines["Settings"]
    window.show_settings(True)
    assert pump(window, app, lambda: "Sessions folder" in window.drawer.diagnostics.text())


# -- Qt starts on the chosen platform ----------------------------------------------------------


class Started:
    """Stands in for ``QApplication``: notes what Qt would have read when it started."""

    seen = {}

    def __init__(self, arguments):
        Started.seen = {"QT_QPA_PLATFORM": os.environ.get("QT_QPA_PLATFORM")}

    def setProperty(self, name, value):
        Started.seen[name] = value

    def platformName(self):
        return Started.seen["QT_QPA_PLATFORM"] or "chosen by Qt"


def session(monkeypatch, environ):
    for variable in (
        "QT_QPA_PLATFORM",
        "GMV_QT_PLATFORM",
        "WAYLAND_DISPLAY",
        "DISPLAY",
        "GAMESCOPE_WAYLAND_DISPLAY",
        "XDG_SESSION_TYPE",
        "XDG_CURRENT_DESKTOP",
    ):
        monkeypatch.delenv(variable, raising=False)
    for variable, value in environ.items():
        monkeypatch.setenv(variable, value)
    monkeypatch.setattr(sys, "argv", ["gmv-visualizer"])
    monkeypatch.setattr(app_window, "QApplication", Started)
    monkeypatch.setattr(app_window, "platform_plugin_problem", lambda name: "")


def test_qt_is_started_only_after_the_platform_is_chosen(linux, monkeypatch, capsys):
    session(monkeypatch, GAMESCOPE)
    app_window.start_qt()
    assert Started.seen["QT_QPA_PLATFORM"] == "xcb"  # already there when Qt started
    assert "Gamescope" in Started.seen["gmv_qt_platform"]
    assert "gmv: Qt platform xcb (inside Gamescope" in capsys.readouterr().err
    session(monkeypatch, {**GAMESCOPE, "GMV_QT_PLATFORM": "auto"})
    app_window.start_qt()
    assert Started.seen["QT_QPA_PLATFORM"] is None
    session(monkeypatch, {**WAYLAND_DESKTOP, "GMV_QT_PLATFORM": "xcb"})
    app_window.start_qt()
    assert Started.seen["QT_QPA_PLATFORM"] == "xcb"
    assert Started.seen["gmv_qt_platform"] == "GMV_QT_PLATFORM=xcb"


@pytest.mark.parametrize("desktop", [WAYLAND_DESKTOP, X11_DESKTOP], ids=["wayland", "x11"])
def test_a_desktop_session_starts_qt_with_nothing_forced(linux, monkeypatch, capsys, desktop):
    session(monkeypatch, desktop)
    app_window.start_qt()
    assert Started.seen["QT_QPA_PLATFORM"] is None and "QT_QPA_PLATFORM" not in os.environ
    assert Started.seen["gmv_qt_platform"] == "Qt's own choice"
    assert "gmv: Qt platform chosen by Qt (Qt's own choice)" in capsys.readouterr().err


def test_windows_starts_qt_as_before_and_says_nothing(linux, monkeypatch, capsys):
    monkeypatch.setattr(host, "system", lambda: "windows")
    session(monkeypatch, {})
    app_window.start_qt()
    assert Started.seen["QT_QPA_PLATFORM"] is None
    assert capsys.readouterr().err == ""


def test_a_platform_that_cannot_start_is_explained_before_qt_ends_the_process(
    linux, monkeypatch
):
    def never(arguments):
        raise AssertionError("Qt was started on a platform known not to work")

    session(monkeypatch, {"GMV_QT_PLATFORM": "xcb"})  # and no DISPLAY
    monkeypatch.setattr(app_window, "platform_plugin_problem", real_problem)
    monkeypatch.setattr(app_window, "QApplication", never)
    with pytest.raises(SystemExit) as stopped:
        app_window.start_qt()
    message = str(stopped.value)
    assert 'cannot start on the Qt platform "xcb" (GMV_QT_PLATFORM=xcb)' in message
    assert "DISPLAY is not set" in message and "GMV_QT_PLATFORM=auto" in message
    assert "no platform plugin of that name" in real_problem("no-such-platform")
    assert real_problem("wayland;xcb") == ""  # a list is Qt's to work through
    monkeypatch.setattr(host, "system", lambda: "windows")
    assert real_problem("xcb") == ""  # only looked into on Linux


real_problem = app_window.platform_plugin_problem


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="loads Linux's Qt plugins")
def test_on_linux_the_plugins_this_qt_has_are_found_loadable():
    assert real_problem("offscreen") == ""  # the plugin these tests run on
    # "" where the system has the libraries a plugin needs; otherwise the missing one is named
    assert real_problem("wayland") == "" or "missing" in real_problem("wayland")
    if os.environ.get("DISPLAY"):
        assert real_problem("xcb") == "" or "missing" in real_problem("xcb")


def test_the_override_reaches_a_real_qt():
    code = (
        "import sys\n"
        "from gmv.app.window import application\n"
        "app = application()\n"
        "print(app.platformName())\n"
    )
    done = run_python(code, QT_QPA_PLATFORM=None, GMV_QT_PLATFORM="offscreen")
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "offscreen"
    assert "gmv: Qt platform offscreen (GMV_QT_PLATFORM=offscreen)" in done.stderr
    done = run_python(code, QT_QPA_PLATFORM="minimal", GMV_QT_PLATFORM="offscreen")
    assert done.returncode == 0 and done.stdout.strip() == "minimal", done.stderr


# -- settings ----------------------------------------------------------------------------------

REMEMBER = """
import json, sys
from PySide6.QtCore import QSettings
from gmv.app.window import MainWindow, application

app = application()
if sys.argv[2] != "native":  # a store of the test's own, never the user's
    QSettings.setDefaultFormat(QSettings.Format.IniFormat)
    QSettings.setPath(QSettings.Format.IniFormat, QSettings.Scope.UserScope, sys.argv[1])
if sys.argv[3] == "spoil":
    QSettings().setValue("settings", "{not json")
    raise SystemExit(0)
window = MainWindow(session_root=sys.argv[1], discover_sources=False)
drawer = window.drawer
if sys.argv[3] == "save":
    drawer.source.setCurrentIndex(drawer.source.findData("file"))
    drawer.file.setText("song.flac")
    drawer.mode.setCurrentIndex(drawer.mode.findData("radial"))
    drawer.expressive.setChecked(True)
print(json.dumps({
    "store": QSettings().fileName(), "problem": window.problem, "file": drawer.file.text(),
    "source": drawer.source.currentData(), "mode": drawer.mode.currentData(),
    "expressive": drawer.expressive.isChecked(),
    "unreadable": QSettings().value("settings_unreadable", ""),
}))
window.close()
"""


def remember(folder, step, store="own", prelude="", **environment):
    done = run_python(prelude + REMEMBER, str(folder), store, step, **environment)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout.splitlines()[-1]) if done.stdout.strip() else None


@pytest.mark.parametrize("prelude", ["", AS_LINUX], ids=["this system", "as Linux"])
def test_settings_are_remembered_between_runs_and_unreadable_ones_are_kept(tmp_path, prelude):
    first = remember(tmp_path, "save", prelude=prelude)
    assert Path(first["store"]).parent.parent == tmp_path and first["problem"] == ""
    again = remember(tmp_path, "open", prelude=prelude)
    assert again["file"] == "song.flac" and again["source"] == "file"
    assert again["mode"] == "radial" and again["expressive"] and again["problem"] == ""
    remember(tmp_path, "spoil", prelude=prelude)
    spoiled = remember(tmp_path, "open", prelude=prelude)
    assert "could not be read" in spoiled["problem"] and spoiled["unreadable"] == "{not json"
    assert spoiled["file"] == "" and spoiled["mode"] == "bars"  # the defaults, not a crash
    after = remember(tmp_path, "open", prelude=prelude)
    assert after["problem"] == "" and after["unreadable"] == "{not json"  # still kept


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="Qt's own settings file on Linux")
def test_on_linux_the_settings_file_is_in_the_xdg_configuration_folder(tmp_path):
    environment = {
        "HOME": str(tmp_path / "home"),
        "XDG_CONFIG_HOME": str(tmp_path / "config"),
        "XDG_DATA_HOME": str(tmp_path / "data"),
        "XDG_CACHE_HOME": str(tmp_path / "cache"),
    }
    (tmp_path / "home").mkdir()
    first = remember(tmp_path / "sessions", "save", store="native", **environment)
    expected = tmp_path / "config" / "GranularMusicVisualizer" / "Granular Music Visualizer.conf"
    assert Path(first["store"]) == expected and expected.is_file()
    assert expected.parent == Path(
        run_python(
            "from gmv import host; print(host.config_folder())", **environment
        ).stdout.strip()
    )
    again = remember(tmp_path / "sessions", "open", store="native", **environment)
    assert again["file"] == "song.flac" and again["mode"] == "radial" and again["expressive"]
    assert not (tmp_path / "home" / ".config").exists()  # not the default place, when told another


# -- the command line --------------------------------------------------------------------------

COMMAND = "import sys\nfrom gmv.cli import main\nmain(sys.argv[1:])\n"


def test_the_linux_command_line_records_in_the_data_folder_from_any_directory(tmp_path):
    audio = write_clicks(tmp_path / "clicks.wav", seconds=1)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    environment = {"XDG_DATA_HOME": str(tmp_path / "data")}
    done = run_python(AS_LINUX + COMMAND, "file", str(audio), cwd=elsewhere, **environment)
    assert done.returncode == 0, done.stderr
    sessions = tmp_path / "data" / "GranularMusicVisualizer" / "sessions"
    recorded = [folder for folder in sessions.iterdir() if folder.name.startswith("session_")]
    assert len(recorded) == 1 and (recorded[0] / "messages.jsonl").stat().st_size > 0
    assert f"Session: {recorded[0]}" in done.stdout
    assert list(elsewhere.iterdir()) == []  # nothing in the working directory
    named = run_python(
        AS_LINUX + COMMAND, "file", str(audio), "--sessions", "mine", cwd=elsewhere, **environment
    )
    assert named.returncode == 0 and len(list((elsewhere / "mine").iterdir())) == 1


def test_the_linux_command_line_explains_capture_it_cannot_do(tmp_path):
    # In the tests no sound server answers (conftest), and on a system pretending to be
    # Linux pactl may not even be installed: either way one sentence naming pactl.
    done = run_python(AS_LINUX + COMMAND, "capture", "--seconds", "1", cwd=tmp_path)
    assert done.returncode == 1 and "pactl" in done.stderr
    assert "Traceback" not in done.stderr and list(tmp_path.iterdir()) == []
    device = run_python(AS_LINUX + COMMAND, "capture", "--device", "Speakers", cwd=tmp_path)
    assert device.returncode == 1 and device.stderr.strip().startswith(
        "Capture of one output device is not available on Linux"
    )
    devices = run_python(AS_LINUX + COMMAND, "devices", cwd=tmp_path)
    assert devices.returncode == 0 and json.loads(devices.stdout) == []
