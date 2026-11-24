"""The platform boundary: ``gmv.host`` and what ``gmv.paths`` and ``gmv.sources`` build on it.

Each system's answers are asked on whatever system the tests run on, by telling ``host``
which system it is. Homes, XDG folders and ``%LOCALAPPDATA%`` are temporary folders: nothing
here reads or writes the user's own settings, sessions or models.
"""

import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from gmv import host, paths, sources

REPOSITORY = Path(__file__).resolve().parents[1]
XDG = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")
CAPTURE_MODULES = (
    "gmv.sources.windows",
    "gmv.sources.system_native",
    "gmv.sources.application",
    "pyaudiowpatch",
    "pycaw",
    "comtypes",
)
AS_LINUX = "import gmv.host as host; host.system = lambda: 'linux'\n"


def run_python(code, *arguments, cwd=None, **environment):
    """Run ``code`` in an interpreter of its own; a value of None removes a variable."""
    env = {**os.environ, "QT_QPA_PLATFORM": "offscreen", **environment}
    env = {key: value for key, value in env.items() if value is not None}
    return subprocess.run(
        [sys.executable, "-c", code, *arguments],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


# -- which system ------------------------------------------------------------------------------


def test_the_system_is_named_from_the_platform(monkeypatch):
    for platform, expected in (("win32", "windows"), ("linux", "linux"), ("darwin", "other")):
        monkeypatch.setattr(sys, "platform", platform)
        assert host.system() == expected
    monkeypatch.undo()
    assert host.system() in ("windows", "linux", "other")


def test_the_clock_every_timestamp_comes_from_is_monotonic():
    assert time.get_clock_info("perf_counter").monotonic


# -- Linux folders -----------------------------------------------------------------------------


def test_linux_folders_default_to_the_xdg_locations_under_home(linux):
    assert host.config_folder() == linux / ".config" / "GranularMusicVisualizer"
    assert host.data_folder() == linux / ".local" / "share" / "GranularMusicVisualizer"
    assert host.cache_folder() == linux / ".cache" / "GranularMusicVisualizer"
    assert paths.sessions_folder() == host.data_folder() / "sessions"
    assert paths.user_models_folder() == host.data_folder() / "models"


def test_linux_folders_follow_the_xdg_variables(linux, monkeypatch, tmp_path):
    for variable in XDG:
        monkeypatch.setenv(variable, str(tmp_path / variable.lower()))
    assert host.config_folder() == tmp_path / "xdg_config_home" / "GranularMusicVisualizer"
    assert host.data_folder() == tmp_path / "xdg_data_home" / "GranularMusicVisualizer"
    assert host.cache_folder() == tmp_path / "xdg_cache_home" / "GranularMusicVisualizer"
    assert paths.sessions_folder() == tmp_path / "xdg_data_home/GranularMusicVisualizer/sessions"
    assert paths.command_line_sessions() == paths.sessions_folder()


@pytest.mark.parametrize("value", ["", "relative/folder", "."])
def test_an_empty_or_relative_xdg_variable_is_ignored(linux, monkeypatch, value):
    for variable in XDG:
        monkeypatch.setenv(variable, value)
    assert host.config_folder() == linux / ".config" / "GranularMusicVisualizer"
    assert host.data_folder() == linux / ".local" / "share" / "GranularMusicVisualizer"
    assert host.cache_folder() == linux / ".cache" / "GranularMusicVisualizer"


def test_linux_keeps_what_the_user_records_out_of_the_repository_and_the_program(
    linux, monkeypatch, tmp_path
):
    for folder in (paths.sessions_folder(), paths.user_models_folder(), host.cache_folder()):
        assert folder.is_absolute() and REPOSITORY not in folder.parents
    # the same when the program is a packaged one: beside the executable is not the place
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "opt" / "gmv" / "gmv"))
    assert paths.sessions_folder() == host.data_folder() / "sessions"
    assert tmp_path / "opt" not in paths.sessions_folder().parents


def test_asking_for_a_folder_makes_nothing(linux):
    host.config_folder(), host.data_folder(), host.cache_folder()
    paths.sessions_folder(), paths.model_folders(), paths.default_model("activity_v2.npz")
    assert list(linux.iterdir()) == []


# -- Linux models ------------------------------------------------------------------------------


def test_linux_looks_for_models_in_the_users_folder_first_then_the_programs(
    linux, monkeypatch, tmp_path
):
    program = tmp_path / "checkout" / ".model-cache"
    monkeypatch.setattr(paths, "models_folder", lambda: program)
    user = paths.user_models_folder()
    assert paths.model_folders() == (user, program)
    assert paths.default_model("activity_v2.npz") is None  # neither folder exists yet
    assert paths.models_start_folder() == program
    program.mkdir(parents=True)
    (program / "activity_v2.npz").write_bytes(b"program")
    assert paths.default_model("activity_v2.npz") == program / "activity_v2.npz"
    user.mkdir(parents=True)
    assert paths.models_start_folder() == user
    assert paths.default_model("activity_v2.npz") == program / "activity_v2.npz"
    (user / "activity_v2.npz").write_bytes(b"user")
    assert paths.default_model("activity_v2.npz") == user / "activity_v2.npz"
    # only the named file: other models lying in a folder are never picked up
    (user / "guitar_mn10_v1.json").write_text("{}", encoding="utf-8")
    assert paths.default_model("guitar_v1.npz") is None


# -- making folders ----------------------------------------------------------------------------


def test_a_missing_folder_is_made_when_it_is_needed(linux):
    folder = paths.sessions_folder()
    assert not folder.exists() and not host.data_folder().exists()
    assert paths.writable_folder(folder, "session logs") == folder
    assert folder.is_dir() and list(folder.iterdir()) == []
    assert paths.writable_folder(folder, "session logs") == folder  # and again, when it exists


def test_a_folder_that_cannot_be_made_says_which_what_for_and_why(tmp_path):
    blocker = tmp_path / "not-a-folder"
    blocker.write_text("a file where a folder is wanted", encoding="utf-8")
    with pytest.raises(RuntimeError) as failure:
        paths.writable_folder(blocker / "sessions", "session logs")
    message = str(failure.value)
    assert "Cannot save session logs in" in message and str(blocker / "sessions") in message
    assert message.count(":") >= 1 and "Traceback" not in message
    assert blocker.read_text(encoding="utf-8").startswith("a file")


@pytest.mark.skipif(
    sys.platform == "win32" or getattr(os, "geteuid", lambda: 0)() == 0,
    reason="needs POSIX permissions and a user they apply to",
)
def test_a_read_only_folder_is_refused_with_the_reason(tmp_path):
    folder = tmp_path / "sessions"
    folder.mkdir()
    folder.chmod(0o555)
    try:
        with pytest.raises(RuntimeError, match="Permission denied"):
            paths.writable_folder(folder, "session logs")
        with pytest.raises(RuntimeError, match="Permission denied"):
            paths.writable_folder(folder / "deeper", "session logs")
    finally:
        folder.chmod(0o755)


# -- Windows folders: as they always were ------------------------------------------------------


def test_a_windows_checkout_keeps_its_folders_in_the_repository(windows):
    assert paths.sessions_folder() == REPOSITORY / "sessions"
    assert paths.models_folder() == REPOSITORY / ".model-cache"
    assert paths.user_models_folder() is None
    assert paths.model_folders() == (REPOSITORY / ".model-cache",)
    assert paths.models_start_folder() == REPOSITORY / ".model-cache"
    assert paths.command_line_sessions() == Path("sessions")
    assert host.config_folder() is None  # settings are in the registry
    assert paths.icon() in (None, REPOSITORY / "packaging" / "app.ico")


def test_a_packaged_windows_build_keeps_its_folders_where_it_always_did(
    windows, monkeypatch, tmp_path, home
):
    program = tmp_path / "app"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(program / "app.exe"))
    assert paths.sessions_folder() == windows / "GranularMusicVisualizer" / "sessions"
    assert paths.models_folder() == program / "models"
    assert paths.model_folders() == (program / "models",)
    assert paths.capture_helper() == program / "ProcessLoopback" / "ProcessLoopback.exe"
    monkeypatch.delenv("LOCALAPPDATA")
    assert paths.sessions_folder() == home / "GranularMusicVisualizer" / "sessions"


def test_the_capture_helper_is_only_ever_available_on_windows(monkeypatch, tmp_path):
    helper = tmp_path / "ProcessLoopback.exe"
    helper.write_bytes(b"")
    monkeypatch.setattr(paths, "capture_helper", lambda: helper)
    monkeypatch.setattr(host, "system", lambda: "windows")
    assert paths.capture_helper_available()
    monkeypatch.setattr(host, "system", lambda: "linux")
    assert not paths.capture_helper_available()


# -- the working directory ---------------------------------------------------------------------


@pytest.mark.parametrize("prelude", ["", AS_LINUX], ids=["this system", "as Linux"])
def test_no_path_depends_on_the_working_directory(tmp_path, prelude):
    code = prelude + (
        "from gmv import host, paths\n"
        "print(paths.sessions_folder()); print(paths.models_folder()); print(host.data_folder())\n"
        "print(paths.model_folders()); print(paths.capture_helper()); print(paths.icon())\n"
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    here, there = run_python(code, cwd=REPOSITORY), run_python(code, cwd=elsewhere)
    assert here.returncode == 0 and there.returncode == 0, here.stderr + there.stderr
    assert here.stdout == there.stdout
    for line in here.stdout.splitlines()[:3]:
        assert Path(line).is_absolute()
    assert list(elsewhere.iterdir()) == []


# -- audio sources -----------------------------------------------------------------------------


def test_windows_has_every_capture_source_and_the_file(windows):
    assert host.audio_sources() == ("system", "device", "application", "file")
    assert host.live_capture() and host.capture_note() == ""
    for key in host.audio_sources():
        sources.require(key)  # nothing is refused


def test_a_system_without_a_capture_backend_has_the_file_and_nothing_that_pretends(monkeypatch):
    monkeypatch.setattr(host, "system", lambda: "other")
    assert host.audio_sources() == ("file",)
    assert not host.live_capture()
    sources.require("file")
    for key in ("system", "device", "application"):
        with pytest.raises(sources.Unavailable, match="not available on"):
            sources.require(key)
    assert sources.list_output_devices() == [] and sources.list_audio_applications() == []


def test_linux_has_system_capture_and_the_file_but_no_device_or_application(linux):
    assert host.audio_sources() == ("system", "file")
    assert host.live_capture() and host.capture_note() == ""
    sources.require("system")
    sources.require("file")
    for key, name in (("device", "one output device"), ("application", "one application")):
        with pytest.raises(sources.Unavailable, match=f"Capture of {name} is not available"):
            sources.require(key)
    assert sources.list_output_devices() == [] and sources.list_audio_applications() == []


def test_linux_capture_is_built_by_the_pulseaudio_backend(linux, monkeypatch):
    from gmv.sources import system_pulse

    asked = []
    monkeypatch.setattr(
        system_pulse, "system_audio_source", lambda *a: asked.append(a) or "pulse source"
    )
    assert sources.system_audio_source() == "pulse source"
    assert asked == [(None, "auto")]
    for build in (
        lambda: sources.system_audio_source("Speakers [Loopback]"),
        lambda: sources.application_audio_source(pid=os.getpid()),
    ):
        with pytest.raises(sources.Unavailable) as refused:
            build()
        message = str(refused.value)
        assert "is not available on Linux" in message
        assert "audio file" in message and "replay" in message and "WASAPI" not in message
    assert asked == [(None, "auto")]  # nothing more was built


def test_windows_capture_is_built_by_the_windows_backend(windows, monkeypatch):
    from gmv.sources import application, system_native

    asked = []
    monkeypatch.setattr(
        system_native, "system_audio_source", lambda *a: asked.append(a) or "system source"
    )
    monkeypatch.setattr(
        application, "ApplicationAudioSource", lambda **k: asked.append(k) or "application source"
    )
    assert sources.system_audio_source() == "system source"
    assert sources.system_audio_source("Speakers [Loopback]", "portaudio") == "system source"
    assert sources.application_audio_source(pid=7) == "application source"
    assert asked == [(None, "auto"), ("Speakers [Loopback]", "portaudio"), {"pid": 7}]


def test_linux_never_loads_a_windows_capture_backend(tmp_path):
    """Starting the application, looking for sources, asking for system capture and being
    refused the sources Linux does not have load nothing of the Windows backends: not the
    modules, not their libraries. (No sound server answers in the tests, so system capture
    is refused here too, in a sentence about pactl.)"""
    code = AS_LINUX + (
        "import sys\n"
        "from gmv.app.controller import Settings, audio_source, available_sources, discover\n"
        "from gmv.app.window import MainWindow, application\n"
        "app = application()\n"
        f"window = MainWindow(session_root=r'{tmp_path}', remember=False, discover_sources=False)\n"
        "assert discover() == ([], [], ''), discover()\n"
        "assert [key for key, _ in available_sources()] == ['system', 'file']\n"
        "expected = {'system': 'pactl', 'device': 'not available on Linux',\n"
        "            'application': 'not available on Linux'}\n"
        "for kind, said in expected.items():\n"
        "    try:\n"
        "        audio_source(Settings(source=kind, device='x', application={'pid': 1}))\n"
        "    except ValueError as refused:\n"
        "        assert said in str(refused), refused\n"
        "    else:\n"
        "        raise SystemExit(kind + ' was built')\n"
        "window.close()\n"
        f"loaded = [name for name in {CAPTURE_MODULES!r} if name in sys.modules]\n"
        "assert not loaded, loaded\n"
    )
    done = run_python(code)
    assert done.returncode == 0, done.stderr


# -- the Qt platform plugin --------------------------------------------------------------------

WAYLAND_DESKTOP = {
    "XDG_SESSION_TYPE": "wayland",
    "XDG_CURRENT_DESKTOP": "KDE",
    "WAYLAND_DISPLAY": "wayland-0",
    "DISPLAY": ":1",  # its Xwayland
}
X11_DESKTOP = {"XDG_SESSION_TYPE": "x11", "XDG_CURRENT_DESKTOP": "KDE", "DISPLAY": ":0"}
# What Gamescope gives the programs it runs (its main.cpp): its own socket's name, its
# Xwayland, and no WAYLAND_DISPLAY.
GAMESCOPE = {
    "XDG_SESSION_TYPE": "x11",
    "XDG_CURRENT_DESKTOP": "gamescope",
    "GAMESCOPE_WAYLAND_DISPLAY": "gamescope-0",
    "DISPLAY": ":1",
}


@pytest.mark.parametrize("session", [WAYLAND_DESKTOP, X11_DESKTOP], ids=["wayland", "x11"])
def test_an_ordinary_desktop_session_is_left_to_qt(linux, session):
    environ = dict(session)
    assert host.qt_platform(environ) == (None, "Qt's own choice")
    assert host.select_qt_platform(environ) == (None, "Qt's own choice")
    assert environ == session and "QT_QPA_PLATFORM" not in environ
    assert not host.gamescope(environ)


def test_gamescope_gets_the_x11_plugin(linux):
    environ = dict(GAMESCOPE)
    assert host.gamescope(environ)
    chosen, reason = host.select_qt_platform(environ)
    assert chosen == "xcb" and "Gamescope" in reason
    assert environ["QT_QPA_PLATFORM"] == "xcb"
    # a session script may leave the login session's type in place: still Gamescope
    assert host.qt_platform({**GAMESCOPE, "XDG_SESSION_TYPE": "wayland"})[0] == "xcb"


@pytest.mark.parametrize(
    "environ",
    [
        {**WAYLAND_DESKTOP, "GAMESCOPE_WAYLAND_DISPLAY": "gamescope-0"},  # a stray variable
        {**GAMESCOPE, "WAYLAND_DISPLAY": "gamescope-0"},  # gamescope --expose-wayland
        {key: value for key, value in GAMESCOPE.items() if key != "DISPLAY"},  # no X11 at all
        {**X11_DESKTOP, "GAMESCOPE_WAYLAND_DISPLAY": ""},
        {**X11_DESKTOP, "XDG_CURRENT_DESKTOP": "gamescope"},  # the name alone proves nothing
    ],
    ids=["stray variable", "exposed wayland", "no display", "empty variable", "desktop name"],
)
def test_nothing_short_of_gamescope_with_only_x11_forces_the_plugin(linux, environ):
    assert not host.gamescope(environ)
    assert host.qt_platform(environ)[0] is None


def test_the_override_names_a_plugin_or_switches_the_rule_off(linux):
    chosen, reason = host.qt_platform({**WAYLAND_DESKTOP, "GMV_QT_PLATFORM": "xcb"})
    assert chosen == "xcb" and reason == "GMV_QT_PLATFORM=xcb"
    assert host.qt_platform({**GAMESCOPE, "GMV_QT_PLATFORM": "wayland"})[0] == "wayland"
    chosen, reason = host.qt_platform({**GAMESCOPE, "GMV_QT_PLATFORM": "auto"})
    assert chosen is None and "GMV_QT_PLATFORM=auto" in reason
    assert host.qt_platform({**GAMESCOPE, "GMV_QT_PLATFORM": " AUTO "})[0] is None


def test_qts_own_ways_of_choosing_are_never_overruled(linux):
    for extra in ({}, {"GMV_QT_PLATFORM": "xcb"}):
        environ = {**GAMESCOPE, **extra, "QT_QPA_PLATFORM": "wayland;xcb"}
        chosen, reason = host.select_qt_platform(environ)
        assert chosen is None and "QT_QPA_PLATFORM=wayland;xcb" in reason
        assert environ["QT_QPA_PLATFORM"] == "wayland;xcb"
    environ = dict(GAMESCOPE)
    chosen, reason = host.select_qt_platform(environ, ["gmv", "-platform", "wayland"])
    assert chosen is None and "-platform" in reason and "QT_QPA_PLATFORM" not in environ


def test_the_gamescope_rule_is_for_linux_and_the_override_for_everyone(windows):
    assert host.qt_platform(dict(GAMESCOPE)) == (None, "Qt's own choice")
    assert host.qt_platform({"GMV_QT_PLATFORM": "offscreen"})[0] == "offscreen"
    assert host.qt_platform({}) == (None, "Qt's own choice")
