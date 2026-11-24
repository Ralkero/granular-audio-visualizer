"""Fixtures shared by the tests: homes and systems of their own, and the ones that draw."""

import json
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# No test records what this machine plays: the PulseAudio client programs (pactl, parec)
# are pointed at a server that does not exist, in this process and in every process it
# starts. Tests of the Linux capture backend use tests/fake_pulse.py instead.
os.environ["PULSE_SERVER"] = "unix:/nonexistent/gmv-tests-have-no-sound-server"

import pytest


XDG = ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME")


@pytest.fixture
def home(monkeypatch, tmp_path):
    """A home directory of the test's own, with no XDG variable set."""
    folder = tmp_path / "home"
    folder.mkdir()
    for variable in ("HOME", "USERPROFILE"):
        monkeypatch.setenv(variable, str(folder))
    for variable in XDG:
        monkeypatch.delenv(variable, raising=False)
    assert Path.home() == folder
    return folder


@pytest.fixture
def linux(monkeypatch, home):
    """This is Linux, with ``home`` as the home directory."""
    from gmv import host

    monkeypatch.setattr(host, "system", lambda: "linux")
    return home


@pytest.fixture
def windows(monkeypatch, home, tmp_path):
    """This is Windows, with ``home`` as the home directory; returns ``%LOCALAPPDATA%``."""
    from gmv import host

    monkeypatch.setattr(host, "system", lambda: "windows")
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    return tmp_path / "local"


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


@pytest.fixture(scope="session")
def measured(tmp_path_factory):
    """What ``plasma_gpu_probe.py`` measured on this machine's graphics card. It is run
    once, in a process of its own, on the real Qt platform (these tests use the
    ``offscreen`` one, which has no OpenGL on Windows)."""
    out = tmp_path_factory.mktemp("plasma") / "probe.json"
    environment = {k: v for k, v in os.environ.items() if k != "QT_QPA_PLATFORM"}
    began = perf_counter()
    done = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("plasma_gpu_probe.py")), str(out)],
        env=environment, capture_output=True, text=True, timeout=900,
    )  # fmt: skip
    if not out.is_file():
        return {"available": False, "reason": f"the probe did not finish (exit {done.returncode}): {done.stderr[-400:]}"}
    return json.loads(out.read_text(encoding="utf-8")) | {"seconds": perf_counter() - began}


@pytest.fixture
def gpu(measured):
    """The probe's measurements; the test is skipped, with the reason, without a card."""
    if not measured["available"]:
        pytest.skip(f"no OpenGL 3.3 context here: {measured['reason']}")
    return measured
