"""The Steam Deck helper, ``packaging/steamdeck/gmv-deck.sh``, without the network.

What is checked here: the commands it knows, the folders it refuses to install into or
remove, the launcher it writes (the library path Steam may set, and the log it keeps when
started without a terminal), and the menu entry and icons. Downloading GMV, uv and Python
needs the network and is not done by the test suite; docs/STEAM_DECK_FOR_CLAUDE.md says how
the whole install was run.

Every run has a home folder of the test's own, so nothing of the machine's is read or
written, and no D-Bus session, so no running desktop is told about a menu entry.
"""

import os
import platform
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "packaging" / "steamdeck" / "gmv-deck.sh"
APP_ID = "granular-music-visualizer"
SIZES = (16, 22, 24, 32, 48, 64, 128, 256, 512)

pytestmark = pytest.mark.skipif(
    sys.platform != "linux" or not shutil.which("bash"),
    reason="the Steam Deck helper is a bash script for Linux",
)


def environment(home, **extra):
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_CACHE_HOME", "GMV_DECK_HOME")
        and not key.startswith(("LD_", "SYSTEM_LD_", "STEAM", "Steam"))
    }
    env |= {"HOME": str(home), "DBUS_SESSION_BUS_ADDRESS": "unix:path=/nonexistent"}
    return env | {key: str(value) for key, value in extra.items()}


def helper(home, *arguments, **extra):
    return subprocess.run(
        ["bash", str(SCRIPT), *arguments],
        env=environment(home, **extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def functions(home, code, **extra):
    """Run ``code`` in bash with the helper's functions loaded."""
    return subprocess.run(
        ["bash", "-c", f'GMV_DECK_SOURCED=1 source "{SCRIPT}"\n{code}'],
        env=environment(home, **extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=60,
    )


def fake_install(root):
    """The folders an install leaves, with a stand-in for the program."""
    (root / "venv" / "bin").mkdir(parents=True)
    (root / "app" / "src" / "gmv").mkdir(parents=True)
    (root / "BUILD").write_text("installed by: gmv-deck.sh\n", encoding="utf-8")
    program = root / "venv" / "bin" / "gmv-visualizer"
    program.write_text(
        "#!/bin/sh\n"
        'echo "LD_LIBRARY_PATH=${LD_LIBRARY_PATH-(unset)}"\n'
        'echo "arguments: $*"\n'
        'echo "gmv: Qt platform stand-in" >&2\n',
        encoding="utf-8",
    )
    program.chmod(program.stat().st_mode | stat.S_IXUSR)
    return root


@pytest.fixture
def home(tmp_path):
    folder = tmp_path / "home"
    folder.mkdir()
    return folder


def test_the_script_is_valid_bash():
    done = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr


def test_help_names_every_command(home):
    done = helper(home, "help")
    assert done.returncode == 0
    for command in ("install", "check", "smoke", "listen", "remove"):
        assert f"gmv-deck.sh {command}" in done.stdout
    assert helper(home, "frobnicate").returncode == 2


def test_check_runs_without_a_sound_server_or_an_install(home):
    # tests/conftest.py points pactl at a server that does not exist
    done = helper(home, "check")
    assert done.returncode == 0, done.stderr
    assert "Sound server: did not answer" in done.stdout
    assert f"Not installed in {home}/granular-music-visualizer" in done.stdout


@pytest.mark.skipif(platform.machine() != "x86_64", reason="install refuses other machines first")
def test_install_leaves_a_folder_it_did_not_make_alone(home):
    foreign = home / "granular-music-visualizer"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("mine", encoding="utf-8")
    done = helper(home, "install")
    assert done.returncode == 1
    assert "was not made by this script" in done.stderr
    assert sorted(p.name for p in foreign.iterdir()) == ["keep.txt"]


@pytest.mark.skipif(platform.machine() != "x86_64", reason="install refuses other machines first")
def test_install_refuses_a_folder_path_a_menu_entry_cannot_hold(home):
    done = helper(home, "install", GMV_DECK_HOME=home / "with space")
    assert done.returncode == 1
    assert "may only hold letters" in done.stderr
    assert not (home / "with space").exists()


def test_install_and_listen_refuse_unknown_input(home):
    assert "Unknown option for install: --bogus" in helper(home, "install", "--bogus").stderr
    assert "listen takes a number of seconds" in helper(home, "listen", "abc").stderr


def test_smoke_and_listen_need_an_install(home):
    for command in ("smoke", "listen"):
        done = helper(home, command)
        assert done.returncode == 1
        assert "is not installed" in done.stderr


def test_remove_asks_first_and_keeps_the_owners_files(home):
    root = fake_install(home / "granular-music-visualizer")
    share = home / ".local" / "share"
    entry = share / "applications" / f"{APP_ID}.desktop"
    entry.parent.mkdir(parents=True)
    entry.write_text("[Desktop Entry]\nX-GMV-Installed-By=gmv-deck.sh\n", encoding="utf-8")
    icon = share / "icons" / "hicolor" / "48x48" / "apps" / f"{APP_ID}.png"
    icon.parent.mkdir(parents=True)
    icon.write_bytes(b"png")
    settings = home / ".config" / "GranularMusicVisualizer"
    sessions = share / "GranularMusicVisualizer" / "sessions"
    settings.mkdir(parents=True)
    sessions.mkdir(parents=True)

    done = helper(home, "remove")  # no terminal to answer on, no --yes
    assert done.returncode == 1 and "Nothing was removed" in done.stderr
    assert root.is_dir() and entry.is_file() and icon.is_file()

    done = helper(home, "remove", "--yes")
    assert done.returncode == 0, done.stderr
    assert not root.exists() and not entry.exists() and not icon.exists()
    assert settings.is_dir() and sessions.is_dir()


def test_remove_leaves_a_folder_it_did_not_make_alone(home):
    foreign = home / "elsewhere"
    foreign.mkdir()
    (foreign / "keep.txt").write_text("mine", encoding="utf-8")
    done = helper(home, "remove", "--yes", GMV_DECK_HOME=foreign)
    assert done.returncode == 1
    assert (foreign / "keep.txt").is_file()


def launch(home, root, **extra):
    done = functions(home, "write_launcher", GMV_DECK_HOME=root)
    assert done.returncode == 0, done.stderr
    launcher = root / "launch.sh"
    assert os.access(launcher, os.X_OK)
    # Started from the menu or by Steam: no terminal, so everything goes to the log.
    started = subprocess.run(
        [str(launcher), "--replay", "x"],
        env=environment(home, **extra),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert started.returncode == 0 and started.stdout == "" and started.stderr == ""
    return (root / "last-launch.log").read_text(encoding="utf-8")


def test_the_launcher_puts_back_the_library_path_steam_replaced(home):
    root = fake_install(home / "granular-music-visualizer")
    log = launch(
        home,
        root,
        LD_LIBRARY_PATH="/steam/ubuntu12_32:/steam/ubuntu12_32/steam-runtime/usr/lib",
        SYSTEM_LD_LIBRARY_PATH="/opt/own/lib",
        SteamGameId="1234",
        GAMESCOPE_WAYLAND_DISPLAY="gamescope-0",
    )
    assert "library path reset: yes" in log
    assert "LD_LIBRARY_PATH=/opt/own/lib" in log
    assert "arguments: --replay x" in log
    assert "gmv: Qt platform stand-in" in log  # standard error is kept too
    assert "SteamGameId=1234" in log and "GAMESCOPE_WAYLAND_DISPLAY=gamescope-0" in log


def test_the_launcher_clears_steams_path_when_there_was_none_before(home):
    root = fake_install(home / "granular-music-visualizer")
    log = launch(home, root, LD_LIBRARY_PATH="/steam/ubuntu12_64")
    assert "library path reset: yes" in log and "LD_LIBRARY_PATH=(unset)" in log


def test_the_launcher_leaves_any_other_library_path_alone(home):
    root = fake_install(home / "granular-music-visualizer")
    log = launch(home, root, LD_LIBRARY_PATH="/opt/mine/lib")
    assert "library path reset: no" in log and "LD_LIBRARY_PATH=/opt/mine/lib" in log
    assert "GAMESCOPE_WAYLAND_DISPLAY=(not set)" in log


def test_the_menu_entry_starts_the_launcher_with_the_applications_icon(home):
    root = home / "granular-music-visualizer"
    packaging = root / "app" / "packaging"
    shutil.copytree(REPOSITORY / "packaging" / "arch", packaging / "arch")
    shutil.copy(REPOSITORY / "packaging" / "icon.png", packaging / "icon.png")
    done = functions(home, "install_menu_entry", GMV_DECK_HOME=root)
    assert done.returncode == 0, done.stderr

    share = home / ".local" / "share"
    entry = share / "applications" / f"{APP_ID}.desktop"
    lines = entry.read_text(encoding="utf-8").splitlines()
    assert f"Exec={root}/launch.sh" in lines
    assert f"Icon={APP_ID}" in lines
    assert "X-GMV-Installed-By=gmv-deck.sh" in lines
    for size in SIZES:
        assert (share / "icons" / "hicolor" / f"{size}x{size}" / "apps" / f"{APP_ID}.png").is_file()
    # The program calls itself by this name on Wayland (setDesktopFileName), which is how
    # the taskbar finds the entry.
    from gmv import paths

    assert entry.stem == paths.APP_ID
    if shutil.which("desktop-file-validate"):
        checked = subprocess.run(
            ["desktop-file-validate", str(entry)], capture_output=True, text=True
        )
        assert checked.returncode == 0, checked.stdout + checked.stderr


def test_the_pinned_build_is_a_commit_with_the_files_the_install_uses():
    text = SCRIPT.read_text(encoding="utf-8")
    commit = next(
        line.split("=", 1)[1] for line in text.splitlines() if line.startswith("GMV_COMMIT=")
    )
    checksum = next(
        line.split("=", 1)[1] for line in text.splitlines() if line.startswith("GMV_SHA256=")
    )
    assert len(commit) == 40 and len(checksum) == 64
    if not (REPOSITORY / ".git").exists() or not shutil.which("git"):
        pytest.skip("not a git checkout")
    known = subprocess.run(["git", "-C", str(REPOSITORY), "cat-file", "-e", f"{commit}^{{commit}}"])
    if known.returncode != 0:
        pytest.skip("the pinned commit is not in this clone")
    for path in (
        "uv.lock",
        f"packaging/arch/{APP_ID}.desktop",
        "packaging/icon.png",
        "scripts/validate_linux.py",
        "scripts/capture_probe_linux.py",
        *(f"packaging/arch/icons/{size}.png" for size in SIZES if size != 512),
    ):
        found = subprocess.run(["git", "-C", str(REPOSITORY), "cat-file", "-e", f"{commit}:{path}"])
        assert found.returncode == 0, f"{path} is not in the pinned commit"
