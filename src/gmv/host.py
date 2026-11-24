"""The machine GMV runs on: the one place that asks which operating system this is.

The engine, the messages, the renderers and replay are the same program everywhere. What
differs between systems is answered here, and two modules build on it: ``gmv.paths`` (where
the application's files are) and ``gmv.sources`` (how the capture backends are reached).

* Windows: capture of the system output, of one output device and of one application.
  Settings are in the registry; a checkout keeps sessions and models in the repository, a
  packaged build in ``%LOCALAPPDATA%`` and beside its executable.
* Linux: capture of the system output (the default output's monitor, through the PulseAudio
  service PipeWire provides), audio files and replay of recorded sessions. No capture of one
  output device or of one application. Settings, sessions and models are in the XDG base
  directories.

Nothing here imports Qt or a capture library, and nothing here touches the disk: these are
answers, not actions.
"""

import os
import sys
from pathlib import Path

APPLICATION = "GranularMusicVisualizer"  # the folder name; also the settings' organization
QT_OVERRIDE = "GMV_QT_PLATFORM"

# The sources a system can capture from, in the order they are offered. A system that is
# not listed has no live capture; an audio file can be analysed everywhere. Being listed
# says the program has the backend, not that the machine can use it now: Linux needs a
# PulseAudio-compatible server and its client programs, which the backend checks.
LIVE_CAPTURE = {"windows": ("system", "device", "application"), "linux": ("system",)}


def system():
    """``"windows"``, ``"linux"`` or ``"other"``."""
    if sys.platform == "win32":
        return "windows"
    return "linux" if sys.platform.startswith("linux") else "other"


def name():
    return {"windows": "Windows", "linux": "Linux"}.get(system(), "this system")


# -- the user's folders ------------------------------------------------------------------------


def xdg(variable, default):
    """An XDG base directory: the variable when it holds an absolute path, otherwise the
    default under the home directory. The specification says an empty or relative value is
    to be ignored."""
    value = os.environ.get(variable, "")
    return Path(value) if value and Path(value).is_absolute() else Path.home() / default


def config_folder():
    """Where the settings file is on Linux (Qt writes it there by itself: ``QSettings``
    follows ``XDG_CONFIG_HOME``). None on other systems: Windows keeps settings in the
    registry."""
    if system() != "linux":
        return None
    return xdg("XDG_CONFIG_HOME", ".config") / APPLICATION


def data_folder():
    """The user's own folder for what the application records and for files they add to
    it. Never inside the program or the repository, so it does not need them writable."""
    if system() == "linux":
        return xdg("XDG_DATA_HOME", ".local/share") / APPLICATION
    return Path(os.environ.get("LOCALAPPDATA") or str(Path.home())) / APPLICATION


def cache_folder():
    """For files that can be made again. Nothing is kept there yet."""
    if system() == "linux":
        return xdg("XDG_CACHE_HOME", ".cache") / APPLICATION
    return data_folder() / "cache"


# -- audio sources -----------------------------------------------------------------------------


def audio_sources():
    """Keys of the audio sources this system has, in the order they are offered."""
    return (*LIVE_CAPTURE.get(system(), ()), "file")


def live_capture():
    return system() in LIVE_CAPTURE


def capture_note():
    """Why there is no live capture here, in words for the interface; empty where there is."""
    if live_capture():
        return ""
    return f"Live system audio capture is not available on {name()} yet."


# -- the Qt platform plugin --------------------------------------------------------------------


def gamescope(environ=None):
    """Whether this process draws inside Valve's Gamescope compositor with only its X11
    display to draw on: Steam's Gaming Mode, or ``gamescope -- program`` on a desktop.

    Gamescope gives what it runs ``GAMESCOPE_WAYLAND_DISPLAY`` (its own, private socket)
    and ``DISPLAY`` (its Xwayland), and removes ``WAYLAND_DISPLAY`` unless it was started
    with ``--expose-wayland``. All three are checked, so a desktop session, which has no
    ``GAMESCOPE_WAYLAND_DISPLAY`` and on Wayland has a ``WAYLAND_DISPLAY``, is never taken
    for one. Not confirmed on a Steam Deck: see ``docs/LINUX.md``."""
    environ = os.environ if environ is None else environ
    return bool(
        environ.get("GAMESCOPE_WAYLAND_DISPLAY")
        and environ.get("DISPLAY")
        and not environ.get("WAYLAND_DISPLAY")
    )


def qt_platform(environ=None, arguments=()):
    """The Qt platform plugin to ask for, and why: ``(name, reason)``. A name of None
    leaves the choice to Qt, which on a Linux desktop takes Wayland when the session has
    it and X11 otherwise.

    In order: Qt's own ``-platform`` argument and ``QT_QPA_PLATFORM`` variable are never
    overruled; ``GMV_QT_PLATFORM`` names a plugin (``xcb``, ``wayland``, ...) or, as
    ``auto``, switches the next rule off; inside Gamescope on Linux the X11 plugin ``xcb``
    is asked for, because Gamescope shows X11 windows."""
    environ = os.environ if environ is None else environ
    if "-platform" in arguments:
        return None, "the -platform argument"
    if environ.get("QT_QPA_PLATFORM"):
        return None, f"QT_QPA_PLATFORM={environ['QT_QPA_PLATFORM']}, set outside the application"
    wanted = environ.get(QT_OVERRIDE, "").strip()
    if wanted.lower() == "auto":
        return None, f"Qt's own choice ({QT_OVERRIDE}=auto)"
    if wanted:
        return wanted, f"{QT_OVERRIDE}={wanted}"
    if system() == "linux" and gamescope(environ):
        return "xcb", "inside Gamescope, which shows X11 windows (GAMESCOPE_WAYLAND_DISPLAY is set)"
    return None, "Qt's own choice"


def select_qt_platform(environ=None, arguments=()):
    """Put the choice of ``qt_platform`` where Qt reads it. This must happen before the
    ``QApplication`` exists; afterwards it changes nothing. Returns the choice."""
    environ = os.environ if environ is None else environ
    chosen, reason = qt_platform(environ, arguments)
    if chosen:
        environ["QT_QPA_PLATFORM"] = chosen
    return chosen, reason
