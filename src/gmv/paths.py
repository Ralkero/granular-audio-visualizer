"""Where the application finds its files, run from a checkout or as a packaged build.

On Windows, from a checkout everything is relative to the repository. A packaged build
(PyInstaller) has its models and the capture helper beside the executable and writes
sessions to the user's local application data, because the program folder may be read-only.

On Linux, what the user records or adds is never in the repository or the program folder:
sessions and the user's own models are under ``$XDG_DATA_HOME`` (``gmv.host``), so the
application works from a read-only checkout and from any working directory. None of these
paths depends on the working directory.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from . import host

REPOSITORY = Path(__file__).resolve().parents[2]


def packaged():
    return bool(getattr(sys, "frozen", False))


def program_folder():
    return Path(sys.executable).resolve().parent if packaged() else REPOSITORY


def models_folder():
    """The program's own models: ``.model-cache`` in a checkout, ``models`` beside a
    packaged executable."""
    return program_folder() / ("models" if packaged() else ".model-cache")


def user_models_folder():
    """Where a Linux user keeps model files of their own, or None: on Windows weights are
    in the program's folder or chosen by path."""
    return host.data_folder() / "models" if host.system() == "linux" else None


def model_folders():
    """The folders looked in for the weights the application starts with, in order: the
    user's own, then the program's. A file chosen in Settings is used as it is and is never
    replaced by one found here."""
    user = user_models_folder()
    return (user, models_folder()) if user else (models_folder(),)


def default_model(name):
    """The first file called ``name`` in ``model_folders()``, or None."""
    return next((f / name for f in model_folders() if (f / name).is_file()), None)


def models_start_folder():
    """Where a dialog for choosing weights opens."""
    folders = model_folders()
    return next((folder for folder in folders if folder.is_dir()), folders[-1])


def sessions_folder():
    """Where sessions are recorded when no other folder is named."""
    if host.system() == "linux" or packaged():
        return host.data_folder() / "sessions"
    return REPOSITORY / "sessions"


def command_line_sessions():
    """The ``--sessions`` default of the command line: ``sessions`` in the working
    directory on Windows, as it always was; the user's sessions folder on Linux, where the
    working directory may be anything and may not be writable."""
    return sessions_folder() if host.system() == "linux" else Path("sessions")


def writable_folder(path, purpose):
    """``path``, made if it is not there, once a file has been written in it. A folder
    that cannot be used raises an error that says which folder, what for and why."""
    path = Path(path)
    try:
        path.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryFile(dir=path):
            pass
    except OSError as exc:
        raise RuntimeError(
            f"Cannot save {purpose} in {path}: {exc.strerror or exc}. "
            "Make that folder writable or choose another."
        ) from exc
    return path


def icon():
    """The application's icon file (made by ``packaging/make_icon.py``), or None."""
    name = "icon.png" if host.system() == "linux" else "app.ico"
    path = program_folder() / (name if packaged() else f"packaging/{name}")
    return path if path.is_file() else None


def capture_helper():
    """The Windows capture helper (``native/ProcessLoopback``). A path on every system,
    because modules name it when they are imported; only Windows has the program."""
    if packaged():
        return program_folder() / "ProcessLoopback" / "ProcessLoopback.exe"
    return REPOSITORY / "native/ProcessLoopback/bin/Release/net9.0-windows/ProcessLoopback.exe"


def capture_helper_available():
    return host.system() == "windows" and capture_helper().is_file()


def helper_environment():
    """Environment for the capture helper. A private .NET runtime beside it (``dotnet``,
    put there by ``Build App.ps1``) is used when present, so capture does not depend on
    the .NET installed on the machine."""
    runtime = capture_helper().parent / "dotnet"
    if not (runtime / "host" / "fxr").is_dir():
        return None
    return {**os.environ, "DOTNET_ROOT": str(runtime), "DOTNET_MULTILEVEL_LOOKUP": "0"}


def build_defaults():
    """Settings a packaged build was made with (``models/defaults.json``): which weights it
    starts with. A checkout has none, so experimental models stay off there."""
    if not packaged():
        return {}
    try:
        found = json.loads((models_folder() / "defaults.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return {}
    out = {}
    for key in ("activity_model", "guitar_model"):
        path = models_folder() / str(found.get(key) or "-")
        if path.is_file():
            out[key] = str(path)
    return out
