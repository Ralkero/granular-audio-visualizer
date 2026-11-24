"""Where the application finds its files: run from a checkout, as a packaged build, or as an
installed package.

The program is run in one of three ways, and each has its own answers:

* **checkout**: from the repository (``src/gmv`` beside ``pyproject.toml``), the way it is
  developed. On Windows everything is relative to the repository.
* **packaged build** (PyInstaller, Windows): models and the capture helper are beside the
  executable; sessions go to the user's local application data, because the program folder
  may be read-only.
* **installed package**: a wheel installed by a package manager, as the Arch package puts it
  in ``/usr/lib/pythonX.Y/site-packages``. There is no repository beside it, the program's
  files are read-only and belong to the package manager, and it carries no model weights.
  Its icon is installed into the system's icon theme (``APP_ID``), not looked for as a file.

On Linux, what the user records or adds is never in the repository or the program folder:
sessions and the user's own models are under ``$XDG_DATA_HOME`` (``gmv.host``), so the
application works from a read-only checkout, from an installed package and from any working
directory. None of these paths depends on the working directory.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

from . import host

REPOSITORY = Path(__file__).resolve().parents[2]
# The name of the desktop entry (``APP_ID.desktop``) and of the icon in the icon theme that
# an installed package provides. On Linux the application says it is this program by it.
APP_ID = "granular-music-visualizer"


def packaged():
    return bool(getattr(sys, "frozen", False))


def checkout():
    """Whether this is run from the repository: ``REPOSITORY`` holds ``pyproject.toml`` and
    the package's source, ``src/gmv``. An installed package has neither beside it."""
    return (REPOSITORY / "pyproject.toml").is_file() and (REPOSITORY / "src" / "gmv").is_dir()


def installed():
    """Whether this is an installed package: neither a packaged build nor a checkout."""
    return not packaged() and not checkout()


def program_folder():
    return Path(sys.executable).resolve().parent if packaged() else REPOSITORY


def models_folder():
    """The program's own models: ``.model-cache`` in a checkout, ``models`` beside a
    packaged executable, and None for an installed package, which carries no weights (its
    program folder is a system one that the user cannot add to)."""
    if installed():
        return None
    return program_folder() / ("models" if packaged() else ".model-cache")


def user_models_folder():
    """Where a Linux user keeps model files of their own, or None: on Windows weights are
    in the program's folder or chosen by path."""
    return host.data_folder() / "models" if host.system() == "linux" else None


def model_folders():
    """The folders looked in for the weights the application starts with, in order: the
    user's own, then the program's. A file chosen in Settings is used as it is and is never
    replaced by one found here."""
    return tuple(folder for folder in (user_models_folder(), models_folder()) if folder)


def default_model(name):
    """The first file called ``name`` in ``model_folders()``, or None."""
    return next((f / name for f in model_folders() if (f / name).is_file()), None)


def models_start_folder():
    """Where a dialog for choosing weights opens: the first of ``model_folders()`` that
    exists, otherwise the last of them (for an installed package on Linux, the user's own
    models folder, which the dialog makes), or the home folder when there is none."""
    folders = model_folders()
    if not folders:
        return Path.home()
    return next((folder for folder in folders if folder.is_dir()), folders[-1])


def sessions_folder():
    """Where sessions are recorded when no other folder is named."""
    if host.system() == "linux" or packaged() or installed():
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
    """The application's icon file (made by ``packaging/make_icon.py``), or None. An
    installed package has no file of its own to point at: its icon is the icon theme's
    ``APP_ID``."""
    if installed():
        return None
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
