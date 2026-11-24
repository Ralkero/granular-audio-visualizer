"""The three ways the program is run, and what the Arch package needs from it.

A **checkout** (the repository, as it is developed), a **packaged build** (PyInstaller, on
Windows) and an **installed package** (a wheel a package manager put in the system's Python,
as the Arch package does, with no repository beside it). An installed package is simulated
by pointing ``paths.REPOSITORY`` where an installed ``gmv`` would find it, or by copying the
files a wheel holds into a folder of their own and running them from there. Homes and XDG
folders are temporary: nothing here reads or writes the user's own files.
"""

import configparser
import fnmatch
import importlib
import json
import os
import shutil
import struct
import sys
import tomllib
from pathlib import Path

import pytest

from gmv import paths
from test_host import AS_LINUX, run_python

REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
PROJECT = tomllib.loads((REPOSITORY / "pyproject.toml").read_text(encoding="utf-8"))
ARCH = REPOSITORY / "packaging" / "arch"
DESKTOP = ARCH / f"{paths.APP_ID}.desktop"


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """The program as an installed package sees itself: ``paths.py`` is in
    ``site-packages/gmv``, so ``REPOSITORY`` is the Python library folder, which holds no
    project. Returns that folder."""
    library = tmp_path / "usr" / "lib" / "python3.x"
    library.mkdir(parents=True)
    monkeypatch.setattr(paths, "REPOSITORY", library)
    return library


def png_size(path):
    data = Path(path).read_bytes()[:24]
    assert data[:8] == b"\x89PNG\r\n\x1a\n", f"{path} is not a PNG"
    return struct.unpack(">II", data[16:24])


# -- which way the program is run -------------------------------------------------------------


def test_these_tests_run_from_a_checkout():
    assert paths.checkout() and not paths.installed() and not paths.packaged()


def test_without_a_project_beside_it_the_program_is_an_installed_package(installed):
    assert paths.installed() and not paths.checkout() and not paths.packaged()


def test_a_packaged_build_is_never_taken_for_an_installed_package(
    windows, installed, monkeypatch, tmp_path
):
    program = tmp_path / "app"
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(program / "app.exe"))
    assert paths.packaged() and not paths.installed()
    assert paths.models_folder() == program / "models"
    assert paths.model_folders() == (program / "models",)


# -- an installed package on Linux -------------------------------------------------------------


def test_an_installed_package_looks_for_models_only_in_the_users_folder(linux, installed):
    user = paths.user_models_folder()
    assert user == linux / ".local/share/GranularMusicVisualizer/models"
    assert paths.models_folder() is None  # the package carries no weights
    assert paths.model_folders() == (user,)
    assert paths.default_model("activity_v2.npz") is None
    assert paths.models_start_folder() == user  # the dialog makes it when it opens
    assert list(linux.iterdir()) == []  # asking made nothing
    user.mkdir(parents=True)
    (user / "activity_v2.npz").write_bytes(b"user")
    assert paths.default_model("activity_v2.npz") == user / "activity_v2.npz"


def test_an_installed_package_keeps_everything_the_user_adds_out_of_the_system(
    linux, installed, monkeypatch, tmp_path
):
    from gmv import host

    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert paths.sessions_folder() == tmp_path / "data/GranularMusicVisualizer/sessions"
    assert paths.command_line_sessions() == paths.sessions_folder()
    assert paths.models_start_folder() == tmp_path / "data/GranularMusicVisualizer/models"
    assert host.cache_folder() == tmp_path / "cache/GranularMusicVisualizer"
    assert host.config_folder() == tmp_path / "config/GranularMusicVisualizer"
    for folder in (
        paths.sessions_folder(),
        paths.models_start_folder(),
        *paths.model_folders(),
        host.cache_folder(),
        host.config_folder(),
    ):
        assert installed not in folder.parents and folder != installed


def test_an_installed_package_takes_its_icon_from_the_icon_theme(linux, installed):
    assert paths.icon() is None


def test_an_installed_package_on_windows_writes_nothing_beside_its_python(windows, installed, home):
    # Not a way GMV is offered on Windows, but it must not write into the Python folder.
    assert paths.sessions_folder() == windows / "GranularMusicVisualizer" / "sessions"
    assert paths.model_folders() == ()
    assert paths.default_model("activity_v2.npz") is None
    assert paths.models_start_folder() == home


# -- a checkout, as it always was -------------------------------------------------------------


def test_a_linux_checkout_keeps_its_answers(linux):
    assert paths.models_folder() == REPOSITORY / ".model-cache"
    assert paths.model_folders() == (paths.user_models_folder(), REPOSITORY / ".model-cache")
    assert paths.icon() == REPOSITORY / "packaging" / "icon.png"


# -- what the wheel holds ----------------------------------------------------------------------


def wheel_files():
    """The files a wheel built from this project holds under ``gmv``: every Python module,
    and the other files ``[tool.setuptools.package-data]`` names, relative to ``src``."""
    data = PROJECT["tool"]["setuptools"].get("package-data", {})
    for path in sorted((SOURCE / "gmv").rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix == ".py":
            yield path.relative_to(SOURCE)
            continue
        for package, patterns in data.items():
            folder = SOURCE.joinpath(*package.split("."))
            if folder in path.parents and any(
                fnmatch.fnmatch(path.relative_to(folder).as_posix(), p) for p in patterns
            ):
                yield path.relative_to(SOURCE)
                break


def test_the_wheel_takes_every_file_the_program_reads_that_is_not_python():
    others = {
        p.relative_to(SOURCE)
        for p in (SOURCE / "gmv").rglob("*")
        if p.is_file() and p.suffix not in (".py", ".pyc") and "__pycache__" not in p.parts
    }
    assert others, "expected the Radiant shaders"
    missing = others - set(wheel_files())
    assert not missing, f"left out of the wheel: {sorted(map(str, missing))}"


def test_an_installed_copy_runs_from_anywhere_with_its_shaders_and_writes_nothing_there(
    tmp_path,
):
    site = tmp_path / "usr" / "lib" / "python3.x" / "site-packages"
    for relative in wheel_files():
        (site / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE / relative, site / relative)
    before = sorted(p.relative_to(site) for p in site.rglob("*"))
    elsewhere, home_folder = tmp_path / "elsewhere", tmp_path / "home"
    elsewhere.mkdir()
    home_folder.mkdir()
    code = AS_LINUX + (
        "import json, gmv\n"
        "from gmv import paths\n"
        "from gmv.visualizer.themes.radiant_gpu import resources\n"
        "frags = sorted(p.stem for p in resources.FOLDER.glob('*.frag'))\n"
        "texts = [resources.vertex()] + [resources.fragment(n) for n in frags]\n"
        "print(json.dumps({'module': gmv.__file__, 'installed': paths.installed(),\n"
        "  'frags': frags, 'includes': sum('#include' in t for t in texts),\n"
        "  'models': [str(f) for f in paths.model_folders()],\n"
        "  'start': str(paths.models_start_folder()), 'sessions': str(paths.sessions_folder()),\n"
        "  'icon': paths.icon()}))\n"
    )
    done = run_python(
        code,
        cwd=elsewhere,
        PYTHONPATH=str(site),
        PYTHONDONTWRITEBYTECODE="1",
        HOME=str(home_folder),
        XDG_DATA_HOME=str(tmp_path / "data"),
        XDG_CONFIG_HOME=None,
        XDG_CACHE_HOME=None,
    )
    assert done.returncode == 0, done.stderr
    found = json.loads(done.stdout.splitlines()[-1])
    assert Path(found["module"]).parent == site / "gmv"  # not the checkout's
    assert found["installed"] is True
    shaders = SOURCE / "gmv/visualizer/themes/radiant_gpu/shaders"
    assert found["frags"] == sorted(p.stem for p in shaders.glob("*.frag"))
    assert found["includes"] == 0  # every #include was found and put in
    models = str(tmp_path / "data/GranularMusicVisualizer/models")
    assert found["models"] == [models] and found["start"] == models
    assert found["sessions"] == str(tmp_path / "data/GranularMusicVisualizer/sessions")
    assert found["icon"] is None
    assert sorted(p.relative_to(site) for p in site.rglob("*")) == before
    assert list(elsewhere.iterdir()) == []


def test_every_command_the_package_installs_names_a_function_that_exists():
    commands = {**PROJECT["project"]["scripts"], **PROJECT["project"]["gui-scripts"]}
    assert set(commands) == {"gmv", "gmv-control", "gmv-visualizer"}
    for name, target in commands.items():
        module, function = target.split(":")
        assert callable(getattr(importlib.import_module(module), function)), name


# -- the desktop entry and the icon ------------------------------------------------------------


def test_the_desktop_entry_starts_the_visualizer_with_the_programs_icon():
    entry = configparser.ConfigParser(interpolation=None)
    entry.optionxform = str
    entry.read(DESKTOP, encoding="utf-8")
    fields = entry["Desktop Entry"]
    assert fields["Type"] == "Application"
    assert fields["Name"] == "Granular Music Visualizer"
    assert (
        fields["Exec"] == "gmv-visualizer" and "gmv-visualizer" in PROJECT["project"]["gui-scripts"]
    )
    assert fields["Icon"] == paths.APP_ID == DESKTOP.stem
    assert fields["Terminal"] == "false"
    assert "MimeType" not in fields  # GMV opens no file type of its own


def test_the_icon_theme_pictures_are_the_sizes_their_names_say():
    sizes = {int(p.stem): png_size(p) for p in (ARCH / "icons").glob("*.png")}
    assert 256 in sizes and 48 in sizes
    for size, found in sizes.items():
        assert found == (size, size)
    assert png_size(REPOSITORY / "packaging" / "icon.png") == (512, 512)


def test_on_linux_the_window_names_its_desktop_entry_and_finds_an_installed_icon(
    app, linux, installed, tmp_path
):
    from PySide6.QtGui import QGuiApplication, QIcon

    from gmv.app import window as app_window

    theme = tmp_path / "icons" / "gmv-test"
    (theme / "256x256" / "apps").mkdir(parents=True)
    (theme / "index.theme").write_text(
        "[Icon Theme]\nName=gmv-test\nDirectories=256x256/apps\n\n"
        "[256x256/apps]\nSize=256\nType=Fixed\n",
        encoding="utf-8",
    )
    shutil.copy2(ARCH / "icons" / "256.png", theme / "256x256" / "apps" / f"{paths.APP_ID}.png")
    search, name = QIcon.themeSearchPaths(), QIcon.themeName()
    try:
        QIcon.setThemeSearchPaths([str(tmp_path / "icons")])
        QIcon.setThemeName("gmv-test")
        assert not app_window.window_icon().isNull()
        app_window.application()
        assert QGuiApplication.desktopFileName() == paths.APP_ID
        assert not QGuiApplication.windowIcon().isNull()
    finally:
        QIcon.setThemeSearchPaths(search)
        QIcon.setThemeName(name)


def test_the_weights_dialog_of_an_installed_package_opens_in_the_users_models_folder(
    app, linux, installed, monkeypatch, tmp_path
):
    from gmv.app import window as app_window

    opened = []

    class Dialog:
        @staticmethod
        def getOpenFileName(parent, title, folder, kinds):
            opened.append(Path(folder))
            return "", ""

    monkeypatch.setattr(app_window, "QFileDialog", Dialog)
    main = app_window.MainWindow(
        session_root=tmp_path / "sessions", remember=False, discover_sources=False
    )
    try:
        main.drawer.choose_weights(main.drawer.activity_model)
    finally:
        main.close()
    user = paths.user_models_folder()
    assert opened == [user] and user.is_dir()  # made, so the dialog opens there
    assert os.access(user, os.W_OK)
