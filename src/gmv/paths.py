"""Where the application finds its files, run from a checkout or as a packaged build.

From a checkout everything is relative to the repository. A packaged build (PyInstaller)
has its models and the capture helper beside the executable and writes sessions to the
user's local application data, because the program folder may be read-only.
"""

import json
import os
import sys
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[2]


def packaged():
    return bool(getattr(sys, "frozen", False))


def program_folder():
    return Path(sys.executable).resolve().parent if packaged() else REPOSITORY


def models_folder():
    return program_folder() / ("models" if packaged() else ".model-cache")


def sessions_folder():
    if not packaged():
        return REPOSITORY / "sessions"
    base = os.environ.get("LOCALAPPDATA") or str(Path.home())
    return Path(base) / "GranularMusicVisualizer" / "sessions"


def capture_helper():
    if packaged():
        return program_folder() / "ProcessLoopback" / "ProcessLoopback.exe"
    return REPOSITORY / "native/ProcessLoopback/bin/Release/net9.0-windows/ProcessLoopback.exe"


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
