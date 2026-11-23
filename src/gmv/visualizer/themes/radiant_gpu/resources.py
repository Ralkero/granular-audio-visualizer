"""Shader sources: read from ``shaders/``, with ``#include "file"`` lines replaced by the
file. No OpenGL here, so a missing or broken file is found without a graphics card."""

from functools import lru_cache
from pathlib import Path

FOLDER = Path(__file__).parent / "shaders"


class ShaderError(Exception):
    """A shader could not be read, compiled or linked. The message says which and why."""


@lru_cache(maxsize=32)
def source(name, folder=FOLDER):
    """The text of shader file ``name`` (for example ``core.frag``), includes resolved."""
    path = Path(folder) / name
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as error:
        raise ShaderError(f"shader file {name} cannot be read: {error}") from error
    lines = []
    for line in text.splitlines():
        if line.strip().startswith("#include"):
            included = line.split('"')[1] if line.count('"') == 2 else ""
            if not included or included == name:
                raise ShaderError(f"{name}: bad include line: {line.strip()}")
            lines.append(source(included, folder))
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def vertex(folder=FOLDER):
    return source("fullscreen.vert", folder)


def fragment(name, folder=FOLDER):
    return source(name + ".frag", folder)
