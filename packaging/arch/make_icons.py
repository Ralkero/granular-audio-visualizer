"""Make the Arch package's icon theme pictures from the application's icon.

    python packaging/arch/make_icons.py

Reads ``packaging/icon.png`` (512 pixels, made by ``packaging/make_icon.py``) and writes
``packaging/arch/icons/SIZE.png`` at the sizes the freedesktop icon theme ``hicolor`` has
folders for. The PKGBUILD installs each as
``/usr/share/icons/hicolor/SIZExSIZE/apps/granular-music-visualizer.png``, and ``icon.png``
itself as the 512 pixel one. The picture is the same one, only scaled. Run this again only
when the icon changes; the files are kept in the repository so that building the package
needs no Qt.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtGui import QGuiApplication, QImage

SIZES = (16, 22, 24, 32, 48, 64, 128, 256)  # 512: packaging/icon.png itself
here = Path(__file__).parent
source = here.parent / "icon.png"


def main():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
    picture = QImage(str(source)).convertToFormat(QImage.Format.Format_ARGB32)
    if picture.isNull() or picture.width() != picture.height():
        raise SystemExit(f"Not a square picture Qt can read: {source}")
    out = here / "icons"
    out.mkdir(exist_ok=True)
    for size in SIZES:
        scaled = picture.scaled(
            size,
            size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        if not scaled.save(str(out / f"{size}.png")):
            raise SystemExit(f"Could not write {out / f'{size}.png'}")
    print(out, ", ".join(map(str, SIZES)))


if __name__ == "__main__":
    main()
