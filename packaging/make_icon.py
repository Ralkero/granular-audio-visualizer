"""Make the application's icon file from a picture.

    python packaging/make_icon.py PICTURE

Writes ``packaging/icon.png`` (the picture, square, 512 pixels) and ``packaging/app.ico``
(the same at the sizes Windows asks for). ``Build App.ps1`` gives ``app.ico`` to the
executable and copies it beside it, where the running application takes its window icon
from. Run this again only to change the icon; both files are kept in the repository.
"""

import os
import struct
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, Qt
from PySide6.QtGui import QGuiApplication, QImage, QPainter

SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)
here = Path(__file__).parent


def png(image):
    data = QByteArray()
    buffer = QBuffer(data)
    buffer.open(QIODevice.OpenModeFlag.WriteOnly)
    image.save(buffer, "PNG")
    return bytes(data)


def main():
    app = QGuiApplication.instance() or QGuiApplication(sys.argv[:1])  # noqa: F841
    picture = QImage(sys.argv[1]).convertToFormat(QImage.Format.Format_ARGB32)
    if picture.isNull():
        raise SystemExit(f"Not a picture Qt can read: {sys.argv[1]}")
    side = max(picture.width(), picture.height())
    square = QImage(side, side, QImage.Format.Format_ARGB32)
    square.fill(Qt.GlobalColor.transparent)
    artist = QPainter(square)
    artist.drawImage((side - picture.width()) // 2, (side - picture.height()) // 2, picture)
    artist.end()

    def sized(size):
        return square.scaled(
            size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
        )

    sized(512).save(str(here / "icon.png"))
    # An .ico file: a directory of entries, then each picture as a PNG.
    pictures = [png(sized(size)) for size in SIZES]
    offset = 6 + 16 * len(SIZES)
    parts = [struct.pack("<HHH", 0, 1, len(SIZES))]
    for size, data in zip(SIZES, pictures, strict=True):
        parts.append(struct.pack("<BBBBHHII", size % 256, size % 256, 0, 0, 1, 32, len(data), offset))
        offset += len(data)
    (here / "app.ico").write_bytes(b"".join(parts + pictures))
    print(here / "app.ico", sum(map(len, pictures)), "bytes")


if __name__ == "__main__":
    main()
