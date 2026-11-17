"""Fast drawing helpers for scrolling histories.

Redrawing fifteen seconds of history sixty times a second is what makes a scrolling display
slow. A ``Strip`` keeps the history in a pixmap anchored to audio time: each refresh draws
only the samples that arrived since the last one and then copies the visible part across.
"""

import ctypes
import math

import numpy as np
import shiboken6
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap, QPolygonF


def polygon(xs, ys):
    """QPolygonF from two arrays, written straight into Qt's buffer."""
    count = len(xs)
    result = QPolygonF()
    result.resize(count)
    if count:
        address = shiboken6.getCppPointer(result.data())[0]
        points = np.frombuffer((ctypes.c_double * (2 * count)).from_address(address))
        points[0::2], points[1::2] = xs, ys
    return result


class Strip:
    def __init__(self):
        self.pixmap = self.key = self.until = None
        self.origin = 0.0

    def _blank(self, width, height, ratio):
        pixmap = QPixmap(math.ceil(width * ratio), max(1, math.ceil(height * ratio)))
        pixmap.fill(Qt.GlobalColor.transparent)
        return pixmap

    def paint(self, painter, area, start, per_second, key, latest, draw):
        """Show history from audio time ``start`` across ``area``.

        ``latest`` is any value that grows when there is something new to draw (a time, a
        count). ``draw(painter, to_x, since, height)`` must paint what arrived after
        ``since``, the previous ``latest`` (None: everything), using ``to_x`` to place audio
        times. ``key`` is anything that, when it changes, makes the cached picture wrong.
        """
        ratio = painter.device().devicePixelRatioF()
        width = area.width() * 1.5
        key = (key, round(area.width()), round(area.height()), per_second, ratio)
        if self.pixmap is None or key != self.key or start < self.origin:
            self.key, self.origin, self.until = key, start, None
            self.pixmap = self._blank(width, area.height(), ratio)
        elif (start - self.origin) * per_second + area.width() * 1.2 > width:
            # Out of room on the right: slide the picture left by a whole number of pixels.
            shift = math.floor((start - self.origin) * per_second * ratio)
            fresh = self._blank(width, area.height(), ratio)
            mover = QPainter(fresh)
            mover.drawPixmap(QPointF(-shift, 0), self.pixmap)
            mover.end()
            self.pixmap = fresh
            self.origin += shift / ratio / per_second
        if latest is not None and (self.until is None or latest > self.until):
            origin = self.origin
            artist = QPainter(self.pixmap)
            artist.scale(ratio, ratio)
            draw(artist, lambda t: (t - origin) * per_second, self.until, area.height())
            artist.end()
            self.until = latest
        offset = (start - self.origin) * per_second
        painter.drawPixmap(
            area,
            self.pixmap,
            QRectF(offset * ratio, 0, area.width() * ratio, area.height() * ratio),
        )
