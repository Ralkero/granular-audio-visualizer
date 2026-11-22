"""Fast drawing helpers for scrolling histories.

Redrawing fifteen seconds of history sixty times a second is what makes a scrolling display
slow. A ``Strip`` keeps the history in a pixmap anchored to audio time: each refresh draws
only the samples that arrived since the last one and then copies the visible part across.
"""

import ctypes
import math

import numpy as np
import shiboken6
from PySide6.QtCore import QRectF, Qt
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
        self.shifts = self.redraws = 0  # how often the picture slid, or was drawn afresh

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
            self.redraws += 1
        elif (start - self.origin) * per_second + area.width() * 1.2 > width:
            # Out of room on the right: slide the picture left by a whole number of pixels,
            # in place (no new picture, no second copy), and empty what the slide exposed.
            shift = math.floor((start - self.origin) * per_second * ratio)
            self.pixmap.scroll(-shift, 0, self.pixmap.rect())
            eraser = QPainter(self.pixmap)
            eraser.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            eraser.fillRect(
                self.pixmap.width() - shift,
                0,
                shift,
                self.pixmap.height(),
                Qt.GlobalColor.transparent,
            )
            eraser.end()
            self.origin += shift / ratio / per_second
            self.shifts += 1
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


class Sweep:
    """History on a ring: a hand goes round once per ``period`` and what it passes stays
    where it was drawn until the hand comes round again.

    Nothing ever moves, so nothing is copied or shifted: each refresh draws the rows that
    arrived since the last one at their own angle, after emptying the part of the ring just
    ahead of the hand. ``draw(artist, since)`` paints what arrived after ``since`` (None: the whole
    turn) into a square picture whose centre is the ring's centre.
    """

    def __init__(self):
        self.pixmap = self.key = self.until = self.cleared = None
        self.shifts = self.redraws = 0

    def paint(self, painter, cx, cy, outer, period, gap, key, latest, draw):
        ratio = painter.device().devicePixelRatioF()
        size = 2 * (math.ceil(outer) + 2)
        key = (key, size, period, ratio)
        if self.pixmap is None or key != self.key:
            self.key, self.until, self.cleared = key, None, None
            self.pixmap = QPixmap(math.ceil(size * ratio), math.ceil(size * ratio))
            self.pixmap.fill(Qt.GlobalColor.transparent)
            self.redraws += 1
        if latest is not None and (self.until is None or latest > self.until):
            artist = QPainter(self.pixmap)
            artist.scale(ratio, ratio)
            # Empty the ring a little way ahead of the hand, each part once and before
            # anything of this turn is drawn there: a mark is never cut by a later clearing.
            ahead = latest + gap * period
            if self.cleared is not None and ahead > self.cleared:
                first = (self.cleared % period) / period * 360
                span = min(360.0, (ahead - self.cleared) / period * 360)
                artist.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
                artist.setPen(Qt.PenStyle.NoPen)
                artist.setBrush(Qt.GlobalColor.black)
                # Qt measures angles anticlockwise from three o'clock, in sixteenths.
                artist.drawPie(
                    QRectF(0, 0, size, size), round((90 - first - span) * 16), round(span * 16)
                )
                artist.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            self.cleared = ahead
            draw(artist, self.until)
            artist.end()
            self.until = latest
        half = size / 2
        painter.drawPixmap(
            QRectF(cx - half, cy - half, size, size),
            self.pixmap,
            QRectF(0, 0, self.pixmap.width(), self.pixmap.height()),
        )
