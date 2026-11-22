"""Where an instrument layer's picture goes: the interface between a visualizer mode and
the instrument renderers.

A renderer draws in *band space*: x is time in pixels along the band, y runs from 0 (top
of the band) to the band's height. It never draws with the painter directly; it hands its
shapes to a surface, and the surface puts them where the current mode wants them:

* ``Linear``  the band is a horizontal strip (Spectrum Bars): band space is the picture.
* ``Polar``   the band is a ring (Radial): x goes round the circle, one turn per period,
  and y goes from the ring's outer edge inwards.

A new visualizer mode supports every instrument by providing a surface with these six
methods; no renderer has to be written again.
"""

import math

import numpy as np
from PySide6.QtCore import QLineF, QPointF

from ..strip import polygon


class Linear:
    polar = False

    def polyline(self, artist, xs, ys):
        artist.drawPolyline(polygon(xs, ys))

    def polygon(self, artist, xs, ys):
        artist.drawPolygon(polygon(xs, ys))

    def line(self, artist, x0, y0, x1, y1):
        artist.drawLine(QLineF(x0, y0, x1, y1))

    def lines(self, artist, segments):
        artist.drawLines([QLineF(*segment) for segment in segments])

    def ellipse(self, artist, x, y, radius):
        artist.drawEllipse(QPointF(x, y), radius, radius)


class Polar:
    """Band space wrapped round a centre. ``per_turn`` is the band's length in pixels for one
    full turn; ``outer`` the radius of y = 0."""

    polar = True

    def __init__(self, cx, cy, outer, per_turn):
        self.cx, self.cy, self.outer = cx, cy, outer
        self.k = 2 * math.pi / per_turn
        self.step = max(2.0, per_turn / 180)  # a straight piece spans at most two degrees

    def xy(self, xs, ys):
        angle = np.asarray(xs, dtype=float) * self.k
        radius = self.outer - np.asarray(ys, dtype=float)
        return self.cx + radius * np.sin(angle), self.cy - radius * np.cos(angle)

    def polyline(self, artist, xs, ys):
        artist.drawPolyline(polygon(*self.xy(xs, ys)))

    def polygon(self, artist, xs, ys):
        artist.drawPolygon(polygon(*self.xy(xs, ys)))

    def line(self, artist, x0, y0, x1, y1):
        pieces = max(1, int(abs(x1 - x0) / self.step))
        if pieces == 1:
            (ax, bx), (ay, by) = self.xy((x0, x1), (y0, y1))
            artist.drawLine(QLineF(float(ax), float(ay), float(bx), float(by)))
        else:  # a long run along the band follows the ring
            artist.drawPolyline(
                polygon(*self.xy(np.linspace(x0, x1, pieces + 1), np.linspace(y0, y1, pieces + 1)))
            )

    def lines(self, artist, segments):
        for segment in segments:
            self.line(artist, *segment)

    def ellipse(self, artist, x, y, radius):
        px, py = self.xy((x,), (y,))
        artist.drawEllipse(QPointF(float(px[0]), float(py[0])), radius, radius)
