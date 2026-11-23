"""Radiant: the radial view as concentric luminous structures.

From the centre outwards: a core of light with a halo and rays; a dense ring of spectrum
spikes; a jagged contour; a ring of separate lit segments; the primary ring; the four
instruments, each in a band of its own, drawn by their own layers; the outer primary
ring. Round it all an atmosphere with drifting points of light, and streaks where
something was struck.

Where things are and what moves them is ``radiant_geometry`` (numbers only). What colour
they are is ``radiant_palette`` (one colour per role, all the user's to change). This
module only draws: it has no colour of its own (white appears only as light: a lit thing
is its role's colour moved towards white) and it keeps nothing from one frame to the next
except pictures cached per size and colour.

``scene`` works out one frame's numbers from the visual state and an audio time;
``paint`` draws them. Nothing here takes in analysis messages or runs a clock.
"""

import math
from collections import OrderedDict
from functools import lru_cache
from types import SimpleNamespace

import numpy as np

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen, QPixmap, QPolygonF, QRadialGradient

from .. import layers as layer_registry
from .. import palette as canonical
from ..strip import polygon
from . import radiant_geometry as geometry
from .radiant_palette import INSTRUMENTS, RadiantPalette

LIGHT = "#ffffff"  # what a lit thing moves towards; never drawn as a colour of its own
RING_SECONDS = 6.0  # one turn of the history rings, as in the classic view
PICTURES = 10  # cached pictures kept (halo, centre, rays, glow rings)
RAYS = 28
TICKS = 120  # fine marks inside the outer primary ring
RAY_TURN = 0.035  # radians a second the rays turn, by audio time


@lru_cache(maxsize=4096)
def lit(colour, towards, steps):
    """``colour`` moved ``steps`` sixteenths of the way to ``towards``, mixed as light."""
    if steps <= 0:
        return colour
    if steps >= 16:
        return towards
    return canonical.blend([(colour, 16 - steps), (towards, steps)])


def shade(colour, opacity=1.0, towards=LIGHT, share=0.0):
    """A QColor: ``colour`` moved ``share`` of the way to ``towards``, at ``opacity``."""
    mixed = QColor(lit(colour, towards, round(max(0.0, min(1.0, share)) * 16)))
    mixed.setAlphaF(max(0.0, min(1.0, opacity)))
    return mixed


class Radiant:
    name = "radiant"

    def __init__(self):
        self.palette = RadiantPalette()
        self._pictures = OrderedDict()  # (name, colours, size) -> picture, newest last

    # -- colours ---------------------------------------------------------------------------

    def instrument(self, key):
        """An instrument's colour in this theme."""
        return self.palette[key] if key in INSTRUMENTS else canonical.instrument(key)

    @property
    def neutral(self):
        """The background glow's colour when no instrument leads."""
        return self.palette["atmosphere_primary"]

    @property
    def beat(self):
        """The colour of the beat glow."""
        return self.palette["energy_accents"]

    # -- cached pictures ---------------------------------------------------------------------

    def picture(self, name, colours, size, build):
        """A picture that depends only on colours and a size: built once, a few kept."""
        key = (name, colours, size)
        held = self._pictures.pop(key, None)
        if held is None:
            held = build()
            while len(self._pictures) >= PICTURES:
                self._pictures.popitem(last=False)
        self._pictures[key] = held
        return held

    @staticmethod
    def _blank(pixels):
        picture = QPixmap(pixels, pixels)
        picture.fill(Qt.GlobalColor.transparent)
        artist = QPainter(picture)
        artist.setRenderHint(QPainter.RenderHint.Antialiasing)
        artist.setPen(Qt.PenStyle.NoPen)
        return picture, artist

    def halo_picture(self, pixels):
        colour = self.palette["core_halo"]

        def build():
            picture, artist = self._blank(pixels)
            glow = QRadialGradient(QPointF(pixels / 2, pixels / 2), pixels / 2)
            glow.setColorAt(0.0, shade(colour, 0.85))
            glow.setColorAt(0.35, shade(colour, 0.38))
            glow.setColorAt(1.0, shade(colour, 0.0))
            artist.fillRect(picture.rect(), glow)
            artist.end()
            return picture

        return self.picture("halo", colour, pixels, build)

    def centre_picture(self, pixels):
        centre, halo = self.palette["core_center"], self.palette["core_halo"]

        def build():
            picture, artist = self._blank(pixels)
            glow = QRadialGradient(QPointF(pixels / 2, pixels / 2), pixels / 2)
            glow.setColorAt(0.0, shade(centre, 1.0))
            glow.setColorAt(0.42, shade(centre, 0.95))
            glow.setColorAt(0.72, shade(lit(centre, halo, 10), 0.55))
            glow.setColorAt(1.0, shade(halo, 0.0))
            artist.fillRect(picture.rect(), glow)
            artist.end()
            return picture

        return self.picture("centre", (centre, halo), pixels, build)

    def rays_picture(self, pixels):
        colour = self.palette["core_rays"]

        def build():
            picture, artist = self._blank(pixels)
            half = pixels / 2
            fade = QRadialGradient(QPointF(half, half), half)
            fade.setColorAt(0.0, shade(colour, 0.95))
            fade.setColorAt(0.5, shade(colour, 0.4))
            fade.setColorAt(1.0, shade(colour, 0.0))
            artist.setBrush(QBrush(fade))
            spots = geometry.needles(geometry.seed_of(0.0, 99), RAYS)
            for index, (jitter, length, width) in enumerate(spots):
                angle = 2 * math.pi * (index + 0.6 * jitter) / RAYS
                sin, cos = math.sin(angle), math.cos(angle)
                tip = half * (0.3 + 0.7 * ((length - 0.55) / 0.45) ** 2)
                wide = half * 0.009 * width
                artist.drawConvexPolygon(
                    QPolygonF(
                        [
                            QPointF(half + wide * cos, half + wide * sin),
                            QPointF(half + tip * sin, half - tip * cos),
                            QPointF(half - wide * cos, half - wide * sin),
                        ]
                    )
                )
            artist.end()
            return picture

        return self.picture("rays", colour, pixels, build)

    def ring_picture(self, name, colour, radius, width, ratio):
        """A soft ring of light of ``width`` either side of ``radius``."""
        size = round((radius + width) * 2 * ratio)

        def build():
            picture, artist = self._blank(size)
            half = size / 2
            glow = QRadialGradient(QPointF(half, half), half)
            inner = max(0.0, (radius - width) / (radius + width))
            glow.setColorAt(0.0, shade(colour, 0.0))
            glow.setColorAt(inner, shade(colour, 0.0))
            glow.setColorAt((1 + inner) / 2, shade(colour, 0.5))
            glow.setColorAt(1.0, shade(colour, 0.0))
            artist.fillRect(picture.rect(), glow)
            artist.end()
            return picture

        return self.picture(name, colour, size, build)

    # -- geometry of this frame ----------------------------------------------------------------

    def layout(self, canvas, rect, active):
        cx, cy, reach = canvas.radial_geometry(rect)
        return geometry.layout(
            cx, cy, reach, tuple(layer.profile.weight for layer in active), 5 * canvas.scale
        )

    def scene(self, state, now, lay, *, values, peaks, response, running, beats=True, keys=()):
        """Everything one frame needs, as numbers: a function of the visual state, the
        display spectrum and audio time ``now``. ``keys``: the instruments drawn, outermost
        first; their live states must already be at ``now`` (the layers see to that)."""
        since = getattr(state, "segment_start", None)
        if running and now is not None:
            level, intensity = geometry.mix_levels(state.frames, now)
            flash, pulse, onsets, _ = geometry.impulses(state.markers, now, since, beats)
        else:
            level = intensity = flash = pulse = 0.0
            onsets = []
        radius, halo, rays = geometry.core(lay, level, intensity, flash, pulse, response.breath)
        attacks = {}
        if running:
            for band, key in enumerate(keys):
                live = state.live.get(key)
                if live is None or live.t is None or live.weight <= 0.02:
                    continue
                if key == "bass":
                    events = [(age, strength, live.now - age) for age, strength in live.beats(0.5)]
                else:
                    events = [
                        (age, max(hit.strength, hit.low, hit.mid, hit.high), hit.time)
                        for age, hit in live.recent(0.5)
                        if key != "vocals" or hit.start
                    ]
                if events:
                    attacks[key] = (band, [(a, s * live.weight, t) for a, s, t in events])
        curve = response.spectrum_curve
        return SimpleNamespace(
            now=now,
            running=running,
            level=level,
            intensity=intensity,
            musical=state.musical.value if running else 0.0,
            flash=flash,
            pulse=pulse,
            core=radius,
            halo=halo,
            rays=rays,
            ray_turn=RAY_TURN * (now or 0.0),
            spikes=geometry.spikes(lay, values if running or values is not None else None, curve),
            peaks=geometry.peak_marks(lay, peaks, curve),
            contour=geometry.contour(lay, state.frames if running else None, now, response.motion),
            segments=geometry.segment_levels(values),
            segment_ring=geometry.segment_ring(lay, pulse),
            streaks=geometry.streaks(lay, onsets, attacks, response.impact),
            shocks=geometry.shocks(lay, onsets),
        )

    # -- drawing -----------------------------------------------------------------------------

    def paint_static(self, canvas, painter, rect, active):
        """The atmosphere, into the rarely-changing picture: the base colour over the whole
        view, the two hazes, the faint arcs and the soft light of the primary rings."""
        colours = self.palette
        lay = self.layout(canvas, rect, active)
        centre = QPointF(lay.cx, lay.cy)
        view = QRectF(0, 0, canvas.width(), canvas.height())
        painter.fillRect(view, QColor(colours["background_base"]))
        far = math.hypot(canvas.width(), canvas.height()) / 2
        haze = QRadialGradient(centre, max(lay.reach * 1.9, far))
        haze.setColorAt(0.0, shade(colours["atmosphere_secondary"], 0.0))
        haze.setColorAt(0.5, shade(colours["atmosphere_secondary"], 0.26))
        haze.setColorAt(1.0, shade(colours["atmosphere_secondary"], 0.05))
        painter.fillRect(view, haze)
        near = QRadialGradient(centre, lay.reach * 1.2)
        near.setColorAt(0.0, shade(colours["atmosphere_primary"], 0.62))
        near.setColorAt(0.45, shade(colours["atmosphere_primary"], 0.3))
        near.setColorAt(1.0, shade(colours["atmosphere_primary"], 0.0))
        painter.fillRect(view, near)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for share, start, span in geometry.ambient_arcs():
            radius = lay.reach * share
            painter.setPen(QPen(shade(colours["atmosphere_secondary"], 0.22), 1.2 * canvas.scale))
            painter.drawArc(
                QRectF(lay.cx - radius, lay.cy - radius, 2 * radius, 2 * radius),
                round(start * 5760),
                round(span * 5760),
            )
        painter.setPen(Qt.PenStyle.NoPen)
        # The soft light the rings stand in. It never changes, so it costs nothing a frame.
        for role, radius, width, strength in (
            ("primary_ring", lay.frame, lay.reach * 0.06, 0.5),
            ("primary_ring", lay.rim, lay.reach * 0.045, 0.5),
            ("secondary_ring", lay.contour, lay.zone * 0.09, 0.26),
            ("segmented_ring", lay.segments, lay.zone * 0.07, 0.3),
        ):
            glow = QRadialGradient(centre, radius + width)
            inner = (radius - width) / (radius + width)
            glow.setColorAt(0.0, shade(colours[role], 0.0))
            glow.setColorAt(inner, shade(colours[role], 0.0))
            glow.setColorAt((1 + inner) / 2, shade(colours[role], strength))
            glow.setColorAt(1.0, shade(colours[role], 0.0))
            painter.setBrush(QBrush(glow))
            painter.drawEllipse(centre, radius + width, radius + width)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(shade(colours["primary_ring"], 0.85), 2.4 * canvas.scale))
        painter.drawEllipse(centre, lay.frame, lay.frame)
        painter.drawEllipse(centre, lay.rim, lay.rim)
        # A ring of fine ticks just inside the rim.
        ticks = np.linspace(0.0, 2 * np.pi, TICKS, endpoint=False)
        sines, cosines = np.sin(ticks), np.cos(ticks)
        near, far = lay.rim - lay.reach * 0.024, lay.rim - lay.reach * 0.011
        painter.setPen(QPen(shade(colours["primary_ring"], 0.55), 1.2 * canvas.scale))
        painter.drawLines(
            [
                QLineF(lay.cx + near * s, lay.cy - near * c, lay.cx + far * s, lay.cy - far * c)
                for s, c in zip(sines.tolist(), cosines.tolist(), strict=True)
            ]
        )

    def paint_layers(self, canvas, painter, lay, active):
        """The instruments, each by its own layer, in its band, in its Radiant colour."""
        state = canvas.state
        for layer, (low, high) in zip(active, lay.bands, strict=True):
            colour = self.instrument(layer.key)
            live = state.live.get(layer.key)
            presence = min(1.0, live.weight) if live is not None and live.t is not None else 0.0
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(shade(colour, 0.12 + 0.16 * presence), 0))
            painter.drawEllipse(QPointF(lay.cx, lay.cy), low, low)
            stage = layer_registry.Stage(
                canvas.now, RING_SECONDS, canvas.scale, state.generation, canvas.font,
                ring=(lay.cx, lay.cy, low, high), live=canvas.radial_live,
                gain=0.55 + 0.45 * state.musical.value, response=canvas.response, colour=colour,
            )  # fmt: skip
            with canvas.perf.measure("layer " + layer.key):
                layer.paint_stage(painter, stage, state)

    def paint(self, canvas, painter, rect, active, spectrum=True):
        """One frame of the moving picture, over the atmosphere."""
        colours = self.palette
        response = canvas.response
        running = canvas.condition in ("live", "replay") and canvas.now is not None
        lay = self.layout(canvas, rect, active)
        ratio = canvas.devicePixelRatioF()
        centre = QPointF(lay.cx, lay.cy)
        scale = canvas.scale
        if running and active:
            self.paint_layers(canvas, painter, lay, active)
        values, peaks = canvas.display.values, canvas.display.peaks
        view = self.scene(
            canvas.state, canvas.now, lay, values=values, peaks=peaks, response=response,
            running=running, beats=canvas.show_beats, keys=tuple(layer.key for layer in active),
        )  # fmt: skip
        shine = response.shine
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)

        # Points of light, drifting with audio time.
        xs, ys, sizes = geometry.particles(lay, view.now)
        for size in range(3):
            pen = QPen(shade(colours["particle_accents"], 0.16 + 0.12 * size + 0.2 * view.intensity))
            pen.setWidthF((1.0 + 0.7 * size) * scale)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            keep = sizes == size
            painter.drawPoints(polygon(xs[keep], ys[keep]))

        # Halo and rays: sustained intensity keeps them up, an onset throws them out.
        # They are light, so they are added to what is under them, not laid over it.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Plus)
        painter.setOpacity(min(1.0, (0.3 + 0.6 * view.intensity + 0.3 * view.flash) * shine))
        halo = self.halo_picture(round(lay.zone * 1.2 * ratio))
        painter.drawPixmap(
            QRectF(lay.cx - view.halo, lay.cy - view.halo, 2 * view.halo, 2 * view.halo),
            halo, QRectF(halo.rect()),
        )  # fmt: skip
        painter.setOpacity(min(1.0, (0.3 + 0.5 * view.intensity + 0.4 * view.flash) * shine))
        rays = self.rays_picture(round(lay.zone * 2.0 * ratio))
        painter.save()
        painter.translate(centre)
        painter.rotate(math.degrees(view.ray_turn))
        painter.drawPixmap(
            QRectF(-view.rays, -view.rays, 2 * view.rays, 2 * view.rays), rays, QRectF(rays.rect())
        )
        painter.restore()

        # The spike ring: a soft light on its baseline, then the spikes themselves.
        if spectrum:
            glow = self.ring_picture(
                "spectrum glow", colours["spectrum_body"], lay.base, lay.zone * 0.11, ratio
            )
            painter.setOpacity(min(1.0, (0.3 + 0.6 * view.level + 0.3 * view.flash) * shine))
            spread = lay.base + lay.zone * 0.11
            painter.drawPixmap(
                QRectF(lay.cx - spread, lay.cy - spread, 2 * spread, 2 * spread),
                glow, QRectF(glow.rect()),
            )  # fmt: skip
            painter.setOpacity(1.0)
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            angles, outer, inner = view.spikes
            sines, cosines = np.sin(angles), np.cos(angles)
            ends = np.column_stack(
                (
                    lay.cx + inner * sines,
                    lay.cy - inner * cosines,
                    lay.cx + outer * sines,
                    lay.cy - outer * cosines,
                )
            )
            # How far the body is lit: the music's intensity (when that colouring is on)
            # and an onset's flash.
            lift = min(1.0, 0.55 * canvas.heat + 0.5 * view.flash * response.flash)
            body = lit(colours["spectrum_body"], colours["spectrum_highlights"], round(lift * 16))
            # Plain colours and thin lines: a gradient along 300 lines, or filled spikes,
            # cost several milliseconds a frame. A band's own spike is wider than the
            # finer ones between two bands.
            step = geometry.SUB
            own = np.zeros(len(ends), dtype=bool)
            own[::step] = True
            painter.setBrush(Qt.BrushStyle.NoBrush)
            for mask, width in ((own, 2.2 * scale), (~own, 0)):
                pen = QPen(QColor(body), width)  # opaque: translucent lines cost five times as much
                pen.setCapStyle(Qt.PenCapStyle.FlatCap)
                painter.setPen(pen)
                painter.drawLines([QLineF(*row) for row in ends[mask].tolist()])
            # The tips of the spikes that stand out catch the light: the taller, the larger.
            tall = (outer - lay.base) / (lay.tips - lay.base)
            for mask, width in ((tall > 0.55, 3.4), ((tall > 0.22) & (tall <= 0.55), 2.2)):
                if mask.any():
                    pen = QPen(shade(colours["spectrum_highlights"], 0.95))
                    pen.setWidthF(width * scale)
                    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                    painter.setPen(pen)
                    painter.drawPoints(polygon(ends[mask, 2], ends[mask, 3]))
            painter.setPen(QPen(shade(body, 0.95), 0))
            painter.drawEllipse(centre, lay.base, lay.base)
            painter.drawEllipse(centre, lay.base - lay.thickness, lay.base - lay.thickness)
            marks, radii = view.peaks
            if len(marks):
                pen = QPen(shade(colours["spectrum_highlights"], 0.85))
                pen.setWidthF(1.6 * scale)
                pen.setCapStyle(Qt.PenCapStyle.RoundCap)
                painter.setPen(pen)
                painter.drawPoints(polygon(*geometry.xy(lay, marks, np.minimum(radii, lay.tips))))
        painter.setOpacity(1.0)
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)

        # The core itself, over the rays, and its boundary.
        core = self.centre_picture(round(lay.core_most * 2 * ratio))
        painter.setOpacity(min(1.0, (0.6 + 0.4 * view.intensity + 0.3 * view.flash) * max(1.0, shine)))
        painter.drawPixmap(
            QRectF(lay.cx - view.core, lay.cy - view.core, 2 * view.core, 2 * view.core),
            core, QRectF(core.rect()),
        )  # fmt: skip
        painter.setOpacity(1.0)
        painter.setPen(QPen(shade(colours["core_halo"], 0.35 + 0.45 * view.flash), 0))
        painter.drawEllipse(centre, view.core * 1.12, view.core * 1.12)

        # The contour.
        angles, radii = view.contour
        painter.setPen(
            QPen(
                shade(
                    colours["secondary_ring"],
                    min(1.0, (0.5 + 0.4 * view.level) * shine),
                    share=0.5 * view.flash,
                ),
                0,  # one device pixel: a wide pen along a curve this jagged costs milliseconds
            )
        )
        painter.drawPolyline(polygon(*geometry.xy(lay, np.append(angles, angles[0]), np.append(radii, radii[0]))))
        loud = np.abs(radii - lay.contour) > 0.5 * lay.swing
        if loud.any():  # the strongest teeth catch the light
            pen = QPen(shade(colours["secondary_ring"], 0.9, share=0.6))
            pen.setWidthF(2.2 * scale)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(polygon(*geometry.xy(lay, angles[loud], radii[loud])))

        # The segmented ring: complete at rest, each segment lit by its part of the spectrum.
        low, high = view.segment_ring
        arcs = geometry.segment_arcs()
        middle = (low + high) / 2
        chords = np.column_stack(
            (
                lay.cx + middle * np.sin(arcs[:, 0]),
                lay.cy - middle * np.cos(arcs[:, 0]),
                lay.cx + middle * np.sin(arcs[:, 1]),
                lay.cy - middle * np.cos(arcs[:, 1]),
            )
        ).tolist()
        glow = min(1.0, 0.6 * view.flash * response.flash + 0.5 * view.pulse)
        for chord, strength in zip(chords, view.segments.tolist(), strict=True):
            bright = min(1.0, strength * 1.15 + glow)
            pen = QPen(
                shade(
                    colours["segmented_ring"],
                    min(1.0, (0.28 + 0.72 * bright) * max(1.0, shine)),
                    share=0.55 * bright * bright,
                ),
                high - low,
            )
            pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(pen)
            painter.drawLine(QLineF(*chord))

        # The primary rings are in the still picture; here only the light the music puts
        # on them (one device pixel wide: wide circles this large cost too much a frame).
        painter.setPen(
            QPen(
                shade(
                    colours["primary_ring"],
                    min(1.0, (0.25 + 0.6 * view.musical + 0.4 * view.flash) * shine),
                    share=0.3 + 0.5 * view.flash,
                ),
                0,
            )
        )
        painter.drawEllipse(centre, lay.frame, lay.frame)
        painter.drawEllipse(centre, lay.rim, lay.rim)

        # What was struck: widening circles and radial streaks, fading with their age.
        for radius, strength in view.shocks:
            painter.setPen(QPen(shade(colours["energy_accents"], 0.5 * strength * shine), 0))
            painter.drawEllipse(centre, radius, radius)
        painter.setPen(Qt.PenStyle.NoPen)
        for angle, start, end, width, strength, source in view.streaks:
            colour = colours["energy_accents"]
            if source is not None:
                colour = lit(colour, self.instrument(source), 8)
            painter.setBrush(shade(colour, min(1.0, (0.35 + 0.65 * strength) * shine)))
            sin, cos = math.sin(angle), math.cos(angle)
            waist = start + 0.22 * (end - start)
            wide = width * scale * 0.9
            painter.drawConvexPolygon(
                QPolygonF(
                    [
                        QPointF(lay.cx + start * sin, lay.cy - start * cos),
                        QPointF(lay.cx + waist * sin + wide * cos, lay.cy - waist * cos + wide * sin),
                        QPointF(lay.cx + end * sin, lay.cy - end * cos),
                        QPointF(lay.cx + waist * sin - wide * cos, lay.cy - waist * cos - wide * sin),
                    ]
                )
            )
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, False)
        return view
