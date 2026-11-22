"""The unified view: instruments inside the selected visualizer, one palette, sections,
quick controls, and the performance-related behaviour that must stay correct."""

import colorsys
import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtCore import QPoint, QRectF, Qt
from PySide6.QtGui import QPainter, QPixmap
from PySide6.QtTest import QTest

from gmv.bus import EventBus
from gmv.models import InstrumentEstimate
from gmv.visualizer import layers as layer_registry
from gmv.visualizer import palette
from gmv.visualizer.lanes import KNOWN
from gmv.visualizer.strip import Strip, Sweep
from gmv.visualizer.window import DRAIN_MOST, MIN_MAIN

from test_instrument_layers_phase2 import ALL, activity, play_band
from test_visualizer import frame


def opaque(pixmap, step=2):
    """RGB of the opaque pixels of a picture (sampled)."""
    image = pixmap.toImage()
    found = []
    for x in range(0, image.width(), step):
        for y in range(0, image.height(), step):
            colour = image.pixelColor(x, y)
            if colour.alpha() > 120:
                found.append((colour.red(), colour.green(), colour.blue()))
    return np.array(found).reshape(-1, 3)


def hue(colour):
    r, g, b = (v / 255 for v in palette.rgb(colour))
    return colorsys.rgb_to_hsv(r, g, b)[0] * 360


def hue_gap(a, b):
    gap = abs(hue(a) - hue(b)) % 360
    return min(gap, 360 - gap)


def picture(layer, mode):
    return (layer.sweep if mode == "radial" else layer.strip).pixmap


# -- palette ------------------------------------------------------------------------------


def test_every_instrument_has_one_colour_everywhere():
    colours = palette.INSTRUMENTS
    assert colours == {
        "vocals": "#f472b6",
        "guitar": "#f5c451",
        "bass": "#4cc9f0",
        "drums": "#ff8a5b",
    }
    profiles = layer_registry.profiles()
    assert {key: profile.colour for key, profile in profiles.items()} == colours
    lanes = {lane.key: lane.color for lane in KNOWN}
    assert {key: lanes[key] for key in colours} == colours
    # Regions of the kit are shades of the drums' own colour, not other hues.
    assert all(hue_gap(shade, colours["drums"]) < 25 for shade in palette.DRUM_REGIONS.values())
    assert len(set(palette.DRUM_REGIONS.values())) == 3
    # The four are far apart, predictions are none of them, and the whole-mix spectrum is
    # not the bass.
    keys = list(colours)
    assert all(
        hue_gap(colours[a], colours[b]) > 20 for i, a in enumerate(keys) for b in keys[i + 1 :]
    )
    assert all(hue_gap(palette.PREDICTED, colour) > 30 for colour in colours.values())
    assert all(hue_gap(shade, colours["bass"]) > 30 for shade in palette.SPECTRUM)


# -- both modes, each instrument and all four ---------------------------------------------


@pytest.mark.parametrize("mode", ["radial", "bars"])
def test_each_instrument_is_drawn_inside_the_selected_visualizer_in_its_own_colour(window, mode):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    window.set_mode(mode)
    play_band(window)
    window.grab()
    assert [layer.key for layer in canvas.active_layers()] == ["vocals", "guitar", "bass", "drums"]
    for key, layer in canvas.layers.items():
        pixels = opaque(picture(layer, mode))
        assert len(pixels) > 60, (mode, key)
        # Its picture is in its own hue: of the four instrument colours, its own is nearest
        # to the typical hue of what it drew (whitened highlights left aside).
        hsv = np.array([colorsys.rgb_to_hsv(*(pixel / 255)) for pixel in pixels])
        vivid = hsv[hsv[:, 1] > 0.25]
        assert len(vivid) > 20, (mode, key)
        angle = np.angle(np.exp(2j * np.pi * vivid[:, 0]).mean()) % (2 * np.pi)
        typical = np.degrees(angle)
        gaps = {
            name: min(abs(typical - hue(colour)) % 360, 360 - abs(typical - hue(colour)) % 360)
            for name, colour in palette.INSTRUMENTS.items()
        }
        assert min(gaps, key=gaps.get) == key, (mode, key, gaps)
    # Four different pictures, not one shape in four colours: they differ in how much they
    # cover as well (marks, strings, a slab, strands).
    covered = sorted(len(opaque(picture(layer, mode))) for layer in canvas.layers.values())
    assert covered[-1] > 2 * covered[0]
    # One at a time: only that instrument is drawn, and the others' pictures are not touched.
    for key in ALL:
        for other in ALL:
            window.set_layer(other, enabled=other == key)
        before = {
            name: picture(layer, mode).cacheKey()
            for name, layer in canvas.layers.items()
            if name != key
        }
        window.grab()
        assert [layer.key for layer in canvas.active_layers()] == [key]
        assert len(opaque(picture(canvas.layers[key], mode))) > 60
        assert before == {
            name: picture(layer, mode).cacheKey()
            for name, layer in canvas.layers.items()
            if name != key
        }
    for key in ALL:
        window.set_layer(key, enabled=True)
    with_all = window.grab().toImage()
    window.set_layer("guitar", enabled=False)
    assert window.grab().toImage() != with_all


def test_radial_puts_rings_round_the_spectrum_and_bars_puts_strips_across_it(window):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    play_band(window)
    window.set_mode("radial")
    window.grab()
    rings = [layer.sweep.pixmap for layer in canvas.layers.values()]
    assert all(ring is not None and ring.width() == ring.height() for ring in rings)
    # Outermost first; the six strings get the widest ring.
    bands = layer_registry.rings(100, 300, canvas.active_layers(), 4)
    assert bands[0][1] == 300 and all(a[0] > b[1] for a, b in zip(bands, bands[1:], strict=False))
    widths = {
        layer.key: high - low
        for layer, (low, high) in zip(canvas.active_layers(), bands, strict=True)
    }
    assert max(widths, key=widths.get) == "guitar"
    window.set_mode("bars")
    window.grab()
    strips = {key: layer.strip.pixmap for key, layer in canvas.layers.items()}
    assert all(strip.width() > 3 * strip.height() for strip in strips.values())
    assert strips["guitar"].height() > strips["bass"].height()


def test_switching_modes_and_resizing_keep_the_history(window):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    play_band(window)
    before = {key: track.since().copy() for key, track in window.state.activity.items()}
    generation = window.state.generation
    for mode in ("radial", "bars", "radial", "bars"):
        window.set_mode(mode)
        window.grab()
        assert all(len(opaque(picture(layer, mode))) > 60 for layer in canvas.layers.values())
    window.resize(1000, 640)
    window.grab()
    small = {key: layer.strip.pixmap.width() for key, layer in canvas.layers.items()}
    window.resize(1500, 900)
    window.grab()
    assert all(layer.strip.pixmap.width() > small[key] for key, layer in canvas.layers.items())
    assert all(len(opaque(layer.strip.pixmap)) > 60 for layer in canvas.layers.values())
    assert window.state.generation == generation
    for key, rows in before.items():
        assert np.array_equal(window.state.activity[key].since(), rows, equal_nan=True)


# -- redraws -----------------------------------------------------------------------------------


def test_one_instruments_judgement_redraws_that_instrument_only(window):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    window.set_mode("bars")
    play_band(window)
    window.grab()
    drawn = {key: layer.strip.redraws for key, layer in canvas.layers.items()}
    judged = dict(window.state.board.judged)
    # The recognition model changes its mind about guitar: detected -> absent.
    window.state.consume(
        InstrumentEstimate(
            session_id="s", source="synthetic", timestamp=4.0, interval_end=4.5, detected_monotonic=0,
            backend="fake", scores=(("guitar", 0.1),), states=(("guitar", "absent"),),
        )
    )  # fmt: skip
    for index in range(4):
        time = 4.0 + index * 0.02
        window.state.consume(
            frame(
                time,
                scores=(("drums", 0.9), ("bass", 0.9), ("vocals", 0.9)),
                active=("drums", "bass", "vocals"),
                method="activity_test",
            ),
            arrival=perf_counter(),
        )
        for key in ALL:
            window.state.consume(activity(time, key))
    canvas.advance(0.016, perf_counter())
    window.grab()
    after = {key: layer.strip.redraws for key, layer in canvas.layers.items()}
    assert window.state.board.judged["guitar"] == judged["guitar"] + 1
    assert after["guitar"] == drawn["guitar"] + 1
    assert {key: after[key] for key in ("vocals", "bass", "drums")} == {
        key: drawn[key] for key in ("vocals", "bass", "drums")
    }


def test_a_strip_slides_in_place_without_seams_or_lost_history():
    """Marks at known times, drawn as they arrive while the strip scrolls for a long time,
    are exactly where a strip drawn in one go puts them."""
    width, height, per_second = 300, 20, 100.0
    times = np.arange(0.0, 14.0, 0.37)

    def render(strip, steps):
        pixmap = QPixmap(width, height)
        for now in steps:
            pixmap.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pixmap)
            done = {"until": strip.until}

            def draw(artist, to_x, since, _tall, now=now):
                first = -1.0 if since is None else since
                for time in times[(times > first) & (times <= now)]:
                    artist.fillRect(QRectF(to_x(time), 0, 2, height), Qt.GlobalColor.white)

            strip.paint(
                painter,
                QRectF(0, 0, width, height),
                now - width / per_second,
                per_second,
                "k",
                now,
                draw,
            )
            painter.end()
            del done
        return pixmap.toImage()

    scrolled = Strip()
    live = render(scrolled, np.arange(0.5, 14.0, 0.05))
    whole = render(Strip(), [13.95])
    assert scrolled.shifts >= 5 and scrolled.redraws == 1  # it slid; it was never redrawn
    columns = lambda image: [x for x in range(width) if image.pixelColor(x, 10).alpha() > 0]  # noqa: E731
    assert columns(live) == columns(whole) and len(columns(live)) > 10


def test_a_ring_keeps_what_was_drawn_until_the_hand_comes_round_again():
    sweep = Sweep()
    target = QPixmap(220, 220)
    target.fill(Qt.GlobalColor.transparent)
    drawn = []

    def paint(now):
        painter = QPainter(target)

        def draw(artist, since):
            drawn.append(since)
            angle = 2 * np.pi * (now % 4.0) / 4.0
            x, y = 102 + 80 * np.sin(angle), 102 - 80 * np.cos(angle)
            artist.fillRect(QRectF(x - 2, y - 2, 4, 4), Qt.GlobalColor.white)

        sweep.paint(painter, 110, 110, 100, 4.0, 0.06, "k", now, draw)
        painter.end()

    def lit(now):  # is the mark drawn at ``now`` still on the ring?
        angle = 2 * np.pi * (now % 4.0) / 4.0
        return (
            sweep.pixmap.toImage()
            .pixelColor(round(102 + 80 * np.sin(angle)), round(102 - 80 * np.cos(angle)))
            .alpha()
            > 0
        )

    for now in np.arange(0.5, 3.0, 0.5):
        paint(float(now))
    assert drawn[0] is None and drawn[1] == 0.5  # first everything, then only what is new
    assert lit(0.5) and lit(2.5) and sweep.redraws == 1
    for now in np.arange(3.0, 4.31, 0.1):  # the hand comes round to just before where 0.5 was
        paint(float(now))
    assert not lit(0.5) and lit(2.5) and sweep.redraws == 1


def test_message_intake_is_bounded_in_order_and_accounted_for(window):
    bus = EventBus()
    window.attach(bus)
    for index in range(1500):
        bus.publish(frame(index * 0.02))
    window.tick()
    taken = len(window.state.frames)
    assert 0 < taken <= DRAIN_MOST and len(window.subscription) == 1500 - taken
    rows = window.state.frames.since()[:, 0]
    assert np.all(np.diff(rows) > 0) and rows[0] == 0.0  # oldest first, none skipped
    assert window.canvas.perf.counts.get("frames that left messages queued", 0) >= 0
    while len(window.subscription):
        window.tick()
    assert len(window.state.frames) == 1500 and window.subscription.dropped == 0
    report = window.performance()
    assert report["tick"]["samples"] >= 2 and "messages per tick" in report
    assert report["dropped by this window"] == 0


# -- sections ----------------------------------------------------------------------------------


def test_sections_can_be_hidden_shown_and_resized_and_the_visualizer_keeps_its_room(window):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    play_band(window)
    window.grab()
    assert window.layout_state() == {
        "timeline": {"shown": True, "height": None},
        "details": {"shown": False, "height": None},
        "notes": {"shown": False, "height": None},
    }
    assert set(canvas._edges) == {"timeline"}
    canvas.perf.clear()
    window.grab()
    assert "details" not in canvas.perf.series  # a hidden section is not painted at all
    window.set_section("details", shown=True)
    window.grab()
    assert set(canvas._edges) == {"timeline", "details"} and "details" in canvas.perf.series
    rows = len(window.state.activity["guitar"])
    window.set_section("timeline", shown=False)
    window.grab()
    assert set(canvas._edges) == {"details"} and len(window.state.activity["guitar"]) == rows
    # Resizing: by number, and by dragging the section's upper edge.
    window.set_section("details", height=300)
    window.grab()
    assert canvas._heights["details"] == pytest.approx(300, abs=1)
    edge = canvas._edges["details"]
    QTest.mousePress(canvas, Qt.MouseButton.LeftButton, pos=QPoint(400, round(edge)))
    QTest.mouseMove(canvas, QPoint(400, round(edge) + 60))
    QTest.mouseRelease(canvas, Qt.MouseButton.LeftButton, pos=QPoint(400, round(edge) + 60))
    window.grab()
    assert canvas._heights["details"] == pytest.approx(240, abs=3)
    # However much is asked for, the main visualizer keeps its minimum.
    window.set_section("timeline", shown=True, height=2000)
    window.set_section("details", height=2000)
    window.grab()
    main = min(canvas._edges.values())
    assert main - 58 * canvas.scale >= MIN_MAIN * canvas.scale - 30
    window.reset_layout()
    window.grab()
    assert window.layout_state()["details"] == {"shown": False, "height": None}
    assert set(canvas._edges) == {"timeline"}


# -- the application -----------------------------------------------------------------------------


def test_quick_controls_follow_settings_and_layout_survives_a_restart(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        assert list(first.chips) == ["vocals", "guitar", "bass", "drums"]
        assert all(chip.isChecked() for chip in first.chips.values())
        engine = first.controller.engine
        first.chips["bass"].click()  # the button hides the layer and the checkbox follows
        assert not first.view.canvas.layers["bass"].enabled
        assert not first.drawer.layer_controls["bass"]["shown"].isChecked()
        first.drawer.layer_controls["bass"]["shown"].setChecked(True)  # and the other way
        assert first.chips["bass"].isChecked() and first.view.canvas.layers["bass"].enabled
        first.chips["drums"].click()
        assert first.controller.engine is engine  # nothing was restarted
        assert not first.apply.isVisible()  # and no restart is asked for
        first.section_buttons["details"].click()
        first.section_buttons["timeline"].click()
        first.view.set_section("details", height=220)
        first.drawer.mode.setCurrentIndex(first.drawer.mode.findData("radial"))
        saved = first.drawer.remembered()
    finally:
        first.close()
    assert saved["sections"]["details"] == {"shown": True, "height": 220.0}
    assert saved["layers"]["drums"]["shown"] is False and saved["mode"] == "radial"
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        second.drawer.restore(saved)
        second.sync_layout_controls()
        assert second.view.layout_state() == saved["sections"]
        assert (
            second.section_buttons["details"].isChecked()
            and not second.section_buttons["timeline"].isChecked()
        )
        assert (
            not second.chips["drums"].isChecked() and not second.view.canvas.layers["drums"].enabled
        )
        assert second.view.canvas.mode == "radial"
        # Settings saved before any of this existed load with sensible defaults.
        second.drawer.restore({"source": "system", "mode": "bars", "spectrum_place": "beside"})
        second.sync_layout_controls()
        assert second.view.canvas.mode == "bars" and second.view.canvas.spectrum_place == "beside"
        second.reset_layout()
        assert second.view.layout_state()["timeline"] == {"shown": True, "height": None}
        assert (
            second.section_buttons["timeline"].isChecked()
            and not second.section_buttons["details"].isChecked()
        )
        assert "analysis keeps running" in second.chips["guitar"].toolTip()
    finally:
        second.close()
