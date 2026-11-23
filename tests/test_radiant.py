"""The Radiant radial theme: its geometry, what moves it, its palette, and that it leaves
everything else (Classic, Spectrum Bars, the analysis, the canonical colours) alone."""

import math
import os
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor, QImage

from gmv.visualizer import palette, themes
from gmv.visualizer.layers import drums as drum_layer
from gmv.visualizer.response import BASELINE, EXPRESSIVE
from gmv.visualizer.state import VisualState
from gmv.visualizer.themes import radiant as radiant_module
from gmv.visualizer.themes import radiant_geometry as geometry
from gmv.visualizer.themes.radiant import Radiant
from gmv.visualizer.themes.radiant_palette import (
    CUSTOM,
    FACTORY,
    INSTRUMENTS,
    KEYS,
    PRESETS,
    RadiantPalette,
    valid,
)

from test_instrument_layers_phase2 import ALL, activity, play_band
from test_live_segments import Stream, play_segment
from test_presentation import bass_led, led
from test_visualizer import event, frame

HOP = 0.02
WEIGHTS = (1.0, 1.25, 0.9, 1.0)  # of the four layers, outermost first
VALUES = np.linspace(0.9, 0.1, 48)
# A palette with nothing in common with the factory one, every role a hue of its own.
MIXED = dict(
    background_base="#140a02", atmosphere_primary="#7a1fa2", atmosphere_secondary="#0e7490",
    core_center="#ff2d95", core_halo="#ffe600", core_rays="#ff7a00",
    spectrum_body="#2563eb", spectrum_highlights="#f97316", primary_ring="#22c55e",
    secondary_ring="#e11d48", segmented_ring="#facc15", energy_accents="#fb923c",
    particle_accents="#ffffff", vocals="#22d3ee", guitar="#ff00aa", bass="#a3e635",
    drums="#c084fc",
)  # fmt: skip


def lay(reach=450.0, weights=WEIGHTS):
    return geometry.layout(640.0, 400.0, reach, weights, 5.0)


def picture(window):
    """What the window draws now, as an array of blue, green, red."""
    image = window.grab().toImage().convertToFormat(QImage.Format.Format_RGB32)
    data = np.frombuffer(image.constBits(), dtype=np.uint8)
    data = data.reshape(image.height(), image.bytesPerLine() // 4, 4)
    return data[:, : image.width(), :3].astype(int)


def playing(window, theme="radiant", onset=True):
    """A window on four seconds of four instruments, drawn once so that nothing moves
    between two pictures of it."""
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    window.set_mode("radial")
    window.set_radial_theme(theme)
    play_band(window)
    if onset:
        window.state.consume(event(3.95, "onset", "observed", intensity=0.9))
        window.state.consume(event(3.9, "beat_prediction", "predicted"))
    canvas.advance(0.016, perf_counter())
    window.grab()
    return canvas


# -- geometry ---------------------------------------------------------------------------------


@pytest.mark.parametrize("reach", [236.0, 396.0, 536.0, 716.0])  # 480 to 1440 pixels high
@pytest.mark.parametrize("weights", [WEIGHTS, (1.25, 1.0), (0.9,), ()])
def test_the_structures_are_concentric_in_order_and_inside_the_circle(reach, weights):
    plan = lay(reach, weights)
    push = plan.zone * 0.014  # how far a beat pushes the segmented ring
    inward = (plan.tips - plan.base) * geometry.INWARD
    order = [
        0.0,
        plan.core,
        plan.core_most,
        plan.base - plan.thickness - inward,
        plan.base,
        plan.tips,
        plan.contour - plan.swing,
        plan.contour + plan.swing,
        plan.segments - plan.depth / 2,
        plan.segments + plan.depth / 2 + push,
        plan.frame,
        *[edge for band in reversed(plan.bands) for edge in band],
        plan.rim,
        plan.reach,
    ]
    assert all(np.isfinite(order)) and order == sorted(order)
    assert all(b > a for a, b in zip(order, order[1:], strict=False)), order
    assert len(plan.bands) == len(weights)
    for (low, high), weight in zip(plan.bands, weights, strict=True):
        assert high - low >= 0.03 * reach * weight  # every instrument keeps room to draw in


def test_hidden_layers_give_their_room_to_the_others_and_none_gives_it_to_the_centre():
    four, two, none = lay(), lay(weights=(1.25, 1.0)), lay(weights=())
    span = lambda plan: plan.bands[0][1] - plan.bands[-1][0]  # noqa: E731
    assert span(two) == pytest.approx(span(four))  # the same zone, shared by fewer
    assert two.bands[0][1] - two.bands[0][0] > four.bands[1][1] - four.bands[1][0]
    assert two.zone == four.zone and none.zone > 1.5 * four.zone and none.bands == ()
    assert none.rim == four.rim  # the outer ring stays where it is


@pytest.mark.parametrize("size", [(720, 480), (1280, 800), (1920, 1080), (2560, 1440), (900, 1400)])
def test_any_window_shape_gives_a_circle_round_its_middle(window, size):
    canvas = playing(window)
    window.resize(*size)
    window.grab()
    rect = canvas.rect().adjusted(28, 58, -28, -28)
    plan = canvas.radiant.layout(canvas, rect, canvas.active_layers())
    assert (plan.cx, plan.cy) == (rect.center().x(), rect.center().y())
    assert plan.rim < min(rect.width(), rect.height()) / 2 and plan.core > 3
    view = canvas.radiant.scene(
        canvas.state, canvas.now, plan, values=VALUES, peaks=VALUES, response=EXPRESSIVE,
        running=True, keys=ALL,
    )  # fmt: skip
    for angles, radii in (view.spikes[:2], view.spikes[::2], view.peaks, view.contour):
        x, y = geometry.xy(plan, angles, radii)
        assert np.isfinite(x).all() and np.isfinite(y).all()
        assert (np.hypot(x - plan.cx, y - plan.cy) <= plan.frame + 1e-6).all()
    for _angle, start, end, width, strength, _source in view.streaks:
        assert plan.tips <= start <= end <= plan.rim and width > 0 and 0 < strength <= 1


def test_damaged_or_missing_input_still_gives_a_whole_picture():
    plan = lay()
    for values in (None, (), np.full(48, np.nan), np.array([np.inf, -3.0, 0.5] * 16)):
        angles, outer, inner = geometry.spikes(plan, values)
        assert np.isfinite(outer).all() and np.isfinite(inner).all() and len(angles) == len(outer)
        assert (outer >= plan.base).all() and (outer <= plan.tips).all()
    assert (geometry.segment_levels(None) == 0).all() and len(geometry.segment_levels(())) == 40
    angles, radii = geometry.contour(plan, None, None)
    assert (radii == plan.contour).all()  # a plain circle
    assert geometry.mix_levels(None, 1.0) == (0.0, 0.0)
    assert geometry.impulses([], None) == (0.0, 0.0, [], [])
    # Nothing running: the theme at rest, every structure there, nothing struck.
    view = Radiant().scene(
        VisualState(), None, plan, values=None, peaks=None, response=BASELINE, running=False
    )
    assert view.core == plan.core and not view.streaks and not view.shocks
    assert len(view.segments) == geometry.SEGMENTS and len(view.spikes[0]) == 2 * 48 * geometry.SUB


def test_caches_stay_small_whatever_is_changed_however_often(window):
    canvas = playing(window)
    theme = canvas.radiant
    for index in range(40):
        window.set_radiant_colour("core_halo", f"#{index * 6:02x}40ff")
        window.set_radiant_colour("spectrum_body", f"#ff{index * 6:02x}40")
        if index % 8 == 0:
            window.resize(900 + index * 10, 700)
        window.grab()
    assert len(theme._pictures) <= radiant_module.PICTURES
    assert len(canvas._glow_pictures) <= 4 and len(canvas._glow_colours) <= 513
    assert radiant_module.lit.cache_info().maxsize == 4096
    assert geometry.needles.cache_info().maxsize == 512
    for time in range(2000):
        geometry.needles(geometry.seed_of(time * 0.02, 1), 3)
    assert geometry.needles.cache_info().currsize <= 512


# -- spectrum ---------------------------------------------------------------------------------


def test_spikes_are_the_spectrum_in_its_order_and_stronger_bands_reach_further():
    plan = lay()
    values = np.array([0.1, 0.9, 0.3, 0.6] + [0.05] * 44)
    angles, outer, _ = geometry.spikes(plan, values)
    count = len(values) * geometry.SUB
    assert len(outer) == 2 * count
    own = (outer[:count:geometry.SUB] - plan.base) / (plan.tips - plan.base)
    assert own[:4] == pytest.approx(values[:4])  # a band's own spike is the band
    assert own[1] > own[3] > own[2] > own[0]
    # Lows at the top, highs at the bottom, down the right side; the left mirrors it.
    assert angles[0] < angles[count - 1] < math.pi < angles[count] < angles[-1]
    assert outer[count:] == pytest.approx(outer[:count][::-1])
    assert angles[count:] == pytest.approx((2 * math.pi - angles[:count])[::-1])
    # Louder everywhere: every spike at least as long.
    assert (geometry.spikes(plan, np.minimum(1.0, values * 1.5))[1] >= outer - 1e-9).all()


def test_the_spikes_between_two_bands_are_never_above_the_line_between_them():
    random = np.random.default_rng(3)
    for _ in range(20):
        values = random.random(48)
        lengths = geometry.spike_lengths(values)
        places = np.arange(len(lengths)) / geometry.SUB
        line = np.interp(places, np.arange(48), values)
        assert (lengths <= line + 1e-12).all()  # nothing is added to what was measured
        assert lengths[:: geometry.SUB] == pytest.approx(values)
        neighbours = np.maximum(values, np.append(values[1:], values[-1]))
        assert (lengths.reshape(48, -1).max(axis=1) <= neighbours + 1e-12).all()


def test_peak_marks_sit_on_their_own_bands_spike():
    plan = lay()
    peaks = np.zeros(48)
    peaks[5], peaks[30] = 0.8, 0.5
    angles, radii = geometry.peak_marks(plan, peaks)
    spike_angles = geometry.spikes(plan, peaks)[0]
    count = 48 * geometry.SUB
    assert len(angles) == 4  # two bands, both sides
    assert angles[0] == pytest.approx(spike_angles[5 * geometry.SUB])
    assert angles[1] == pytest.approx(spike_angles[30 * geometry.SUB])
    assert angles[2] == pytest.approx(2 * math.pi - angles[1]) and angles[2] > math.pi
    assert radii[0] > radii[1] > plan.base and count == len(spike_angles) // 2
    assert len(geometry.peak_marks(plan, None)[0]) == 0


def test_segments_are_lit_by_their_part_of_the_spectrum_and_a_beat_pushes_the_ring_out():
    values = np.zeros(48)
    values[:6] = 1.0  # only the lowest frequencies
    levels = geometry.segment_levels(values)
    assert len(levels) == geometry.SEGMENTS
    assert levels[0] > 0.9 and levels[-1] > 0.9  # top of the ring, both sides
    assert levels[10] == 0 and levels[geometry.SEGMENTS // 2] == 0  # the bottom stays dark
    plan = lay()
    rest, pushed = geometry.segment_ring(plan, 0.0), geometry.segment_ring(plan, 1.0)
    assert pushed[0] > rest[0] and pushed[1] < plan.frame
    assert rest[1] - rest[0] == pytest.approx(pushed[1] - pushed[0])
    arcs = geometry.segment_arcs()
    assert arcs.shape == (geometry.SEGMENTS, 2) and (np.diff(arcs.ravel()) > 0).all()  # gaps between


# -- what moves it ----------------------------------------------------------------------------


def stream(loud=-20, intensity=0.6, seconds=3.0):
    feed = Stream()
    for _ in range(round(seconds / HOP)):
        feed.time += HOP
        feed.state.consume(
            frame(feed.time, segment=1, loudness_dbfs=loud, dynamic_intensity=intensity),
            arrival=feed.clock + feed.time,
        )
    return feed


def scene(state, now, response=BASELINE, keys=(), beats=True, values=VALUES):
    return Radiant().scene(
        state, now, lay(), values=values, peaks=values, response=response, running=True,
        beats=beats, keys=keys,
    )  # fmt: skip


def test_the_core_follows_the_level_and_intensity_of_the_music():
    quiet, loud = stream(-50, 0.15), stream(-8, 0.9)
    low, high = scene(quiet.state, quiet.time), scene(loud.state, loud.time)
    assert high.level > low.level and high.intensity > low.intensity
    assert high.core > low.core and high.halo > low.halo and high.rays > low.rays
    plan = lay()
    assert plan.core <= low.core and high.core <= plan.core_most < plan.base
    # It is the frames at or before the display's present, and nothing after.
    early = scene(loud.state, loud.time - 1.0)
    assert early.level == pytest.approx(high.level) and early.intensity == pytest.approx(high.intensity)
    assert scene(loud.state, 0.5).level == 0.0  # before anything was heard


def test_an_onset_flashes_and_dies_away_by_audio_time_alone():
    feed = stream()
    feed.state.consume(event(3.0, "onset", "observed", intensity=1.0))
    before, at = scene(feed.state, 2.99), scene(feed.state, 3.0)
    assert before.flash == 0.0 and not before.shocks and at.flash == pytest.approx(1.0)
    assert at.core > before.core and at.rays > before.rays
    ages = [0.02, 0.1, 0.3, 0.44]
    seen = [scene(feed.state, 3.0 + age) for age in ages]
    assert [view.flash for view in seen] == pytest.approx(
        [math.exp(-age / geometry.FLASH_S) for age in ages]
    )
    assert [len(view.streaks) for view in seen] == [6, 6, 6, 6]  # the same streaks, fading
    strengths = [view.streaks[0][4] for view in seen]
    assert strengths == sorted(strengths, reverse=True)
    assert seen[0].streaks[0][0] == seen[3].streaks[0][0]  # each keeps its direction
    assert seen[3].streaks[0][1] > seen[0].streaks[0][1]  # and moves outwards
    assert not scene(feed.state, 3.0 + geometry.ONSET_LIFE + 0.01).streaks
    # The same question twice has the same answer: nothing is rolled per frame.
    again = scene(feed.state, 3.1)
    assert again.streaks == seen[1].streaks and again.shocks == seen[1].shocks


def test_stronger_events_make_more_and_longer_streaks():
    plan = lay()
    weak = geometry.streaks(plan, [(0.05, 0.37, 2.0)], {})
    strong = geometry.streaks(plan, [(0.05, 1.0, 2.0)], {})
    assert len(strong) > len(weak) > 0
    assert max(row[2] - row[1] for row in strong) > max(row[2] - row[1] for row in weak)
    assert max(row[4] for row in strong) > max(row[4] for row in weak)
    soft = geometry.streaks(plan, [], {"guitar": (1, [(0.05, 0.3, 2.0)])})
    hard = geometry.streaks(plan, [], {"guitar": (1, [(0.05, 0.95, 2.0)])})
    assert len(hard) == 3 and len(soft) == 1 and {row[5] for row in hard} == {"guitar"}
    assert all(plan.frame <= row[1] <= row[2] <= plan.rim for row in hard + soft)
    many = geometry.streaks(plan, [(0.01, 1.0, 2.0 + i * 0.02) for i in range(40)], {})
    assert len(many) == geometry.MOST_STREAKS


def test_nothing_is_struck_that_the_analysis_did_not_report():
    feed = stream()
    for now in np.arange(1.0, 4.0, 0.013):
        view = scene(feed.state, float(now))
        assert view.flash == 0.0 and view.pulse == 0.0 and not view.streaks and not view.shocks
    # A beat is only a predicted beat that has been reached, and only when beats are shown.
    feed.state.consume(event(3.5, "beat_prediction", "predicted"))
    feed.state.consume(event(3.2, "onset", "context", analysis_tier="advanced"))  # delayed context
    feed.state.consume(event(3.2, "energy_rise", "observed"))  # not an onset
    assert scene(feed.state, 3.4).pulse == 0.0
    assert scene(feed.state, 3.5).pulse == pytest.approx(1.0)
    assert scene(feed.state, 3.5, beats=False).pulse == 0.0
    assert scene(feed.state, 3.3).flash == 0.0 and not scene(feed.state, 3.3).streaks
    ring = scene(feed.state, 3.5).segment_ring
    assert ring[0] > scene(feed.state, 3.5, beats=False).segment_ring[0]


def test_instrument_attacks_strike_in_the_instruments_own_band_and_only_real_ones():
    feed = Stream()
    play_segment(feed, 1, 138)  # ends just after a bass pulse and a drum hit
    state, now = feed.state, feed.time
    for live in state.live.values():
        live.advance(now, 1.0, False)
    keys = ("vocals", "guitar", "bass", "drums")
    view = scene(state, now, keys=keys)
    sources = {row[5] for row in view.streaks}
    assert sources <= {"bass", "drums", "vocals"} and "bass" in sources and "drums" in sources
    plan = lay()
    for _angle, start, end, _width, _strength, source in view.streaks:
        low, high = plan.bands[keys.index(source)]
        assert start < high and end > low  # it crosses its own band
    # A hidden instrument strikes nothing; one the presence model does not hear, nothing.
    assert {row[5] for row in scene(state, now, keys=("drums",)).streaks} == {"drums"}
    state.live["drums"].weight = 0.0
    assert "drums" not in {row[5] for row in scene(state, now, keys=keys).streaks}


def test_a_new_segment_starts_with_nothing_left_over():
    feed = Stream()
    play_segment(feed, 1, 100)
    feed.state.consume(event(feed.time - 0.02, "onset", "observed", intensity=1.0))
    for live in feed.state.live.values():
        live.advance(feed.time, 1.0, False)
    keys = ("vocals", "guitar", "bass", "drums")
    before = scene(feed.state, feed.time, keys=keys)
    assert before.flash > 0.5 and before.streaks and before.shocks
    assert feed.state.segment_start is None
    feed.frame(2)  # the next frame is of a new segment; the timestamps carry on
    assert feed.state.segment_start == pytest.approx(feed.time)
    for live in feed.state.live.values():
        live.advance(feed.time, 1.0, False)
    after = scene(feed.state, feed.time, keys=keys)
    assert after.flash == 0.0 and not after.streaks and not after.shocks
    feed.state.reset()
    assert feed.state.segment_start is None


def test_the_contour_is_the_envelope_of_the_last_seconds_newest_at_the_top():
    feed = Stream()
    for index in range(200):
        feed.time += HOP
        loud = index >= 150  # the last second is loud, with strong onsets
        feed.state.consume(
            frame(feed.time, segment=1, loudness_dbfs=-6 if loud else -50, onset_strength=0.9 if loud else 0.0),
            arrival=feed.clock + feed.time,
        )
    plan = lay()
    angles, radii = geometry.contour(plan, feed.state.frames, feed.time)
    swing = np.abs(radii - plan.contour)
    half = len(angles) // 2
    assert (swing <= plan.swing + 1e-9).all() and (np.diff(angles) >= 0).all()
    recent = swing[(angles > 0.02) & (angles < math.pi / 3 - 0.05)]  # the newest second
    older = swing[(angles > math.pi / 3 + 0.05) & (angles < math.pi - 0.02)]
    assert recent.min() > 4 * older.max() and older.max() < 0.25 * plan.swing
    assert radii[:half] == pytest.approx(radii[half:][::-1])  # the left side mirrors it
    sides = np.sign(radii[1:40] - plan.contour)
    assert (sides[:-1] != sides[1:]).all()  # in and out by turns: a waveform's look
    # It is not the spike ring over again: the spectrum does not move it at all.
    assert np.array_equal(geometry.contour(plan, feed.state.frames, feed.time)[1], radii)
    later = geometry.contour(plan, feed.state.frames, feed.time + 0.5)[0]
    assert later[1] == pytest.approx(angles[1] + math.pi * 0.5 / geometry.CONTOUR_S)  # it flows


def test_expressive_mode_draws_the_same_events_larger_and_stays_in_bounds():
    feed = Stream()
    play_segment(feed, 1, 150)
    feed.state.consume(event(feed.time - 0.04, "onset", "observed", intensity=0.8))
    for live in feed.state.live.values():
        live.advance(feed.time, 1.0, False)
    keys = ("vocals", "guitar", "bass", "drums")
    plain = scene(feed.state, feed.time, BASELINE, keys)
    bold = scene(feed.state, feed.time, EXPRESSIVE, keys)
    plan = lay()
    assert (bold.spikes[1] >= plain.spikes[1]).all() and bold.spikes[1].sum() > plain.spikes[1].sum()
    assert bold.spikes[1].max() <= plan.tips
    assert bold.core > plain.core and bold.core <= plan.core_most
    wide = lambda view: np.abs(view.contour[1] - plan.contour)  # noqa: E731
    assert wide(bold).sum() > wide(plain).sum() and wide(bold).max() <= plan.swing + 1e-9
    assert len(bold.streaks) == len(plain.streaks)  # the same events
    assert [row[0] for row in bold.streaks] == [row[0] for row in plain.streaks]
    assert sum(row[2] - row[1] for row in bold.streaks) > sum(row[2] - row[1] for row in plain.streaks)
    assert (bold.flash, bold.pulse, bold.level) == (plain.flash, plain.pulse, plain.level)


# -- the same at any frame rate ---------------------------------------------------------------


def messages(seconds=3.2):
    found = []
    for index in range(round(seconds / HOP)):
        time = round(1.0 + index * HOP, 6)
        beat = index % 25
        found.append(frame(time, segment=1, loudness_dbfs=-30 + 15 * (beat < 5), onset_strength=0.8 * (beat == 0)))
        if beat == 0:
            found.append(event(time, "onset", "observed", intensity=0.9))
        if beat == 12:
            found.append(event(time, "beat_prediction", "predicted"))
        found.append(activity(time - 0.08, "bass", level=0.6, energy=0.6, lead_s=0.08, segment=1,
                              pitch_midi=33.0, pulse=0.9 * (beat == 5)))  # fmt: skip
        found.append(activity(time - 0.06, "drums", attack=0.9 * (beat == 10), flux=0.9 * (beat == 10),
                              components=(("low", 0.9 * (beat == 10)), ("mid", 0.0), ("high", 0.0)), segment=1))  # fmt: skip
        found.append(activity(time - 0.02, "guitar", attack=0.7 * (beat == 18), centre=0.4, spread=0.5, segment=1))
        found.append(activity(time, "vocals", level=0.7, pitch_midi=60.0, boundary="start" if index == 120 else "", segment=1))
    return found


def at_rate(rate, target):
    """The scene at audio time ``target`` when the display has been stepping at ``rate``."""
    state, waiting, keys = VisualState(), messages(), ("vocals", "guitar", "bass", "drums")
    base, tick = 1000.0, 0
    while True:
        now = 1.0 + tick / rate
        while waiting and waiting[0].timestamp <= now:
            message = waiting.pop(0)
            state.consume(message, arrival=base + message.timestamp)
        for live in state.live.values():
            live.advance(now, 1.0, False)
        if now >= target - 1e-9:
            return scene(state, now, keys=keys)
        tick += 1


def test_the_same_recording_gives_the_same_scene_at_any_frame_rate():
    for target in (3.0, 3.5, 4.0):  # whole numbers of frames at all four rates
        first = at_rate(60, target)
        assert first.streaks and first.contour[1].std() > 0  # there is something to compare
        for rate in (72, 120, 144):
            other = at_rate(rate, target)
            for name in ("level", "intensity", "musical", "flash", "pulse", "core", "halo", "rays", "ray_turn"):
                assert getattr(other, name) == pytest.approx(getattr(first, name), abs=1e-9), (name, rate)
            for mine, theirs in zip(first.spikes + first.contour + first.peaks, other.spikes + other.contour + other.peaks, strict=True):
                assert np.allclose(mine, theirs, atol=1e-9)
            assert np.allclose(first.segments, other.segments) and first.segment_ring == pytest.approx(other.segment_ring)
            assert len(other.streaks) == len(first.streaks) and len(other.shocks) == len(first.shocks)
            for mine, theirs in zip(first.streaks, other.streaks, strict=True):
                assert mine[5] == theirs[5] and mine[:5] == pytest.approx(theirs[:5], abs=1e-6)
            assert np.allclose(geometry.particles(lay(), other.now)[0], geometry.particles(lay(), first.now)[0])


# -- the palette ------------------------------------------------------------------------------


def test_every_role_takes_any_colour_and_changing_one_changes_no_other():
    colours = RadiantPalette()
    assert set(MIXED) == set(KEYS) and len(KEYS) == 17
    assert {key: colours[key] for key in INSTRUMENTS} == palette.INSTRUMENTS  # canonical to start
    for index, role in enumerate(KEYS):
        before, version = dict(colours.colours), colours.version
        for colour in ("#ff1493", "#00ff00", "#000000", "#ffffff", f"#{index * 15:02x}80c0"):
            assert colours.set(role, colour) and colours[role] == colour
        assert {key: value for key, value in colours.colours.items() if key != role} == {
            key: value for key, value in before.items() if key != role
        }
        assert colours.version > version and colours.preset == CUSTOM
    assert colours.set("core_center", "ABC") and colours["core_center"] == "#aabbcc"
    assert colours.set("core_center", " #FF00aa ") and colours["core_center"] == "#ff00aa"
    for bad in ("", "red", "#12", "#gggggg", "#ff00aa80", None, 255, "rgb(1,2,3)"):
        assert valid(bad) is None and not colours.set("core_center", bad)
    assert not colours.set("no_such_role", "#ffffff") and colours["core_center"] == "#ff00aa"


def test_presets_are_starting_points_and_the_factory_palette_comes_back():
    colours = RadiantPalette()
    assert colours.preset == FACTORY == "electric_blue" and len(PRESETS) == 4
    for name, (title, preset) in PRESETS.items():
        assert set(preset) == set(KEYS) and title and all(valid(c) == c for c in preset.values())
        assert colours.apply(name) and colours.preset == name and colours.colours == preset
        assert {key: preset[key] for key in INSTRUMENTS} == palette.INSTRUMENTS
    # The presets are different pictures, not one hue turned: a preset that is all one
    # colour family would have every role's hue close together.
    hues = {QColor(PRESETS["multicolor_energy"][1][role]).hue() // 40 for role in KEYS[:13]}
    assert len(hues) >= 5
    assert colours.set("segmented_ring", "#123456") and colours.preset == CUSTOM  # then anything goes
    assert colours["primary_ring"] == PRESETS["multicolor_energy"][1]["primary_ring"]
    assert not colours.apply("no_such_preset") and colours["segmented_ring"] == "#123456"
    colours.restore()
    assert colours.preset == FACTORY and colours.colours == PRESETS[FACTORY][1]


def test_saved_colours_come_back_and_damage_falls_back_role_by_role():
    colours = RadiantPalette()
    for role, colour in MIXED.items():
        colours.set(role, colour)
    saved = colours.saved()
    again = RadiantPalette()
    again.load(saved)
    assert again.colours == MIXED and again.preset == CUSTOM
    # A preset's name is only believed while the colours are that preset's.
    again.load({"preset": "aurora_green", "colours": MIXED})
    assert again.preset == CUSTOM and again.colours == MIXED
    again.load(RadiantPalette().saved() | {"preset": CUSTOM})
    assert again.preset == FACTORY
    factory = PRESETS[FACTORY][1]
    damaged = {"preset": 7, "colours": {"core_center": "#ff00aa", "core_halo": "blue", "bass": None, "extra": "#000000"}}
    again.load(damaged)
    assert again["core_center"] == "#ff00aa" and again["core_halo"] == factory["core_halo"]
    assert again["bass"] == factory["bass"] and set(again.colours) == set(KEYS)
    for nothing in (None, {}, [], "palette", {"colours": "none"}, {"colours": []}):
        again.load(nothing)
        assert again.colours == factory and again.preset == FACTORY


def test_every_role_shows_in_the_picture_and_none_is_replaced_by_a_default(window):
    canvas = playing(window)
    window.set_section("timeline", shown=False)
    window.grab()
    for role in KEYS:
        window.radiant_palette.restore()
        before = picture(window)
        # As unlike the factory colour as a colour can be: its opposite.
        factory = QColor(window.radiant_palette[role])
        other = QColor(255 - factory.red(), 255 - factory.green(), 255 - factory.blue()).name()
        assert window.set_radiant_colour(role, other)
        changed = np.abs(picture(window) - before).max(axis=2)
        assert (changed > 24).sum() >= 12, role
    assert canvas.themed() is canvas.radiant


def test_unrelated_colours_sit_side_by_side_each_where_it_belongs(window):
    canvas = playing(window)
    window.set_section("timeline", shown=False)
    for role, colour in MIXED.items():
        assert window.set_radiant_colour(role, colour)
    image = picture(window)
    rect = canvas.rect().adjusted(28, 58, -28, -28)
    plan = canvas.radiant.layout(canvas, rect, canvas.active_layers())

    def near(colour, pixel, slack=70):
        wanted = QColor(colour)
        return np.abs(np.array([wanted.blue(), wanted.green(), wanted.red()]) - pixel).max() <= slack

    cx, cy = round(plan.cx), round(plan.cy)
    assert near(MIXED["core_center"], image[cy, cx], 40)  # a hot-pink core, not a white one
    ring = image[cy - 3 : cy + 4, round(plan.cx + plan.frame) - 3 : round(plan.cx + plan.frame) + 4]
    assert any(near(MIXED["primary_ring"], pixel) for pixel in ring.reshape(-1, 3))  # green ring
    corner = image[image.shape[0] - 12, 12]
    assert corner[2] >= corner[0] and near(MIXED["background_base"], corner, 60)  # warm, dark
    # The spikes are the spectrum's blue; somewhere among their tips its orange.
    angles, outer, _ = canvas.radiant.scene(
        canvas.state, canvas.now, plan, values=canvas.display.values, peaks=canvas.display.peaks,
        response=BASELINE, running=True,
    ).spikes  # fmt: skip
    tallest = int(np.argmax(outer))
    x, y = geometry.xy(plan, angles[tallest], (plan.base + outer[tallest]) / 2)
    patch = image[round(y) - 2 : round(y) + 3, round(x) - 2 : round(x) + 3].reshape(-1, 3)
    assert any(near(MIXED["spectrum_body"], pixel, 90) for pixel in patch)
    # No role has been pulled towards another's hue.
    assert window.radiant_palette.colours == MIXED


def test_gradients_between_unlike_hues_do_not_go_dark_or_grey():
    for a, b in (("#ff1493", "#00ff66"), ("#2563eb", "#f97316"), ("#ffe600", "#7a1fa2")):
        light = lambda colour: QColor(colour).lightnessF()  # noqa: E731
        steps = [radiant_module.lit(a, b, step) for step in range(17)]
        assert steps[0] == a and steps[-1] == b
        assert min(light(step) for step in steps) >= min(light(a), light(b)) - 0.02
        assert all(max(palette.rgb(step)) >= 150 for step in steps)  # still a colour, not mud
        assert radiant_module.shade(a, 0.5, b, 0.5).alphaF() == pytest.approx(0.5, abs=0.01)


# -- the instruments --------------------------------------------------------------------------


def test_each_instrument_keeps_its_own_picture_and_colour_and_takes_a_new_one_alone(window):
    canvas = playing(window)
    tones = {key: layer.tone for key, layer in canvas.layers.items()}
    assert tones == palette.INSTRUMENTS == {key: canvas.radiant.instrument(key) for key in ALL}
    assert {key: type(layer).__name__ for key, layer in canvas.layers.items()} == {
        "vocals": "VocalsLayer", "guitar": "GuitarLayer", "bass": "BassLayer", "drums": "DrumsLayer",
    }  # fmt: skip
    plan = canvas.radiant.layout(canvas, canvas.rect().adjusted(28, 58, -28, -28), canvas.active_layers())
    assert len(plan.bands) == 4 and [layer.key for layer in canvas.active_layers()] == list(canvas.layers)
    before = picture(window)
    for key, colour in (("guitar", "#ff00aa"), ("vocals", "#00e5ff"), ("bass", "#b6ff00"), ("drums", "#b388ff")):
        others = {name: layer.tone for name, layer in canvas.layers.items() if name != key}
        assert window.set_radiant_colour(key, colour)
        window.grab()
        assert canvas.layers[key].tone == colour == window.instrument_colour(key)
        assert {name: layer.tone for name, layer in canvas.layers.items() if name != key} == others
    assert (np.abs(picture(window) - before).max(axis=2) > 40).sum() > 200
    # The drums' three regions are shades of whatever the drums' colour is.
    assert drum_layer.shades(palette.INSTRUMENTS["drums"]) is drum_layer.SHADES
    shades = drum_layer.shades("#b388ff")
    assert shades["mid"] == "#b388ff" and len(set(shades.values())) == 3
    # Nothing canonical moved, and a hidden layer gives its band away.
    assert palette.INSTRUMENTS == {"vocals": "#f472b6", "guitar": "#f5c451", "bass": "#4cc9f0", "drums": "#ff8a5b"}
    assert {key: layer.profile.colour for key, layer in canvas.layers.items()} == palette.INSTRUMENTS
    assert palette.DRUM_REGIONS["mid"] == "#ff8a5b"
    window.set_layer("guitar", enabled=False)
    window.set_layer("drums", enabled=False)
    fewer = canvas.radiant.layout(canvas, canvas.rect().adjusted(28, 58, -28, -28), canvas.active_layers())
    assert len(fewer.bands) == 2 and fewer.bands[0][1] == plan.bands[0][1]
    window.grab()
    for key in canvas.layers:
        window.set_layer(key, enabled=False)
    assert canvas.radiant.layout(canvas, canvas.rect(), canvas.active_layers()).bands == ()
    assert not window.grab().isNull()  # the theme is complete without any instrument


def test_live_states_are_what_the_layers_draw_from_in_radiant_as_in_classic(window):
    canvas = playing(window)
    bass = canvas.layers["bass"]
    radiant_time = canvas.state.live["bass"].now
    phase = canvas.state.live["bass"].phase(0.5, 3.0)
    window.set_radial_theme("classic")
    window.grab()
    assert canvas.state.live["bass"].now == radiant_time  # the same present, the same phase
    assert canvas.state.live["bass"].phase(0.5, 3.0) == phase
    assert bass.tone == palette.INSTRUMENTS["bass"]
    window.set_radial_rings("history")
    window.set_radial_theme("radiant")
    window.set_radiant_colour("bass", "#b6ff00")
    assert not window.grab().isNull() and bass.tone == "#b6ff00"  # the swept history as well


# -- other themes and modes -------------------------------------------------------------------


def test_classic_and_spectrum_bars_are_untouched_by_the_theme_and_its_colours(window):
    canvas = playing(window, theme="classic")
    assert canvas.theme == themes.DEFAULT == "classic" and canvas.themed() is None
    classic = picture(window)
    window.set_mode("bars")
    window.grab()
    bars = picture(window)
    window.set_mode("radial")
    window.set_radial_theme("radiant")
    for role, colour in MIXED.items():
        window.set_radiant_colour(role, colour)
    radiant = picture(window)
    assert (np.abs(radiant - classic).max(axis=2) > 40).mean() > 0.2  # a different picture
    window.set_mode("bars")  # the radial theme stays chosen while the bars are shown
    assert canvas.theme == "radiant" and canvas.themed() is None
    assert np.array_equal(picture(window), bars)
    window.set_mode("radial")
    assert canvas.themed() is canvas.radiant and np.array_equal(picture(window), radiant)
    window.set_radial_theme("classic")
    assert np.array_equal(picture(window), classic)
    assert window.radiant_palette.colours == MIXED  # kept while another theme is shown
    assert {key: layer.tone for key, layer in canvas.layers.items()} == palette.INSTRUMENTS
    assert palette.SPECTRUM == ("#5b62a8", "#8f8fd6", "#cfd0f2") and palette.PREDICTED == "#a3e635"
    assert window.instrument_colour("guitar") == palette.INSTRUMENTS["guitar"]
    window.set_radial_theme("no such theme")
    assert canvas.theme == "classic"


def test_changing_theme_or_colours_touches_no_stream_state(window):
    canvas = playing(window)
    state = canvas.state
    kept = (state.generation, state.live_epoch, state.segment, len(state.frames), len(state.markers),
            state.marked, dict(state.dominance.scores), state.dominance.leader, state.musical.value)  # fmt: skip
    hits = {key: list(live.hits) for key, live in state.live.items()}
    window.set_radial_theme("classic")
    window.set_radial_theme("radiant")
    window.set_radiant_preset("plasma_magenta")
    window.set_radiant_colour("drums", "#00ff00")
    window.grab()
    assert kept == (state.generation, state.live_epoch, state.segment, len(state.frames), len(state.markers),
                    state.marked, dict(state.dominance.scores), state.dominance.leader, state.musical.value)  # fmt: skip
    assert hits == {key: list(live.hits) for key, live in state.live.items()}
    assert canvas.condition == "live"


# -- the background glow ----------------------------------------------------------------------


def test_the_glow_takes_the_leaders_radiant_colour_and_scoring_never_sees_a_colour(window):
    canvas = bass_led(window)
    scores = dict(window.state.dominance.scores)
    assert window.state.dominance.leader == "bass"
    assert canvas.glow_colour() == palette.INSTRUMENTS["bass"]  # classic: the canonical colour
    window.set_radial_theme("radiant")
    assert canvas.glow_colour() == palette.INSTRUMENTS["bass"]  # Radiant's own copy, still cyan
    window.set_radiant_colour("bass", "#39ff14")
    assert canvas.glow_colour() == "#39ff14"
    window.set_radiant_colour("vocals", "#ff1493")
    assert canvas.glow_colour() == "#39ff14"  # another instrument's colour is not the leader's
    assert dict(window.state.dominance.scores) == scores and window.state.dominance.leader == "bass"
    # Nobody leading: the palette's own neutral, and the beat glow its accents.
    led(window)
    assert canvas.glow_colour() == window.radiant_palette["atmosphere_primary"]
    window.set_radiant_colour("atmosphere_primary", "#402000")
    window.set_radiant_colour("energy_accents", "#00ffcc")
    parts = {name: QColor(colour).name() for name, colour, _ in canvas.glow_parts()}
    assert parts == {"glow": "#402000", "beat glow": "#00ffcc"}
    window.set_reactive_background(False)
    led(window, bass=1.0)
    assert canvas.glow_colour() == "#402000"
    window.set_reactive_background(True)
    # Classic has its own neutral and its own beat colour again.
    window.set_radial_theme("classic")
    led(window)
    assert canvas.glow_colour() == palette.SPECTRUM[1]
    assert QColor(canvas.glow_parts()[1][1]).name() == palette.PREDICTED


def test_the_glow_moves_evenly_between_two_unrelated_colours(window):
    canvas = bass_led(window)
    window.set_radial_theme("radiant")
    window.set_radiant_colour("bass", "#39ff14")  # green
    window.set_radiant_colour("vocals", "#ff1493")  # pink
    seen = []
    for step in range(13):
        led(window, bass=1 - step / 12, vocals=step / 12)
        seen.append(palette.rgb(canvas.glow_colour()))
    assert seen[0] == palette.rgb("#39ff14") and seen[-1] == palette.rgb("#ff1493")
    path = np.array(seen, dtype=float)
    for channel in range(3):
        steps = np.diff(path[:, channel])
        assert (steps >= 0).all() or (steps <= 0).all()  # each channel goes one way
    assert path.max(axis=1).min() >= 180  # and it never passes through a dark, muddy middle
    led(window, bass=0.5, vocals=0.5)
    assert canvas.glow_colour() == canvas.glow_colour()  # the same shares, the same colour
    assert len(canvas._glow_colours) <= 40


# -- settings ---------------------------------------------------------------------------------


def test_theme_and_palette_are_set_in_the_drawer_and_remembered(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        drawer, view = first.drawer, first.view
        saved = drawer.remembered()
        assert saved["radial_theme"] == "classic" and saved["radiant_palette"]["preset"] == FACTORY
        assert view.canvas.theme == "classic" and drawer.radiant_box.isHidden()
        assert [drawer.radial_theme.itemData(i) for i in range(drawer.radial_theme.count())] == ["classic", "radiant"]
        assert set(drawer.radiant_controls) == set(KEYS)
        generation = view.state.generation
        drawer.mode.setCurrentIndex(drawer.mode.findData("radial"))
        drawer.radial_theme.setCurrentIndex(drawer.radial_theme.findData("radiant"))
        assert view.canvas.theme == "radiant" and not drawer.radiant_box.isHidden()
        assert view.state.generation == generation  # nothing restarted, nothing reset
        # A code typed into a role's field: taken at once, and the preset is now "custom".
        swatch, code = drawer.radiant_controls["guitar"]
        code.setText("#FF00AA")
        code.editingFinished.emit()
        assert view.radiant_palette["guitar"] == "#ff00aa" and code.text() == "#ff00aa"
        assert "#ff00aa" in swatch.styleSheet() and drawer.radiant_preset.currentData() == CUSTOM
        assert first.chips["guitar"].property("colour") == "#ff00aa"  # the button follows
        assert first.chips["bass"].property("colour") == palette.INSTRUMENTS["bass"]
        code.setText("not a colour")
        code.editingFinished.emit()
        assert view.radiant_palette["guitar"] == "#ff00aa" and code.text() == "#ff00aa"
        # A preset, then one colour of it changed.
        drawer.choose_radiant_preset(preset="aurora_green")
        assert view.radiant_palette.preset == "aurora_green" and drawer.radiant_preset.currentData() == "aurora_green"
        assert drawer.set_radiant_colour("core_center", "#ff1493")
        assert drawer.radiant_preset.currentData() == CUSTOM
        assert view.radiant_palette["core_halo"] == PRESETS["aurora_green"][1]["core_halo"]
        saved = drawer.remembered()
        # Spectrum bars and Classic leave the choice and the colours alone.
        drawer.mode.setCurrentIndex(drawer.mode.findData("bars"))
        assert drawer.remembered()["radial_theme"] == "radiant"
        assert first.chips["guitar"].property("colour") == palette.INSTRUMENTS["guitar"]
        drawer.radial_theme.setCurrentIndex(drawer.radial_theme.findData("classic"))
        assert drawer.remembered()["radiant_palette"] == saved["radiant_palette"]
        assert drawer.radiant_box.isHidden()
    finally:
        first.close()
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        drawer, view = second.drawer, second.view
        drawer.restore(saved)
        assert view.canvas.theme == "radiant" and view.canvas.mode == "radial"
        assert view.radiant_palette.colours == saved["radiant_palette"]["colours"]
        assert view.radiant_palette["core_center"] == "#ff1493" and view.radiant_palette.preset == CUSTOM
        assert drawer.radiant_controls["core_center"][1].text() == "#ff1493"
        # Restoring the defaults changes Radiant's palette and nothing else.
        expressive, theme = drawer.expressive.isChecked(), view.canvas.theme
        drawer.radiant_restore.click()
        assert view.radiant_palette.colours == PRESETS[FACTORY][1] and view.canvas.theme == theme
        assert drawer.expressive.isChecked() == expressive and drawer.radiant_preset.currentData() == FACTORY
        # Settings from before the themes: Classic, the factory palette.
        drawer.restore(saved)
        drawer.restore({"source": "system", "mode": "radial", "expressive": True})
        assert view.canvas.theme == "classic" and view.radiant_palette.colours == PRESETS[FACTORY][1]
        assert drawer.radiant_box.isHidden() and drawer.expressive.isChecked()
        # Damaged settings start the application all the same.
        drawer.restore({"radial_theme": "plasma", "radiant_palette": {"preset": 3, "colours": {"bass": "x", "drums": "#00ff00"}}})
        assert view.canvas.theme == "classic" and view.radiant_palette["drums"] == "#00ff00"
        assert view.radiant_palette["bass"] == palette.INSTRUMENTS["bass"]
        drawer.restore({"radial_theme": 5, "radiant_palette": "none"})
        assert view.canvas.theme == "classic" and view.radiant_palette.colours == PRESETS[FACTORY][1]
    finally:
        second.close()
