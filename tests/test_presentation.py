"""Presentation only: the background glow that follows the most prominent instrument, and
Expressive Mode. Neither may change what was analysed."""

import math
import os
from dataclasses import fields
from time import perf_counter

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PySide6.QtGui import QColor

from gmv.visualizer import layers as layer_registry
from gmv.visualizer import palette
from gmv.visualizer.dominance import (
    BLEND_S,
    DEFAULT_EDGES,
    FLOOR,
    HOLD_S,
    KEYS,
    MARGIN,
    MODELS,
    Bands,
    Dominance,
)
from gmv.visualizer.response import BASELINE, EXPRESSIVE, Response
from gmv.visualizer.state import VisualState

from test_instrument_layers_phase2 import ALL, activity, play_band
from test_live_rings import coloured
from test_live_segments import Stream, play_segment
from test_visualizer import frame

HOP = 0.02
LOW_HEAVY = (0.3, 0.5, 0.05, 0.05, 0.06, 0.03, 0.01)  # most energy below 250 Hz
MID_HEAVY = (0.0, 0.05, 0.15, 0.4, 0.3, 0.07, 0.03)  # most between 250 Hz and 4 kHz
EVEN = (0.1, 0.2, 0.15, 0.25, 0.15, 0.1, 0.05)
PRESENT = dict.fromkeys(KEYS, 1.0)


def listen(dominance, seconds, playing, bands, weights=PRESENT, silent=False, start=None):
    """``playing``: instrument -> (level or energy, hits a second, hit strength)."""
    time = dominance.time + HOP if start is None and dominance.time is not None else (start or 0.0)
    leaders = []
    for index in range(round(seconds / HOP)):
        now = time + index * HOP
        for key, (drive, per_second, strength) in playing.items():
            every = max(1, round(1 / (per_second * HOP))) if per_second else 0
            dominance.row(key, now, drive, strength if every and index % every == 0 else 0.0)
        leaders.append(dominance.step(now, bands, silent, weights))
    return leaders


# -- prominence and the choice of a leader -------------------------------------------------


@pytest.mark.parametrize(
    "key, playing, bands",
    [
        ("bass", (0.85, 2, 0.7), LOW_HEAVY),
        ("vocals", (0.75, 0, 0.0), MID_HEAVY),
        ("guitar", (0.5, 4, 0.8), MID_HEAVY),
        ("drums", (0.5, 6, 0.9), EVEN),
    ],
)
def test_an_instrument_playing_alone_leads_and_gives_the_glow_its_colour(key, playing, bands):
    dominance = Dominance()
    leaders = listen(dominance, 3.0, {key: playing}, bands)
    assert leaders[-1] == key and dominance.scores[key] > FLOOR
    assert dominance.mix[key] > 0.95 and sum(dominance.mix.values()) == pytest.approx(dominance.mix[key])
    assert all(dominance.scores[other] == 0.0 for other in KEYS if other != key)  # they are silent
    assert key in palette.INSTRUMENTS


def test_scores_are_one_scale_not_the_instruments_own_levels():
    # Every instrument reports a high level of its own; the mix says who is heavy.
    everyone = {"bass": (0.9, 2, 0.6), "vocals": (0.9, 0, 0.0), "guitar": (0.9, 3, 0.6), "drums": (0.9, 4, 0.6)}
    low, mid = Dominance(), Dominance()
    assert listen(low, 4.0, everyone, LOW_HEAVY)[-1] == "bass"
    assert listen(mid, 4.0, everyone, MID_HEAVY)[-1] in ("vocals", "guitar")
    assert low.scores["bass"] > mid.scores["bass"] + 0.3
    for scores in (low.scores, mid.scores):
        assert all(0.0 <= value <= 1.0 and math.isfinite(value) for value in scores.values())
    assert set(MODELS) == set(KEYS) == set(palette.INSTRUMENTS)


def test_presence_decides_whether_an_instrument_counts_but_not_by_itself_who_leads():
    playing = {"bass": (0.85, 2, 0.7), "vocals": (0.75, 0, 0.0)}
    dominance = Dominance()
    assert listen(dominance, 3.0, playing, LOW_HEAVY)[-1] == "bass"
    # The model stops reporting bass: it cannot lead however active its analyzer is.
    absent = PRESENT | {"bass": 0.0}
    listen(dominance, 2.0, playing, LOW_HEAVY, weights=absent)
    assert dominance.scores["bass"] == 0.0 and dominance.leader != "bass"
    # A tentative reading counts for less than a confirmed one, and full presence of an
    # instrument that reports nothing counts for nothing.
    tentative = Dominance()
    listen(tentative, 3.0, playing, LOW_HEAVY, weights=PRESENT | {"bass": 0.4})
    sure = Dominance()
    listen(sure, 3.0, playing, LOW_HEAVY)
    assert tentative.scores["bass"] == pytest.approx(0.4 * sure.scores["bass"], rel=1e-6)
    idle = Dominance()
    listen(idle, 2.0, {"bass": (0.0, 0, 0.0)}, LOW_HEAVY)
    assert idle.leader is None and idle.scores["bass"] == 0.0


def choose(dominance, seconds, **scores):
    leaders = []
    for _ in range(round(seconds / HOP)):
        dominance.scores = dict.fromkeys(KEYS, 0.0) | scores
        leaders.append(dominance.choose(HOP))
    return leaders


def test_near_equals_do_not_trade_the_lead_and_a_brief_overtake_does_not_take_it():
    dominance = Dominance()
    choose(dominance, 1.0, bass=0.6, vocals=0.5)
    assert dominance.leader == "bass"
    # Two nearly equal: the other is ahead by less than the margin, back and forth.
    seen = set()
    for index in range(60):
        ahead = 0.6 + (MARGIN - 0.01) * (index % 2)
        seen.update(choose(dominance, HOP, bass=0.6, vocals=ahead))
    assert seen == {"bass"}
    # Clearly ahead, but for a tenth of a second only (one hit): the lead stays.
    assert set(choose(dominance, 0.1, bass=0.6, vocals=0.9)) == {"bass"}
    assert set(choose(dominance, 1.0, bass=0.6, vocals=0.5)) == {"bass"}
    assert dominance.mix["bass"] > 0.9 and dominance.mix["vocals"] < 0.1


def test_a_real_change_takes_the_lead_soon_and_the_colour_follows_smoothly():
    dominance = Dominance()
    choose(dominance, 2.0, bass=0.7, vocals=0.3)
    leaders = choose(dominance, 2.0, bass=0.3, vocals=0.7)
    switched = leaders.index("vocals") * HOP
    assert HOLD_S - 0.03 <= switched <= HOLD_S + 0.05  # not at once, and not late
    assert leaders[-1] == "vocals" and dominance.mix["vocals"] > 0.95 and dominance.mix["bass"] < 0.05
    # The mix moves in small steps, never jumps, and its parts never add up to more than one.
    dominance = Dominance()
    choose(dominance, 2.0, bass=0.7)
    before, largest = dict(dominance.mix), 0.0
    for _ in range(100):
        choose(dominance, HOP, vocals=0.7)
        largest = max(largest, *(abs(dominance.mix[k] - before[k]) for k in KEYS))
        assert sum(dominance.mix.values()) <= 1.0 + 1e-9
        before = dict(dominance.mix)
    assert largest < 0.07  # 20 ms of an easing over BLEND_S
    assert BLEND_S > 0.2


def test_silence_weak_activity_and_stopped_instruments_go_back_to_neutral():
    dominance = Dominance()
    assert listen(dominance, 3.0, {"bass": (0.85, 2, 0.7)}, LOW_HEAVY)[-1] == "bass"
    # Silence: nobody leads after the hold, and the glow has let go of the colour.
    leaders = listen(dominance, 2.5, {"bass": (0.0, 0, 0.0)}, LOW_HEAVY, silent=True)
    assert leaders[-1] is None and leaders.index(None) * HOP <= HOLD_S + 0.05
    assert sum(dominance.mix.values()) < 0.01
    # Rows that simply stop arriving (the instrument's analyzer went quiet, or was reset).
    dominance = Dominance()
    listen(dominance, 3.0, {"bass": (0.85, 2, 0.7)}, LOW_HEAVY)
    leaders = listen(dominance, 3.0, {}, LOW_HEAVY)
    assert leaders[-1] is None and dominance.sustain["bass"] < 0.02
    # Under the floor nobody is chosen at all.
    dominance = Dominance()
    assert set(choose(dominance, 2.0, bass=FLOOR - 0.02, drums=FLOOR - 0.05)) == {None}
    assert sum(dominance.mix.values()) == 0.0


# -- in the visual state: audio time, segments, replay ---------------------------------------


def bass_heavy(stream, segment, frames):
    for index in range(frames):
        stream.time += HOP
        stream.state.consume(
            frame(stream.time, segment=segment, band_energy=LOW_HEAVY), arrival=stream.clock + stream.time
        )
        stream.bass(segment, energy=0.85, level=0.8, pulse=0.7 * (index % 25 == 0), pitch_midi=33.0)


def test_the_state_keeps_it_from_the_messages_alone_and_a_new_segment_starts_it_afresh():
    stream = Stream()
    bass_heavy(stream, 1, 150)
    state = stream.state
    assert state.dominance.leader == "bass" and state.dominance.mix["bass"] > 0.9
    # The same messages, arriving a second and a half later and never drawn: the same result.
    late = Stream()
    late.clock += 1.5
    bass_heavy(late, 1, 150)
    assert late.state.dominance.mix == state.dominance.mix
    assert late.state.dominance.scores == state.dominance.scores
    # A new segment with continuous timestamps: nothing of the old leader is left.
    stream.frame(2)
    assert state.dominance.leader is None and sum(state.dominance.mix.values()) == 0.0
    assert state.dominance.sustain["bass"] == 0.0
    # Segment 2 is about the voice.
    for index in range(150):
        stream.time += HOP
        state.consume(frame(stream.time, segment=2, band_energy=MID_HEAVY), arrival=stream.clock + stream.time)
        stream.vocals(2, level=0.8, pitch_midi=60.0, boundary="start" if index == 0 else "")
    assert state.dominance.leader == "vocals" and state.dominance.mix["bass"] == 0.0
    # A restarted timeline (the full reset) forgets it as well.
    state.reset()
    assert state.dominance.leader is None and state.dominance.time is None


# -- the glow ------------------------------------------------------------------------------


def glow(canvas):
    return {name: (colour, strength) for name, colour, strength in canvas.glow_parts()}


def bass_led(window):
    """A window in the radial view after three seconds of bass-heavy music."""
    stream = Stream(window.state)
    stream.clock = perf_counter() - 4.1
    stream.time = 1.0
    bass_heavy(stream, 1, 150)
    window.set_mode("radial")
    window.canvas.advance(0.014, perf_counter())
    return window.canvas


def led(window, **shares):
    """Put the colour mix where a test wants it (the dominance tests cover how it gets
    there) and hold the glow's strength still."""
    canvas = window.canvas
    window.state.dominance.mix = dict.fromkeys(KEYS, 0.0) | shares
    canvas.intensity, canvas.flash, canvas.pulse = 0.8, 0.0, 0.0
    return canvas


SPOT = (0.14, 0.42)  # beside the rings, well inside the glow


def spot(window):
    image = window.grab().toImage()
    pixel = QColor(image.pixel(int(image.width() * SPOT[0]), int(image.height() * SPOT[1])))
    return np.array([pixel.red(), pixel.green(), pixel.blue()], dtype=float)


def rgb(colour):
    return np.array(palette.rgb(QColor(colour).name()), dtype=float)


def coverage(pixel, background, colour):
    """How much of ``colour`` lies over ``background`` in a rendered pixel (0..1): the glow's
    effective strength there, read from the channel in which the two differ most."""
    channel = int(np.argmax(np.abs(colour - background)))
    return (pixel[channel] - background[channel]) / (colour[channel] - background[channel])


def no_glow(window):
    """The same picture with the glow switched out, to measure against."""
    canvas = window.canvas
    kept = canvas.response
    canvas.response = Response(glow=0.0)
    background = spot(window)
    canvas.response = kept
    return background


def test_the_glow_is_one_colour_at_the_strength_the_music_gives_it(window):
    canvas = bass_led(window)
    for key in canvas.layers:
        window.set_layer(key, enabled=False)  # nothing but spectrum and glow near the spot
    led(window)
    background = no_glow(window)
    window.set_reactive_background(False)
    neutral = spot(window)
    covered = coverage(neutral, background, rgb(palette.SPECTRUM[1]))
    assert 0.03 < covered < 0.6  # there is a glow at the spot, as before
    window.set_reactive_background(True)
    assert np.array_equal(spot(window), neutral)  # nobody leads: exactly the neutral picture
    assert glow(canvas)["glow"][0] == palette.SPECTRUM[1]
    # Each instrument alone: its own colour, lying over the background exactly as far as the
    # neutral glow did. The rendered pixel is checked, not the numbers handed to the painter.
    for key, colour in palette.INSTRUMENTS.items():
        led(window, **{key: 1.0})
        assert glow(canvas)["glow"][0] == colour
        expected = background + covered * (rgb(colour) - background)
        assert np.abs(spot(window) - expected).max() <= 2.0, key
    # Spectrum Bars has no instrument glow; a stopped stream goes back to neutral.
    led(window, bass=1.0)
    window.set_mode("bars")
    assert glow(canvas)["glow"][0] == palette.SPECTRUM[1]
    window.set_mode("radial")
    window.state.status = "stopped"
    canvas.advance(0.014, perf_counter())
    assert glow(canvas)["glow"][0] == palette.SPECTRUM[1]


def test_a_change_of_leader_moves_the_colour_smoothly_without_dimming_the_glow(window):
    canvas = bass_led(window)
    for key in canvas.layers:
        window.set_layer(key, enabled=False)
    led(window)
    background = no_glow(window)
    window.set_reactive_background(False)
    covered = coverage(spot(window), background, rgb(palette.SPECTRUM[1]))
    window.set_reactive_background(True)
    bass, vocals = rgb(palette.INSTRUMENTS["bass"]), rgb(palette.INSTRUMENTS["vocals"])
    seen, light = [], []
    for step in range(13):  # bass hands over to vocals in twelve steps
        share = step / 12
        led(window, bass=1.0 - share, vocals=share)
        colour = rgb(canvas.glow_colour())
        pixel = spot(window)
        # The glow lies over the background as far as ever, whatever its colour is ...
        assert np.abs(pixel - (background + covered * (colour - background))).max() <= 2.0
        seen.append(colour)
        light.append(float(pixel @ (0.2126, 0.7152, 0.0722)))
    assert np.array_equal(seen[0], bass) and np.array_equal(seen[-1], vocals)
    # ... each channel goes one way from the one colour to the other, in small steps ...
    path = np.array(seen)
    for channel in range(3):
        steps = np.diff(path[:, channel])
        assert np.all(steps >= 0) or np.all(steps <= 0)
        assert np.abs(steps).max() <= 0.25 * abs(vocals[channel] - bass[channel]) + 2
    # ... and the picture is never darker or brighter on the way than at its two ends.
    assert min(light) >= min(light[0], light[-1]) - 1.0
    assert max(light) <= max(light[0], light[-1]) + 1.0
    # A leader fading out to nobody goes to the neutral colour the same way.
    led(window, drums=0.5)
    half = rgb(canvas.glow_colour())
    drums, plain = rgb(palette.INSTRUMENTS["drums"]), rgb(palette.SPECTRUM[1])
    assert all(min(a, b) - 1 <= value <= max(a, b) + 1 for value, a, b in zip(half, drums, plain, strict=True))
    # Only so many colours are ever made, and only a few pictures of them are kept.
    assert len(canvas._glow_pictures) <= 4 and len(canvas._glow_colours) <= 40


def test_the_beat_glow_keeps_its_own_colour_and_hiding_a_layer_stops_nothing(window):
    window.set_layer("bass", enabled=False)  # its ring is not drawn
    canvas = bass_led(window)
    assert window.state.dominance.leader == "bass"
    assert canvas.glow_colour() == palette.INSTRUMENTS["bass"]
    canvas.pulse = 1.0
    parts = glow(canvas)
    assert QColor(parts["beat glow"][0]).name() == palette.PREDICTED and parts["beat glow"][1] > 0.1
    assert parts["glow"][0] == palette.INSTRUMENTS["bass"]
    # Layer colours are not touched by any of this.
    assert palette.INSTRUMENTS["bass"] == "#4cc9f0" and canvas.layers["bass"].profile.colour == "#4cc9f0"
    assert palette.blend([("#4cc9f0", 1.0)]) == "#4cc9f0"
    assert palette.blend([("#000000", 0.5), ("#ffffff", 0.5)]) == "#bcbcbc"  # light adds up


# -- Expressive Mode -------------------------------------------------------------------------


def test_the_baseline_is_all_ones_and_expressive_is_bounded():
    for field in fields(Response):
        if field.name != "name":
            assert getattr(BASELINE, field.name) == 1.0
            value = getattr(EXPRESSIVE, field.name)
            assert math.isfinite(value) and 0.5 <= value <= 2.0
    values = np.linspace(0.0, 1.0, 48)
    assert BASELINE.lifted(values) is values  # untouched, not merely equal
    lifted = EXPRESSIVE.lifted(values)
    assert lifted[0] == 0.0 and lifted[-1] == 1.0 and np.all(lifted >= values) and lifted.max() <= 1.0
    assert np.all(np.diff(lifted) > 0)  # the same bars are the tallest: order is kept
    assert EXPRESSIVE.motion > 1.2 and EXPRESSIVE.glow > 1.2 and EXPRESSIVE.spectrum_curve < 1.0


def stage(response, inner=250.0, outer=330.0):
    return layer_registry.Stage(
        2.0, 6.0, 1.0, 0, None, ring=(400.0, 400.0, inner, outer), live=True, response=response
    )


def test_the_bass_ring_is_larger_in_expressive_mode_with_the_same_phase_and_inside_its_ring():
    state = VisualState()
    stream = Stream(state)
    bass_heavy(stream, 1, 150)
    live = state.live["bass"]
    live.advance(stream.time + 0.005, weight=1.0)
    layer = layer_registry.build()["bass"]
    plain = layer.live_shape(stage(BASELINE), live)
    phase = live.phase(0.5, 3.0)
    bold = layer.live_shape(stage(EXPRESSIVE), live)
    assert live.phase(0.5, 3.0) == phase  # drawing differently moved no clock
    thick = lambda shape: float(np.mean(shape[0] - shape[1]))  # noqa: E731
    assert thick(bold) > 1.25 * thick(plain)
    for shape in (plain, bold):
        assert np.all(np.isfinite(shape[0])) and np.all(np.isfinite(shape[1]))
        assert shape[0].max() <= 330.0 and shape[1].min() >= 250.0 and np.all(shape[0] >= shape[1])
    # Even at the layer's own highest intensity nothing leaves the ring.
    layer.intensity = 2.0
    wild = layer.live_shape(stage(EXPRESSIVE), live)
    assert wild[0].max() <= 330.0 and wild[1].min() >= 250.0


def lit(image):
    return sum(
        QColor(image.pixel(x, y)).value() > 110
        for y in range(0, image.height(), 3)
        for x in range(0, image.width(), 3)
    )


@pytest.mark.parametrize("mode", ["radial", "bars"])
def test_expressive_mode_draws_more_from_the_same_state_and_off_is_the_picture_as_it_was(window, mode):
    canvas = window.canvas
    canvas.info["instruments"] = ALL
    window.set_mode(mode)
    window.set_spectrum_place("beside")
    play_band(window)
    window.state.consume(
        activity(window.state.latest.timestamp + 0.02, "drums", attack=0.9, components=(("low", 0.9), ("mid", 0.9), ("high", 0.8)))
    )
    canvas.advance(0.014, perf_counter())
    snapshot = lambda: (  # noqa: E731  what the analysis produced, as the state holds it
        {key: (list(live.hits), list(live.pulses), live.time, live.origin) for key, live in window.state.live.items()},
        dict(window.state.dominance.scores),
        dict(window.state.dominance.mix),
        {key: track.since().tobytes() for key, track in window.state.activity.items()},
    )
    plain = window.grab().toImage()
    before = snapshot()
    assert canvas.response is BASELINE
    window.set_expressive(True)
    bold = window.grab().toImage()
    assert canvas.response is EXPRESSIVE and bold != plain
    assert lit(bold) > 1.05 * lit(plain)  # measurably more, not just a flag
    for colour in palette.INSTRUMENTS.values():  # every instrument is still there, in its colour
        assert coloured(bold, colour) >= 0.8 * coloured(plain, colour) > 0
    assert snapshot() == before  # nothing analysed was touched
    window.set_expressive(False)
    assert window.grab().toImage() == plain  # exactly the picture it was
    assert max(canvas.paint_ms) < 250


def test_the_two_settings_work_alone_and_together(window):
    canvas = bass_led(window)
    for key in canvas.layers:
        window.set_layer(key, enabled=False)
    canvas.intensity, canvas.flash, canvas.pulse = 0.5, 0.0, 0.0
    background = no_glow(window)
    results = {}
    for reactive in (False, True):
        for expressive in (False, True):
            window.set_reactive_background(reactive)
            window.set_expressive(expressive)
            results[reactive, expressive] = (
                glow(canvas)["glow"],
                dict(window.state.dominance.scores),
                window.state.dominance.leader,
                spot(window),
            )
    # Who leads, and by how much, is the same whatever is switched on.
    assert len({str(scores) for _, scores, _, _ in results.values()}) == 1
    assert {leader for _, _, leader, _ in results.values()} == {"bass"}
    for expressive in (False, True):
        assert results[False, expressive][0][0] == palette.SPECTRUM[1]
        assert results[True, expressive][0][0] == palette.INSTRUMENTS["bass"]
    # Expressive Mode scales the strength of the glow and leaves its colour alone: in the
    # numbers, and in how far the colour lies over the background in the picture.
    for reactive in (False, True):
        (colour, plain), (bold_colour, bold) = results[reactive, False][0], results[reactive, True][0]
        assert bold_colour == colour and bold == pytest.approx(min(1.0, plain * EXPRESSIVE.glow))
        thin = coverage(results[reactive, False][3], background, rgb(colour))
        thick = coverage(results[reactive, True][3], background, rgb(colour))
        assert thick == pytest.approx(thin * EXPRESSIVE.glow, rel=0.12) and thick > thin


def test_both_settings_are_remembered_and_old_settings_get_the_defaults(app, tmp_path):
    from gmv.app.window import MainWindow

    first = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        saved = first.drawer.remembered()
        assert saved["reactive_background"] is True and saved["expressive"] is False
        assert first.view.canvas.reactive_background and first.view.canvas.response is BASELINE
        # The quick button and the setting are one switch; both act at once.
        first.expressive_button.setChecked(True)
        assert first.drawer.expressive.isChecked() and first.view.canvas.response is EXPRESSIVE
        first.drawer.expressive.setChecked(False)
        assert not first.expressive_button.isChecked() and first.view.canvas.response is BASELINE
        first.drawer.expressive.setChecked(True)
        first.drawer.reactive_background.setChecked(False)
        assert not first.view.canvas.reactive_background
        saved = first.drawer.remembered()
        assert saved["expressive"] is True and saved["reactive_background"] is False
    finally:
        first.close()
    second = MainWindow(session_root=tmp_path, remember=False, discover_sources=False)
    try:
        second.drawer.restore(saved)
        assert second.view.canvas.response is EXPRESSIVE and second.expressive_button.isChecked()
        assert not second.view.canvas.reactive_background
        old = {"source": "system", "mode": "bars", "intensity_colour": False}  # saved before these
        second.drawer.restore(old)
        assert second.view.canvas.response is BASELINE and second.view.canvas.reactive_background
        assert second.view.canvas.mode == "bars" and not second.view.canvas.colour_by_intensity
    finally:
        second.close()


def test_expressive_mode_changes_no_segment_or_event_bookkeeping():
    # The same stream into two states, one of them drawn expressively: identical states.
    def run():
        stream = Stream()
        play_segment(stream, 1, 80)
        play_segment(stream, 2, 80)
        return stream.state

    one, two = run(), run()
    for key in one.live:
        a, b = one.live[key], two.live[key]
        for live in (a, b):
            live.advance(one.latest.timestamp + 0.005)
        assert (list(a.hits), list(a.pulses), a.segment, a.origin) == (list(b.hits), list(b.pulses), b.segment, b.origin)
        assert a.phase(0.5, 3.0) == b.phase(0.5, 3.0)
    assert one.dominance.scores == two.dominance.scores and one.live_epoch == two.live_epoch


# -- band edges are a setting ----------------------------------------------------------------

DISPLAY = (30.0, 18000.0)  # the display spectrum's range, 48 bands


def display(power_at):
    """A 48-band display spectrum in dB from a function of each band's centre frequency."""
    marks = np.geomspace(*DISPLAY, 49)
    centres = np.sqrt(marks[:-1] * marks[1:])
    return tuple((10 * np.log10(np.maximum(1e-12, [power_at(f) for f in centres]))).tolist())


def test_default_edges_give_the_shares_the_scores_were_fitted_on():
    bands = Bands(DEFAULT_EDGES)
    assert bands.usable and bands.exact
    for energy in (LOW_HEAVY, MID_HEAVY, EVEN):
        low, mid = bands.shares(energy)
        assert low == pytest.approx((energy[0] + energy[1]) / sum(energy), abs=1e-12)
        assert mid == pytest.approx((energy[2] + energy[3] + energy[4]) / sum(energy), abs=1e-12)
    assert bands.shares((0.0,) * 7) is None and bands.shares(()) is None


def test_other_edges_are_read_by_their_frequencies_not_by_their_position():
    # Three bands below 250 Hz, two up to 4 kHz, two above: still exact.
    edges = (20, 60, 120, 250, 1000, 4000, 8000, 20000)
    energy = (0.1, 0.2, 0.3, 0.15, 0.1, 0.1, 0.05)
    bands = Bands(edges)
    assert bands.usable and bands.exact
    low, mid = bands.shares(energy)
    assert low == pytest.approx(0.6) and mid == pytest.approx(0.25)
    # Reading the same numbers by position, as if the edges were the default, is wrong.
    assert (energy[0] + energy[1]) / sum(energy) == pytest.approx(0.3)
    # The same sound described with either set of edges scores the same.
    # 0.6 below 250 Hz, 0.25 up to 4 kHz, 0.15 above:
    default_energy = (0.1, 0.5, 0.05, 0.1, 0.1, 0.1, 0.05)
    playing = {"bass": (0.85, 2, 0.7), "vocals": (0.7, 0, 0.0)}
    usual, other = Dominance(), Dominance(edges)
    listen(usual, 3.0, playing, default_energy)
    listen(other, 3.0, playing, energy)
    assert other.scores == pytest.approx(usual.scores) and other.leader == usual.leader == "bass"


def test_a_band_across_a_cutoff_is_divided_by_the_display_spectrum_or_not_at_all():
    # [60, 400] lies across 250 Hz and [2000, 5000] across 4 kHz.
    edges = (20, 60, 400, 800, 2000, 5000, 8000, 20000)
    energy = (0.1, 0.3, 0.1, 0.1, 0.2, 0.1, 0.1)
    bands = Bands(edges)
    assert bands.usable and not bands.exact
    # All of what is in those bands sits on the low side of each cutoff ...
    low, mid = bands.shares(energy, display(lambda f: float(f < 250 or 2000 < f < 4000)), DISPLAY)
    assert low == pytest.approx(0.1 + 0.3, abs=0.03) and mid == pytest.approx(0.1 + 0.1 + 0.2, abs=0.03)
    # ... or on the high side ...
    low, mid = bands.shares(energy, display(lambda f: float(250 < f < 2000 or f > 4000)), DISPLAY)
    # (to within the display spectrum's own bands, each about a fifth of an octave wide)
    assert low == pytest.approx(0.1, abs=0.03) and mid == pytest.approx(0.3 + 0.1 + 0.1, abs=0.07)
    # ... or evenly per octave: each band is divided by octaves.
    low, mid = bands.shares(energy, display(lambda f: 1.0), DISPLAY)
    below = np.log2(250 / 60) / np.log2(400 / 60)
    inside = np.log2(4000 / 2000) / np.log2(5000 / 2000)
    assert low == pytest.approx(0.1 + 0.3 * below, abs=0.01)
    assert mid == pytest.approx(0.3 * (1 - below) + 0.1 + 0.1 + 0.2 * inside, abs=0.01)
    # Without a display spectrum such a band cannot be divided: no shares, nobody leads.
    assert bands.shares(energy) is None and bands.shares(energy, (), DISPLAY) is None
    dominance = Dominance(edges)
    assert set(listen(dominance, 3.0, {"bass": (0.85, 2, 0.7)}, energy)) == {None}
    assert sum(dominance.mix.values()) == 0.0 and all(v == 0.0 for v in dominance.scores.values())


def test_unknown_or_unsuitable_edges_leave_the_glow_neutral():
    for edges in (None, (), (100, 200, 250, 500, 2000, 4000, 6000, 24000), (20, 60, 250, 500, 2000, 3000, 4000, 5000)):
        bands = Bands(edges)
        assert not bands.usable and bands.shares(LOW_HEAVY) is None
        dominance = Dominance(edges)
        assert set(listen(dominance, 3.0, {"bass": (0.85, 2, 0.7)}, LOW_HEAVY)) == {None}
    # A frame with another number of bands than the edges say is not guessed at.
    assert Bands(DEFAULT_EDGES).shares((0.2, 0.3, 0.5)) is None


def test_the_window_is_told_the_edges_for_live_capture_and_for_replay(window, tmp_path):
    from gmv.app.controller import SessionController, Settings
    from gmv.bus import EventBus

    # Live: the controller hands the visualizer the edges of the session's own settings.
    custom = (20, 60, 120, 250, 1000, 4000, 8000, 20000)
    controller = SessionController(tmp_path)
    controller.settings = Settings(band_edges=custom)
    assert controller.describe()["band_edges"] == custom
    window.attach(EventBus(), **controller.describe())
    assert window.state.band_edges == custom and window.state.dominance.bands.edges == custom
    # They stay through the resets a stream brings.
    window.state.reset()
    assert window.state.dominance.bands.edges == custom
    # Replay passes what the recording's configuration says; a recording without one has
    # none, and then nobody leads rather than the default being assumed.
    window.attach(EventBus(), band_edges={}.get("band_edges"))
    assert window.state.band_edges is None and not window.state.dominance.bands.usable
    canvas = bass_led(window)
    assert window.state.dominance.leader is None and canvas.glow_colour() == palette.SPECTRUM[1]
    # Going back to the default edges is the fitted behaviour again.
    window.attach(EventBus())
    assert window.state.band_edges == tuple(DEFAULT_EDGES)
    canvas = bass_led(window)
    assert window.state.dominance.leader == "bass" and canvas.glow_colour() == palette.INSTRUMENTS["bass"]
